"""pykrx로 실제 일봉 OHLCV 데이터를 stock_candles 테이블에 삽입.

Usage:
    docker-compose run --rm app python -m scripts.seed_candles
    docker-compose run --rm app python -m scripts.seed_candles --symbol 005930 --days 365
    docker-compose run --rm app python -m scripts.seed_candles --all --days 5475
    docker-compose run --rm app python -m scripts.seed_candles --all --days 5475 --skip-existing
"""

import argparse
import time
from datetime import datetime, timedelta

from pykrx import stock as krx
from sqlalchemy import create_engine, text

from app.core.config import settings

DEFAULT_SYMBOLS = [
    "005930", "000660", "035720", "005380",
    "035420", "051910", "006400", "068270",
]


def seed_one_stock(engine, symbol: str, days: int, skip_existing: bool = False):
    if skip_existing:
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT count(*) FROM stock_candles WHERE symbol = :symbol AND interval = '1d'"),
                {"symbol": symbol},
            )
            existing = result.scalar()
            if existing and existing > 0:
                return -1  # 스킵

    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")

    df = krx.get_market_ohlcv_by_date(start, end, symbol)
    if df.empty:
        return 0

    rows = []
    for date_idx, row in df.iterrows():
        dt = date_idx.to_pydatetime()
        rows.append({
            "symbol": symbol,
            "dt": dt,
            "interval": "1d",
            "open": float(row["시가"]),
            "high": float(row["고가"]),
            "low": float(row["저가"]),
            "close": float(row["종가"]),
            "volume": int(row["거래량"]),
        })

    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM stock_candles WHERE symbol = :symbol AND interval = '1d'"),
            {"symbol": symbol},
        )
        if rows:
            conn.execute(
                text(
                    "INSERT INTO stock_candles (symbol, dt, interval, open, high, low, close, volume) "
                    "VALUES (:symbol, :dt, :interval, :open, :high, :low, :close, :volume)"
                ),
                rows,
            )
    return len(rows)


def get_all_symbols(engine) -> list[dict]:
    """stock_masters 테이블에서 전 종목 조회."""
    with engine.connect() as conn:
        result = conn.execute(text("SELECT symbol, name FROM stock_masters ORDER BY symbol"))
        return [{"symbol": row[0], "name": row[1]} for row in result.fetchall()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default=None, help="Single symbol to seed")
    parser.add_argument("--all", action="store_true", help="Seed all stocks from stock_masters")
    parser.add_argument("--days", type=int, default=365, help="Number of days of history")
    parser.add_argument("--skip-existing", action="store_true", help="Skip symbols that already have data")
    args = parser.parse_args()

    engine = create_engine(settings.sync_database_url)

    if args.all:
        all_stocks = get_all_symbols(engine)
        if not all_stocks:
            print("stock_masters is empty! Run seed_stock_masters first.")
            return
        print(f"[전 종목 일봉 수집] {len(all_stocks)}개 종목, {args.days}일")
    elif args.symbol:
        all_stocks = [{"symbol": args.symbol, "name": ""}]
    else:
        all_stocks = [{"symbol": s, "name": ""} for s in DEFAULT_SYMBOLS]

    total = 0
    skipped = 0
    failed = 0

    for i, stock in enumerate(all_stocks, 1):
        sym = stock["symbol"]
        name = stock["name"]
        try:
            count = seed_one_stock(engine, sym, args.days, skip_existing=args.skip_existing)
            if count == -1:
                skipped += 1
                if i % 100 == 0:
                    print(f"  [{i}/{len(all_stocks)}] ... (skipped {skipped})")
            elif count == 0:
                print(f"  [{i}/{len(all_stocks)}] {sym} {name}: no data")
            else:
                total += count
                print(f"  [{i}/{len(all_stocks)}] {sym} {name}: {count}건")
        except Exception as e:
            failed += 1
            print(f"  [{i}/{len(all_stocks)}] {sym} {name}: ERROR - {e}")

        time.sleep(1)  # pykrx IP 차단 방지

    print(f"\nDone! Total {total} candles, {len(all_stocks)} stocks "
          f"(skipped: {skipped}, failed: {failed})")


if __name__ == "__main__":
    main()
