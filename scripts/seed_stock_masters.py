"""KRX 전체 종목 마스터 데이터를 stock_masters 테이블에 삽입.

Usage:
    docker-compose run --rm app python -m scripts.seed_stock_masters
"""

from datetime import datetime, timedelta

from pykrx import stock as krx
from sqlalchemy import create_engine, text

from app.core.config import settings


def _find_recent_trading_date() -> str:
    """최근 거래일 찾기 (오늘부터 10일 전까지 탐색)."""
    dt = datetime.now()
    for i in range(10):
        d = (dt - timedelta(days=i)).strftime("%Y%m%d")
        tickers = krx.get_market_ticker_list(d, market="KOSPI")
        if tickers:
            return d
    raise RuntimeError("최근 10일 내 거래일을 찾을 수 없습니다.")


def main():
    engine = create_engine(settings.sync_database_url)
    date = _find_recent_trading_date()
    print(f"Using trading date: {date}")

    rows = []

    # KOSPI + KOSDAQ 주식
    for market in ("KOSPI", "KOSDAQ"):
        print(f"Fetching {market} tickers...")
        tickers = krx.get_market_ticker_list(date, market=market)
        print(f"  {market}: {len(tickers)} tickers")

        for ticker in tickers:
            name = krx.get_market_ticker_name(ticker)
            rows.append({"symbol": ticker, "name": name, "market": market})

    # ETF/ETN
    print("Fetching ETF tickers...")
    try:
        etf_tickers = krx.get_etf_ticker_list(date)
        print(f"  ETF: {len(etf_tickers)} tickers")
        for ticker in etf_tickers:
            name = krx.get_etf_ticker_name(ticker)
            rows.append({"symbol": ticker, "name": name, "market": "ETF"})
    except Exception as e:
        print(f"  ETF fetch failed: {e}, skipping ETFs")

    print(f"Inserting {len(rows)} stocks into stock_masters...")
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM stock_masters"))
        if rows:
            conn.execute(
                text(
                    "INSERT INTO stock_masters (symbol, name, market) "
                    "VALUES (:symbol, :name, :market)"
                ),
                rows,
            )

    kospi_count = sum(1 for r in rows if r["market"] == "KOSPI")
    kosdaq_count = sum(1 for r in rows if r["market"] == "KOSDAQ")
    etf_count = sum(1 for r in rows if r["market"] == "ETF")
    print(f"Done! Seeded {len(rows)} stocks (KOSPI: {kospi_count}, KOSDAQ: {kosdaq_count}, ETF: {etf_count})")


if __name__ == "__main__":
    main()
