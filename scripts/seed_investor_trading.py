"""투자자 주체별 순매수 데이터 시딩.

pykrx의 get_market_net_purchases_of_equities_by_ticker()를 사용하여
일별 전체 종목의 외국인/기관/개인 순매수 데이터를 수집한다.

Usage:
    docker-compose run --rm app python -m scripts.seed_investor_trading
    docker-compose run --rm app python -m scripts.seed_investor_trading --start 2023-01-01 --end 2025-12-31
    docker-compose run --rm app python -m scripts.seed_investor_trading --market KOSDAQ
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from datetime import date, datetime, timedelta

import asyncpg
from pykrx import stock

from app.core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("seed_investor")


def _dsn() -> str:
    s = settings
    return (
        f"postgresql://{s.POSTGRES_USER}:{s.POSTGRES_PASSWORD}"
        f"@{s.POSTGRES_HOST}:{s.POSTGRES_PORT}/{s.POSTGRES_DB}"
    )


async def seed(
    start_date: date,
    end_date: date,
    market: str = "KOSPI",
    batch_size: int = 500,
) -> None:
    """투자자 수급 데이터 시딩."""
    conn = await asyncpg.connect(_dsn())

    # 테이블 생성 (마이그레이션 전 실행 가능하도록)
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS investor_trading (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            dt DATE NOT NULL,
            foreign_net BIGINT DEFAULT 0,
            inst_net BIGINT DEFAULT 0,
            retail_net BIGINT DEFAULT 0,
            CONSTRAINT uq_investor_trading UNIQUE (symbol, dt)
        )
    """)
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_investor_trading_lookup ON investor_trading (symbol, dt)"
    )

    total_inserted = 0
    total_updated = 0
    current = start_date

    while current <= end_date:
        date_str = current.strftime("%Y%m%d")

        try:
            # 일별 전체 종목 외국인 순매수
            df_foreign = stock.get_market_net_purchases_of_equities_by_ticker(
                date_str, date_str, market=market, investor="외국인"
            )
            # 일별 전체 종목 기관 순매수
            df_inst = stock.get_market_net_purchases_of_equities_by_ticker(
                date_str, date_str, market=market, investor="기관합계"
            )
            # 일별 전체 종목 개인 순매수
            df_retail = stock.get_market_net_purchases_of_equities_by_ticker(
                date_str, date_str, market=market, investor="개인"
            )
        except Exception as e:
            logger.debug("Date %s fetch failed: %s", date_str, e)
            current += timedelta(days=1)
            continue

        if df_foreign.empty:
            # 비영업일
            current += timedelta(days=1)
            continue

        # 데이터 병합
        rows_to_insert = []
        for ticker in df_foreign.index:
            try:
                foreign_net = int(df_foreign.loc[ticker, "순매수"] if "순매수" in df_foreign.columns else 0)
            except (ValueError, KeyError):
                foreign_net = 0

            try:
                inst_net = int(df_inst.loc[ticker, "순매수"] if ticker in df_inst.index and "순매수" in df_inst.columns else 0)
            except (ValueError, KeyError):
                inst_net = 0

            try:
                retail_net = int(df_retail.loc[ticker, "순매수"] if ticker in df_retail.index and "순매수" in df_retail.columns else 0)
            except (ValueError, KeyError):
                retail_net = 0

            rows_to_insert.append((ticker, current, foreign_net, inst_net, retail_net))

        # 배치 upsert
        for i in range(0, len(rows_to_insert), batch_size):
            batch = rows_to_insert[i : i + batch_size]
            await conn.executemany(
                """
                INSERT INTO investor_trading (symbol, dt, foreign_net, inst_net, retail_net)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (symbol, dt) DO UPDATE
                SET foreign_net = EXCLUDED.foreign_net,
                    inst_net = EXCLUDED.inst_net,
                    retail_net = EXCLUDED.retail_net
                """,
                batch,
            )
            total_inserted += len(batch)

        logger.info(
            "Date %s: %d symbols processed (total %d)",
            date_str, len(rows_to_insert), total_inserted,
        )

        current += timedelta(days=1)
        time.sleep(0.3)  # pykrx API 쓰로틀링

    await conn.close()
    logger.info("=== Seed complete: %d rows inserted/updated ===", total_inserted)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="투자자 수급 데이터 시딩")
    parser.add_argument(
        "--start", type=str, default="2023-01-01",
        help="시작일 (YYYY-MM-DD, 기본 2023-01-01)",
    )
    parser.add_argument(
        "--end", type=str, default="",
        help="종료일 (YYYY-MM-DD, 기본 오늘)",
    )
    parser.add_argument(
        "--market", type=str, default="ALL",
        choices=["KOSPI", "KOSDAQ", "ALL"],
        help="시장 (기본 ALL)",
    )
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else date.today()

    if args.market == "ALL":
        for mkt in ["KOSPI", "KOSDAQ"]:
            logger.info("=== Seeding %s ===", mkt)
            asyncio.run(seed(start, end, market=mkt))
    else:
        asyncio.run(seed(start, end, market=args.market))
