"""신용잔고 / 공매도 일별 데이터 시딩 (KIS API 기반).

pykrx 공매도 API가 KRX 로그인 필수화(2025-12)로 작동하지 않아
KIS Open API를 통해 데이터를 수집한다.

- 공매도 일별추이: FHPST04830000 (output2 배열, 90일 단위 분할)
- 신용잔고 일별추이: FHPST04760000 (output 배열, 30건/회)

주의: 앱이 실행 중이면 KIS 토큰 경쟁 발생 가능.
      `docker compose stop app` 후 실행을 권장하거나,
      앱 내 `/scheduler/trigger` POST {"job": "margin_short"} 사용.

Usage:
    docker compose exec app python -m scripts.seed_margin_short
    docker compose exec app python -m scripts.seed_margin_short --start 2024-01-01 --end 2025-12-31
    docker compose exec app python -m scripts.seed_margin_short --symbols 005930,000660
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import date, timedelta

import asyncpg
import httpx

from app.core.config import settings
from app.trading.kis_client import KISClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("seed_margin_short")


def _dsn() -> str:
    s = settings
    return (
        f"postgresql://{s.POSTGRES_USER}:{s.POSTGRES_PASSWORD}"
        f"@{s.POSTGRES_HOST}:{s.POSTGRES_PORT}/{s.POSTGRES_DB}"
    )


async def _ensure_token(kis: KISClient, max_retries: int = 3) -> None:
    """KIS 토큰 발급 (1분당 1회 제한 대응).

    403 EGW00133 에러 시 65초 대기 후 재시도.
    """
    for attempt in range(max_retries):
        try:
            await kis._get_token()
            logger.info("KIS 토큰 발급 성공")
            return
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 403 and attempt < max_retries - 1:
                logger.info(
                    "토큰 1분 제한 (EGW00133) — 65초 대기 (attempt %d/%d)",
                    attempt + 1, max_retries,
                )
                await asyncio.sleep(65)
            else:
                raise
        except Exception:
            if attempt < max_retries - 1:
                logger.info("토큰 발급 실패 — 10초 후 재시도 (attempt %d/%d)", attempt + 1, max_retries)
                await asyncio.sleep(10)
            else:
                raise


async def _get_universe(conn: asyncpg.Connection, symbols: list[str] | None) -> list[str]:
    """시딩 대상 종목 목록 결정."""
    if symbols:
        return symbols

    rows = await conn.fetch(
        "SELECT symbol FROM stock_masters WHERE market IN ('KOSPI', 'KOSDAQ') ORDER BY symbol"
    )
    return [r["symbol"] for r in rows]


async def _ensure_table(conn: asyncpg.Connection) -> None:
    """테이블 생성 (마이그레이션 전 실행 가능하도록)."""
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS margin_short_daily (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            dt DATE NOT NULL,
            margin_balance BIGINT DEFAULT 0,
            margin_rate FLOAT DEFAULT 0,
            short_volume BIGINT DEFAULT 0,
            short_balance BIGINT DEFAULT 0,
            short_balance_rate FLOAT DEFAULT 0,
            CONSTRAINT uq_margin_short UNIQUE (symbol, dt)
        )
    """)
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_margin_short_lookup ON margin_short_daily (symbol, dt)"
    )


async def _seed_short_sale(
    conn: asyncpg.Connection,
    kis: KISClient,
    symbol: str,
    start_date: date,
    end_date: date,
) -> int:
    """한 종목의 공매도 데이터를 KIS API로 수집 + DB 저장."""
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")

    try:
        rows = await kis.inquire_daily_short_sale(symbol, start_str, end_str)
    except Exception as e:
        logger.warning("  %s short sale API failed: %s", symbol, e)
        return 0

    if not rows:
        return 0

    batch = []
    for row in rows:
        dt_str = row["dt"]
        try:
            dt_val = date(int(dt_str[:4]), int(dt_str[4:6]), int(dt_str[6:8]))
        except (ValueError, IndexError):
            continue
        batch.append((
            symbol, dt_val,
            0,  # margin_balance (신용잔고는 별도 API)
            0.0,  # margin_rate
            row["short_volume"],
            0,  # short_balance
            row["short_volume_ratio"],
        ))

    if batch:
        await conn.executemany(
            """
            INSERT INTO margin_short_daily (symbol, dt, margin_balance, margin_rate,
                short_volume, short_balance, short_balance_rate)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (symbol, dt) DO UPDATE
            SET short_volume = EXCLUDED.short_volume,
                short_balance_rate = EXCLUDED.short_balance_rate
            """,
            batch,
        )

    return len(batch)


async def _seed_credit_balance(
    conn: asyncpg.Connection,
    kis: KISClient,
    symbol: str,
    end_date: date,
) -> int:
    """한 종목의 신용잔고 데이터를 KIS API로 수집 + DB 저장 (UPSERT)."""
    settle_str = end_date.strftime("%Y%m%d")

    try:
        rows = await kis.inquire_daily_credit_balance(symbol, settle_str)
    except Exception as e:
        logger.warning("  %s credit balance API failed: %s", symbol, e)
        return 0

    if not rows:
        return 0

    batch = []
    for row in rows:
        dt_str = row["dt"]
        try:
            dt_val = date(int(dt_str[:4]), int(dt_str[4:6]), int(dt_str[6:8]))
        except (ValueError, IndexError):
            continue
        batch.append((
            symbol, dt_val,
            row["margin_balance"],
            row["margin_rate"],
        ))

    if batch:
        await conn.executemany(
            """
            INSERT INTO margin_short_daily (symbol, dt, margin_balance, margin_rate,
                short_volume, short_balance, short_balance_rate)
            VALUES ($1, $2, $3, $4, 0, 0, 0)
            ON CONFLICT (symbol, dt) DO UPDATE
            SET margin_balance = EXCLUDED.margin_balance,
                margin_rate = EXCLUDED.margin_rate
            """,
            batch,
        )

    return len(batch)


async def seed(
    start_date: date,
    end_date: date,
    symbols: list[str] | None = None,
    batch_days: int = 90,
) -> None:
    """KIS API 기반 신용잔고/공매도 데이터 시딩."""
    conn = await asyncpg.connect(_dsn())
    await _ensure_table(conn)

    universe = await _get_universe(conn, symbols)
    logger.info("Universe: %d symbols", len(universe))

    # KIS 실전 클라이언트 (시세 조회는 실전 서버만 가능)
    kis = KISClient(is_mock=False)

    # 토큰 발급 (403 재시도 포함)
    await _ensure_token(kis)

    total_short = 0
    total_credit = 0

    for i, symbol in enumerate(universe):
        logger.info("[%d/%d] %s", i + 1, len(universe), symbol)

        # 1) 공매도: batch_days 단위로 분할 조회
        current = start_date
        sym_short = 0
        while current <= end_date:
            chunk_end = min(current + timedelta(days=batch_days - 1), end_date)
            cnt = await _seed_short_sale(conn, kis, symbol, current, chunk_end)
            sym_short += cnt
            current = chunk_end + timedelta(days=1)
            await asyncio.sleep(0.12)  # KIS rate limit (15req/s)

        # 2) 신용잔고: 최신 30건 (end_date 기준)
        sym_credit = await _seed_credit_balance(conn, kis, symbol, end_date)
        await asyncio.sleep(0.12)

        if sym_short or sym_credit:
            logger.info("  short=%d, credit=%d", sym_short, sym_credit)

        total_short += sym_short
        total_credit += sym_credit

    await kis.close()
    await conn.close()
    logger.info("=== Seed complete: short=%d, credit=%d ===", total_short, total_credit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="신용잔고/공매도 일별 데이터 시딩 (KIS API)")
    parser.add_argument(
        "--start", type=str, default="2020-01-01",
        help="시작일 (YYYY-MM-DD, 기본 2020-01-01)",
    )
    parser.add_argument(
        "--end", type=str, default="",
        help="종료일 (YYYY-MM-DD, 기본 오늘)",
    )
    parser.add_argument(
        "--symbols", type=str, default="",
        help="종목코드 (쉼표 구분, 기본 전체 유니버스)",
    )
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else date.today()
    sym_list = [s.strip() for s in args.symbols.split(",") if s.strip()] or None

    asyncio.run(seed(start, end, symbols=sym_list))
