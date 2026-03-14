"""KIS API 기반 프로그램 매매 데이터 수집기.

장 중(09:00~15:30) N분 간격으로 구독 종목의 프로그램 매매 현황을 폴링하여 DB에 저장.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone, timedelta

from app.core.config import settings

logger = logging.getLogger(__name__)

_KST = timezone(timedelta(hours=9))
_running = False
_consecutive_failures = 0
_MAX_FAILURES = 3


def _is_market_hours() -> bool:
    """현재 KST 기준 장 시간(09:00~15:30)인지 확인."""
    now = datetime.now(_KST)
    hour_min = now.hour * 100 + now.minute
    weekday = now.weekday()
    return weekday < 5 and 900 <= hour_min <= 1530


async def _save_snapshot(
    symbol: str,
    data: dict,
    dt: datetime,
) -> None:
    """프로그램 매매 스냅샷을 DB에 저장."""
    import asyncpg

    dsn = (
        f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
        f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
    )
    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute(
            """
            INSERT INTO program_trading (symbol, dt, pgm_buy_qty, pgm_sell_qty,
                pgm_net_qty, pgm_buy_amount, pgm_sell_amount, pgm_net_amount)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (symbol, dt) DO UPDATE
            SET pgm_buy_qty = EXCLUDED.pgm_buy_qty,
                pgm_sell_qty = EXCLUDED.pgm_sell_qty,
                pgm_net_qty = EXCLUDED.pgm_net_qty,
                pgm_buy_amount = EXCLUDED.pgm_buy_amount,
                pgm_sell_amount = EXCLUDED.pgm_sell_amount,
                pgm_net_amount = EXCLUDED.pgm_net_amount
            """,
            symbol,
            dt,
            data.get("pgm_buy_qty", 0),
            data.get("pgm_sell_qty", 0),
            data.get("pgm_net_qty", 0),
            data.get("pgm_buy_amount", 0),
            data.get("pgm_sell_amount", 0),
            data.get("pgm_net_amount", 0),
        )
    finally:
        await conn.close()


async def _ensure_table() -> None:
    """테이블이 없으면 생성."""
    import asyncpg

    dsn = (
        f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
        f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
    )
    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS program_trading (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                dt TIMESTAMPTZ NOT NULL,
                pgm_buy_qty BIGINT DEFAULT 0,
                pgm_sell_qty BIGINT DEFAULT 0,
                pgm_net_qty BIGINT DEFAULT 0,
                pgm_buy_amount BIGINT DEFAULT 0,
                pgm_sell_amount BIGINT DEFAULT 0,
                pgm_net_amount BIGINT DEFAULT 0,
                CONSTRAINT uq_program_trading UNIQUE (symbol, dt)
            )
        """)
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_program_trading_lookup ON program_trading (symbol, dt)"
        )
    finally:
        await conn.close()


async def start_collector() -> None:
    """프로그램 매매 수집기 시작.

    장 시간 중 N분 간격으로 구독 종목의 프로그램 매매 데이터를 수집.
    서킷 브레이커: 연속 실패 시 지수 백오프.
    """
    global _running, _consecutive_failures
    _running = True
    _consecutive_failures = 0

    interval_minutes = getattr(settings, "PGM_TRADING_COLLECT_INTERVAL_MINUTES", 5)

    await _ensure_table()
    logger.info("프로그램 매매 수집기 시작 (간격: %d분)", interval_minutes)

    from app.trading.kis_client import get_kis_client
    client = get_kis_client(is_mock=False)  # 실전 API만 지원

    from app.services.ws_manager import manager

    while _running:
        if not _is_market_hours():
            await asyncio.sleep(60)
            continue

        # 구독 중인 종목 또는 감시 목록
        symbols = manager.get_subscribed_symbols("ticks")
        if not symbols:
            await asyncio.sleep(interval_minutes * 60)
            continue

        now = datetime.now(_KST)
        # 분 단위 truncate (스냅샷 시점)
        dt = now.replace(second=0, microsecond=0)

        success_count = 0
        for symbol in symbols:
            try:
                data = await client.inquire_program_trading(symbol)
                await _save_snapshot(symbol, data, dt)
                success_count += 1
            except Exception as e:
                logger.warning("프로그램 매매 조회 실패 (%s): %s", symbol, e)

        if success_count > 0:
            _consecutive_failures = 0
            logger.info(
                "프로그램 매매 수집 완료: %d/%d 종목", success_count, len(symbols),
            )
        else:
            _consecutive_failures += 1
            if _consecutive_failures >= _MAX_FAILURES:
                backoff = min(2 ** _consecutive_failures, 300)
                logger.warning(
                    "프로그램 매매 수집 %d연속 실패, %ds 대기",
                    _consecutive_failures, backoff,
                )
                await asyncio.sleep(backoff)
                continue

        await asyncio.sleep(interval_minutes * 60)


def stop_collector() -> None:
    """수집기 중지."""
    global _running
    _running = False
