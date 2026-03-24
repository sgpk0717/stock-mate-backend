"""프로그램 매매 일별 수집 (장후 배치) — KIS API.

장 마감 후 전 종목의 당일 프로그램 매매 데이터를 수집한다.
장중 폴링(program_trading_collector.py)에서 빠진 종목을 보충하는 역할.

- 종목별 프로그램매매추이(일별): FHPPG04650201
- /uapi/domestic-stock/v1/quotations/program-trade-by-stock-daily
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date as date_type, datetime, timedelta, timezone
from typing import Callable

from sqlalchemy import text

from app.core.database import async_session
from app.scheduler.circuit_breaker import CircuitBreaker, CircuitBreakerOpen
from app.scheduler.collectors import LogCb, ProgressCb
from app.scheduler.schemas import CollectionResult
from app.trading.kis_client import get_kis_client

logger = logging.getLogger(__name__)

_cb = CircuitBreaker(name="program_trading_daily", failure_threshold=5, reset_timeout=300)


async def _get_symbols() -> list[str]:
    """stock_masters에서 KOSPI/KOSDAQ 전 종목."""
    async with async_session() as db:
        result = await db.execute(
            text("SELECT symbol FROM stock_masters WHERE market IN ('KOSPI', 'KOSDAQ') ORDER BY symbol"),
        )
        return [r[0] for r in result.fetchall()]


async def _get_existing_symbols(target_date: str) -> set[str]:
    """이미 수집된 종목 (빠진 데이터만 보충하기 위해)."""
    async with async_session() as db:
        result = await db.execute(
            text("""
                SELECT DISTINCT symbol FROM program_trading
                WHERE dt::date = :dt AND (pgm_buy_amount != 0 OR pgm_sell_amount != 0)
            """),
            {"dt": target_date},
        )
        return {r[0] for r in result.fetchall()}


async def collect_program_trading_daily(
    target_date: date_type | None = None,
    log_cb: LogCb | None = None,
    progress_cb: ProgressCb | None = None,
) -> CollectionResult:
    """장후 프로그램 매매 일일 수집 (빠진 종목 보충).

    Args:
        target_date: 수집 대상일 (None이면 오늘)
    """
    KST = timezone(timedelta(hours=9))
    if target_date is None:
        target_date = datetime.now(KST).date()
    date_str = target_date.strftime("%Y%m%d")

    if log_cb:
        await log_cb(f"프로그램 매매 일일 수집 시작: {target_date}")

    # 전 종목 로드
    all_symbols = await _get_symbols()
    if not all_symbols:
        return CollectionResult(source="program_trading_daily", success=False, message="종목 없음")

    # 이미 수집된 종목 제외
    existing = await _get_existing_symbols(date_str[:4] + "-" + date_str[4:6] + "-" + date_str[6:8])
    missing = [s for s in all_symbols if s not in existing]

    if log_cb:
        await log_cb(f"전체 {len(all_symbols)}종목, 기존 {len(existing)}종목, 보충 {len(missing)}종목")

    if not missing:
        return CollectionResult(
            source="program_trading_daily",
            success=True,
            rows_affected=0,
            message=f"보충 필요 없음 (전체 {len(all_symbols)}종목 수집 완료)",
        )

    client = get_kis_client()
    collected = 0
    failed = 0
    total = len(missing)

    for i, symbol in enumerate(missing):
        if progress_cb and i % 50 == 0:
            await progress_cb(int((i / total) * 100))

        try:
            _cb.check()
        except CircuitBreakerOpen:
            if log_cb:
                await log_cb("서킷 브레이커 오픈 — 수집 중단")
            break

        try:
            data = await client.inquire_program_trading(symbol, date=date_str)

            # 값이 전부 0이면 스킵 (데이터 없는 종목)
            if data["pgm_buy_amount"] == 0 and data["pgm_sell_amount"] == 0:
                continue

            # DB UPSERT
            dt = datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=KST)
            async with async_session() as db:
                await db.execute(
                    text("""
                        INSERT INTO program_trading (symbol, dt, pgm_buy_amount, pgm_sell_amount,
                            pgm_net_amount, arbt_buy_amount, arbt_sell_amount,
                            nabt_buy_amount, nabt_sell_amount, collected_at)
                        VALUES (:symbol, :dt, :buy, :sell, :net, :arbt_buy, :arbt_sell,
                            :nabt_buy, :nabt_sell, NOW())
                        ON CONFLICT (symbol, dt) DO UPDATE
                        SET pgm_buy_amount = EXCLUDED.pgm_buy_amount,
                            pgm_sell_amount = EXCLUDED.pgm_sell_amount,
                            pgm_net_amount = EXCLUDED.pgm_net_amount,
                            collected_at = EXCLUDED.collected_at
                    """),
                    {
                        "symbol": symbol, "dt": dt,
                        "buy": data["pgm_buy_amount"],
                        "sell": data["pgm_sell_amount"],
                        "net": data["pgm_net_amount"],
                        "arbt_buy": data.get("arbt_buy_amount", 0),
                        "arbt_sell": data.get("arbt_sell_amount", 0),
                        "nabt_buy": data.get("nabt_buy_amount", 0),
                        "nabt_sell": data.get("nabt_sell_amount", 0),
                    },
                )
                await db.commit()
            collected += 1
            _cb.record_success()

        except Exception as e:
            failed += 1
            _cb.record_failure()
            if failed <= 3:
                logger.warning("프로그램 매매 수집 실패 (%s): %s", symbol, e)

        # Rate limit (15req/s)
        await asyncio.sleep(0.08)

    if progress_cb:
        await progress_cb(100)
    if log_cb:
        await log_cb(f"프로그램 매매 수집 완료: {collected}종목 수집, {failed}실패")

    return CollectionResult(
        source="program_trading_daily",
        success=True,
        rows_affected=collected,
        message=f"{collected}종목 수집 ({failed}실패), 기존 {len(existing)}종목 보존",
    )
