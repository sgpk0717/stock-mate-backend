"""신용잔고 / 공매도 일별 수집 — KIS API.

당일 전 종목의 공매도 거래량 + 신용잔고를 수집한다.
- 공매도: FHPST04830000 (date~date 범위로 당일 1건)
- 신용잔고: FHPST04760000 (결제일 기준 최신 1건)

pykrx 공매도 API가 KRX 로그인 필수화(2025-12)로 작동하지 않아
KIS Open API를 통해 데이터를 수집한다.
"""

from __future__ import annotations

from app.core.timezone import KST, now_kst

import asyncio
import logging
from datetime import date as date_type, datetime, timedelta, timezone
from typing import Callable

from sqlalchemy import text

from app.core.database import async_session
from app.scheduler.circuit_breaker import CircuitBreaker, CircuitBreakerOpen
from app.scheduler.collectors import LogCb, ProgressCb, load_symbol_name_map
from app.scheduler.schemas import CollectionResult
from app.trading.kis_client import get_kis_client

logger = logging.getLogger(__name__)


async def _get_all_symbols() -> list[str]:
    """stock_masters에서 전 종목 코드 조회."""
    async with async_session() as db:
        result = await db.execute(
            text("SELECT symbol FROM stock_masters ORDER BY symbol"),
        )
        return [r[0] for r in result.fetchall()]


async def _upsert_margin_short(
    symbol: str,
    date_str: str,
    short_rows: list[dict],
    credit_rows: list[dict],
) -> int:
    """margin_short_daily에 UPSERT."""
    upserted = 0

    async with async_session() as db:
        # 공매도 데이터
        for row in short_rows:
            dt_str = row["dt"]
            if dt_str != date_str:
                continue
            await db.execute(
                text("""
                    INSERT INTO margin_short_daily
                        (symbol, dt, short_volume, short_balance_rate, collected_at)
                    VALUES (:symbol, :dt, :short_volume, :short_balance_rate, :collected_at)
                    ON CONFLICT (symbol, dt) DO UPDATE
                    SET short_volume = EXCLUDED.short_volume,
                        short_balance_rate = EXCLUDED.short_balance_rate,
                        collected_at = EXCLUDED.collected_at
                """),
                {
                    "symbol": symbol,
                    "dt": date_type(int(dt_str[:4]), int(dt_str[4:6]), int(dt_str[6:8])),
                    "short_volume": row["short_volume"],
                    "short_balance_rate": row["short_volume_ratio"],
                    "collected_at": now_kst(),
                },
            )
            upserted += 1

        # 신용잔고 데이터
        for row in credit_rows:
            dt_str = row["dt"]
            if dt_str != date_str:
                continue
            await db.execute(
                text("""
                    INSERT INTO margin_short_daily
                        (symbol, dt, margin_balance, margin_rate, collected_at)
                    VALUES (:symbol, :dt, :margin_balance, :margin_rate, :collected_at)
                    ON CONFLICT (symbol, dt) DO UPDATE
                    SET margin_balance = EXCLUDED.margin_balance,
                        margin_rate = EXCLUDED.margin_rate,
                        collected_at = EXCLUDED.collected_at
                """),
                {
                    "symbol": symbol,
                    "dt": date_type(int(dt_str[:4]), int(dt_str[4:6]), int(dt_str[6:8])),
                    "margin_balance": row["margin_balance"],
                    "margin_rate": row["margin_rate"],
                    "collected_at": now_kst(),
                },
            )
            upserted += 1

        await db.commit()

    return upserted


async def collect_margin_short(
    date: str,
    *,
    progress_cb: ProgressCb = None,
    log_cb: LogCb = None,
    cb: CircuitBreaker,
    cancel_event: asyncio.Event | None = None,
) -> CollectionResult:
    """전 종목 당일 공매도+신용잔고 수집.

    Args:
        date: YYYYMMDD (당일).
        progress_cb: 진행률 콜백.
        log_cb: UI 로그 콜백.
        cb: KIS API 서킷 브레이커.
    """
    logger.info("[신용잔고/공매도] 수집 시작 (date=%s)", date)

    symbols = await _get_all_symbols()
    name_map = await load_symbol_name_map()
    client = get_kis_client(is_mock=False)

    if log_cb:
        await log_cb(f"총 {len(symbols)}종목 대상 (공매도 + 신용잔고)")
        await log_cb(f"API: KIS 공매도(FHPST04830000) + 신용잔고(FHPST04760000)")

    # 토큰 warmup: tick_simulator 등 다른 모듈이 먼저 토큰을 발급받아
    # 1분 쿼터를 소진했을 수 있으므로, 403 시 65초 대기 후 재시도
    for attempt in range(3):
        try:
            await client._get_token()
            break
        except Exception as e:
            if "403" in str(e) and attempt < 2:
                logger.info("[신용잔고/공매도] 토큰 1분 제한 — 65초 대기 (attempt %d)", attempt + 1)
                if log_cb:
                    await log_cb(f"KIS 토큰 1분 제한 — 65초 대기 (시도 {attempt + 1}/3)")
                for remaining in range(65, 0, -10):
                    await asyncio.sleep(min(10, remaining))
                    if log_cb and remaining > 10:
                        await log_cb(f"  토큰 대기 중... {remaining - 10}초 남음")
            else:
                logger.error("[신용잔고/공매도] 토큰 발급 실패: %s", e)
                if log_cb:
                    await log_cb(f"토큰 발급 실패: {e}")
                return CollectionResult(
                    job="margin_short",
                    total=len(symbols),
                    error=f"토큰 발급 실패: {e}",
                )

    completed = 0
    failed = 0

    for i, sym in enumerate(symbols):
        if cancel_event and cancel_event.is_set():
            if log_cb:
                await log_cb("· 사용자 중단 요청 감지 — 수집 중단")
            break
        name = name_map.get(sym, sym)
        try:
            # 공매도 일별추이 (당일 1건)
            short_rows = await cb.call(
                client.inquire_daily_short_sale, sym, date, date,
            )
            await asyncio.sleep(0.12)

            # 신용잔고 일별추이 (결제일 기준 최신)
            credit_rows = await cb.call(
                client.inquire_daily_credit_balance, sym, date,
            )
            await asyncio.sleep(0.12)

            await _upsert_margin_short(sym, date, short_rows, credit_rows)
            completed += 1

            if log_cb:
                short_vol = short_rows[0].get("short_volume", 0) if short_rows else 0
                credit_bal = credit_rows[0].get("margin_balance", 0) if credit_rows else 0
                await log_cb(
                    f"  [{i+1}/{len(symbols)}] {name}({sym}) — "
                    f"공매도 {int(short_vol):,}주, 신용잔고 {int(credit_bal):,}"
                )

        except CircuitBreakerOpen:
            logger.warning(
                "[신용잔고/공매도] 서킷 OPEN — 나머지 %d종목 스킵",
                len(symbols) - i,
            )
            if log_cb:
                await log_cb(f"서킷 브레이커 OPEN — 나머지 {len(symbols) - i}종목 스킵")
            return CollectionResult(
                job="margin_short",
                total=len(symbols),
                completed=completed,
                failed=failed,
                skipped=len(symbols) - i,
                error="KIS 서킷 브레이커 OPEN",
            )
        except Exception as e:
            failed += 1
            logger.warning("[신용잔고/공매도] %s 실패: %s", sym, e)
            if log_cb:
                await log_cb(f"  [{i+1}/{len(symbols)}] {name}({sym}) — 실패: {str(e)[:60]}")

        if progress_cb and (i + 1) % 10 == 0:
            await progress_cb(len(symbols), i + 1, sym)

    logger.info("[신용잔고/공매도] 완료: %d종목 성공, %d 실패", completed, failed)
    return CollectionResult(
        job="margin_short",
        total=len(symbols),
        completed=completed,
        failed=failed,
    )
