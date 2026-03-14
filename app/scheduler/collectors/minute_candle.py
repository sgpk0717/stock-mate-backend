"""당일 분봉 수집 — KIS API.

전 종목의 당일 1분봉만 수집한다 (max_days=1).
종목당 최대 4페이지 (380분 / 120건).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable

from sqlalchemy import text

from app.core.database import async_session
from app.scheduler.circuit_breaker import CircuitBreaker, CircuitBreakerOpen
from app.scheduler.schemas import CollectionResult
from app.services.candle_writer import write_candles_bulk
from app.trading.kis_client import get_kis_client

logger = logging.getLogger(__name__)

ProgressCb = Callable[[int, int, str], object] | None


async def _get_all_symbols() -> list[str]:
    """stock_masters에서 전 종목 코드 조회."""
    async with async_session() as db:
        result = await db.execute(
            text("SELECT symbol FROM stock_masters ORDER BY symbol"),
        )
        return [r[0] for r in result.fetchall()]


async def _collect_today(
    client,
    symbol: str,
    date: str,
    cb: CircuitBreaker,
) -> int:
    """한 종목의 당일 분봉 수집.

    date 날짜의 분봉만 가져온다 (이전 날짜 도달 시 중단).
    3회 지수 백오프 + 서킷 브레이커.
    """
    cur_date = date
    hour = "160000"
    total = 0
    consec_errors = 0

    while True:
        try:
            candles, next_date, next_hour = await cb.call(
                client.get_minute_candles, symbol, cur_date, hour,
            )
            consec_errors = 0
        except CircuitBreakerOpen:
            raise
        except Exception as e:
            consec_errors += 1
            if consec_errors <= 3:
                await asyncio.sleep(2 ** consec_errors)
                continue
            logger.warning("%s: 3회 연속 실패, 스킵: %s", symbol, e)
            break

        if not candles:
            break

        rows = []
        for c in candles:
            dt_str = c.get("stck_bsop_date", "") + c.get("stck_cntg_hour", "")
            if len(dt_str) != 14:
                continue
            # 당일 데이터만 (다른 날짜면 제외)
            candle_date = dt_str[:8]
            if candle_date != date:
                continue
            cl = float(c.get("stck_prpr", 0) or 0)
            if cl <= 0:
                continue
            rows.append({
                "dt": dt_str,
                "open": float(c.get("stck_oprc", 0) or 0),
                "high": float(c.get("stck_hgpr", 0) or 0),
                "low": float(c.get("stck_lwpr", 0) or 0),
                "close": cl,
                "volume": int(c.get("cntg_vol", 0) or 0),
            })

        if rows:
            await write_candles_bulk(symbol, rows, "1m")
            total += len(rows)

        # 다음 페이지가 당일 이전이면 중단
        if not next_date or not next_hour:
            break
        if next_date < date:
            break
        # pagination 정체 감지
        if next_date == cur_date and next_hour == hour:
            break

        cur_date, hour = next_date, next_hour

    return total


async def collect_minute_candles(
    date: str,
    *,
    progress_cb: ProgressCb = None,
    cb: CircuitBreaker,
) -> CollectionResult:
    """전 종목 당일 분봉 수집.

    Args:
        date: YYYYMMDD (당일).
        progress_cb: 진행률 콜백.
        cb: KIS API 서킷 브레이커.
    """
    logger.info("[분봉] 수집 시작 (date=%s)", date)

    symbols = await _get_all_symbols()
    client = get_kis_client(is_mock=False)

    completed = 0
    failed = 0
    total_candles = 0

    for i, sym in enumerate(symbols):
        try:
            count = await _collect_today(client, sym, date, cb)
            total_candles += count
            completed += 1
        except CircuitBreakerOpen:
            logger.warning("[분봉] 서킷 OPEN — 나머지 %d종목 스킵", len(symbols) - i)
            return CollectionResult(
                job="minute_candle",
                total=len(symbols),
                completed=completed,
                failed=failed,
                skipped=len(symbols) - i,
                error="KIS 서킷 브레이커 OPEN",
            )
        except Exception as e:
            failed += 1
            logger.warning("[분봉] %s 실패: %s", sym, e)

        if progress_cb and (i + 1) % 50 == 0:
            await progress_cb(len(symbols), i + 1, sym)

    logger.info("[분봉] 완료: %d종목, %d건", completed, total_candles)
    return CollectionResult(
        job="minute_candle",
        total=len(symbols),
        completed=completed,
        failed=failed,
    )
