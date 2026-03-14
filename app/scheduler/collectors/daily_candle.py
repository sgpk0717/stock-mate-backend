"""일봉 수집 — pykrx 벌크 API.

전 종목의 당일 OHLCV를 한 번의 API 호출로 가져온다.
벌크 실패 시 종목별 개별 호출로 fallback.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable

from app.core.config import settings
from app.scheduler.circuit_breaker import CircuitBreaker
from app.scheduler.schemas import CollectionResult
from app.services.candle_writer import write_candles_bulk

logger = logging.getLogger(__name__)

ProgressCb = Callable[[int, int, str], object] | None


async def _bulk_fetch(date: str) -> list[dict]:
    """pykrx 종목별 호출 — 전종목 OHLCV.

    pykrx get_market_ohlcv_by_ticker(벌크)가 불안정하므로,
    stock_masters에서 종목 목록을 가져와 종목별로 조회한다.
    """
    from sqlalchemy import text as sa_text

    from app.core.database import async_session

    # stock_masters에서 종목 목록 조회
    async with async_session() as db:
        result = await db.execute(
            sa_text("SELECT symbol FROM stock_masters ORDER BY symbol"),
        )
        symbols = [r[0] for r in result.fetchall()]

    def _fetch():
        from pykrx import stock as krx

        rows = []
        for i, sym in enumerate(symbols):
            try:
                df = krx.get_market_ohlcv_by_date(date, date, sym)
                if df.empty:
                    continue
                row = df.iloc[0]
                c = float(row.get("종가", 0))
                if c <= 0:
                    continue
                rows.append({
                    "symbol": sym,
                    "dt": date,
                    "open": float(row.get("시가", 0)),
                    "high": float(row.get("고가", 0)),
                    "low": float(row.get("저가", 0)),
                    "close": c,
                    "volume": int(row.get("거래량", 0)),
                })
            except Exception:
                pass  # 개별 종목 실패는 무시
            # IP 차단 방지
            if (i + 1) % 100 == 0:
                import time
                time.sleep(1)
        return rows

    return await asyncio.to_thread(_fetch)


async def _per_stock_fallback(
    date: str,
    progress_cb: ProgressCb,
    cb: CircuitBreaker,
) -> CollectionResult:
    """벌크 실패 시 종목별 개별 호출."""
    from sqlalchemy import text

    from app.core.database import async_session

    async with async_session() as db:
        result = await db.execute(
            text("SELECT symbol FROM stock_masters ORDER BY symbol"),
        )
        symbols = [r[0] for r in result.fetchall()]

    completed = 0
    failed = 0

    for i, sym in enumerate(symbols):
        try:
            candle = await cb.call(_fetch_one, date, sym)
            if candle:
                await write_candles_bulk(sym, [candle], "1d")
                completed += 1
            else:
                completed += 1  # 데이터 없음 (거래 정지 등)
        except Exception as e:
            failed += 1
            logger.warning("일봉 fallback 실패 %s: %s", sym, e)

        if progress_cb and (i + 1) % 100 == 0:
            await progress_cb(len(symbols), completed, sym)

        await asyncio.sleep(settings.DAILY_PYKRX_THROTTLE_SEC)

    return CollectionResult(
        job="daily_candle",
        total=len(symbols),
        completed=completed,
        failed=failed,
    )


async def _fetch_one(date: str, symbol: str) -> dict | None:
    """pykrx 단일 종목 호출."""
    def _f():
        from pykrx import stock as krx

        df = krx.get_market_ohlcv_by_date(date, date, symbol)
        if df.empty:
            return None
        row = df.iloc[0]
        c = float(row.get("종가", 0))
        if c <= 0:
            return None
        return {
            "dt": date,
            "open": float(row.get("시가", 0)),
            "high": float(row.get("고가", 0)),
            "low": float(row.get("저가", 0)),
            "close": c,
            "volume": int(row.get("거래량", 0)),
        }

    return await asyncio.to_thread(_f)


async def collect_daily_candles(
    date: str,
    *,
    progress_cb: ProgressCb = None,
    cb: CircuitBreaker,
) -> CollectionResult:
    """전 종목 당일 일봉 수집.

    1차: 벌크 호출 (~5-10초).
    2차: 실패 시 종목별 fallback (~1시간).
    """
    logger.info("[일봉] 수집 시작 (date=%s)", date)

    # 벌크 시도
    try:
        rows = await cb.call(_bulk_fetch, date)

        if rows:
            completed = 0
            for r in rows:
                sym = r.pop("symbol")
                await write_candles_bulk(sym, [r], "1d")
                completed += 1

            if progress_cb:
                await progress_cb(completed, completed, rows[-1]["symbol"] if rows else "")

            logger.info("[일봉] 벌크 완료: %d종목", completed)
            return CollectionResult(
                job="daily_candle",
                total=completed,
                completed=completed,
            )
    except Exception as e:
        logger.warning("[일봉] 벌크 실패, fallback 전환: %s", e)

    # fallback
    return await _per_stock_fallback(date, progress_cb, cb)
