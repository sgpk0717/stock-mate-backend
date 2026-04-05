"""일봉 수집 — pykrx 벌크 API.

전 종목의 당일 OHLCV를 한 번의 API 호출로 가져온다.
벌크 실패 시 종목별 개별 호출로 fallback.
"""

from __future__ import annotations

import asyncio
import logging

from app.core.config import settings
from app.scheduler.circuit_breaker import CircuitBreaker
from app.scheduler.collectors import LogCb, ProgressCb
from app.scheduler.schemas import CollectionResult
from app.services.candle_writer import write_candles_bulk

logger = logging.getLogger(__name__)


async def _bulk_fetch(date: str, log_cb=None, progress_cb=None) -> list[dict]:
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

    # progress_log: 스레드 안전 진행 메시지 큐
    import queue
    _progress_q: queue.Queue = queue.Queue()

    def _fetch():
        from pykrx import stock as krx
        import signal
        import time

        rows = []
        total = len(symbols)
        failed = 0
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
                failed += 1
            # IP 차단 방지 + 진행 로그 (50종목마다)
            if (i + 1) % 50 == 0:
                _progress_q.put(f"pykrx 진행 {i+1}/{total} (성공 {len(rows)}, 실패 {failed})")
                time.sleep(0.5)
        return rows

    # 벌크 수집을 백그라운드 스레드에서 실행하면서 진행 로그를 폴링
    import asyncio
    task = asyncio.get_event_loop().run_in_executor(None, _fetch)

    while not task.done():
        await asyncio.sleep(5)
        while not _progress_q.empty():
            msg = _progress_q.get_nowait()
            if log_cb:
                await log_cb(msg)
            if progress_cb:
                # 메시지에서 진행률 추출
                try:
                    parts = msg.split("/")
                    current = int(parts[0].split()[-1])
                    await progress_cb(len(symbols), current, "")
                except Exception:
                    pass

    rows = task.result()
    return rows


async def _per_stock_fallback(
    date: str,
    progress_cb: ProgressCb,
    log_cb: LogCb,
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

    if log_cb:
        await log_cb(f"종목별 개별 수집 시작 ({len(symbols)}종목, pykrx)")

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
            if log_cb and failed % 5 == 0:
                await log_cb(f"  누적 실패 {failed}건 (최근: {sym} — {str(e)[:60]})")

        if progress_cb and (i + 1) % 50 == 0:
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
    log_cb: LogCb = None,
    cb: CircuitBreaker,
) -> CollectionResult:
    """전 종목 당일 일봉 수집.

    1차: 벌크 호출 (~5-10초).
    2차: 실패 시 종목별 fallback (~1시간).
    """
    logger.info("[일봉] 수집 시작 (date=%s)", date)

    if log_cb:
        await log_cb("벌크 수집 시작 (pykrx 전종목 OHLCV)")

    # 벌크 시도
    try:
        rows = await cb.call(_bulk_fetch, date, log_cb=log_cb, progress_cb=progress_cb)

        if rows:
            completed = 0
            for r in rows:
                sym = r.pop("symbol")
                await write_candles_bulk(sym, [r], "1d")
                completed += 1

            if progress_cb:
                await progress_cb(completed, completed, rows[-1]["symbol"] if rows else "")

            logger.info("[일봉] 벌크 완료: %d종목", completed)
            if log_cb:
                await log_cb(f"벌크 완료: {completed}종목")
            return CollectionResult(
                job="daily_candle",
                total=completed,
                completed=completed,
            )
    except Exception as e:
        logger.warning("[일봉] 벌크 실패, fallback 전환: %s", e)
        if log_cb:
            await log_cb(f"벌크 실패: {str(e)[:80]} → 종목별 개별 수집 전환")

    # fallback
    return await _per_stock_fallback(date, progress_cb, log_cb, cb)
