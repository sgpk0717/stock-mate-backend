"""VKOSPI 대용 데이터 일일 수집기.

Yahoo Finance ^KS200에서 최근 90일 데이터를 다운로드하고,
실현변동성을 계산하여 stock_candles(VKOSPI_PROXY)에 upsert한다.

일일 배치 스케줄러에 등록하거나 수동 트리거로 실행 가능.
"""

from __future__ import annotations

import asyncio
import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Callable

import polars as pl

from app.scheduler.circuit_breaker import CircuitBreaker
from app.scheduler.schemas import CollectionResult

logger = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9))
SYMBOL = "VKOSPI_PROXY"
TICKER = "^KS200"
ANNUALIZE_FACTOR = math.sqrt(252)

ProgressCb = Callable[[int, int, str], object] | None


def _download_and_compute() -> list[dict]:
    """yfinance 다운로드 + 실현변동성 계산 (동기, to_thread용)."""
    import yfinance as yf

    end = datetime.now()
    # 60일 변동성 계산 위해 90일 + 여유분
    start = end - timedelta(days=180)

    ticker = yf.Ticker(TICKER)
    pdf = ticker.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))

    if pdf.empty:
        logger.warning("yfinance returned empty data for %s", TICKER)
        return []

    pdf = pdf.reset_index()
    df = pl.from_pandas(pdf[["Date", "Close"]]).rename({"Date": "dt", "Close": "close_idx"})
    df = df.with_columns(pl.col("dt").cast(pl.Date))
    df = df.sort("dt")

    # 일간 로그 수익률
    df = df.with_columns(
        (pl.col("close_idx") / pl.col("close_idx").shift(1)).log().alias("log_return")
    )

    # 20일 / 60일 실현변동성
    df = df.with_columns([
        (
            pl.col("log_return")
            .rolling_std(window_size=20, min_periods=15)
            * ANNUALIZE_FACTOR
            * 100.0
        ).alias("realized_vol_20d"),
        (
            pl.col("log_return")
            .rolling_std(window_size=60, min_periods=40)
            * ANNUALIZE_FACTOR
            * 100.0
        ).alias("realized_vol_60d"),
    ])

    # 60일 백분위
    df = df.with_columns([
        pl.col("realized_vol_20d")
        .rolling_min(window_size=60, min_periods=20)
        .alias("_vol_min_60"),
        pl.col("realized_vol_20d")
        .rolling_max(window_size=60, min_periods=20)
        .alias("_vol_max_60"),
    ])
    df = df.with_columns(
        (
            (pl.col("realized_vol_20d") - pl.col("_vol_min_60"))
            / (pl.col("_vol_max_60") - pl.col("_vol_min_60")).clip(lower_bound=1e-10)
            * 100.0
        ).alias("vol_percentile_60d")
    )

    df = df.drop_nulls(subset=["realized_vol_20d"])

    rows = []
    for row in df.iter_rows(named=True):
        dt_val = row["dt"]
        rv20 = row["realized_vol_20d"]
        rv60 = row.get("realized_vol_60d")
        pct = row.get("vol_percentile_60d")

        if rv20 is None or math.isnan(rv20):
            continue

        rows.append({
            "dt": datetime(dt_val.year, dt_val.month, dt_val.day, tzinfo=KST),
            "open": round(rv60, 4) if rv60 is not None and not math.isnan(rv60) else 0.0,
            "high": round(pct, 2) if pct is not None and not math.isnan(pct) else 0.0,
            "low": 0.0,
            "close": round(rv20, 4),
            "volume": 0,
        })

    return rows


async def collect_vkospi(
    date: str,
    *,
    progress_cb: ProgressCb = None,
    cb: CircuitBreaker | None = None,
) -> CollectionResult:
    """VKOSPI_PROXY 일일 수집.

    Args:
        date: YYYYMMDD (호환용, 실제로는 최근 180일 전체 upsert)
        progress_cb: 진행률 콜백
        cb: 서킷 브레이커 (None 가능 — yfinance는 자체 rate limit)
    """
    logger.info("[VKOSPI] 수집 시작 (date=%s)", date)

    try:
        if cb:
            rows = await cb.call(asyncio.to_thread, _download_and_compute)
        else:
            rows = await asyncio.to_thread(_download_and_compute)
    except Exception as e:
        logger.error("[VKOSPI] yfinance 다운로드 실패: %s", e, exc_info=True)
        return CollectionResult(job="vkospi", total=1, failed=1)

    if not rows:
        logger.warning("[VKOSPI] 데이터 없음")
        return CollectionResult(job="vkospi", total=1, completed=0)

    # DB 저장 (candle_writer 재사용)
    from app.services.candle_writer import write_candles_bulk
    try:
        await write_candles_bulk(SYMBOL, rows, "1d")
        if progress_cb:
            await progress_cb(1, 1, SYMBOL)
        logger.info("[VKOSPI] %d건 저장 완료", len(rows))
        return CollectionResult(job="vkospi", total=1, completed=1)
    except Exception as e:
        logger.error("[VKOSPI] DB 저장 실패: %s", e, exc_info=True)
        return CollectionResult(job="vkospi", total=1, failed=1)
