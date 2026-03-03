"""고속 데이터 로더 — asyncpg raw query → Polars DataFrame."""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone

import asyncpg
import polars as pl

from app.core.config import settings

logger = logging.getLogger(__name__)

# DB의 TIMESTAMPTZ를 KST 거래일(date)로 변환하는 데 사용
_KST = timezone(timedelta(hours=9))

# 1분봉에서 집계 가능한 인터벌
_DERIVED_FROM_1M = {"3m", "5m", "15m", "30m", "1h"}

# 백테스트 지원 인터벌
SUPPORTED_INTERVALS = {"1m", "3m", "5m", "15m", "30m", "1h", "1d"}


def _dsn() -> str:
    s = settings
    return (
        f"postgresql://{s.POSTGRES_USER}:{s.POSTGRES_PASSWORD}"
        f"@{s.POSTGRES_HOST}:{s.POSTGRES_PORT}/{s.POSTGRES_DB}"
    )


def _interval_minutes(interval: str) -> int:
    """인터벌 문자열 → 분 단위 정수."""
    if interval.endswith("h"):
        return int(interval[:-1]) * 60
    if interval.endswith("m"):
        return int(interval[:-1])
    raise ValueError(f"Unsupported minute interval: {interval}")


async def load_candles(
    symbols: list[str] | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    interval: str = "1d",
) -> pl.DataFrame:
    """stock_candles에서 OHLCV 벌크 로딩.

    Parameters
    ----------
    symbols : 종목 코드 리스트. None이면 전 종목.
    start_date / end_date : 조회 기간.
    interval : 캔들 인터벌. 1m/3m/5m/15m/30m/1h는 1분봉에서 집계.

    Returns
    -------
    pl.DataFrame  columns=[dt, symbol, open, high, low, close, volume]
                  일봉: dt=Date, 분봉: dt=Datetime
    """
    if interval in _DERIVED_FROM_1M or interval == "1m":
        df = await _load_raw_candles(
            symbols, start_date, end_date, db_interval="1m", as_datetime=True,
        )
        if not df.is_empty() and interval != "1m":
            minutes = _interval_minutes(interval)
            df = _aggregate_to_minutes(df, minutes)
        return df

    # 일봉 이상: 기존 로직
    return await _load_raw_candles(
        symbols, start_date, end_date, db_interval=interval, as_datetime=False,
    )


async def _load_raw_candles(
    symbols: list[str] | None,
    start_date: date | None,
    end_date: date | None,
    db_interval: str,
    as_datetime: bool,
) -> pl.DataFrame:
    """DB에서 캔들 데이터 로딩.

    as_datetime=True: dt를 Datetime으로 (분봉)
    as_datetime=False: dt를 Date로 (일봉)
    """
    conn: asyncpg.Connection = await asyncpg.connect(_dsn())
    try:
        clauses = ["interval = $1"]
        params: list = [db_interval]
        idx = 2

        if start_date is not None:
            clauses.append(f"dt >= ${idx}")
            params.append(start_date)
            idx += 1

        if end_date is not None:
            if as_datetime:
                # 분봉: end_date의 모든 봉을 포함하기 위해 다음날 자정까지 조회
                clauses.append(f"dt < ${idx}")
                params.append(end_date + timedelta(days=1))
            else:
                clauses.append(f"dt <= ${idx}")
                params.append(end_date)
            idx += 1

        if symbols is not None and len(symbols) > 0:
            clauses.append(f"symbol = ANY(${idx})")
            params.append(symbols)
            idx += 1

        where = " AND ".join(clauses)
        query = f"""
            SELECT dt, symbol,
                   open::float8, high::float8, low::float8, close::float8,
                   volume::bigint
            FROM stock_candles
            WHERE {where}
            ORDER BY symbol, dt
        """

        rows = await conn.fetch(query, *params)

        if not rows:
            dt_type = pl.Datetime if as_datetime else pl.Date
            return pl.DataFrame(
                schema={
                    "dt": dt_type,
                    "symbol": pl.Utf8,
                    "open": pl.Float64,
                    "high": pl.Float64,
                    "low": pl.Float64,
                    "close": pl.Float64,
                    "volume": pl.Int64,
                }
            )

        if as_datetime:
            # 분봉: TIMESTAMPTZ → KST datetime (시분초 유지)
            def _to_kst_datetime(dt_val: datetime) -> datetime:
                if hasattr(dt_val, "astimezone"):
                    return dt_val.astimezone(_KST).replace(tzinfo=None)
                return dt_val

            data = {
                "dt": [_to_kst_datetime(r["dt"]) for r in rows],
                "symbol": [r["symbol"] for r in rows],
                "open": [r["open"] for r in rows],
                "high": [r["high"] for r in rows],
                "low": [r["low"] for r in rows],
                "close": [r["close"] for r in rows],
                "volume": [r["volume"] for r in rows],
            }
        else:
            # 일봉: TIMESTAMPTZ → KST date
            def _to_kst_date(dt_val: datetime) -> date:
                if hasattr(dt_val, "astimezone"):
                    return dt_val.astimezone(_KST).date()
                if hasattr(dt_val, "date"):
                    return dt_val.date()
                return dt_val

            data = {
                "dt": [_to_kst_date(r["dt"]) for r in rows],
                "symbol": [r["symbol"] for r in rows],
                "open": [r["open"] for r in rows],
                "high": [r["high"] for r in rows],
                "low": [r["low"] for r in rows],
                "close": [r["close"] for r in rows],
                "volume": [r["volume"] for r in rows],
            }

        df = pl.DataFrame(data)
        # 가격이 0인 행 제거 (거래정지, 결측 등)
        df = df.filter(
            (pl.col("open") > 0)
            & (pl.col("close") > 0)
            & (pl.col("high") > 0)
            & (pl.col("low") > 0)
        )
        # 중복 제거
        df = df.unique(subset=["symbol", "dt"], keep="last").sort(["symbol", "dt"])
        return df
    finally:
        await conn.close()


def _aggregate_to_minutes(df: pl.DataFrame, minutes: int) -> pl.DataFrame:
    """1분봉 → N분봉 집계 (Polars).

    dt를 N분 단위로 truncate → group_by(symbol, dt_bucket) → OHLCV 집계.
    """
    df = df.with_columns(
        pl.col("dt").dt.truncate(f"{minutes}m").alias("dt_bucket")
    )
    agg_df = (
        df.group_by(["symbol", "dt_bucket"])
        .agg([
            pl.col("open").sort_by("dt").first().alias("open"),
            pl.col("high").max().alias("high"),
            pl.col("low").min().alias("low"),
            pl.col("close").sort_by("dt").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
        ])
        .rename({"dt_bucket": "dt"})
        .sort(["symbol", "dt"])
    )
    return agg_df


async def available_minute_symbols() -> list[str]:
    """1분봉 데이터가 있는 종목 코드 목록 조회."""
    conn = await asyncpg.connect(_dsn())
    try:
        rows = await conn.fetch(
            "SELECT DISTINCT symbol FROM stock_candles WHERE interval = '1m' ORDER BY symbol"
        )
        return [r["symbol"] for r in rows]
    finally:
        await conn.close()


async def available_symbols() -> list[str]:
    """stock_masters에서 전체 종목 코드 목록 조회."""
    conn = await asyncpg.connect(_dsn())
    try:
        rows = await conn.fetch(
            "SELECT symbol FROM stock_masters ORDER BY symbol"
        )
        return [r["symbol"] for r in rows]
    finally:
        await conn.close()
