"""고속 데이터 로더 — asyncpg raw query → Polars DataFrame."""

from __future__ import annotations

import asyncio
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


_CHUNK_SIZE = 50  # 분봉 청크 로딩 시 심볼 배치 크기


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
        # DB에 해당 인터벌 데이터가 존재하면 직접 로딩 시도
        if interval != "1m":
            df_direct = await _load_raw_candles(
                symbols, start_date, end_date, db_interval=interval, as_datetime=True,
            )
            if not df_direct.is_empty():
                # 오늘 데이터가 포함되어 있는지 확인
                # 없으면 1분봉 폴백으로 보충 (장중 수집기가 1m만 저장하는 경우)
                max_dt = df_direct["dt"].max()
                today_start = datetime.combine(date.today(), datetime.min.time())
                if max_dt is not None and max_dt >= today_start:
                    return df_direct
                # 오늘 데이터 없음 → 1분봉에서 리샘플링하여 합산
                logger.info(
                    "load_candles(%s): DB %s 데이터에 오늘분 없음 (max=%s), 1분봉 폴백",
                    interval, interval, max_dt,
                )
                df_1m = await _load_raw_candles_chunked(
                    symbols, start_date, end_date, db_interval="1m", as_datetime=True,
                )
                if not df_1m.is_empty():
                    minutes = _interval_minutes(interval)
                    df_resampled = _aggregate_to_minutes(df_1m, minutes)
                    if not df_resampled.is_empty():
                        # 직접 로딩 + 리샘플링 합산, 중복 제거
                        # 컬럼 순서 통일 후 concat
                        cols = df_direct.columns
                        df_resampled = df_resampled.select(cols)
                        combined = pl.concat([df_direct, df_resampled])
                        combined = combined.unique(subset=["symbol", "dt"], keep="last")
                        return combined.sort(["symbol", "dt"])
                return df_direct
        # 폴백: 1분봉에서 집계
        df = await _load_raw_candles_chunked(
            symbols, start_date, end_date, db_interval="1m", as_datetime=True,
        )
        if not df.is_empty() and interval != "1m":
            minutes = _interval_minutes(interval)
            df = _aggregate_to_minutes(df, minutes)
        return df

    # 일봉 이상: 기존 로직 (데이터 양 적음)
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
                    # 의도적 naive KST 변환: 백테스트 엔진이 naive datetime 기대
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


async def _load_raw_candles_chunked(
    symbols: list[str] | None,
    start_date: date | None,
    end_date: date | None,
    db_interval: str,
    as_datetime: bool,
) -> pl.DataFrame:
    """대량 분봉 데이터를 심볼 청크 단위로 로딩.

    950+ 심볼 × 수개월 1분봉 = 수천만 행 → 단일 쿼리로 OOM 발생.
    심볼을 _CHUNK_SIZE 단위로 나눠서 DB 쿼리 → concat.
    """
    # symbols=None이면 전 종목 — 먼저 심볼 목록을 확보
    if symbols is None:
        symbols = await available_minute_symbols()
        logger.info("Resolved %d minute-data symbols for chunked loading", len(symbols))

    # 적은 심볼이면 기존 방식 사용
    if len(symbols) <= _CHUNK_SIZE:
        return await _load_raw_candles(symbols, start_date, end_date, db_interval, as_datetime)

    chunks: list[pl.DataFrame] = []
    total_chunks = (len(symbols) + _CHUNK_SIZE - 1) // _CHUNK_SIZE

    for i in range(0, len(symbols), _CHUNK_SIZE):
        chunk_symbols = symbols[i : i + _CHUNK_SIZE]
        chunk_num = i // _CHUNK_SIZE + 1
        logger.info(
            "Loading candles chunk %d/%d (%d symbols)...",
            chunk_num, total_chunks, len(chunk_symbols),
        )
        chunk_df = await _load_raw_candles(
            chunk_symbols, start_date, end_date, db_interval, as_datetime,
        )
        if not chunk_df.is_empty():
            chunks.append(chunk_df)

        # 이벤트 루프에 제어권 양보 — API 핸들러가 처리될 수 있도록
        await asyncio.sleep(0)

    if not chunks:
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

    result = pl.concat(chunks)
    logger.info(
        "Chunked candle loading complete: %d rows, %d symbols",
        result.height, result["symbol"].n_unique(),
    )
    return result.sort(["symbol", "dt"])


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


# ── 풍부화 데이터 로더 (알파 팩터 전용) ──────────────────────────


async def _load_investor_trading(
    symbols: list[str] | None,
    start_date: date | None,
    end_date: date | None,
) -> pl.DataFrame:
    """investor_trading 테이블에서 수급 데이터 로드."""
    conn: asyncpg.Connection = await asyncpg.connect(_dsn())
    try:
        # 테이블 존재 확인
        exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name='investor_trading')"
        )
        _inv_schema = {
            "dt": pl.Date, "symbol": pl.Utf8,
            "foreign_net": pl.Int64, "inst_net": pl.Int64, "retail_net": pl.Int64,
            "foreign_buy_vol": pl.Int64, "foreign_sell_vol": pl.Int64,
            "inst_buy_vol": pl.Int64, "inst_sell_vol": pl.Int64,
            "retail_buy_vol": pl.Int64, "retail_sell_vol": pl.Int64,
        }

        if not exists:
            return pl.DataFrame(schema=_inv_schema)

        clauses: list[str] = []
        params: list = []
        idx = 1

        if start_date is not None:
            clauses.append(f"dt >= ${idx}")
            params.append(start_date)
            idx += 1
        if end_date is not None:
            clauses.append(f"dt <= ${idx}")
            params.append(end_date)
            idx += 1
        if symbols is not None and len(symbols) > 0:
            clauses.append(f"symbol = ANY(${idx})")
            params.append(symbols)
            idx += 1

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"""
            SELECT dt, symbol,
                   foreign_net::bigint, inst_net::bigint, retail_net::bigint,
                   COALESCE(foreign_buy_vol, 0)::bigint AS foreign_buy_vol,
                   COALESCE(foreign_sell_vol, 0)::bigint AS foreign_sell_vol,
                   COALESCE(inst_buy_vol, 0)::bigint AS inst_buy_vol,
                   COALESCE(inst_sell_vol, 0)::bigint AS inst_sell_vol,
                   COALESCE(retail_buy_vol, 0)::bigint AS retail_buy_vol,
                   COALESCE(retail_sell_vol, 0)::bigint AS retail_sell_vol
            FROM investor_trading
            {where}
            ORDER BY symbol, dt
        """
        rows = await conn.fetch(query, *params)

        if not rows:
            return pl.DataFrame(schema=_inv_schema)

        return pl.DataFrame({
            "dt": [r["dt"] for r in rows],
            "symbol": [r["symbol"] for r in rows],
            "foreign_net": [r["foreign_net"] for r in rows],
            "inst_net": [r["inst_net"] for r in rows],
            "retail_net": [r["retail_net"] for r in rows],
            "foreign_buy_vol": [r["foreign_buy_vol"] for r in rows],
            "foreign_sell_vol": [r["foreign_sell_vol"] for r in rows],
            "inst_buy_vol": [r["inst_buy_vol"] for r in rows],
            "inst_sell_vol": [r["inst_sell_vol"] for r in rows],
            "retail_buy_vol": [r["retail_buy_vol"] for r in rows],
            "retail_sell_vol": [r["retail_sell_vol"] for r in rows],
        })
    except Exception as e:
        logger.debug("investor_trading load failed (table may not exist): %s", e)
        return pl.DataFrame(schema={
            "dt": pl.Date, "symbol": pl.Utf8,
            "foreign_net": pl.Int64, "inst_net": pl.Int64, "retail_net": pl.Int64,
            "foreign_buy_vol": pl.Int64, "foreign_sell_vol": pl.Int64,
            "inst_buy_vol": pl.Int64, "inst_sell_vol": pl.Int64,
            "retail_buy_vol": pl.Int64, "retail_sell_vol": pl.Int64,
        })
    finally:
        await conn.close()


async def _load_dart_financials(
    symbols: list[str] | None,
    start_date: date | None,
    end_date: date | None,
) -> pl.DataFrame:
    """dart_financials 테이블에서 재무 데이터 로드."""
    conn: asyncpg.Connection = await asyncpg.connect(_dsn())
    try:
        exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name='dart_financials')"
        )
        if not exists:
            return pl.DataFrame(schema={
                "disclosure_date": pl.Date, "symbol": pl.Utf8,
                "eps": pl.Float64, "bps": pl.Float64,
                "operating_margin": pl.Float64, "debt_to_equity": pl.Float64,
            })

        clauses: list[str] = []
        params: list = []
        idx = 1

        if symbols is not None and len(symbols) > 0:
            clauses.append(f"symbol = ANY(${idx})")
            params.append(symbols)
            idx += 1
        # 공시일 기준 필터 (약간 넓게 — join_asof에서 정밀 매칭)
        if start_date is not None:
            # 시작일 1년 전부터 (이전 분기 데이터 포함)
            clauses.append(f"disclosure_date >= ${idx}")
            params.append(start_date - timedelta(days=365))
            idx += 1
        if end_date is not None:
            clauses.append(f"disclosure_date <= ${idx}")
            params.append(end_date)
            idx += 1

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"""
            SELECT disclosure_date, symbol,
                   eps::float8, bps::float8,
                   operating_margin::float8, debt_to_equity::float8
            FROM dart_financials
            {where}
            ORDER BY symbol, disclosure_date
        """
        rows = await conn.fetch(query, *params)

        if not rows:
            return pl.DataFrame(schema={
                "disclosure_date": pl.Date, "symbol": pl.Utf8,
                "eps": pl.Float64, "bps": pl.Float64,
                "operating_margin": pl.Float64, "debt_to_equity": pl.Float64,
            })

        return pl.DataFrame({
            "disclosure_date": [r["disclosure_date"] for r in rows],
            "symbol": [r["symbol"] for r in rows],
            "eps": [r["eps"] for r in rows],
            "bps": [r["bps"] for r in rows],
            "operating_margin": [r["operating_margin"] for r in rows],
            "debt_to_equity": [r["debt_to_equity"] for r in rows],
        })
    except Exception as e:
        logger.debug("dart_financials load failed (table may not exist): %s", e)
        return pl.DataFrame(schema={
            "disclosure_date": pl.Date, "symbol": pl.Utf8,
            "eps": pl.Float64, "bps": pl.Float64,
            "operating_margin": pl.Float64, "debt_to_equity": pl.Float64,
        })
    finally:
        await conn.close()


async def _load_sentiment(
    symbols: list[str] | None,
    start_date: date | None,
    end_date: date | None,
) -> pl.DataFrame:
    """뉴스 감성 데이터 로드 (T+1 shift 적용)."""
    try:
        from app.news.backtest_integration import load_sentiment_data
        return await load_sentiment_data(symbols, start_date, end_date)
    except Exception as e:
        logger.debug("sentiment load failed: %s", e)
        return pl.DataFrame(schema={
            "symbol": pl.Utf8, "dt": pl.Date,
            "sentiment_score": pl.Float64, "article_count": pl.Int64,
            "event_score": pl.Float64,
        })


async def _load_sector_mapping(
    symbols: list[str] | None,
) -> pl.DataFrame:
    """섹터 ID 매핑 로드."""
    try:
        from app.alpha.confounders import load_sector_mapping
        sector_map = await load_sector_mapping(symbols)
        if not sector_map:
            return pl.DataFrame(schema={"symbol": pl.Utf8, "sector_id": pl.Int64})
        return pl.DataFrame({
            "symbol": list(sector_map.keys()),
            "sector_id": list(sector_map.values()),
        })
    except Exception as e:
        logger.debug("sector mapping load failed: %s", e)
        return pl.DataFrame(schema={"symbol": pl.Utf8, "sector_id": pl.Int64})


async def _load_margin_short(
    symbols: list[str] | None,
    start_date: date | None,
    end_date: date | None,
) -> pl.DataFrame:
    """신용잔고/공매도 일별 데이터 로드."""
    conn: asyncpg.Connection = await asyncpg.connect(_dsn())
    try:
        exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name='margin_short_daily')"
        )
        if not exists:
            return pl.DataFrame(schema={
                "dt": pl.Date, "symbol": pl.Utf8,
                "margin_balance": pl.Int64, "margin_rate": pl.Float64,
                "short_volume": pl.Int64, "short_balance": pl.Int64,
                "short_balance_rate": pl.Float64,
            })

        clauses: list[str] = []
        params: list = []
        idx = 1

        if start_date is not None:
            clauses.append(f"dt >= ${idx}")
            params.append(start_date)
            idx += 1
        if end_date is not None:
            clauses.append(f"dt <= ${idx}")
            params.append(end_date)
            idx += 1
        if symbols is not None and len(symbols) > 0:
            clauses.append(f"symbol = ANY(${idx})")
            params.append(symbols)
            idx += 1

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"""
            SELECT dt, symbol,
                   margin_balance::bigint, margin_rate::float8,
                   short_volume::bigint, short_balance::bigint,
                   short_balance_rate::float8
            FROM margin_short_daily
            {where}
            ORDER BY symbol, dt
        """
        rows = await conn.fetch(query, *params)

        if not rows:
            return pl.DataFrame(schema={
                "dt": pl.Date, "symbol": pl.Utf8,
                "margin_balance": pl.Int64, "margin_rate": pl.Float64,
                "short_volume": pl.Int64, "short_balance": pl.Int64,
                "short_balance_rate": pl.Float64,
            })

        return pl.DataFrame({
            "dt": [r["dt"] for r in rows],
            "symbol": [r["symbol"] for r in rows],
            "margin_balance": [r["margin_balance"] for r in rows],
            "margin_rate": [r["margin_rate"] for r in rows],
            "short_volume": [r["short_volume"] for r in rows],
            "short_balance": [r["short_balance"] for r in rows],
            "short_balance_rate": [r["short_balance_rate"] for r in rows],
        })
    except Exception as e:
        logger.debug("margin_short_daily load failed (table may not exist): %s", e)
        return pl.DataFrame(schema={
            "dt": pl.Date, "symbol": pl.Utf8,
            "margin_balance": pl.Int64, "margin_rate": pl.Float64,
            "short_volume": pl.Int64, "short_balance": pl.Int64,
            "short_balance_rate": pl.Float64,
        })
    finally:
        await conn.close()


async def _load_program_trading(
    symbols: list[str] | None,
    start_date: date | None,
    end_date: date | None,
) -> pl.DataFrame:
    """프로그램 매매 데이터 로드."""
    conn: asyncpg.Connection = await asyncpg.connect(_dsn())
    try:
        exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name='program_trading')"
        )
        if not exists:
            return pl.DataFrame(schema={
                "dt": pl.Date, "symbol": pl.Utf8,
                "pgm_buy_qty": pl.Int64, "pgm_sell_qty": pl.Int64,
                "pgm_net_qty": pl.Int64,
            })

        clauses: list[str] = []
        params: list = []
        idx = 1

        if start_date is not None:
            clauses.append(f"dt >= ${idx}")
            params.append(start_date)
            idx += 1
        if end_date is not None:
            clauses.append(f"dt <= ${idx}")
            params.append(end_date)
            idx += 1
        if symbols is not None and len(symbols) > 0:
            clauses.append(f"symbol = ANY(${idx})")
            params.append(symbols)
            idx += 1

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"""
            SELECT dt::date, symbol,
                   pgm_buy_qty::bigint, pgm_sell_qty::bigint, pgm_net_qty::bigint
            FROM program_trading
            {where}
            ORDER BY symbol, dt
        """
        rows = await conn.fetch(query, *params)

        if not rows:
            return pl.DataFrame(schema={
                "dt": pl.Date, "symbol": pl.Utf8,
                "pgm_buy_qty": pl.Int64, "pgm_sell_qty": pl.Int64,
                "pgm_net_qty": pl.Int64,
            })

        return pl.DataFrame({
            "dt": [r["dt"] for r in rows],
            "symbol": [r["symbol"] for r in rows],
            "pgm_buy_qty": [r["pgm_buy_qty"] for r in rows],
            "pgm_sell_qty": [r["pgm_sell_qty"] for r in rows],
            "pgm_net_qty": [r["pgm_net_qty"] for r in rows],
        })
    except Exception as e:
        logger.debug("program_trading load failed (table may not exist): %s", e)
        return pl.DataFrame(schema={
            "dt": pl.Date, "symbol": pl.Utf8,
            "pgm_buy_qty": pl.Int64, "pgm_sell_qty": pl.Int64,
            "pgm_net_qty": pl.Int64,
        })
    finally:
        await conn.close()


async def load_enriched_candles(
    symbols: list[str] | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    interval: str = "1d",
    include_investor: bool = True,
    include_dart: bool = True,
    include_sentiment: bool = True,
    include_sector: bool = True,
    include_margin_short: bool = True,
    include_program_trading: bool = True,
) -> pl.DataFrame:
    """풍부화된 캔들 데이터: OHLCV + 투자자 수급 + DART 재무 + 뉴스 감성 + 섹터 + 신용/공매도 + 프로그램 매매.

    기존 load_candles()를 변경하지 않고, 알파 팩터 탐색 전용으로 사용한다.
    """
    df = await load_candles(symbols, start_date, end_date, interval)

    if df.is_empty():
        return df

    # 분봉인 경우 dt가 Datetime. 일별 데이터 JOIN 시 date 컬럼이 필요.
    is_intraday = interval != "1d"
    if is_intraday:
        df = df.with_columns(pl.col("dt").cast(pl.Date).alias("dt_date"))

    # 투자자 수급 데이터 JOIN (분봉: T-1 전일 기준, 일봉: 당일)
    if include_investor:
        inv_df = await _load_investor_trading(symbols, start_date, end_date)
        if not inv_df.is_empty():
            if is_intraday:
                # T-1 shift: 전일 투자자 데이터를 오늘 분봉에 매칭 (look-ahead bias 방지)
                inv_shifted = inv_df.with_columns(
                    (pl.col("dt").cast(pl.Date) + pl.duration(days=1)).alias("dt_next")
                )
                df = df.join(inv_shifted, left_on=["symbol", "dt_date"], right_on=["symbol", "dt_next"], how="left")
                if "dt_right" in df.columns:
                    df = df.drop("dt_right")
                if "dt" in df.columns and df["dt"].dtype == inv_df["dt"].dtype:
                    pass  # 원본 dt 유지
            else:
                df = df.join(inv_df, on=["symbol", "dt"], how="left")
            for col in [
                "foreign_net", "inst_net", "retail_net",
                "foreign_buy_vol", "foreign_sell_vol",
                "inst_buy_vol", "inst_sell_vol",
                "retail_buy_vol", "retail_sell_vol",
            ]:
                if col in df.columns:
                    df = df.with_columns(pl.col(col).fill_null(0).alias(col))
            logger.info("Enriched candles with investor trading data (%d rows)", inv_df.height)
        await asyncio.sleep(0)  # yield to event loop

    # DART 재무 데이터 JOIN (join_asof: 가장 최근 공시)
    if include_dart:
        dart_df = await _load_dart_financials(symbols, start_date, end_date)
        if not dart_df.is_empty():
            join_col = "dt_date" if is_intraday else "dt"
            df = df.sort(["symbol", join_col])
            dart_df = dart_df.sort(["symbol", "disclosure_date"])
            df = df.join_asof(
                dart_df,
                left_on=join_col,
                right_on="disclosure_date",
                by="symbol",
                strategy="backward",
            )
            logger.info("Enriched candles with DART financials (%d records)", dart_df.height)
        else:
            logger.info("No DART financial data found — adding placeholder columns")
        # 컬럼이 없으면 0.0 placeholder 추가
        for col in ["eps", "bps", "operating_margin", "debt_to_equity"]:
            if col not in df.columns:
                df = df.with_columns(pl.lit(0.0).alias(col))
            else:
                df = df.with_columns(pl.col(col).fill_null(0.0).alias(col))
        await asyncio.sleep(0)  # yield to event loop

    # 뉴스 감성 데이터 JOIN
    if include_sentiment:
        sent_df = await _load_sentiment(symbols, start_date, end_date)
        if not sent_df.is_empty():
            if is_intraday:
                df = df.join(sent_df, left_on=["symbol", "dt_date"], right_on=["symbol", "dt"], how="left")
            else:
                df = df.join(sent_df, on=["symbol", "dt"], how="left")
            logger.info("Enriched candles with news sentiment (%d rows)", sent_df.height)
        else:
            logger.info("No news sentiment data found — adding placeholder columns")
        # 컬럼이 없으면 0.0 placeholder 추가 (데이터 부재 시에도 수식 에러 방지)
        for col in ["sentiment_score", "article_count", "event_score"]:
            if col not in df.columns:
                df = df.with_columns(pl.lit(0.0).alias(col))
            else:
                df = df.with_columns(pl.col(col).fill_null(0.0).alias(col))

    # 섹터 ID 매핑 JOIN
    if include_sector:
        sector_df = await _load_sector_mapping(symbols)
        if not sector_df.is_empty():
            df = df.join(sector_df, on="symbol", how="left")
            logger.info("Enriched candles with sector mapping (%d symbols)", sector_df.height)

    # 신용잔고/공매도 JOIN (분봉: T-1 전일 기준, 일봉: 당일)
    if include_margin_short:
        ms_df = await _load_margin_short(symbols, start_date, end_date)
        if not ms_df.is_empty():
            if is_intraday:
                ms_shifted = ms_df.with_columns(
                    (pl.col("dt").cast(pl.Date) + pl.duration(days=1)).alias("dt_next")
                )
                df = df.join(ms_shifted, left_on=["symbol", "dt_date"], right_on=["symbol", "dt_next"], how="left")
                if "dt_right" in df.columns:
                    df = df.drop("dt_right")
            else:
                df = df.join(ms_df, on=["symbol", "dt"], how="left")
            for col in ["margin_balance", "margin_rate", "short_volume", "short_balance", "short_balance_rate"]:
                if col in df.columns:
                    df = df.with_columns(pl.col(col).fill_null(0).alias(col))
            logger.info("Enriched candles with margin/short data (%d rows)", ms_df.height)
        await asyncio.sleep(0)  # yield to event loop

    # 프로그램 매매 JOIN (분봉: T-1 전일 기준, 일봉: 당일)
    if include_program_trading:
        pgm_df = await _load_program_trading(symbols, start_date, end_date)
        if not pgm_df.is_empty():
            if is_intraday:
                pgm_shifted = pgm_df.with_columns(
                    (pl.col("dt").cast(pl.Date) + pl.duration(days=1)).alias("dt_next")
                )
                df = df.join(pgm_shifted, left_on=["symbol", "dt_date"], right_on=["symbol", "dt_next"], how="left")
                if "dt_right" in df.columns:
                    df = df.drop("dt_right")
            else:
                df = df.join(pgm_df, on=["symbol", "dt"], how="left")
            logger.info("Enriched candles with program trading (%d rows)", pgm_df.height)
        else:
            logger.info("No program trading data found — adding placeholder columns")
        # 컬럼이 없으면 0 placeholder 추가
        for col in ["pgm_buy_qty", "pgm_sell_qty", "pgm_net_qty"]:
            if col not in df.columns:
                df = df.with_columns(pl.lit(0).alias(col))
            else:
                df = df.with_columns(pl.col(col).fill_null(0).alias(col))

    # 분봉 헬퍼 컬럼 제거
    if is_intraday and "dt_date" in df.columns:
        df = df.drop("dt_date")

    # 원래 정렬 복원
    df = df.sort(["symbol", "dt"])

    return df


def load_enriched_candles_sync(
    symbols: list[str] | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    interval: str = "1d",
    **kwargs,
) -> pl.DataFrame:
    """동기 래퍼 — asyncio.to_thread()에서 호출용.

    별도 스레드에서 새 이벤트 루프를 생성하여 async 함수를 실행.
    메인 이벤트 루프를 블로킹하지 않음.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            load_enriched_candles(symbols, start_date, end_date, interval, **kwargs)
        )
    finally:
        loop.close()


def load_candles_sync(
    symbols: list[str] | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    interval: str = "1d",
) -> pl.DataFrame:
    """동기 래퍼 — asyncio.to_thread()에서 호출용."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            load_candles(symbols, start_date, end_date, interval)
        )
    finally:
        loop.close()
