"""고속 데이터 로더 — asyncpg raw query → Polars DataFrame."""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timedelta

import asyncpg
import polars as pl

from app.core.config import settings
from app.core.timezone import KST, to_kst

logger = logging.getLogger(__name__)

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
                    return to_kst(dt_val).replace(tzinfo=None)
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
                    return to_kst(dt_val).date()
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
                "pgm_buy_amount": pl.Int64, "pgm_sell_amount": pl.Int64,
                "pgm_net_amount": pl.Int64,
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
                   pgm_buy_amount::bigint, pgm_sell_amount::bigint, pgm_net_amount::bigint
            FROM program_trading
            {where}
            ORDER BY symbol, dt
        """
        rows = await conn.fetch(query, *params)

        if not rows:
            return pl.DataFrame(schema={
                "dt": pl.Date, "symbol": pl.Utf8,
                "pgm_buy_amount": pl.Int64, "pgm_sell_amount": pl.Int64,
                "pgm_net_amount": pl.Int64,
            })

        return pl.DataFrame({
            "dt": [r["dt"] for r in rows],
            "symbol": [r["symbol"] for r in rows],
            "pgm_buy_amount": [r["pgm_buy_amount"] for r in rows],
            "pgm_sell_amount": [r["pgm_sell_amount"] for r in rows],
            "pgm_net_amount": [r["pgm_net_amount"] for r in rows],
        })
    except Exception as e:
        logger.debug("program_trading load failed (table may not exist): %s", e)
        return pl.DataFrame(schema={
            "dt": pl.Date, "symbol": pl.Utf8,
            "pgm_buy_amount": pl.Int64, "pgm_sell_amount": pl.Int64,
            "pgm_net_amount": pl.Int64,
        })
    finally:
        await conn.close()


def _shift_weekend_discussion(disc_df: pl.DataFrame) -> pl.DataFrame:
    """주말+금요일 장마감 후 종토방 데이터를 다음 거래일(월요일)로 리맵.

    - 금요일 15:30 이후 → 다음 월요일 09:00
    - 토요일 전체 → 다음 월요일 09:00
    - 일요일 전체 → 다음 월요일 09:00
    - 그 외 → 그대로 유지

    리맵 후 동일 (symbol, dt)로 재집계하여 월요일에 주말 전체 누적 반영.
    """
    if disc_df.is_empty() or "dt" not in disc_df.columns:
        return disc_df

    # weekday: 1=월 ~ 7=일 (Polars 기본)
    disc_df = disc_df.with_columns(
        pl.col("dt").dt.weekday().alias("_wd"),
        pl.col("dt").dt.hour().alias("_hr"),
        pl.col("dt").dt.minute().alias("_mn"),
    )

    # 리맵 조건: 금요일(5) 15:30+ / 토요일(6) / 일요일(7)
    disc_df = disc_df.with_columns(
        pl.when(
            (pl.col("_wd") == 5) & ((pl.col("_hr") > 15) | ((pl.col("_hr") == 15) & (pl.col("_mn") >= 30)))
        ).then(
            # 금요일 15:30+ → 월요일 (+3일) 09:00
            pl.col("dt").dt.truncate("1d") + pl.duration(days=3, hours=9)
        ).when(
            pl.col("_wd") == 6
        ).then(
            # 토요일 → 월요일 (+2일) 09:00
            pl.col("dt").dt.truncate("1d") + pl.duration(days=2, hours=9)
        ).when(
            pl.col("_wd") == 7
        ).then(
            # 일요일 → 월요일 (+1일) 09:00
            pl.col("dt").dt.truncate("1d") + pl.duration(days=1, hours=9)
        ).otherwise(
            pl.col("dt")
        ).alias("dt")
    ).drop(["_wd", "_hr", "_mn"])

    # 동일 (symbol, dt)로 재집계
    agg_exprs = [
        pl.col("disc_count").sum(),
        pl.col("disc_sentiment").mean(),
        pl.col("disc_positive_ratio").mean(),
        pl.col("disc_negative_ratio").mean(),
    ]
    if "disc_velocity" in disc_df.columns:
        agg_exprs.append(pl.col("disc_velocity").mean())

    disc_df = disc_df.group_by(["symbol", "dt"]).agg(agg_exprs)

    return disc_df


async def _load_discussion(
    symbols: list[str] | None,
    start_date: date | None,
    end_date: date | None,
) -> pl.DataFrame:
    """종토방 시간별 집계 데이터 로드."""
    _empty = pl.DataFrame(schema={
        "dt": pl.Datetime("us", "Asia/Seoul"), "symbol": pl.Utf8,
        "disc_count": pl.Int64, "disc_sentiment": pl.Float64,
        "disc_positive_ratio": pl.Float64, "disc_negative_ratio": pl.Float64,
        "disc_velocity": pl.Float64,
    })
    conn: asyncpg.Connection = await asyncpg.connect(_dsn())
    try:
        exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name='discussion_sentiment_hourly')"
        )
        if not exists:
            return _empty

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
            SELECT dt, symbol, post_count, avg_sentiment,
                   positive_ratio, negative_ratio, total_likes, total_dislikes
            FROM discussion_sentiment_hourly
            {where}
            ORDER BY symbol, dt
        """
        rows = await conn.fetch(query, *params)
        if not rows:
            return _empty

        df = pl.DataFrame({
            "dt": [r["dt"] for r in rows],
            "symbol": [r["symbol"] for r in rows],
            "disc_count": [r["post_count"] for r in rows],
            "disc_sentiment": [r["avg_sentiment"] for r in rows],
            "disc_positive_ratio": [r["positive_ratio"] for r in rows],
            "disc_negative_ratio": [r["negative_ratio"] for r in rows],
        })
        # disc_velocity: 전 시간 대비 게시글 수 변화율
        df = df.with_columns(
            (pl.col("disc_count") / pl.col("disc_count").shift(1).over("symbol").clip(1, None)).alias("disc_velocity")
        )
        return df
    except Exception as e:
        logger.debug("discussion load failed: %s", e)
        return _empty
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
    include_discussion: bool = True,
) -> pl.DataFrame:
    """풍부화된 캔들 데이터: OHLCV + 투자자 수급 + DART 재무 + 뉴스 감성 + 섹터 + 신용/공매도 + 프로그램 매매.

    기존 load_candles()를 변경하지 않고, 알파 팩터 탐색 전용으로 사용한다.
    """
    df = await load_candles(symbols, start_date, end_date, interval)

    if df.is_empty():
        return df

    # ── Phase 1: DB 조회 (async — 이벤트 루프 차단 안 함) ──
    is_intraday = interval != "1d"

    # Phase 1: DB 조회 (async — 병렬 가능)
    inv_df = await _load_investor_trading(symbols, start_date, end_date) if include_investor else pl.DataFrame()
    dart_df = await _load_dart_financials(symbols, start_date, end_date) if include_dart else pl.DataFrame()
    sent_df = await _load_sentiment(symbols, start_date, end_date) if include_sentiment else pl.DataFrame()
    sector_df = await _load_sector_mapping(symbols) if include_sector else pl.DataFrame()
    ms_df = await _load_margin_short(symbols, start_date, end_date) if include_margin_short else pl.DataFrame()
    pgm_df = await _load_program_trading(symbols, start_date, end_date) if include_program_trading else pl.DataFrame()
    disc_df = await _load_discussion(symbols, start_date, end_date) if include_discussion else pl.DataFrame()

    # Phase 2: Polars JOIN/enrich (CPU 바운드 — to_thread로 이벤트 루프 해방)
    def _enrich_sync(
        df: pl.DataFrame,
        inv_df: pl.DataFrame, dart_df: pl.DataFrame, sent_df: pl.DataFrame,
        sector_df: pl.DataFrame, ms_df: pl.DataFrame, pgm_df: pl.DataFrame,
        disc_df: pl.DataFrame,
        is_intraday: bool,
    ) -> pl.DataFrame:
        if is_intraday:
            df = df.with_columns(pl.col("dt").cast(pl.Date).alias("dt_date"))

        # 투자자 수급
        if not inv_df.is_empty():
            if is_intraday:
                inv_shifted = inv_df.with_columns(
                    (pl.col("dt").cast(pl.Date) + pl.duration(days=1)).alias("dt_next")
                )
                df = df.join(inv_shifted, left_on=["symbol", "dt_date"], right_on=["symbol", "dt_next"], how="left")
                if "dt_right" in df.columns:
                    df = df.drop("dt_right")
            else:
                df = df.join(inv_df, on=["symbol", "dt"], how="left")
            # NaN 전파: enrichment 결측은 null 유지 (0-fill 제거)
            # GP가 0↔비제로 구조적 단절을 허위 알파로 착각하는 문제 방지
            logger.info("Enriched candles with investor trading data (%d rows)", inv_df.height)

        # DART 재무
        if not dart_df.is_empty():
            join_col = "dt_date" if is_intraday else "dt"
            df = df.sort(["symbol", join_col])
            dart_df = dart_df.sort(["symbol", "disclosure_date"])
            df = df.join_asof(dart_df, left_on=join_col, right_on="disclosure_date", by="symbol", strategy="backward")
            logger.info("Enriched candles with DART financials (%d records)", dart_df.height)
        for col in ["eps", "bps", "operating_margin", "debt_to_equity"]:
            if col not in df.columns:
                df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

        # 뉴스 감성
        if not sent_df.is_empty():
            if is_intraday:
                df = df.join(sent_df, left_on=["symbol", "dt_date"], right_on=["symbol", "dt"], how="left")
            else:
                df = df.join(sent_df, on=["symbol", "dt"], how="left")
            logger.info("Enriched candles with news sentiment (%d rows)", sent_df.height)
        for col in ["sentiment_score", "article_count", "event_score"]:
            if col not in df.columns:
                df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

        # 섹터 매핑
        if not sector_df.is_empty():
            df = df.join(sector_df, on="symbol", how="left")
            logger.info("Enriched candles with sector mapping (%d symbols)", sector_df.height)

        # 신용/공매도
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
            # NaN 전파: 신용/공매도 결측 null 유지
            logger.info("Enriched candles with margin/short data (%d rows)", ms_df.height)

        # 프로그램 매매
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
        for col in ["pgm_buy_amount", "pgm_sell_amount", "pgm_net_amount"]:
            if col not in df.columns:
                df = df.with_columns(pl.lit(None).cast(pl.Int64).alias(col))

        # 종토방 여론
        if not disc_df.is_empty():
            # [2026-04-06] 주말 종토방 데이터를 다음 거래일(월요일)로 리맵
            disc_df = _shift_weekend_discussion(disc_df)

            if is_intraday:
                # 분봉: dt의 시간(hour) 기준 JOIN — timezone 통일
                _disc_tz = disc_df["dt"].dtype
                df = df.with_columns(
                    pl.col("dt").dt.truncate("1h").alias("dt_hour")
                )
                # disc_df의 dt가 timezone-aware이면 dt_hour도 맞춤
                if hasattr(_disc_tz, "time_zone") and _disc_tz.time_zone:
                    df = df.with_columns(pl.col("dt_hour").dt.replace_time_zone(_disc_tz.time_zone))
                df = df.join(disc_df, left_on=["symbol", "dt_hour"], right_on=["symbol", "dt"], how="left")
                if "dt_right" in df.columns:
                    df = df.drop("dt_right")
                df = df.drop("dt_hour")
            else:
                # 일봉: disc_df에서 일별 합산 후 T+1 shift JOIN
                disc_daily = disc_df.group_by(
                    [pl.col("symbol"), pl.col("dt").cast(pl.Date).alias("disc_date")]
                ).agg([
                    pl.col("disc_count").sum(),
                    pl.col("disc_sentiment").mean(),
                    pl.col("disc_positive_ratio").mean(),
                    pl.col("disc_negative_ratio").mean(),
                    (pl.col("disc_velocity").mean() if "disc_velocity" in disc_df.columns else pl.lit(None).cast(pl.Float64)).alias("disc_velocity"),
                ])
                # T+1 shift (룩어헤드 편향 방지)
                disc_daily = disc_daily.with_columns(
                    (pl.col("disc_date").cast(pl.Date) + pl.duration(days=1)).alias("dt_shifted")
                )
                df = df.join(
                    disc_daily.drop("disc_date"),
                    left_on=["symbol", "dt"],
                    right_on=["symbol", "dt_shifted"],
                    how="left",
                )
            logger.info("Enriched candles with discussion (%d rows)", disc_df.height)
        for col in ["disc_count", "disc_sentiment", "disc_positive_ratio", "disc_negative_ratio", "disc_velocity"]:
            if col not in df.columns:
                df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

        # 정리
        if is_intraday and "dt_date" in df.columns:
            df = df.drop("dt_date")
        df = df.sort(["symbol", "dt"])
        return df

    df = await asyncio.to_thread(
        _enrich_sync, df, inv_df, dart_df, sent_df, sector_df, ms_df, pgm_df, disc_df, is_intraday,
    )

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
