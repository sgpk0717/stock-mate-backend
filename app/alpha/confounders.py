"""교란 변수(Confounder) 데이터 로더.

DoWhy 인과 검증에 필요한 교란 변수 7종을 로드:
- market_return: KOSPI 200 ETF(069500) 일간 수익률
- market_volatility: 20일 실현 변동성
- market_momentum_12m: 252일 시장 모멘텀 (Fama-French UMD 대용, 딥리서치 권고)
- base_rate: 한국은행 기준금리 (일별 forward-fill)
- smb: Small Minus Big (소형주-대형주 수익률 스프레드, 거래대금 기반 분류)
- hml: High Minus Low (가치주-성장주 수익률 스프레드, DART BPS/종가 기반)
- sector_id: 종목의 섹터 정수 ID
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path

import asyncpg
import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)

_BOK_JSON_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "bok_base_rate.json"
_KOSPI_ETF_SYMBOL = "069500"  # KODEX 200


async def load_confounders(
    start_date: date,
    end_date: date,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    """교란 변수 통합 DataFrame 반환.

    Returns
    -------
    pd.DataFrame
        columns: [dt, market_return, market_volatility, market_momentum_12m,
                  base_rate, smb, hml]
        (sector_id는 종목별이므로 별도 매핑 dict로 반환)
    """
    market_df = await _load_market_data(start_date, end_date)
    rate_df = _load_base_rate(start_date, end_date)
    smb_df = await _compute_smb(start_date, end_date)
    hml_df = await _compute_hml(start_date, end_date)

    # dt 기준 병합
    merged = pd.merge(market_df, rate_df, on="dt", how="left")
    merged = pd.merge(merged, smb_df, on="dt", how="left")
    merged = pd.merge(merged, hml_df, on="dt", how="left")

    # forward-fill로 빈 날짜 채우기
    merged["base_rate"] = merged["base_rate"].ffill().bfill()
    # SMB/HML: 결측은 0.0으로 채움 (데이터 부재일에 neutral 가정)
    merged["smb"] = merged["smb"].fillna(0.0)
    merged["hml"] = merged["hml"].fillna(0.0)

    merged = merged.dropna(subset=["market_return", "market_volatility"])

    return merged


async def _load_market_data(
    start_date: date, end_date: date
) -> pd.DataFrame:
    """KOSPI 200 ETF(069500) 일봉에서 시장 수익률 + 변동성 계산."""
    from app.backtest.data_loader import load_candles

    # 변동성 계산을 위해 시작일 이전 30일 추가 로드
    extended_start = start_date - timedelta(days=45)

    candles = await load_candles(
        symbols=[_KOSPI_ETF_SYMBOL],
        start_date=extended_start,
        end_date=end_date,
    )

    if candles.height == 0:
        logger.warning("No KOSPI ETF data found for confounder loading")
        return pd.DataFrame(columns=["dt", "market_return", "market_volatility"])

    # Polars로 계산 후 pandas 변환
    df = candles.sort("dt")
    df = df.with_columns([
        (pl.col("close") / pl.col("close").shift(1) - 1.0).alias("market_return"),
        (pl.col("close") / pl.col("close").shift(1) - 1.0)
        .rolling_std(window_size=20, min_samples=5)
        .alias("market_volatility"),
        # 12개월(252일) 시장 모멘텀 (딥리서치 권고: Fama-French UMD 대용)
        (pl.col("close") / pl.col("close").shift(252) - 1.0).alias("market_momentum_12m"),
    ])

    # 원래 요청 범위로 필터
    df = df.filter(
        (pl.col("dt") >= start_date) & (pl.col("dt") <= end_date)
    )

    result = df.select(["dt", "market_return", "market_volatility", "market_momentum_12m"]).to_pandas()
    result["dt"] = pd.to_datetime(result["dt"]).dt.date
    return result


def _load_base_rate(start_date: date, end_date: date) -> pd.DataFrame:
    """BOK 기준금리 JSON → 일별 forward-fill DataFrame."""
    if not _BOK_JSON_PATH.exists():
        logger.warning("BOK base rate JSON not found: %s", _BOK_JSON_PATH)
        # 기본값 반환
        dates = pd.date_range(start_date, end_date, freq="B")
        return pd.DataFrame({
            "dt": [d.date() for d in dates],
            "base_rate": [3.0] * len(dates),
        })

    with open(_BOK_JSON_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    # 이벤트 날짜 → rate
    events = pd.DataFrame(raw)
    events["date"] = pd.to_datetime(events["date"]).dt.date

    # 영업일 시리즈 생성
    all_dates = pd.date_range(start_date, end_date, freq="B")
    daily = pd.DataFrame({"dt": [d.date() for d in all_dates]})

    # 이벤트를 머지하고 forward-fill
    daily = pd.merge(
        daily,
        events.rename(columns={"date": "dt", "rate": "base_rate"}),
        on="dt",
        how="left",
    )

    # 시작일 이전 가장 최근 금리 찾기
    past_rates = [e for e in raw if date.fromisoformat(e["date"]) <= start_date]
    initial_rate = past_rates[-1]["rate"] if past_rates else raw[0]["rate"]

    # 첫 행이 NaN이면 초기값 설정 후 forward-fill
    if pd.isna(daily["base_rate"].iloc[0]):
        daily.loc[daily.index[0], "base_rate"] = initial_rate
    daily["base_rate"] = daily["base_rate"].ffill()

    return daily


def _dsn() -> str:
    """asyncpg 연결 DSN."""
    from app.core.config import settings
    s = settings
    return (
        f"postgresql://{s.POSTGRES_USER}:{s.POSTGRES_PASSWORD}"
        f"@{s.POSTGRES_HOST}:{s.POSTGRES_PORT}/{s.POSTGRES_DB}"
    )


async def _compute_smb(
    start_date: date, end_date: date,
) -> pd.DataFrame:
    """SMB 팩터: 소형주 수익률 - 대형주 수익률 (거래대금 기반 분류).

    실제 시가총액 데이터 대신 일별 close×volume(거래대금)을
    유동성/규모 대용 변수로 사용한다.
    거래대금과 시가총액의 상관관계는 ~0.7-0.8로 합리적인 프록시이다.

    각 거래일마다:
    1. 종목별 거래대금(close×volume) 산출
    2. 중위수 기준 소형주/대형주 분류
    3. SMB = mean_return(소형) - mean_return(대형)
    """
    # 수익률 계산을 위해 시작일보다 30일 이전부터 로딩
    extended_start = start_date - timedelta(days=45)

    conn: asyncpg.Connection = await asyncpg.connect(_dsn())
    try:
        rows = await conn.fetch(
            """
            SELECT dt::date AS dt, symbol,
                   close::float8, volume::bigint
            FROM stock_candles
            WHERE interval = '1d'
              AND dt >= $1 AND dt <= $2
              AND close > 0 AND volume > 0
            ORDER BY symbol, dt
            """,
            extended_start, end_date,
        )
    finally:
        await conn.close()

    if not rows:
        logger.warning("No daily candle data for SMB computation")
        return pd.DataFrame(columns=["dt", "smb"])

    # Polars로 벡터 연산
    df = pl.DataFrame({
        "dt": [r["dt"] for r in rows],
        "symbol": [r["symbol"] for r in rows],
        "close": [r["close"] for r in rows],
        "volume": [r["volume"] for r in rows],
    })

    # 일별 수익률 (종목별 전일 대비)
    df = df.sort(["symbol", "dt"])
    df = df.with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("symbol") - 1.0)
        .alias("daily_return"),
        (pl.col("close") * pl.col("volume").cast(pl.Float64))
        .alias("trading_value"),
    )

    # NaN 수익률 제거 (첫 날)
    df = df.drop_nulls(subset=["daily_return"])

    # 각 날짜별 거래대금 중위수 → 소형/대형 분류
    median_tv = (
        df.group_by("dt")
        .agg(pl.col("trading_value").median().alias("median_tv"))
    )
    df = df.join(median_tv, on="dt", how="left")
    df = df.with_columns(
        pl.when(pl.col("trading_value") < pl.col("median_tv"))
        .then(pl.lit("small"))
        .otherwise(pl.lit("big"))
        .alias("size_group")
    )

    # 소형/대형 평균 수익률
    smb_df = (
        df.group_by(["dt", "size_group"])
        .agg(pl.col("daily_return").mean().alias("mean_return"))
        .pivot(on="size_group", index="dt", values="mean_return")
    )

    # pivot 결과에서 컬럼명이 size_group 값("small", "big")이 됨
    if "small" in smb_df.columns and "big" in smb_df.columns:
        smb_df = smb_df.with_columns(
            (pl.col("small").fill_null(0.0) - pl.col("big").fill_null(0.0))
            .alias("smb")
        ).select(["dt", "smb"])
    else:
        logger.warning("SMB pivot missing expected columns: %s", smb_df.columns)
        return pd.DataFrame(columns=["dt", "smb"])

    # 원래 요청 범위로 필터
    smb_df = smb_df.filter(
        (pl.col("dt") >= start_date) & (pl.col("dt") <= end_date)
    ).sort("dt")

    result = smb_df.to_pandas()
    result["dt"] = pd.to_datetime(result["dt"]).dt.date
    logger.info(
        "SMB computed: %d trading days, mean=%.6f, std=%.6f",
        len(result),
        result["smb"].mean() if len(result) > 0 else 0.0,
        result["smb"].std() if len(result) > 0 else 0.0,
    )
    return result


async def _compute_hml(
    start_date: date, end_date: date,
) -> pd.DataFrame:
    """HML 팩터: 가치주 수익률 - 성장주 수익률 (B/M 기반 분류).

    dart_financials의 BPS(주당순자산)를 종가로 나누어 B/M ratio를 구한 후:
    1. B/M 상위 30% = 가치주(Value)
    2. B/M 하위 30% = 성장주(Growth)
    3. HML = mean_return(가치주) - mean_return(성장주)

    BPS 데이터 커버리지는 ~68%이므로 데이터가 있는 종목만 사용한다.
    BPS는 공시일 기준 가장 최근 값을 사용하여 look-ahead bias를 방지한다.
    """
    conn: asyncpg.Connection = await asyncpg.connect(_dsn())
    try:
        # 1. dart_financials에서 BPS 로드 (공시일 기준 정렬)
        exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM information_schema.tables "
            "WHERE table_name='dart_financials')"
        )
        if not exists:
            logger.warning("dart_financials table not found — HML will be 0.0")
            return pd.DataFrame(columns=["dt", "hml"])

        # BPS 데이터 전체 로드 (join_asof에서 look-ahead bias 방지)
        # disclosure_date 필터 제거: 공시일이 end_date 이후일 수 있음
        # (DART 데이터는 결산일 기준이 아닌 공시일 기준으로 저장)
        dart_rows = await conn.fetch(
            """
            SELECT symbol, disclosure_date, bps::float8
            FROM dart_financials
            WHERE bps IS NOT NULL AND bps > 0
            ORDER BY symbol, disclosure_date
            """,
        )

        if not dart_rows:
            logger.warning("No BPS data in dart_financials — HML will be 0.0")
            return pd.DataFrame(columns=["dt", "hml"])

        # 2. 일봉 로드
        extended_start = start_date - timedelta(days=10)
        candle_rows = await conn.fetch(
            """
            SELECT dt::date AS dt, symbol,
                   close::float8, volume::bigint
            FROM stock_candles
            WHERE interval = '1d'
              AND dt >= $1 AND dt <= $2
              AND close > 0
            ORDER BY symbol, dt
            """,
            extended_start, end_date,
        )
    finally:
        await conn.close()

    if not candle_rows:
        logger.warning("No daily candle data for HML computation")
        return pd.DataFrame(columns=["dt", "hml"])

    # Polars로 처리
    candles = pl.DataFrame({
        "dt": [r["dt"] for r in candle_rows],
        "symbol": [r["symbol"] for r in candle_rows],
        "close": [r["close"] for r in candle_rows],
        "volume": [r["volume"] for r in candle_rows],
    }).sort(["symbol", "dt"])

    # 일별 수익률
    candles = candles.with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("symbol") - 1.0)
        .alias("daily_return")
    )
    candles = candles.drop_nulls(subset=["daily_return"])

    # BPS를 Polars로 변환
    dart_df = pl.DataFrame({
        "symbol": [r["symbol"] for r in dart_rows],
        "disclosure_date": [r["disclosure_date"] for r in dart_rows],
        "bps": [r["bps"] for r in dart_rows],
    }).sort(["symbol", "disclosure_date"])

    # join_asof: 각 캔들의 dt 시점에서 가장 최근 공시된 BPS를 매칭
    # (look-ahead bias 방지: 미래 공시 BPS를 사용하지 않음)
    candles = candles.sort(["symbol", "dt"])
    dart_df = dart_df.sort(["symbol", "disclosure_date"])
    candles = candles.join_asof(
        dart_df,
        left_on="dt",
        right_on="disclosure_date",
        by="symbol",
        strategy="backward",
    )

    # BPS가 있는 종목만 필터
    candles_with_bps = candles.drop_nulls(subset=["bps"])
    if candles_with_bps.height == 0:
        logger.warning("No BPS-matched candle data — HML will be 0.0")
        return pd.DataFrame(columns=["dt", "hml"])

    # B/M ratio = BPS / close
    candles_with_bps = candles_with_bps.with_columns(
        (pl.col("bps") / pl.col("close")).alias("bm_ratio")
    )

    # 각 날짜별 B/M 30/70 분위수 계산
    quantiles = (
        candles_with_bps.group_by("dt")
        .agg([
            pl.col("bm_ratio").quantile(0.3, interpolation="linear").alias("q30"),
            pl.col("bm_ratio").quantile(0.7, interpolation="linear").alias("q70"),
            pl.len().alias("stock_count"),
        ])
    )

    candles_with_bps = candles_with_bps.join(quantiles, on="dt", how="left")

    # 가치주(Value): B/M >= q70, 성장주(Growth): B/M <= q30
    candles_with_bps = candles_with_bps.with_columns(
        pl.when(pl.col("bm_ratio") >= pl.col("q70"))
        .then(pl.lit("value"))
        .when(pl.col("bm_ratio") <= pl.col("q30"))
        .then(pl.lit("growth"))
        .otherwise(pl.lit("neutral"))
        .alias("bm_group")
    )

    # 가치/성장 평균 수익률
    hml_groups = (
        candles_with_bps.filter(pl.col("bm_group") != "neutral")
        .group_by(["dt", "bm_group"])
        .agg(pl.col("daily_return").mean().alias("mean_return"))
        .pivot(on="bm_group", index="dt", values="mean_return")
    )

    if "value" in hml_groups.columns and "growth" in hml_groups.columns:
        hml_df = hml_groups.with_columns(
            (pl.col("value").fill_null(0.0) - pl.col("growth").fill_null(0.0))
            .alias("hml")
        ).select(["dt", "hml"])
    else:
        logger.warning(
            "HML pivot missing expected columns: %s (need 'value' and 'growth')",
            hml_groups.columns,
        )
        return pd.DataFrame(columns=["dt", "hml"])

    # 원래 요청 범위로 필터
    hml_df = hml_df.filter(
        (pl.col("dt") >= start_date) & (pl.col("dt") <= end_date)
    ).sort("dt")

    result = hml_df.to_pandas()
    result["dt"] = pd.to_datetime(result["dt"]).dt.date

    # 커버리지 진단 로그
    bps_symbols = candles_with_bps["symbol"].n_unique()
    total_symbols = candles["symbol"].n_unique()
    logger.info(
        "HML computed: %d trading days, BPS coverage=%d/%d (%.0f%%), "
        "mean=%.6f, std=%.6f",
        len(result),
        bps_symbols, total_symbols,
        bps_symbols / total_symbols * 100 if total_symbols > 0 else 0,
        result["hml"].mean() if len(result) > 0 else 0.0,
        result["hml"].std() if len(result) > 0 else 0.0,
    )
    return result


async def load_sector_mapping(
    symbols: list[str] | None = None,
) -> dict[str, int]:
    """stock_masters.sector → 정수 ID 매핑.

    기존 SQLAlchemy 엔진의 커넥션 풀을 재사용하여 연결 누수를 방지한다.

    Returns
    -------
    dict[str, int]
        symbol → sector_id (정수 인코딩)
    """
    from sqlalchemy import text

    from app.core.database import async_session

    async with async_session() as session:
        if symbols:
            result = await session.execute(
                text(
                    "SELECT symbol, sector FROM stock_masters "
                    "WHERE symbol = ANY(:symbols)"
                ),
                {"symbols": symbols},
            )
        else:
            result = await session.execute(
                text("SELECT symbol, sector FROM stock_masters")
            )
        rows = result.fetchall()

    # sector 문자열 → 정수 인코딩
    unique_sectors = sorted(set(
        r[1] for r in rows if r[1]
    ))
    sector_to_id = {s: i for i, s in enumerate(unique_sectors)}

    return {
        r[0]: sector_to_id.get(r[1], 0)
        for r in rows
    }
