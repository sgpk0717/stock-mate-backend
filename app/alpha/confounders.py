"""교란 변수(Confounder) 데이터 로더.

DoWhy 인과 검증에 필요한 교란 변수를 로드:
- market_return: KOSPI 200 ETF(069500) 일간 수익률
- market_volatility: 20일 실현 변동성
- base_rate: 한국은행 기준금리 (일별 forward-fill)
- sector_id: 종목의 섹터 정수 ID
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path

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
        columns: [dt, market_return, market_volatility, base_rate]
        (sector_id는 종목별이므로 별도 매핑 dict로 반환)
    """
    market_df = await _load_market_data(start_date, end_date)
    rate_df = _load_base_rate(start_date, end_date)

    # dt 기준 병합
    merged = pd.merge(market_df, rate_df, on="dt", how="left")
    # forward-fill로 빈 날짜 채우기
    merged["base_rate"] = merged["base_rate"].ffill().bfill()
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
    ])

    # 원래 요청 범위로 필터
    df = df.filter(
        (pl.col("dt") >= start_date) & (pl.col("dt") <= end_date)
    )

    result = df.select(["dt", "market_return", "market_volatility"]).to_pandas()
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
