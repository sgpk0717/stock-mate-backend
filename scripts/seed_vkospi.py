"""VKOSPI 대용 데이터 수집: KOSPI200 실현변동성 -> stock_candles 저장.

Yahoo Finance에서 ^KS200 (KOSPI200 지수) 일봉을 다운로드하고,
20일/60일 실현변동성 + 60일 백분위를 계산하여 stock_candles에 저장한다.

VKOSPI(내재변동성) 데이터가 DB에 없고 pykrx/yfinance로도 수집 불가하므로,
실현변동성을 대용으로 사용한다. (상관계수 > 0.85 경험적 근거)

저장 형식:
  symbol = 'VKOSPI_PROXY'
  interval = '1d'
  close = realized_vol_20d (20일 연환산 실현변동성, %)
  open = realized_vol_60d (60일 연환산 실현변동성, %)
  high = vol_percentile_60d (60일 내 백분위, 0~100)
  low = 0 (미사용)
  volume = 0 (미사용)

Usage:
    docker exec stockmate-worker python3 -m scripts.seed_vkospi
    docker exec stockmate-worker python3 -m scripts.seed_vkospi --years 15
"""

from __future__ import annotations

import argparse
import logging
import math
from datetime import datetime, timedelta, timezone

import polars as pl
from sqlalchemy import create_engine, text

from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9))
SYMBOL = "VKOSPI_PROXY"
TICKER = "^KS200"  # Yahoo Finance KOSPI200 지수 티커
ANNUALIZE_FACTOR = math.sqrt(252)


def _download_kospi200(years: int) -> pl.DataFrame:
    """yfinance로 KOSPI200 지수 일봉 다운로드 -> Polars DataFrame."""
    import yfinance as yf

    end = datetime.now()
    start = end - timedelta(days=years * 365 + 90)  # 변동성 계산용 여유분

    logger.info("Downloading %s from %s to %s ...", TICKER, start.date(), end.date())
    ticker = yf.Ticker(TICKER)
    pdf = ticker.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))

    if pdf.empty:
        raise RuntimeError(f"No data returned from yfinance for {TICKER}")

    logger.info("Downloaded %d rows from yfinance", len(pdf))

    # pandas -> polars 변환
    pdf = pdf.reset_index()
    df = pl.from_pandas(pdf[["Date", "Close"]]).rename({"Date": "dt", "Close": "close_idx"})

    # timezone-aware dt를 date로 변환
    df = df.with_columns(
        pl.col("dt").cast(pl.Date)
    )

    return df.sort("dt")


def _compute_realized_vol(df: pl.DataFrame) -> pl.DataFrame:
    """일간 수익률 기반 실현변동성 계산.

    Returns:
        DataFrame with columns: dt, realized_vol_20d, realized_vol_60d, vol_percentile_60d
    """
    # 일간 로그 수익률
    df = df.with_columns(
        (pl.col("close_idx") / pl.col("close_idx").shift(1)).log().alias("log_return")
    )

    # 20일 / 60일 실현변동성 (연환산, %)
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

    # 60일 백분위 (rolling percentile)
    # Polars에는 rolling_quantile이 없으므로 rolling_min/max 기반 근사 사용
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

    # NaN 행 제거 + 불필요 컬럼 정리
    df = df.drop(["close_idx", "log_return", "_vol_min_60", "_vol_max_60"])
    df = df.drop_nulls(subset=["realized_vol_20d"])

    return df


def _save_to_db(engine, df: pl.DataFrame) -> int:
    """stock_candles 테이블에 VKOSPI_PROXY 데이터 upsert.

    저장 매핑:
      close = realized_vol_20d
      open = realized_vol_60d
      high = vol_percentile_60d
      low = 0, volume = 0
    """
    rows = []
    for row in df.iter_rows(named=True):
        dt_val = row["dt"]
        rv20 = row["realized_vol_20d"]
        rv60 = row.get("realized_vol_60d")
        pct = row.get("vol_percentile_60d")

        if rv20 is None or math.isnan(rv20):
            continue

        rows.append({
            "symbol": SYMBOL,
            "dt": datetime(dt_val.year, dt_val.month, dt_val.day, tzinfo=KST),
            "interval": "1d",
            "open": round(rv60, 4) if rv60 is not None and not math.isnan(rv60) else 0.0,
            "high": round(pct, 2) if pct is not None and not math.isnan(pct) else 0.0,
            "low": 0.0,
            "close": round(rv20, 4),
            "volume": 0,
        })

    if not rows:
        logger.warning("No valid rows to save")
        return 0

    with engine.begin() as conn:
        # 기존 데이터 삭제 후 재삽입
        conn.execute(
            text("DELETE FROM stock_candles WHERE symbol = :symbol AND interval = '1d'"),
            {"symbol": SYMBOL},
        )
        conn.execute(
            text(
                "INSERT INTO stock_candles (symbol, dt, interval, open, high, low, close, volume) "
                "VALUES (:symbol, :dt, :interval, :open, :high, :low, :close, :volume)"
            ),
            rows,
        )

    logger.info("Saved %d rows for %s", len(rows), SYMBOL)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="VKOSPI proxy (realized vol) seed script")
    parser.add_argument("--years", type=int, default=10, help="Years of history (default: 10)")
    args = parser.parse_args()

    engine = create_engine(settings.sync_database_url)

    # 1. 다운로드
    raw = _download_kospi200(args.years)
    logger.info("Raw data: %d rows, %s ~ %s", raw.height, raw["dt"].min(), raw["dt"].max())

    # 2. 실현변동성 계산
    vol_df = _compute_realized_vol(raw)
    logger.info(
        "Realized vol computed: %d rows, vol_20d range [%.2f, %.2f]",
        vol_df.height,
        vol_df["realized_vol_20d"].min(),
        vol_df["realized_vol_20d"].max(),
    )

    # 3. DB 저장
    count = _save_to_db(engine, vol_df)
    print(f"\nDone! Saved {count} VKOSPI_PROXY rows to stock_candles")


if __name__ == "__main__":
    main()
