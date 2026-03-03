"""테스트 공통 fixture."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest


@pytest.fixture
def sample_ohlcv_df() -> pl.DataFrame:
    """200행 합성 OHLCV Polars DataFrame (시드 고정)."""
    rng = np.random.default_rng(42)
    n = 200
    base_price = 50000.0

    # 랜덤 워크로 현실적 가격 생성
    returns = rng.normal(0.0005, 0.02, n)
    prices = base_price * np.cumprod(1 + returns)

    close = prices
    open_ = close * (1 + rng.normal(0, 0.005, n))
    high = np.maximum(close, open_) * (1 + rng.uniform(0, 0.01, n))
    low = np.minimum(close, open_) * (1 - rng.uniform(0, 0.01, n))
    volume = rng.integers(100_000, 10_000_000, n).astype(float)

    dates = pl.date_range(
        pl.date(2024, 1, 2),
        pl.date(2024, 1, 2) + pl.duration(days=n - 1),
        eager=True,
    )

    return pl.DataFrame({
        "dt": dates,
        "symbol": ["005930"] * n,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


@pytest.fixture
def sample_ohlcv_with_indicators(sample_ohlcv_df: pl.DataFrame) -> pl.DataFrame:
    """ensure_alpha_features() 적용된 DF."""
    from app.alpha.ast_converter import ensure_alpha_features

    return ensure_alpha_features(sample_ohlcv_df)
