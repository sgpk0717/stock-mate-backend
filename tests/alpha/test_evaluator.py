"""IC 평가기 단위 테스트 (T07-T14)."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from app.alpha.evaluator import (
    compute_factor_metrics,
    compute_forward_returns,
    compute_ic_series,
)


def _make_single_symbol_df(
    n: int = 100,
    factor_fn=None,
    seed: int = 42,
) -> pl.DataFrame:
    """단일 종목 테스트용 DF 생성.

    factor_fn: close, fwd_return를 받아 팩터 값 반환하는 함수.
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.001, 0.02, n)
    prices = 50000.0 * np.cumprod(1 + returns)

    dates = pl.date_range(
        pl.date(2024, 1, 2),
        pl.date(2024, 1, 2) + pl.duration(days=n - 1),
        eager=True,
    )

    df = pl.DataFrame({
        "dt": dates,
        "close": prices,
    })

    # fwd_return 추가
    df = compute_forward_returns(df, periods=1)

    if factor_fn is not None:
        close_arr = df["close"].to_numpy()
        fwd_arr = df["fwd_return"].to_numpy()
        factor_arr = factor_fn(close_arr, fwd_arr)
        df = df.with_columns(pl.Series("alpha_factor", factor_arr))

    return df


class TestICCalculation:
    """IC 계산 정확성 테스트."""

    def test_t07_perfect_positive_correlation(self):
        """T07: 완벽 양의 상관 → IC ≈ 1.0."""
        # 팩터 = fwd_return (완벽 상관)
        df = _make_single_symbol_df(
            n=200,
            factor_fn=lambda c, r: r,
        )

        ic_series = compute_ic_series(df, factor_col="alpha_factor")
        assert len(ic_series) > 0

        metrics = compute_factor_metrics(ic_series)
        # 완벽 상관이므로 IC 평균이 0.8 이상
        assert metrics.ic_mean > 0.8, f"Expected IC > 0.8, got {metrics.ic_mean}"

    def test_t08_random_noise_factor(self):
        """T08: 랜덤 노이즈 팩터 → |IC| < 0.15."""
        rng = np.random.default_rng(123)

        df = _make_single_symbol_df(
            n=200,
            factor_fn=lambda c, r: rng.normal(0, 1, len(c)),
        )

        ic_series = compute_ic_series(df, factor_col="alpha_factor")
        metrics = compute_factor_metrics(ic_series)

        assert abs(metrics.ic_mean) < 0.15, (
            f"Random factor IC should be near 0, got {metrics.ic_mean}"
        )

    def test_t09_negative_correlation(self):
        """T09: 음의 상관 팩터 → IC < 0."""
        df = _make_single_symbol_df(
            n=200,
            factor_fn=lambda c, r: -r,  # 역상관
        )

        ic_series = compute_ic_series(df, factor_col="alpha_factor")
        metrics = compute_factor_metrics(ic_series)

        assert metrics.ic_mean < -0.5, f"Expected IC < -0.5, got {metrics.ic_mean}"

    def test_t10_insufficient_data(self):
        """T10: 데이터 10행 미만 → 빈 ic_series."""
        rng = np.random.default_rng(42)
        df = pl.DataFrame({
            "dt": pl.date_range(pl.date(2024, 1, 2), pl.date(2024, 1, 8), eager=True),
            "close": [100.0, 101.0, 99.0, 102.0, 98.0, 103.0, 97.0],
            "alpha_factor": rng.normal(0, 1, 7),
        })
        df = compute_forward_returns(df, periods=1)

        ic_series = compute_ic_series(df, factor_col="alpha_factor")
        assert ic_series == []


class TestNaNHandling:
    """NaN/엣지 케이스 테스트."""

    def test_t11_nan_values_filtered(self):
        """T11: NaN 포함 데이터 → 필터 후 정상 계산."""
        rng = np.random.default_rng(42)
        n = 100
        factor_vals = rng.normal(0, 1, n).tolist()
        # 20% NaN 삽입
        for i in range(0, n, 5):
            factor_vals[i] = float("nan")

        df = pl.DataFrame({
            "dt": pl.date_range(
                pl.date(2024, 1, 2),
                pl.date(2024, 1, 2) + pl.duration(days=n - 1),
                eager=True,
            ),
            "close": 50000.0 * np.cumprod(1 + rng.normal(0, 0.02, n)),
            "alpha_factor": factor_vals,
        })
        df = compute_forward_returns(df, periods=1)

        ic_series = compute_ic_series(df, factor_col="alpha_factor")
        # NaN 행이 필터되어도 계산은 되어야
        # (데이터가 충분하면 결과 반환)
        metrics = compute_factor_metrics(ic_series)
        assert not math.isnan(metrics.ic_mean)

    def test_t12_constant_factor(self):
        """T12: 상수 팩터 (모든 값 동일) → NaN 아닌 결과."""
        n = 100
        rng = np.random.default_rng(42)

        df = pl.DataFrame({
            "dt": pl.date_range(
                pl.date(2024, 1, 2),
                pl.date(2024, 1, 2) + pl.duration(days=n - 1),
                eager=True,
            ),
            "close": 50000.0 * np.cumprod(1 + rng.normal(0, 0.02, n)),
            "alpha_factor": [1.0] * n,  # 상수
        })
        df = compute_forward_returns(df, periods=1)

        ic_series = compute_ic_series(df, factor_col="alpha_factor")
        metrics = compute_factor_metrics(ic_series)

        # 상수 팩터는 IC 계산 불가 → 빈 시리즈 또는 0 반환
        assert not math.isnan(metrics.ic_mean)
        assert not math.isinf(metrics.ic_mean)

    def test_t13_no_nan_inf_in_metrics(self):
        """T13: compute_factor_metrics 반환에 NaN/Inf 없음."""
        # 다양한 입력
        test_cases = [
            [],
            [0.0],
            [0.1, -0.1, 0.05, -0.05],
            [float("nan"), 0.1, 0.2],  # NaN이 ic_series에 있을 수 있음
        ]

        for ic_series in test_cases:
            # NaN 필터
            clean = [x for x in ic_series if not math.isnan(x)]
            metrics = compute_factor_metrics(clean)

            assert not math.isnan(metrics.ic_mean), f"NaN ic_mean for {ic_series}"
            assert not math.isnan(metrics.ic_std), f"NaN ic_std for {ic_series}"
            assert not math.isnan(metrics.icir), f"NaN icir for {ic_series}"
            assert not math.isnan(metrics.sharpe), f"NaN sharpe for {ic_series}"
            assert not math.isnan(metrics.turnover), f"NaN turnover for {ic_series}"
            assert not math.isnan(metrics.max_drawdown), f"NaN mdd for {ic_series}"

            assert not math.isinf(metrics.ic_mean)
            assert not math.isinf(metrics.icir)
            assert not math.isinf(metrics.sharpe)


class TestMDD:
    """MDD 계산 테스트."""

    def test_t14_mdd_known_sequence(self):
        """T14: L/S 수익률 기반 MDD 정확성 검증."""
        # ls_returns가 없으면 MDD=0.0
        ic_series = [0.1, 0.2, -0.1, 0.15]
        metrics = compute_factor_metrics(ic_series)
        assert metrics.max_drawdown == 0.0, (
            "Without ls_returns, MDD should be 0.0"
        )

        # ls_returns 제공 시 실제 포트폴리오 MDD 계산
        # 수익률: [0.05, 0.03, -0.10, 0.02]
        # 누적: [1.05, 1.0815, 0.97335, 0.992697]
        # peak: [1.05, 1.0815, 1.0815, 1.0815]
        # dd:   [0.0, 0.0, -0.0999..., -0.0820...]
        ls_returns = [0.05, 0.03, -0.10, 0.02]
        metrics2 = compute_factor_metrics(ic_series, ls_returns=ls_returns)
        assert metrics2.max_drawdown < 0, "MDD should be negative"
        assert abs(metrics2.max_drawdown - (-0.0999)) < 0.01, (
            f"Expected MDD≈-0.10, got {metrics2.max_drawdown}"
        )
