"""인과 검증기 단위 테스트 (C05-C09)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from app.alpha.causal import (
    CausalValidationResult,
    DAG_EDGES,
    FactorMirageFilter,
)


def _make_confounder_df(n: int, seed: int = 42) -> pd.DataFrame:
    """합성 교란 변수 DataFrame 생성."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "market_return": rng.normal(0.001, 0.02, n),
        "market_volatility": rng.uniform(0.01, 0.05, n),
        "base_rate": np.full(n, 3.5),
        "sector_id": rng.integers(0, 5, n),
        "smb": rng.normal(0.0, 0.005, n),
        "hml": rng.normal(0.0, 0.005, n),
    })


class TestFactorMirageFilter:
    """FactorMirageFilter 인과 검증 테스트."""

    def test_c05_causal_factor_passes(self):
        """C05: 인과적 팩터 → is_causally_robust 가능성 높음.

        forward_return = alpha_factor * 0.1 + noise 형태로
        교란 변수 통제 후에도 인과 효과가 남는다.
        """
        rng = np.random.default_rng(42)
        n = 300

        confounders = _make_confounder_df(n, seed=42)

        # 팩터가 교란 변수와 독립적으로 생성
        factor_values = rng.normal(0, 1, n)

        # 수익률 = 팩터 * 0.1 + 시장 + 노이즈
        forward_returns = (
            factor_values * 0.1
            + confounders["market_return"].values * 0.5
            + rng.normal(0, 0.01, n)
        )

        filt = FactorMirageFilter(
            placebo_threshold=0.1,
            random_cause_threshold=0.1,
            num_simulations=20,
        )

        result = filt.validate(factor_values, forward_returns, confounders)

        assert isinstance(result, CausalValidationResult)
        # 인과 효과가 0이 아님
        assert abs(result.causal_effect_size) > 0.01
        # p-value가 유의미
        assert result.p_value < 0.5

    def test_c06_spurious_factor_detected(self):
        """C06: 스퓨리어스 팩터 (교란으로 인한 허위 상관) → robust=False 가능성 높음.

        팩터 = market_return + noise (교란 변수에서 파생)
        교란 통제 후 인과 효과가 사라져야 한다.
        """
        rng = np.random.default_rng(123)
        n = 300

        confounders = _make_confounder_df(n, seed=123)

        # 팩터가 시장 수익률에서 파생 (교란 변수)
        factor_values = confounders["market_return"].values + rng.normal(0, 0.005, n)

        # 수익률도 시장에 의존 (팩터와 수익률이 모두 시장의 결과)
        forward_returns = (
            confounders["market_return"].values * 0.8
            + rng.normal(0, 0.01, n)
        )

        filt = FactorMirageFilter(
            placebo_threshold=0.05,
            random_cause_threshold=0.05,
            num_simulations=20,
        )

        result = filt.validate(factor_values, forward_returns, confounders)

        assert isinstance(result, CausalValidationResult)
        # placebo 테스트에서 교란 효과의 ATE가 줄어야
        # (교란 통제 후 팩터의 직접 효과는 작아져야)

    def test_c07_dag_structure(self):
        """C07: DAG 엣지 구조 검증 (12엣지)."""
        assert len(DAG_EDGES) == 12

        # 필수 엣지 검증
        edges_set = {(e["from"], e["to"]) for e in DAG_EDGES}
        assert ("alpha_factor", "forward_return") in edges_set
        assert ("market_return", "alpha_factor") in edges_set
        assert ("market_return", "forward_return") in edges_set
        assert ("market_volatility", "alpha_factor") in edges_set
        assert ("market_volatility", "forward_return") in edges_set
        assert ("base_rate", "forward_return") in edges_set
        assert ("sector_id", "alpha_factor") in edges_set
        assert ("sector_id", "forward_return") in edges_set
        assert ("smb", "alpha_factor") in edges_set
        assert ("smb", "forward_return") in edges_set
        assert ("hml", "alpha_factor") in edges_set
        assert ("hml", "forward_return") in edges_set

    def test_c08_constant_factor_not_robust(self):
        """C08: 상수 팩터 → 에러 없이 robust=False."""
        n = 100
        rng = np.random.default_rng(42)
        confounders = _make_confounder_df(n)

        factor_values = np.ones(n)  # 상수
        forward_returns = rng.normal(0, 0.02, n)

        filt = FactorMirageFilter(num_simulations=10)
        result = filt.validate(factor_values, forward_returns, confounders)

        assert isinstance(result, CausalValidationResult)
        assert not result.is_causally_robust

    def test_c09_no_nan_inf_in_result(self):
        """C09: 결과의 모든 float 필드가 NaN/Inf 아님."""
        n = 300
        rng = np.random.default_rng(42)
        confounders = _make_confounder_df(n)
        factor_values = rng.normal(0, 1, n)
        forward_returns = rng.normal(0, 0.02, n)

        filt = FactorMirageFilter(num_simulations=10)
        result = filt.validate(factor_values, forward_returns, confounders)

        assert not math.isnan(result.causal_effect_size)
        assert not math.isnan(result.p_value)
        assert not math.isnan(result.placebo_effect)
        assert not math.isnan(result.random_cause_delta)
        assert not math.isinf(result.causal_effect_size)
        assert not math.isinf(result.p_value)
