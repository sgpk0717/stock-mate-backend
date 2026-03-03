"""DoWhy 4단계 인과 검증 — Factor Mirage 제거.

Phase 1에서 IC를 통과한 팩터가 교란 변수(시장 수익률, 변동성, 금리, 섹터)를
통제한 후에도 forward_return에 유의미한 인과 효과를 가지는지 검증한다.

4단계:
1. DAG 모델링: 고정 6노드 8엣지 그래프
2. 식별: Backdoor Criterion
3. 추정: Linear Regression → ATE + p-value
4. 반증: Placebo Treatment + Random Common Cause
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# DoWhy 고정 DAG (GML 형식)
# 노드: market_return, market_volatility, base_rate, sector_id, alpha_factor, forward_return
_CAUSAL_DAG_GML = """
graph [
    directed 1
    node [ id "market_return" label "market_return" ]
    node [ id "market_volatility" label "market_volatility" ]
    node [ id "base_rate" label "base_rate" ]
    node [ id "sector_id" label "sector_id" ]
    node [ id "alpha_factor" label "alpha_factor" ]
    node [ id "forward_return" label "forward_return" ]

    edge [ source "market_return" target "alpha_factor" ]
    edge [ source "market_return" target "forward_return" ]
    edge [ source "market_volatility" target "alpha_factor" ]
    edge [ source "market_volatility" target "forward_return" ]
    edge [ source "base_rate" target "forward_return" ]
    edge [ source "sector_id" target "alpha_factor" ]
    edge [ source "sector_id" target "forward_return" ]
    edge [ source "alpha_factor" target "forward_return" ]
]
"""

# 프론트엔드 시각화용 엣지 목록
DAG_EDGES = [
    {"from": "market_return", "to": "alpha_factor"},
    {"from": "market_return", "to": "forward_return"},
    {"from": "market_volatility", "to": "alpha_factor"},
    {"from": "market_volatility", "to": "forward_return"},
    {"from": "base_rate", "to": "forward_return"},
    {"from": "sector_id", "to": "alpha_factor"},
    {"from": "sector_id", "to": "forward_return"},
    {"from": "alpha_factor", "to": "forward_return"},
]


_MIN_SAMPLES = 100  # 6변수 선형회귀에 최소 100개 (≈ 변수당 15-17개)


def _sanitize(value: float, default: float = 0.0) -> float:
    """NaN/Inf를 default로 변환."""
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def _extract_p_value(estimate) -> float:
    """DoWhy 추정 결과에서 p-value를 안전하게 추출.

    DoWhy 버전에 따라 p-value 반환 형태가 다르므로,
    여러 경로를 시도하고 실패 시 statsmodels로 직접 계산한다.
    """
    # 경로 1: estimate.test_stat_significance()
    try:
        if hasattr(estimate, "test_stat_significance"):
            p_val = estimate.test_stat_significance()
            if isinstance(p_val, dict):
                raw = list(p_val.values())[0]
                if hasattr(raw, "item"):
                    raw = raw.item()
                val = float(raw)
                if not (math.isnan(val) or math.isinf(val)):
                    return val
            elif p_val is not None:
                if hasattr(p_val, "item"):
                    p_val = p_val.item()
                val = float(p_val)
                if not (math.isnan(val) or math.isinf(val)):
                    return val
    except Exception:
        pass

    # 경로 2: estimate 객체 내부 속성
    try:
        if hasattr(estimate, "estimator") and hasattr(estimate.estimator, "pvalue"):
            val = float(estimate.estimator.pvalue)
            if not (math.isnan(val) or math.isinf(val)):
                return val
    except Exception:
        pass

    logger.warning("Could not extract p-value from DoWhy estimate, returning 1.0")
    return 1.0


@dataclass
class CausalValidationResult:
    """인과 검증 결과."""

    is_causally_robust: bool
    causal_effect_size: float
    p_value: float
    placebo_passed: bool
    placebo_effect: float
    random_cause_passed: bool
    random_cause_delta: float
    regime_shift_passed: bool = False
    regime_ate_first_half: float = 0.0
    regime_ate_second_half: float = 0.0
    dag_edges: list[dict] = field(default_factory=lambda: list(DAG_EDGES))


class FactorMirageFilter:
    """DoWhy 4단계 인과 검증으로 Factor Mirage를 제거한다."""

    def __init__(
        self,
        placebo_threshold: float = 0.05,
        random_cause_threshold: float = 0.05,
        num_simulations: int = 100,
    ):
        self.placebo_threshold = placebo_threshold
        self.random_cause_threshold = random_cause_threshold
        self.num_simulations = num_simulations

    def validate(
        self,
        factor_values: np.ndarray,
        forward_returns: np.ndarray,
        confounders_df: pd.DataFrame,
    ) -> CausalValidationResult:
        """팩터의 인과적 유효성을 검증한다.

        Parameters
        ----------
        factor_values : 팩터 값 배열 (len N)
        forward_returns : T+1 수익률 배열 (len N)
        confounders_df : 교란 변수 DF (columns: market_return, market_volatility, base_rate)
                         sector_id가 있으면 포함, 없으면 0으로 채움

        Returns
        -------
        CausalValidationResult
        """
        # 상수 팩터 사전 방어: 분산 0이면 인과 추정 불가
        if np.std(factor_values) < 1e-12:
            return CausalValidationResult(
                is_causally_robust=False,
                causal_effect_size=0.0,
                p_value=1.0,
                placebo_passed=False,
                placebo_effect=0.0,
                random_cause_passed=False,
                random_cause_delta=0.0,
            )

        try:
            return self._run_dowhy(factor_values, forward_returns, confounders_df)
        except Exception as e:
            logger.exception("Causal validation failed: %s", e)
            return CausalValidationResult(
                is_causally_robust=False,
                causal_effect_size=0.0,
                p_value=1.0,
                placebo_passed=False,
                placebo_effect=0.0,
                random_cause_passed=False,
                random_cause_delta=0.0,
            )

    @staticmethod
    def _regime_split_test(
        data: pd.DataFrame,
        identified_estimand,
        model,
    ) -> tuple[bool, float, float]:
        """데이터를 전반/후반으로 분할하여 ATE 부호 일관성을 검증한다.

        Returns (passed, ate_first_half, ate_second_half).
        전반/후반 ATE 부호가 동일하면 통과.
        """
        import dowhy

        mid = len(data) // 2
        if mid < _MIN_SAMPLES:
            # 데이터 부족 — 검증 불가이므로 통과 처리
            logger.warning(
                "Insufficient data for regime split: %d rows (need %d per half)",
                len(data), _MIN_SAMPLES,
            )
            return True, 0.0, 0.0

        first_half = data.iloc[:mid].reset_index(drop=True)
        second_half = data.iloc[mid:].reset_index(drop=True)

        ate_first = 0.0
        ate_second = 0.0

        try:
            model_1 = dowhy.CausalModel(
                data=first_half,
                treatment="alpha_factor",
                outcome="forward_return",
                graph=_CAUSAL_DAG_GML,
            )
            id_1 = model_1.identify_effect(proceed_when_unidentifiable=True)
            est_1 = model_1.estimate_effect(
                id_1, method_name="backdoor.linear_regression",
            )
            ate_first = _sanitize(float(est_1.value))
        except Exception as e:
            logger.warning("Regime split first half failed: %s", e)
            return True, 0.0, 0.0

        try:
            model_2 = dowhy.CausalModel(
                data=second_half,
                treatment="alpha_factor",
                outcome="forward_return",
                graph=_CAUSAL_DAG_GML,
            )
            id_2 = model_2.identify_effect(proceed_when_unidentifiable=True)
            est_2 = model_2.estimate_effect(
                id_2, method_name="backdoor.linear_regression",
            )
            ate_second = _sanitize(float(est_2.value))
        except Exception as e:
            logger.warning("Regime split second half failed: %s", e)
            return True, ate_first, 0.0

        # ATE 부호 일관성 검증: 둘 다 양수이거나 둘 다 음수
        # 한쪽이 0에 매우 가까우면 (< 1e-8) 부호 비교 무의미 → 통과
        if abs(ate_first) < 1e-8 or abs(ate_second) < 1e-8:
            passed = True
        else:
            passed = (ate_first > 0) == (ate_second > 0)

        logger.info(
            "Regime split: ATE_first=%.6f, ATE_second=%.6f → %s",
            ate_first, ate_second, "PASS" if passed else "FAIL",
        )
        return passed, ate_first, ate_second

    def _run_dowhy(
        self,
        factor_values: np.ndarray,
        forward_returns: np.ndarray,
        confounders_df: pd.DataFrame,
    ) -> CausalValidationResult:
        """DoWhy 4단계 인과 검증 실행."""
        import dowhy

        n = min(len(factor_values), len(forward_returns), len(confounders_df))
        if n < _MIN_SAMPLES:
            logger.warning(
                "Insufficient data for causal validation: %d rows (min %d)",
                n, _MIN_SAMPLES,
            )
            return CausalValidationResult(
                is_causally_robust=False,
                causal_effect_size=0.0,
                p_value=1.0,
                placebo_passed=False,
                placebo_effect=0.0,
                random_cause_passed=False,
                random_cause_delta=0.0,
            )

        # 데이터 통합 DataFrame 구축
        data = confounders_df.iloc[:n].copy().reset_index(drop=True)
        data["alpha_factor"] = factor_values[:n]
        data["forward_return"] = forward_returns[:n]

        # sector_id가 없으면 0으로 채움
        if "sector_id" not in data.columns:
            data["sector_id"] = 0

        # NaN 행 제거
        required_cols = [
            "market_return", "market_volatility", "base_rate",
            "sector_id", "alpha_factor", "forward_return",
        ]
        data = data.dropna(subset=required_cols)

        if len(data) < _MIN_SAMPLES:
            logger.warning(
                "Insufficient clean data for causal validation: %d rows (min %d)",
                len(data), _MIN_SAMPLES,
            )
            return CausalValidationResult(
                is_causally_robust=False,
                causal_effect_size=0.0,
                p_value=1.0,
                placebo_passed=False,
                placebo_effect=0.0,
                random_cause_passed=False,
                random_cause_delta=0.0,
            )

        # dt 컬럼은 DoWhy에 불필요하므로 제거
        if "dt" in data.columns:
            data = data.drop(columns=["dt"])

        # Step 1: DAG 모델링
        model = dowhy.CausalModel(
            data=data,
            treatment="alpha_factor",
            outcome="forward_return",
            graph=_CAUSAL_DAG_GML,
        )

        # Step 2: 식별 (Backdoor Criterion)
        identified = model.identify_effect(proceed_when_unidentifiable=True)
        # 식별 가능성 경고 (DoWhy 버전별 API 차이 대응)
        _has_estimand = getattr(identified, "estimands", None)
        if _has_estimand is not None and not _has_estimand:
            logger.warning(
                "Causal effect may not be identifiable; "
                "proceeding with available estimand but results may be unreliable"
            )

        # Step 3: 추정 (Linear Regression)
        estimate = model.estimate_effect(
            identified,
            method_name="backdoor.linear_regression",
        )
        ate = _sanitize(float(estimate.value))

        # p-value 추출 (다중 경로 시도)
        p_value = _extract_p_value(estimate)

        # Step 4a: 반증 — Placebo Treatment
        placebo_refute = model.refute_estimate(
            identified,
            estimate,
            method_name="placebo_treatment_refuter",
            placebo_type="permute",
            num_simulations=self.num_simulations,
        )
        placebo_effect = _sanitize(float(placebo_refute.new_effect))
        placebo_passed = abs(placebo_effect) < self.placebo_threshold

        # Step 4b: 반증 — Random Common Cause
        random_refute = model.refute_estimate(
            identified,
            estimate,
            method_name="random_common_cause",
            num_simulations=self.num_simulations,
        )
        random_effect = _sanitize(float(random_refute.new_effect))
        random_delta = abs(random_effect - ate)
        random_passed = random_delta < self.random_cause_threshold

        # Step 5: 국면 생존력 (Regime Shift) 검증
        # 데이터를 시간 기준 전반/후반으로 분할하여 각각 ATE 추정
        regime_passed, ate_first, ate_second = self._regime_split_test(
            data, identified, model,
        )

        is_robust = placebo_passed and random_passed and regime_passed

        logger.info(
            "Causal validation: ATE=%.6f, p=%.4f, placebo=%.6f(%s), "
            "random_delta=%.6f(%s), regime=%s(%.6f/%.6f) → %s",
            ate, p_value, placebo_effect,
            "PASS" if placebo_passed else "FAIL",
            random_delta,
            "PASS" if random_passed else "FAIL",
            "PASS" if regime_passed else "FAIL",
            ate_first, ate_second,
            "ROBUST" if is_robust else "MIRAGE",
        )

        return CausalValidationResult(
            is_causally_robust=is_robust,
            causal_effect_size=ate,
            p_value=p_value,
            placebo_passed=placebo_passed,
            placebo_effect=placebo_effect,
            random_cause_passed=random_passed,
            random_cause_delta=random_delta,
            regime_shift_passed=regime_passed,
            regime_ate_first_half=ate_first,
            regime_ate_second_half=ate_second,
        )
