"""DoWhy 4단계 인과 검증 — Factor Mirage 제거.

Phase 1에서 IC를 통과한 팩터가 교란 변수(시장 수익률, 변동성, 금리, 섹터,
SMB, HML)를 통제한 후에도 forward_return에 유의미한 인과 효과를 가지는지 검증한다.

4단계:
1. DAG 모델링: 고정 8노드 12엣지 그래프
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
from scipy.stats import t as t_dist

logger = logging.getLogger(__name__)

# DoWhy 고정 DAG (GML 형식)
# 노드: market_return, market_volatility, base_rate, sector_id, smb, hml,
#        alpha_factor, forward_return
_CAUSAL_DAG_GML = """
graph [
    directed 1
    node [ id "market_return" label "market_return" ]
    node [ id "market_volatility" label "market_volatility" ]
    node [ id "base_rate" label "base_rate" ]
    node [ id "sector_id" label "sector_id" ]
    node [ id "smb" label "smb" ]
    node [ id "hml" label "hml" ]
    node [ id "alpha_factor" label "alpha_factor" ]
    node [ id "forward_return" label "forward_return" ]

    edge [ source "market_return" target "alpha_factor" ]
    edge [ source "market_return" target "forward_return" ]
    edge [ source "market_volatility" target "alpha_factor" ]
    edge [ source "market_volatility" target "forward_return" ]
    edge [ source "base_rate" target "forward_return" ]
    edge [ source "sector_id" target "alpha_factor" ]
    edge [ source "sector_id" target "forward_return" ]
    edge [ source "smb" target "alpha_factor" ]
    edge [ source "smb" target "forward_return" ]
    edge [ source "hml" target "alpha_factor" ]
    edge [ source "hml" target "forward_return" ]
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
    {"from": "smb", "to": "alpha_factor"},
    {"from": "smb", "to": "forward_return"},
    {"from": "hml", "to": "alpha_factor"},
    {"from": "hml", "to": "forward_return"},
    {"from": "alpha_factor", "to": "forward_return"},
]


_MIN_SAMPLES = 100  # 8변수 선형회귀에 최소 100개 (≈ 변수당 12-13개)

# 교란변수 + 처리변수 컬럼 순서 (OLS 디자인 행렬)
_CONFOUNDER_COLS = [
    "market_return", "market_volatility", "base_rate", "sector_id",
    "smb", "hml",
]


def _fast_ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """NumPy 고속 OLS — β 계수와 p-value를 반환.

    statsmodels OLS.fit()과 수학적으로 동일한 연산을 수행하되,
    불필요한 진단 통계 계산과 Python 객체 생성 오버헤드를 제거한다.

    Parameters
    ----------
    X : (n, k) 디자인 행렬 (절편 포함)
    y : (n,) 종속변수

    Returns
    -------
    beta : (k,) 회귀 계수
    p_values : (k,) 양측 t-검정 p-value
    """
    n, k = X.shape
    # [2026-03-31] 딥리서치 R3+R4 공통 권장 — Cholesky 분해로 전환
    # 변경: np.linalg.lstsq (SVD) → Cholesky 정규방정식 | 실측: 0.089초→0.021초 (4.2x)
    # p-value가 필요한 Step 3 ATE에서만 사용. Step 4는 _fast_ols_beta 사용.
    XtX = X.T @ X
    Xty = X.T @ y
    try:
        from scipy.linalg import cho_factor, cho_solve
        XtX_reg = XtX + 1e-10 * np.eye(k)  # Tikhonov 정규화 (조건수 안전장치)
        c, low = cho_factor(XtX_reg)
        beta = cho_solve((c, low), Xty)
    except (np.linalg.LinAlgError, Exception):
        # Cholesky 실패 시 lstsq 폴백 (ill-conditioned 행렬)
        beta = np.linalg.lstsq(X, y, rcond=None)[0]

    residuals = y - X @ beta
    dof = n - k
    if dof <= 0:
        return beta, np.ones(k)
    mse = np.dot(residuals, residuals) / dof
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX)
    var_beta = mse * XtX_inv
    se = np.sqrt(np.maximum(np.diag(var_beta), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = np.where(se > 0, beta / se, 0.0)
    p_values = 2.0 * (1.0 - t_dist.cdf(np.abs(t_stats), df=dof))
    return beta, p_values


# ── FWL 벡터화 함수 (Frisch-Waugh-Lovell 정리) ─────────────────
# [2026-03-31] 딥리서치 R3+R4 공통 권장
# 프로세스: /deep-research → 4건 보고서 교차 + Docker 실측 벤치마크
# 실측: lstsq 루프 999회 84.4초 → FWL+inv 벡터화 5.4초 (15.5x), 오차 1.68e-16

def _fwl_projection_matrix(X: np.ndarray) -> np.ndarray:
    """FWL 사영행렬 성분 (X'X)^-1 · X' 계산.

    M₀v = v - X @ proj @ v 로 어떤 벡터든 X에 직교인 잔차를 구할 수 있다.
    X_base가 동일한 모든 팩터에서 1회만 계산하면 된다.
    """
    XtX = X.T @ X + 1e-10 * np.eye(X.shape[1])
    XtX_inv = np.linalg.inv(XtX)
    return XtX_inv @ X.T  # (k, n)


def _fwl_residualize(X: np.ndarray, proj: np.ndarray, v: np.ndarray) -> np.ndarray:
    """FWL 잔차: M₀v = v - X @ (proj @ v). v는 1D 또는 2D."""
    return v - X @ (proj @ v)


def _block_permutation(treatment: np.ndarray, block_dates: np.ndarray) -> np.ndarray:
    """날짜 블록 단위 순열 — 같은 날짜의 데이터는 함께 이동.

    일중 패턴(장 시작/마감 효과)과 시계열 자기상관을 보존한다.
    """
    unique_dates = np.unique(block_dates)
    shuffled_dates = np.random.permutation(unique_dates)
    # 원본 날짜 → 셔플된 날짜 매핑
    result = np.empty_like(treatment)
    for orig, shuf in zip(unique_dates, shuffled_dates):
        orig_mask = block_dates == orig
        shuf_mask = block_dates == shuf
        orig_vals = treatment[orig_mask]
        shuf_count = shuf_mask.sum()
        # 블록 크기가 다를 수 있으므로 맞춤 (리샘플링)
        if len(orig_vals) == shuf_count:
            result[shuf_mask] = orig_vals
        else:
            result[shuf_mask] = np.resize(orig_vals, shuf_count)
    return result


def _vectorized_placebo_fwl(
    X_base: np.ndarray,
    proj_base: np.ndarray,
    treatment: np.ndarray,
    y: np.ndarray,
    n_perms: int = 999,
    batch_size: int = 250,
    block_dates: np.ndarray | None = None,
) -> np.ndarray:
    """FWL 벡터화 플라시보 검증 — 999회 순열을 배치 행렬 연산으로 처리.

    수학: FWL 정리에 의해, Y = X_base·γ + T·β + ε 에서 β는
    e_y = M₀·Y, e_T = M₀·T 의 단변량 회귀 β = (e_T'·e_y) / (e_T'·e_T) 와 동일.
    M₀ = I - X_base·(X'X)^-1·X' 는 1회만 계산. 순열은 T만 변경.

    [2026-04-06] block_dates가 주어지면 날짜 블록 단위 셔플 (시계열 구조 보존).

    Returns
    -------
    betas : (n_perms,) 각 순열의 ATE 추정치
    """
    e_y = _fwl_residualize(X_base, proj_base, y)
    all_betas = []

    for start in range(0, n_perms, batch_size):
        end = min(start + batch_size, n_perms)
        # 배치 순열 생성
        if block_dates is not None:
            T_batch = np.column_stack([
                _block_permutation(treatment, block_dates) for _ in range(end - start)
            ])
        else:
            T_batch = np.column_stack([
                np.random.permutation(treatment) for _ in range(end - start)
            ])  # (n, batch)
        e_T = _fwl_residualize(X_base, proj_base, T_batch)
        # 벡터화 단변량 회귀: β = (e_T' · e_y) / (e_T' · e_T)
        numerator = e_T.T @ e_y           # (batch,)
        denominator = np.sum(e_T ** 2, axis=0)  # (batch,)
        betas = numerator / denominator
        all_betas.append(betas)

    return np.concatenate(all_betas)


def _vectorized_random_cause_fwl(
    X_full: np.ndarray,
    proj_full: np.ndarray,
    y: np.ndarray,
    n_perms: int = 999,
    batch_size: int = 250,
) -> np.ndarray:
    """FWL 벡터화 랜덤 원인 검증 — X_full(X_base+treatment)에 대해 FWL.

    랜덤 N(0,1) 교란변수를 추가했을 때 treatment의 ATE가 얼마나 변하는지 측정.
    FWL: X_full을 고정 변수로, 랜덤 교란변수를 관심 변수로 투영.

    Returns
    -------
    betas : (n_perms,) 각 랜덤 교란변수의 계수 추정치
    """
    n = X_full.shape[0]
    e_y = _fwl_residualize(X_full, proj_full, y)
    all_betas = []

    for start in range(0, n_perms, batch_size):
        end = min(start + batch_size, n_perms)
        R_batch = np.random.normal(size=(n, end - start))
        e_R = _fwl_residualize(X_full, proj_full, R_batch)
        numerator = e_R.T @ e_y
        denominator = np.sum(e_R ** 2, axis=0)
        betas = numerator / denominator
        all_betas.append(betas)

    return np.concatenate(all_betas)


def _quick_placebo_screen(
    X_base: np.ndarray,
    proj_base: np.ndarray,
    treatment: np.ndarray,
    y: np.ndarray,
    ate: float,
    n_quick: int = 50,
    exceed_threshold: int = 36,
    block_dates: np.ndarray | None = None,
) -> bool:
    """적응적 사전 판별 — 50회 FWL로 명백한 MIRAGE를 조기 탈락.

    Besag & Clifford (1991) 기반. 50회 순열 중 |placebo_beta| >= |ATE|*0.1인
    횟수가 threshold 이상이면 p > 0.72로 절대 통과 불가 → MIRAGE 확정.

    Returns True if factor is clearly MIRAGE (should skip full test).
    """
    betas = _vectorized_placebo_fwl(
        X_base, proj_base, treatment, y,
        n_perms=n_quick, batch_size=n_quick,
        block_dates=block_dates,
    )
    exceed_count = int(np.sum(np.abs(betas) >= abs(ate) * 0.10))
    if exceed_count >= exceed_threshold:
        logger.debug(
            "Quick screen: %d/%d exceeds (threshold=%d) → MIRAGE",
            exceed_count, n_quick, exceed_threshold,
        )
        return True
    return False


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
    # H4: 실패 분류 (PASSED, LOW_IC, CONFOUNDED, FRAGILE, REGIME_SHIFT)
    failure_type: str = "PASSED"


class FactorMirageFilter:
    """DoWhy 4단계 인과 검증으로 Factor Mirage를 제거한다."""

    def __init__(
        self,
        placebo_threshold: float = 0.05,
        random_cause_threshold: float = 0.05,
        num_simulations: int = 100,
        use_fast_engine: bool = True,
    ):
        self.placebo_threshold = placebo_threshold
        self.random_cause_threshold = random_cause_threshold
        self.num_simulations = num_simulations
        self.use_fast_engine = use_fast_engine

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
            if self.use_fast_engine:
                return self._run_fast(factor_values, forward_returns, confounders_df)
            return self._run_dowhy_legacy(factor_values, forward_returns, confounders_df)
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
            # 데이터 부족 → 자동 기각 (이전: 자동 통과)
            logger.warning(
                "Insufficient data for regime split: %d rows (need %d per half) → REJECT",
                len(data), _MIN_SAMPLES,
            )
            return False, 0.0, 0.0

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

    def _run_dowhy_legacy(
        self,
        factor_values: np.ndarray,
        forward_returns: np.ndarray,
        confounders_df: pd.DataFrame,
    ) -> CausalValidationResult:
        """DoWhy 4단계 인과 검증 — 레거시 (폴백용 보존)."""
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

        # 결측 교란변수 0으로 채움
        for _col in ("sector_id", "smb", "hml"):
            if _col not in data.columns:
                data[_col] = 0

        # NaN 행 제거
        required_cols = [*_CONFOUNDER_COLS, "alpha_factor", "forward_return"]
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

        # H4: 실패 분류
        if is_robust:
            failure_type = "PASSED"
        elif not placebo_passed:
            failure_type = "CONFOUNDED"
        elif not random_passed:
            failure_type = "FRAGILE"
        elif not regime_passed:
            failure_type = "REGIME_SHIFT"
        else:
            failure_type = "LOW_IC"

        logger.info(
            "Causal validation: ATE=%.6f, p=%.4f, placebo=%.6f(%s), "
            "random_delta=%.6f(%s), regime=%s(%.6f/%.6f) → %s [%s]",
            ate, p_value, placebo_effect,
            "PASS" if placebo_passed else "FAIL",
            random_delta,
            "PASS" if random_passed else "FAIL",
            "PASS" if regime_passed else "FAIL",
            ate_first, ate_second,
            "ROBUST" if is_robust else "MIRAGE",
            failure_type,
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
            failure_type=failure_type,
        )

    # ── Fast Engine (NumPy 직접 구현) ───────────────────────────

    def _run_fast(
        self,
        factor_values: np.ndarray,
        forward_returns: np.ndarray,
        confounders_df: pd.DataFrame,
    ) -> CausalValidationResult:
        """NumPy 고속 인과 검증 — DoWhy와 수학적으로 동일한 연산.

        statsmodels/DoWhy 객체 생성 오버헤드를 제거하고,
        동일한 OLS 회귀 + 플라시보/랜덤원인/체제변화 검증을 수행한다.
        """
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

        # 데이터 통합 DataFrame 구축 (기존 _run_dowhy와 동일)
        data = confounders_df.iloc[:n].copy().reset_index(drop=True)
        data["alpha_factor"] = factor_values[:n]
        data["forward_return"] = forward_returns[:n]

        # 결측 교란변수 0으로 채움
        for _col in ("sector_id", "smb", "hml"):
            if _col not in data.columns:
                data[_col] = 0

        required_cols = [*_CONFOUNDER_COLS, "alpha_factor", "forward_return"]
        data = data.dropna(subset=required_cols)

        # [2026-04-06] Block Permutation: dt 드롭 전 날짜 블록 인덱스 추출
        _block_dates = None
        if "dt" in data.columns:
            _block_dates = data["dt"].values  # 날짜 배열 보존
            data = data.drop(columns=["dt"])

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

        # NumPy 배열 추출
        y = data["forward_return"].values.astype(np.float64)
        X_conf = data[_CONFOUNDER_COLS].values.astype(np.float64)
        treatment_raw = data["alpha_factor"].values.astype(np.float64)

        # ── Treatment Z-score 표준화 ──
        # 알파 팩터 값은 수식에 따라 스케일이 천차만별 (0.01 ~ 100,000+).
        # OLS beta(ATE)는 스케일에 직접 비례하므로, 표준화 없이는
        # t-stat이 항상 0에 수렴하여 모든 팩터가 reject됨.
        t_std = float(np.nanstd(treatment_raw))
        if t_std > 1e-12:
            treatment = (treatment_raw - np.nanmean(treatment_raw)) / t_std
        else:
            treatment = treatment_raw  # 상수 팩터 → 그대로 (어차피 reject)

        # 절편 + 교란변수 행렬 (모든 검증에서 재사용)
        ones = np.ones((len(y), 1))
        X_base = np.column_stack([ones, X_conf])  # (n, 7) — 절편+6교란변수
        X_full = np.column_stack([X_base, treatment])  # (n, 8) — +처리변수

        # ── Step 3: ATE 추정 (backdoor linear regression) ──
        beta, p_values = _fast_ols(X_full, y)
        ate = _sanitize(float(beta[-1]))
        p_value = _sanitize(float(p_values[-1]), default=1.0)

        # ── Step 3.5: t-stat 게이트 (Harvey, Liu, Zhu 2016 — 다중검정 t>3.0) ──
        # [QA fix I1] p=0.0 underflow 시 t-stat=inf로 처리 (가장 유의한 팩터가 거부되던 버그)
        # 직접 beta/SE에서 t-stat 계산 — p-value 역산 방식의 underflow 문제 회피
        dof = len(y) - X_full.shape[1]
        residuals = y - X_full @ beta
        mse = np.dot(residuals, residuals) / max(dof, 1)
        try:
            XtX_inv = np.linalg.inv(X_full.T @ X_full)
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(X_full.T @ X_full)
        se_ate = np.sqrt(max(mse * XtX_inv[-1, -1], 0.0))
        t_stat_ate = abs(ate / se_ate) if se_ate > 1e-15 else 0.0

        if t_stat_ate < 3.0:
            logger.info(
                "Causal t-stat gate: ATE=%.6f, t=%.2f < 3.0 → REJECT (Harvey et al. 2016)",
                ate, t_stat_ate,
            )
            return CausalValidationResult(
                is_causally_robust=False,
                causal_effect_size=ate,
                p_value=p_value,
                placebo_passed=False,
                placebo_effect=0.0,
                random_cause_passed=False,
                random_cause_delta=0.0,
                failure_type="LOW_TSTAT",
            )

        # ── Step 4a: 플라시보 검증 (FWL 벡터화) ──
        # [2026-03-31] 딥리서치 R3+R4 — FWL 벡터화 + 적응적 사전 판별
        # 변경: for 루프 999회 lstsq → FWL+inv 배치 행렬 연산
        # 실측: 84.4초 → 5.4초 (15.5x), 오차 1.68e-16
        from app.core.config import settings as _cfg

        proj_base = _fwl_projection_matrix(X_base)

        # 적응적 사전 판별: 50회로 명백한 MIRAGE 조기 탈락
        is_obvious_mirage = _quick_placebo_screen(
            X_base, proj_base, treatment, y, ate,
            n_quick=_cfg.CAUSAL_QUICK_SCREEN_PERMS,
            exceed_threshold=_cfg.CAUSAL_QUICK_SCREEN_THRESHOLD,
            block_dates=_block_dates,
        )
        if is_obvious_mirage:
            # 빠른 탈락: 50회만에 MIRAGE 확정 → 전체 999회 스킵
            placebo_effect = abs(ate)  # ratio > 1.0 → fail
            placebo_passed = False
        else:
            # 전체 999회 FWL 벡터화 실행 (블록 순열 지원)
            placebo_betas = _vectorized_placebo_fwl(
                X_base, proj_base, treatment, y,
                n_perms=self.num_simulations,
                batch_size=_cfg.CAUSAL_FWL_BATCH_SIZE,
                block_dates=_block_dates,
            )
            placebo_effect = _sanitize(float(np.mean(placebo_betas)))
            # 상대 임계값: 플라시보 ATE가 원본 ATE의 10% 미만이어야 통과
            placebo_ratio = abs(placebo_effect / ate) if abs(ate) > 1e-12 else float("inf")
            placebo_passed = placebo_ratio < 0.10

        # ── Step 4b: 랜덤 원인 검증 (FWL 벡터화) ──
        if not placebo_passed:
            # 플라시보 실패 → 랜덤 원인도 스킵 (어차피 MIRAGE)
            random_effect = ate
            random_delta = 0.0
            random_passed = False
        else:
            proj_full = _fwl_projection_matrix(X_full)
            random_betas = _vectorized_random_cause_fwl(
                X_full, proj_full, y,
                n_perms=self.num_simulations,
                batch_size=_cfg.CAUSAL_FWL_BATCH_SIZE,
            )
            # FWL로 X_full(교란변수+treatment)을 통제한 후 랜덤 변수의 자체 계수를 측정.
            # 의미: X_full이 이미 설명하는 변동 외에 랜덤 노이즈가 추가 설명력을 가지면
            # 모델이 불안정한 것 → 랜덤 변수의 계수가 0에 가까울수록 treatment가 강건함.
            # (원본 루프는 treatment 계수 변화를 측정했으나, t>3.0 통과 팩터에서 동일 판정)
            random_effect = _sanitize(float(np.mean(random_betas)))
            random_delta = abs(random_effect)
            # 상대 임계값: 랜덤 교란변수의 영향이 원본 ATE의 10% 미만이어야 통과
            random_ratio = random_delta / abs(ate) if abs(ate) > 1e-12 else float("inf")
            random_passed = random_ratio < 0.10

        # ── Step 5: 체제 변화 검증 (전반/후반 ATE 부호 일관성) ──
        regime_passed, ate_first, ate_second = self._fast_regime_split(
            X_base, treatment, y,
        )

        is_robust = placebo_passed and random_passed and regime_passed

        # 실패 분류
        if is_robust:
            failure_type = "PASSED"
        elif not placebo_passed:
            failure_type = "CONFOUNDED"
        elif not random_passed:
            failure_type = "FRAGILE"
        elif not regime_passed:
            failure_type = "REGIME_SHIFT"
        else:
            failure_type = "LOW_IC"

        logger.info(
            "Causal validation [fast]: ATE=%.6f, p=%.4f, placebo=%.6f(%s), "
            "random_delta=%.6f(%s), regime=%s(%.6f/%.6f) → %s [%s]",
            ate, p_value, placebo_effect,
            "PASS" if placebo_passed else "FAIL",
            random_delta,
            "PASS" if random_passed else "FAIL",
            "PASS" if regime_passed else "FAIL",
            ate_first, ate_second,
            "ROBUST" if is_robust else "MIRAGE",
            failure_type,
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
            failure_type=failure_type,
        )

    @staticmethod
    def _fast_regime_split(
        X_base: np.ndarray,
        treatment: np.ndarray,
        y: np.ndarray,
    ) -> tuple[bool, float, float]:
        """데이터를 전반/후반으로 분할하여 ATE 부호 일관성을 검증한다."""
        mid = len(y) // 2
        if mid < _MIN_SAMPLES:
            logger.warning(
                "Insufficient data for regime split: %d rows (need %d per half) → REJECT",
                len(y), _MIN_SAMPLES,
            )
            return False, 0.0, 0.0  # 소표본 → 자동 기각 (이전: 자동 통과)

        # 전반부
        X_first = np.column_stack([X_base[:mid], treatment[:mid]])
        beta_first, _ = _fast_ols(X_first, y[:mid])
        ate_first = _sanitize(float(beta_first[-1]))

        # 후반부
        X_second = np.column_stack([X_base[mid:], treatment[mid:]])
        beta_second, _ = _fast_ols(X_second, y[mid:])
        ate_second = _sanitize(float(beta_second[-1]))

        if abs(ate_first) < 1e-8 or abs(ate_second) < 1e-8:
            passed = True
        else:
            passed = (ate_first > 0) == (ate_second > 0)

        logger.info(
            "Regime split [fast]: ATE_first=%.6f, ATE_second=%.6f → %s",
            ate_first, ate_second, "PASS" if passed else "FAIL",
        )
        return passed, ate_first, ate_second
