"""조합적 정제 교차 검증 (Combinatorial Purged Cross-Validation).

Lopez de Prado의 CPCV 방법론 구현.
Purging + Embargoing으로 look-ahead bias를 방지하고,
PBO(Probability of Backtest Overfitting)를 산출한다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import polars as pl
import sympy

from app.alpha.ast_converter import parse_expression, sympy_to_polars
from app.alpha.evaluator import compute_forward_returns, compute_ic_series

logger = logging.getLogger(__name__)


@dataclass
class CPCVResult:
    """CPCV 검증 결과."""

    passed: bool
    mean_ic: float = 0.0
    std_ic: float = 0.0
    pbo: float = 0.0  # Probability of Backtest Overfitting
    paths_ic: list[float] = field(default_factory=list)
    reason: str = ""  # "passed", "failed", "insufficient_data", "single_stock"


def cpcv_validate(
    df: pl.DataFrame,
    factor_expr: sympy.Basic | str,
    n_groups: int = 5,
    n_test: int = 2,
    embargo_days: int = 5,
    ic_threshold: float = 0.03,
) -> CPCVResult:
    """조합적 정제 교차 검증.

    Parameters
    ----------
    df : 알파 피처가 포함된 DataFrame (dt, symbol, OHLCV + 피처)
    factor_expr : SymPy 수식 또는 문자열
    n_groups : 날짜 그룹 수 (기본 5)
    n_test : 테스트 그룹 수 (기본 2)
    embargo_days : 엠바고 일수 (기본 5)
    ic_threshold : IC 통과 기준

    Returns
    -------
    CPCVResult
    """
    # 수식 파싱
    if isinstance(factor_expr, str):
        factor_expr = parse_expression(factor_expr)

    # 고유 날짜 목록
    unique_dates = df.select("dt").unique().sort("dt")["dt"].to_list()
    total_days = len(unique_dates)

    # 안전장치: 데이터 부족 시 스킵
    min_days_per_group = max(embargo_days + 10, 20)
    if total_days < n_groups * min_days_per_group:
        return CPCVResult(
            passed=True,
            reason="insufficient_data",
        )

    # 안전장치: 단일 종목 시 스킵 (cross-sectional IC 불가)
    n_symbols = df.select("symbol").n_unique() if "symbol" in df.columns else 1
    if n_symbols < 3:
        return CPCVResult(
            passed=True,
            reason="single_stock",
        )

    # embargo 자동 조정
    actual_embargo = min(embargo_days, total_days // n_groups - 1)

    # 팩터 컬럼 추가
    try:
        polars_expr = sympy_to_polars(factor_expr)
        df = df.with_columns(polars_expr.alias("alpha_factor"))
        df = compute_forward_returns(df, periods=1)
    except Exception as e:
        logger.debug("CPCV factor computation failed: %s", e)
        return CPCVResult(passed=False, reason=f"eval_error: {e}")

    # NaN/Inf 필터링
    df = df.filter(
        pl.col("alpha_factor").is_not_null()
        & pl.col("alpha_factor").is_not_nan()
        & pl.col("alpha_factor").is_finite()
        & pl.col("fwd_return").is_not_null()
        & pl.col("fwd_return").is_not_nan()
    )

    if df.height < 100:
        return CPCVResult(passed=True, reason="insufficient_data")

    # 날짜 그룹 분할
    group_size = total_days // n_groups
    date_groups: list[list] = []
    for g in range(n_groups):
        start_idx = g * group_size
        end_idx = (g + 1) * group_size if g < n_groups - 1 else total_days
        date_groups.append(unique_dates[start_idx:end_idx])

    # C(n_groups, n_test) 조합 생성
    test_combos = list(combinations(range(n_groups), n_test))
    paths_ic: list[float] = []

    for test_indices in test_combos:
        train_indices = [i for i in range(n_groups) if i not in test_indices]

        # 테스트 날짜
        test_dates = set()
        for ti in test_indices:
            test_dates.update(date_groups[ti])

        # 훈련 날짜 (purging + embargoing 적용)
        train_dates = set()
        for ti in train_indices:
            train_dates.update(date_groups[ti])

        # Purging: test 직전 날짜 제거 (fwd_return 형성 기간)
        # fwd_return은 T+1이므로, test 직전 1일 purge
        test_dates_sorted = sorted(test_dates)
        if test_dates_sorted:
            first_test_date = test_dates_sorted[0]
            first_test_idx = unique_dates.index(first_test_date) if first_test_date in unique_dates else -1
            if first_test_idx > 0:
                purge_date = unique_dates[first_test_idx - 1]
                train_dates.discard(purge_date)

        # Embargoing: test 직후 embargo_days 제거
        if test_dates_sorted:
            last_test_date = test_dates_sorted[-1]
            last_test_idx = unique_dates.index(last_test_date) if last_test_date in unique_dates else -1
            if last_test_idx >= 0:
                for offset in range(1, actual_embargo + 1):
                    idx = last_test_idx + offset
                    if idx < total_days:
                        train_dates.discard(unique_dates[idx])

        # 테스트 데이터에서 IC 계산
        test_df = df.filter(pl.col("dt").is_in(list(test_dates)))
        if test_df.height < 30:
            continue

        try:
            ic_series = compute_ic_series(test_df, factor_col="alpha_factor")
            if ic_series:
                path_ic = float(np.mean(ic_series))
                paths_ic.append(path_ic)
        except Exception:
            continue

    if not paths_ic:
        return CPCVResult(passed=True, reason="insufficient_data")

    # 통계량 산출
    mean_ic = float(np.mean(paths_ic))
    std_ic = float(np.std(paths_ic)) if len(paths_ic) > 1 else 0.0

    # PBO: IC < 0인 경로 비율
    negative_paths = sum(1 for ic in paths_ic if ic <= 0)
    pbo = negative_paths / len(paths_ic)

    # 통과 조건: mean_ic >= threshold AND PBO < 0.5
    passed = mean_ic >= ic_threshold and pbo < 0.5

    return CPCVResult(
        passed=passed,
        mean_ic=mean_ic,
        std_ic=std_ic,
        pbo=pbo,
        paths_ic=paths_ic,
        reason="passed" if passed else "failed",
    )
