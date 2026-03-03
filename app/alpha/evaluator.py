"""알파 팩터 IC(Information Coefficient) 평가기.

Polars로 데이터 전처리, scipy로 Spearman 상관계수 계산.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from app.alpha.ast_converter import sympy_to_polars, ensure_alpha_features, parse_expression

logger = logging.getLogger(__name__)


@dataclass
class FactorMetrics:
    """팩터 평가 메트릭."""

    ic_mean: float
    ic_std: float
    icir: float
    turnover: float
    sharpe: float
    max_drawdown: float
    ic_series: list[float]


def compute_forward_returns(
    df: pl.DataFrame, periods: int = 1
) -> pl.DataFrame:
    """T+periods 수익률 컬럼 추가."""
    return df.with_columns(
        (pl.col("close").shift(-periods) / pl.col("close") - 1.0)
        .alias("fwd_return")
    )


def compute_ic_series(
    df: pl.DataFrame, factor_col: str = "alpha_factor"
) -> list[float]:
    """일별 cross-sectional Spearman IC 시리즈 계산.

    단일 종목일 경우 시계열 IC, 다종목일 경우 날짜별 횡단면 IC.
    """
    # 유효한 행만 필터
    valid = df.filter(
        pl.col(factor_col).is_not_null()
        & pl.col(factor_col).is_not_nan()
        & pl.col("fwd_return").is_not_null()
        & pl.col("fwd_return").is_not_nan()
    )

    if valid.height < 10:
        return []

    # 날짜 컬럼 결정 (dt 또는 date)
    date_col = "dt" if "dt" in valid.columns else "date"

    # symbol 컬럼이 있으면 날짜별 횡단면 IC
    if "symbol" in valid.columns:
        dates = valid.select(pl.col(date_col)).unique().sort(date_col)
        ic_list: list[float] = []

        for row in dates.iter_rows():
            dt_val = row[0]
            daily = valid.filter(pl.col(date_col) == dt_val)
            if daily.height < 3:
                continue

            factor_vals = daily[factor_col].to_numpy()
            return_vals = daily["fwd_return"].to_numpy()

            # 상수 입력 방어: 분산 0이면 Spearman 정의 불가
            if np.std(factor_vals) < 1e-12 or np.std(return_vals) < 1e-12:
                continue

            corr, _ = spearmanr(factor_vals, return_vals)
            if not np.isnan(corr):
                ic_list.append(float(corr))

        return ic_list

    # 단일 종목: 롤링 윈도우 IC (20일)
    window = min(20, valid.height // 3)
    if window < 5:
        # 데이터 부족 시 전체 IC
        factor_vals = valid[factor_col].to_numpy()
        return_vals = valid["fwd_return"].to_numpy()
        corr, _ = spearmanr(factor_vals, return_vals)
        return [float(corr)] if not np.isnan(corr) else []

    ic_list = []
    factor_arr = valid[factor_col].to_numpy()
    return_arr = valid["fwd_return"].to_numpy()

    for i in range(window, len(factor_arr)):
        f_slice = factor_arr[i - window : i]
        r_slice = return_arr[i - window : i]

        if np.std(f_slice) < 1e-12 or np.std(r_slice) < 1e-12:
            continue

        corr, _ = spearmanr(f_slice, r_slice)
        if not np.isnan(corr):
            ic_list.append(float(corr))

    if 0 < len(ic_list) < 20:
        logger.warning(
            "IC series length %d < 20: statistical significance may be weak",
            len(ic_list),
        )

    return ic_list


def _sanitize_float(value: float) -> float:
    """NaN/Inf를 0.0으로 클렌징."""
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return value


def compute_factor_metrics(ic_series: list[float]) -> FactorMetrics:
    """IC 시리즈에서 집계 메트릭 산출.

    - sharpe: IC Sharpe (누적 IC 일변화량 기반 연환산). 전략 Sharpe가 아님.
    - turnover: Signal Flip Rate (IC 부호 전환 비율). 포트폴리오 턴오버가 아님.
    """
    if not ic_series:
        return FactorMetrics(
            ic_mean=0.0, ic_std=0.0, icir=0.0,
            turnover=0.0, sharpe=0.0, max_drawdown=0.0,
            ic_series=[],
        )

    arr = np.array(ic_series)
    ic_mean = float(np.mean(arr))
    ic_std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    icir = ic_mean / ic_std if ic_std > 1e-12 else 0.0

    # IC Sharpe: 누적 IC 일변화량의 연환산 Sharpe 비율
    cum_ic = np.cumsum(arr)
    daily_returns = np.diff(cum_ic, prepend=0.0)
    sharpe = 0.0
    if len(daily_returns) > 1:
        dr_std = float(np.std(daily_returns, ddof=1))
        if dr_std > 1e-12:
            sharpe = float(np.mean(daily_returns)) / dr_std * np.sqrt(252)

    # 누적 IC 기반 MDD
    peak = np.maximum.accumulate(cum_ic)
    drawdown = cum_ic - peak
    max_drawdown = float(np.min(drawdown)) if len(drawdown) > 0 else 0.0

    # Signal Flip Rate: IC 부호 전환 비율
    sign_changes = np.sum(np.diff(np.sign(arr)) != 0)
    turnover = float(sign_changes / max(len(arr) - 1, 1))

    return FactorMetrics(
        ic_mean=_sanitize_float(ic_mean),
        ic_std=_sanitize_float(ic_std),
        icir=_sanitize_float(icir),
        turnover=_sanitize_float(turnover),
        sharpe=_sanitize_float(sharpe),
        max_drawdown=_sanitize_float(max_drawdown),
        ic_series=ic_series,
    )


def evaluate_factor(
    df: pl.DataFrame,
    expression_str: str,
    name: str = "alpha_factor",
) -> FactorMetrics:
    """팩터 하나의 전체 평가 파이프라인.

    1. 수식 파싱 (SymPy)
    2. 기저 지표 보장
    3. 팩터 컬럼 추가 (Polars Expression)
    4. T+1 수익률 추가
    5. IC 시리즈 계산
    6. 메트릭 집계
    """
    # 수식 파싱
    expr = parse_expression(expression_str)
    polars_expr = sympy_to_polars(expr)

    # 기저 지표 보장
    df = ensure_alpha_features(df)

    # 팩터 컬럼 추가
    df = df.with_columns(polars_expr.alias(name))

    # T+1 수익률
    df = compute_forward_returns(df, periods=1)

    # null/nan 드롭
    df = df.drop_nulls(subset=[name, "fwd_return"])

    # IC 계산
    ic_series = compute_ic_series(df, factor_col=name)

    # 메트릭
    return compute_factor_metrics(ic_series)
