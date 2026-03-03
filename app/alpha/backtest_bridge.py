"""알파 팩터 → 백테스트 통합 브릿지.

발견된 팩터를 기존 백테스트 엔진의 지표로 등록하여
ConditionSchema에서 사용할 수 있게 한다.
"""

from __future__ import annotations

import logging

import polars as pl

from app.alpha.ast_converter import (
    ensure_alpha_features,
    parse_expression,
    sympy_to_polars,
)
from app.backtest.indicators import _INDICATOR_FN

logger = logging.getLogger(__name__)


def register_alpha_factor(
    factor_id: str,
    expression_str: str,
) -> str:
    """알파 팩터를 indicators 디스패처에 동적 등록.

    등록 후 ConditionSchema(indicator="alpha_<id>", op=">", value=0.0)로 사용 가능.

    Returns
    -------
    등록된 지표 이름 (alpha_<factor_id_short>)
    """
    short_id = factor_id[:8]
    indicator_name = f"alpha_{short_id}"

    # 이미 등록된 경우 스킵
    if indicator_name in _INDICATOR_FN:
        return indicator_name

    def _add_alpha(df: pl.DataFrame, params: dict) -> pl.DataFrame:
        # 기저 지표 보장
        df = ensure_alpha_features(df)
        # 수식 파싱 + 적용
        expr = parse_expression(expression_str)
        polars_expr = sympy_to_polars(expr)
        return df.with_columns(polars_expr.alias(indicator_name))

    _INDICATOR_FN[indicator_name] = _add_alpha

    logger.info("Registered alpha factor indicator: %s", indicator_name)
    return indicator_name


def add_alpha_indicator(
    df: pl.DataFrame,
    factor_id: str,
    expression_str: str,
) -> pl.DataFrame:
    """DataFrame에 알파 팩터 컬럼을 직접 추가.

    register_alpha_factor와 달리, 디스패처에 등록하지 않고
    즉석에서 팩터를 적용한다.
    """
    df = ensure_alpha_features(df)
    expr = parse_expression(expression_str)
    polars_expr = sympy_to_polars(expr)

    short_id = factor_id[:8]
    col_name = f"alpha_{short_id}"
    return df.with_columns(polars_expr.alias(col_name))
