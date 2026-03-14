"""알파 팩터 → 백테스트 통합 브릿지.

발견된 팩터를 기존 백테스트 엔진의 지표로 등록하여
ConditionSchema에서 사용할 수 있게 한다.

팩터 값은 퍼센타일 랭크로 정규화(0~1)되어
0.7 초과 → 매수, 0.3 미만 → 매도로 해석된다.
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

# 퍼센타일 랭크 롤링 윈도우 (거래일 기준)
_RANK_WINDOW = 60


def register_alpha_factor(
    factor_id: str,
    expression_str: str,
) -> str:
    """알파 팩터를 indicators 디스패처에 동적 등록.

    원시 팩터 값을 60일 롤링 퍼센타일 랭크(0~1)로 변환하여 등록한다.
    - 1.0 = 최근 60일 중 최고값
    - 0.5 = 중앙값
    - 0.0 = 최저값

    이를 통해 NaN/Inf가 포함된 수식도 안정적으로 시그널을 생성한다.

    Returns
    -------
    등록된 지표 이름 (alpha_<factor_id_short>)
    """
    short_id = factor_id[:8]
    indicator_name = f"alpha_{short_id}"

    # 항상 최신 expression_str로 재등록 (서버 재시작 후 캐시 무효)
    def _add_alpha(df: pl.DataFrame, params: dict) -> pl.DataFrame:
        # 기저 지표 보장
        df = ensure_alpha_features(df)
        # 수식 파싱 + 적용
        expr = parse_expression(expression_str)
        polars_expr = sympy_to_polars(expr)

        raw_col = f"_raw_{indicator_name}"
        df = df.with_columns(polars_expr.alias(raw_col))

        # Inf/NaN 정리: 유한하지 않은 값은 null로 변환
        df = df.with_columns(
            pl.when(pl.col(raw_col).is_finite())
            .then(pl.col(raw_col))
            .otherwise(None)
            .alias(raw_col)
        )

        # 롤링 퍼센타일 랭크 (0~1):
        # 현재 값이 최근 N일 중 몇 번째인지 (min_periods=10)
        df = df.with_columns(
            pl.col(raw_col)
            .rolling_quantile(quantile=0.5, window_size=_RANK_WINDOW, min_periods=10)
            .alias(f"_median_{indicator_name}")
        )
        df = df.with_columns(
            pl.col(raw_col)
            .rolling_min(window_size=_RANK_WINDOW, min_periods=10)
            .alias(f"_min_{indicator_name}")
        )
        df = df.with_columns(
            pl.col(raw_col)
            .rolling_max(window_size=_RANK_WINDOW, min_periods=10)
            .alias(f"_max_{indicator_name}")
        )

        # 퍼센타일 랭크 = (value - min) / (max - min)
        range_col = (
            pl.col(f"_max_{indicator_name}") - pl.col(f"_min_{indicator_name}")
        ).clip(lower_bound=1e-10)

        df = df.with_columns(
            (
                (pl.col(raw_col) - pl.col(f"_min_{indicator_name}"))
                / range_col
            )
            .clip(0.0, 1.0)
            .fill_null(0.5)  # 데이터 부족 시 중립
            .alias(indicator_name)
        )

        # 임시 컬럼 정리
        df = df.drop([
            raw_col,
            f"_median_{indicator_name}",
            f"_min_{indicator_name}",
            f"_max_{indicator_name}",
        ])
        return df

    _INDICATOR_FN[indicator_name] = _add_alpha

    logger.info("Registered alpha factor indicator: %s (percentile rank)", indicator_name)
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
