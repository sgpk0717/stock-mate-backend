"""팩터 포트폴리오 — 복합 팩터 생성 + 상관행렬.

다수의 단일 팩터를 IC 가중 또는 동일 가중으로 합성하여
복합 팩터를 만들고, 팩터 간 상관행렬을 계산한다.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import polars as pl
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.alpha.ast_converter import (
    ASTConversionError,
    ensure_alpha_features,
    parse_expression,
    sympy_to_polars,
)
from app.alpha.evaluator import (
    compute_factor_metrics,
    compute_forward_returns,
    compute_ic_series,
    FactorMetrics,
)
from app.alpha.models import AlphaFactor
from app.backtest.data_loader import load_candles

logger = logging.getLogger(__name__)


@dataclass
class CompositeResult:
    """복합 팩터 생성 결과."""

    composite_expression: str
    component_ids: list[str]
    weights: dict[str, float]
    metrics: FactorMetrics
    correlation_matrix: list[list[float]]
    factor_names: list[str]


async def build_composite_factor(
    db: AsyncSession,
    factor_ids: list[str],
    method: str = "ic_weighted",
    name: str = "Composite Alpha",
) -> CompositeResult:
    """복합 팩터 생성.

    1. 팩터 로드 → 캔들 로드 → 피처 생성 → 팩터 컬럼 계산
    2. 상관행렬 계산
    3. 가중 합산 (ic_weighted 또는 equal_weight)
    4. 복합 팩터 IC 평가
    """
    # 팩터 로드
    factor_uuids = [uuid.UUID(fid) for fid in factor_ids]
    result = await db.execute(
        select(AlphaFactor).where(AlphaFactor.id.in_(factor_uuids))
    )
    factors = result.scalars().all()

    if len(factors) < 2:
        raise ValueError(f"최소 2개 팩터가 필요합니다 (로드됨: {len(factors)})")

    # 데이터 로드 — 최근 1년 기본 제한 (전체 DB 로드 방지)
    default_start = date.today() - timedelta(days=365)
    data = await load_candles(start_date=default_start)
    if data.height == 0:
        raise ValueError("캔들 데이터가 없습니다")

    data = ensure_alpha_features(data)
    data = compute_forward_returns(data, periods=1)

    # 팩터 컬럼 계산
    valid_factors: list[AlphaFactor] = []
    col_names: list[str] = []

    for factor in factors:
        col_name = f"factor_{factor.id}"
        try:
            parsed = parse_expression(factor.expression_str)
            polars_expr = sympy_to_polars(parsed)
            data = data.with_columns(polars_expr.alias(col_name))
            valid_factors.append(factor)
            col_names.append(col_name)
        except (ASTConversionError, Exception) as e:
            logger.warning(
                "Factor %s parse failed, skipping: %s", factor.id, e
            )

    if len(valid_factors) < 2:
        raise ValueError(
            f"유효한 팩터가 2개 미만입니다 (유효: {len(valid_factors)})"
        )

    # null 제거
    data = data.drop_nulls(subset=col_names + ["fwd_return"])
    if data.height < 30:
        raise ValueError(f"데이터 행이 부족합니다: {data.height}")

    # 상관행렬 계산
    factor_matrix = data.select(col_names).to_numpy()
    corr_matrix = np.corrcoef(factor_matrix, rowvar=False)
    corr_list = corr_matrix.tolist()

    # 가중치 결정
    weights: dict[str, float] = {}
    if method == "ic_weighted":
        total_ic = sum(
            abs(f.ic_mean) for f in valid_factors if f.ic_mean is not None
        )
        if total_ic == 0:
            # IC 정보 없으면 동일 가중으로 폴백
            for f in valid_factors:
                weights[str(f.id)] = 1.0 / len(valid_factors)
        else:
            for f in valid_factors:
                ic = abs(f.ic_mean) if f.ic_mean is not None else 0.0
                weights[str(f.id)] = ic / total_ic
    else:
        # equal_weight
        for f in valid_factors:
            weights[str(f.id)] = 1.0 / len(valid_factors)

    # 복합 팩터 계산
    composite_expr_parts: list[str] = []
    composite_col = pl.lit(0.0)

    for factor, col_name in zip(valid_factors, col_names):
        w = weights[str(factor.id)]
        composite_col = composite_col + pl.col(col_name) * w
        composite_expr_parts.append(f"{w:.4f} * ({factor.expression_str})")

    composite_expression = " + ".join(composite_expr_parts)

    data = data.with_columns(composite_col.alias("composite_factor"))
    data = data.drop_nulls(subset=["composite_factor", "fwd_return"])

    # 복합 팩터 IC 평가
    ic_series = compute_ic_series(data, factor_col="composite_factor")
    metrics = compute_factor_metrics(ic_series)

    factor_names = [f.name for f in valid_factors]
    component_ids_str = [str(f.id) for f in valid_factors]

    return CompositeResult(
        composite_expression=composite_expression,
        component_ids=component_ids_str,
        weights=weights,
        metrics=metrics,
        correlation_matrix=corr_list,
        factor_names=factor_names,
    )


async def compute_correlation_matrix(
    db: AsyncSession,
    factor_ids: list[str],
) -> dict:
    """팩터 간 Pearson 상관행렬 계산."""
    factor_uuids = [uuid.UUID(fid) for fid in factor_ids]
    result = await db.execute(
        select(AlphaFactor).where(AlphaFactor.id.in_(factor_uuids))
    )
    factors = result.scalars().all()

    default_start = date.today() - timedelta(days=365)
    data = await load_candles(start_date=default_start)
    if data.height == 0:
        raise ValueError("캔들 데이터가 없습니다")

    data = ensure_alpha_features(data)

    valid_factors: list[AlphaFactor] = []
    col_names: list[str] = []

    for factor in factors:
        col_name = f"factor_{factor.id}"
        try:
            parsed = parse_expression(factor.expression_str)
            polars_expr = sympy_to_polars(parsed)
            data = data.with_columns(polars_expr.alias(col_name))
            valid_factors.append(factor)
            col_names.append(col_name)
        except (ASTConversionError, Exception):
            continue

    if len(valid_factors) < 2:
        raise ValueError("유효한 팩터가 2개 미만입니다")

    data = data.drop_nulls(subset=col_names)
    factor_matrix = data.select(col_names).to_numpy()
    corr_matrix = np.corrcoef(factor_matrix, rowvar=False)

    return {
        "factor_ids": [str(f.id) for f in valid_factors],
        "factor_names": [f.name for f in valid_factors],
        "matrix": corr_matrix.tolist(),
    }
