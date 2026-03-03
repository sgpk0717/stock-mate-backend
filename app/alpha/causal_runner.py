"""비동기 인과 검증 러너.

마이닝 완료 후 또는 수동 요청 시 팩터의 인과적 유효성을 검증한다.
"""

from __future__ import annotations

import asyncio
import logging
import uuid

import numpy as np
import polars as pl
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.alpha.ast_converter import ensure_alpha_features, parse_expression, sympy_to_polars
from app.alpha.causal import CausalValidationResult, FactorMirageFilter
from app.alpha.confounders import load_confounders, load_sector_mapping
from app.alpha.evaluator import compute_forward_returns
from app.alpha.models import AlphaFactor, AlphaMiningRun
from app.backtest.data_loader import load_candles
from app.core.config import settings

logger = logging.getLogger(__name__)


async def validate_single_factor(
    factor_id: uuid.UUID,
    db: AsyncSession,
    confounders_cache: dict | None = None,
) -> CausalValidationResult:
    """개별 팩터에 대해 인과 검증을 수행하고 DB를 업데이트한다.

    Parameters
    ----------
    factor_id : 검증할 팩터 UUID
    db : DB 세션
    confounders_cache : 교란 변수 캐시 (배치 호출 시 재사용)

    Returns
    -------
    CausalValidationResult
    """
    # 1. 팩터 조회
    result = await db.execute(
        select(AlphaFactor).where(AlphaFactor.id == factor_id)
    )
    factor = result.scalar_one_or_none()
    if not factor:
        raise ValueError(f"Factor not found: {factor_id}")

    # 2. 마이닝 run에서 날짜 범위/종목 추출
    run_result = await db.execute(
        select(AlphaMiningRun).where(AlphaMiningRun.id == factor.mining_run_id)
    )
    run = run_result.scalar_one_or_none()

    config = run.config if run and run.config else {}
    context = run.context if run and run.context else {}
    start_date_str = config.get("start_date")
    end_date_str = config.get("end_date")
    universe_code = config.get("universe", "") or context.get("universe", "")
    symbols = config.get("symbols", [])  # 이전 호환성

    from datetime import date as date_type, timedelta

    if not start_date_str or not end_date_str:
        # config에 날짜가 없으면 기본 2년 윈도우로 fallback
        logger.warning(
            "Mining run %s config missing date range, using default 2-year window",
            factor.mining_run_id,
        )
        fallback_end = date_type.today()
        fallback_start = fallback_end - timedelta(days=730)
        start_date_str = start_date_str or fallback_start.isoformat()
        end_date_str = end_date_str or fallback_end.isoformat()

    start_date = date_type.fromisoformat(start_date_str)
    end_date = date_type.fromisoformat(end_date_str)

    # 유니버스 리졸브 (새 방식 우선, 이전 symbols 호환)
    if universe_code:
        from app.alpha.universe import Universe, resolve_universe
        resolved_symbols = await resolve_universe(Universe(universe_code))
    elif symbols:
        resolved_symbols = symbols
    else:
        resolved_symbols = None

    # 3. 캔들 데이터 로드 + 팩터값 재계산
    candles = await load_candles(
        symbols=resolved_symbols,
        start_date=start_date,
        end_date=end_date,
    )

    if candles.height == 0:
        raise ValueError("No candle data for factor validation")

    # 기저 지표 + 팩터 컬럼 추가
    df = ensure_alpha_features(candles)
    expr = parse_expression(factor.expression_str)
    polars_expr = sympy_to_polars(expr)
    df = df.with_columns(polars_expr.alias("alpha_factor"))
    df = compute_forward_returns(df, periods=1)

    # NaN 제거
    df = df.filter(
        pl.col("alpha_factor").is_not_null()
        & pl.col("alpha_factor").is_not_nan()
        & pl.col("fwd_return").is_not_null()
        & pl.col("fwd_return").is_not_nan()
    )

    factor_values = df["alpha_factor"].to_numpy()
    forward_returns = df["fwd_return"].to_numpy()

    # 4. 교란 변수 로드 (캐시 사용 가능, 날짜/종목 키로 검증)
    _symbols_for_cache = resolved_symbols or []
    cache_key = f"{start_date}_{end_date}_{','.join(sorted(_symbols_for_cache))}"
    if confounders_cache is not None and confounders_cache.get("key") == cache_key:
        confounders_df = confounders_cache["df"]
        sector_map = confounders_cache.get("sector_map", {})
    else:
        confounders_df = await load_confounders(start_date, end_date, resolved_symbols)
        # sector_id 로드
        try:
            sector_map = await load_sector_mapping(resolved_symbols)
        except Exception as e:
            logger.warning("Failed to load sector mapping: %s", e)
            sector_map = {}
        if confounders_cache is not None:
            confounders_cache["key"] = cache_key
            confounders_cache["df"] = confounders_df
            confounders_cache["sector_map"] = sector_map

    # 4b. confounders를 팩터 데이터와 정렬
    # confounders_df는 일별(per-date), factor_values는 종목×일별(per-symbol-date)
    # dt 기준으로 merge하고 sector_id를 매핑
    import pandas as pd

    factor_dates = df["dt"].to_list()
    factor_symbols = df["symbol"].to_list() if "symbol" in df.columns else [None] * len(factor_dates)

    aligned_confounders = pd.DataFrame({"dt": factor_dates})
    aligned_confounders["dt"] = pd.to_datetime(aligned_confounders["dt"]).dt.date
    confounders_df["dt"] = pd.to_datetime(confounders_df["dt"]).apply(
        lambda x: x.date() if hasattr(x, "date") else x
    )
    aligned_confounders = aligned_confounders.merge(confounders_df, on="dt", how="left")
    aligned_confounders["sector_id"] = [sector_map.get(s, 0) for s in factor_symbols]

    # NaN 행을 ffill로 채우기 (merge 후 미매칭 날짜)
    for col in ["market_return", "market_volatility", "base_rate"]:
        if col in aligned_confounders.columns:
            aligned_confounders[col] = aligned_confounders[col].ffill().bfill()

    confounders_df = aligned_confounders

    # 5. DoWhy 인과 검증 (CPU-bound이므로 thread로 실행)
    causal_filter = FactorMirageFilter(
        placebo_threshold=settings.CAUSAL_PLACEBO_THRESHOLD,
        random_cause_threshold=settings.CAUSAL_RANDOM_CAUSE_THRESHOLD,
        num_simulations=settings.CAUSAL_NUM_SIMULATIONS,
    )

    causal_result = await asyncio.to_thread(
        causal_filter.validate,
        factor_values,
        forward_returns,
        confounders_df,
    )

    # 6. DB 업데이트
    new_status = "validated" if causal_result.is_causally_robust else "mirage"
    await db.execute(
        update(AlphaFactor)
        .where(AlphaFactor.id == factor_id)
        .values(
            causal_robust=causal_result.is_causally_robust,
            causal_effect_size=causal_result.causal_effect_size,
            causal_p_value=causal_result.p_value,
            status=new_status,
        )
    )
    await db.commit()

    logger.info(
        "Factor %s causal validation: %s (ATE=%.6f, p=%.4f)",
        factor_id, new_status,
        causal_result.causal_effect_size, causal_result.p_value,
    )

    return causal_result


async def validate_factors_batch(
    run_id: uuid.UUID,
    db: AsyncSession,
) -> int:
    """마이닝 run의 모든 discovered 팩터를 배치 검증한다.

    교란 변수는 1회만 로드하여 전체 팩터에 공유.

    Returns
    -------
    int
        검증된 팩터 수
    """
    result = await db.execute(
        select(AlphaFactor).where(
            AlphaFactor.mining_run_id == run_id,
            AlphaFactor.status == "discovered",
        )
    )
    factors = result.scalars().all()

    if not factors:
        return 0

    # 교란 변수 캐시 (한 번만 로드)
    confounders_cache: dict = {}
    validated_count = 0

    for factor in factors:
        try:
            await validate_single_factor(
                factor.id, db, confounders_cache=confounders_cache
            )
            validated_count += 1
        except Exception as e:
            logger.warning(
                "Causal validation failed for factor %s: %s",
                factor.id, e,
            )

    logger.info(
        "Batch causal validation for run %s: %d/%d factors validated",
        run_id, validated_count, len(factors),
    )

    return validated_count
