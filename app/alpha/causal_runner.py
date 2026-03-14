"""비동기 인과 검증 러너.

마이닝 완료 후 또는 수동 요청 시 팩터의 인과적 유효성을 검증한다.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import date as date_type, timedelta

import numpy as np
import pandas as pd
import polars as pl
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.alpha.ast_converter import ensure_alpha_features, parse_expression, sympy_to_polars
from app.alpha.causal import CausalValidationResult, FactorMirageFilter
from app.alpha.confounders import load_confounders, load_sector_mapping
from app.alpha.evaluator import compute_forward_returns
from app.alpha.models import AlphaFactor, AlphaMiningRun
from app.backtest.data_loader import load_enriched_candles
from app.core.config import settings

logger = logging.getLogger(__name__)

# ── 진행률 추적 (인-메모리) ──────────────────────────────────

_validation_jobs: dict[str, dict] = {}


def start_validation_job(job_id: str, total: int) -> None:
    """검증 잡 시작 등록."""
    _validation_jobs[job_id] = {
        "status": "running",
        "total": total,
        "completed": 0,
        "failed": 0,
        "robust": 0,
        "mirage": 0,
        "started_at": time.time(),
        "avg_ms_per_factor": None,
        "estimated_remaining_ms": None,
        "current_factor_idx": 0,
    }


def update_validation_job(
    job_id: str, *, completed: int, failed: int, robust: int, mirage: int,
) -> None:
    """검증 진행 상황 업데이트."""
    job = _validation_jobs.get(job_id)
    if not job:
        return
    job["completed"] = completed
    job["failed"] = failed
    job["robust"] = robust
    job["mirage"] = mirage
    job["current_factor_idx"] = completed + failed
    elapsed = time.time() - job["started_at"]
    done = completed + failed
    if done > 0:
        avg_ms = (elapsed / done) * 1000
        remaining = job["total"] - done
        job["avg_ms_per_factor"] = round(avg_ms, 1)
        job["estimated_remaining_ms"] = round(remaining * avg_ms, 0)


def finish_validation_job(job_id: str) -> None:
    """검증 잡 완료 표시."""
    job = _validation_jobs.get(job_id)
    if job:
        job["status"] = "completed"
        job["estimated_remaining_ms"] = 0


def get_validation_progress(job_id: str) -> dict | None:
    """검증 잡 진행 상황 조회."""
    return _validation_jobs.get(job_id)


def get_latest_validation_job() -> dict | None:
    """가장 최근 검증 잡 조회."""
    if not _validation_jobs:
        return None
    latest_id = max(_validation_jobs, key=lambda k: _validation_jobs[k]["started_at"])
    return {"job_id": latest_id, **_validation_jobs[latest_id]}


def _prepare_factor_and_validate_sync(
    base_df: pl.DataFrame,
    expression_str: str,
    confounders_df: pd.DataFrame,
    sector_map: dict[str, int],
) -> CausalValidationResult:
    """CPU-heavy 팩터 계산 + DoWhy 검증을 동기적으로 실행 (스레드용).

    base_df: ensure_alpha_features + compute_forward_returns 적용 완료된 데이터.
    """
    expr = parse_expression(expression_str)
    polars_expr = sympy_to_polars(expr)
    df = base_df.with_columns(polars_expr.alias("alpha_factor"))

    # NaN 제거
    df = df.filter(
        pl.col("alpha_factor").is_not_null()
        & pl.col("alpha_factor").is_not_nan()
        & pl.col("fwd_return").is_not_null()
        & pl.col("fwd_return").is_not_nan()
    )

    if df.height == 0:
        raise ValueError("No valid rows after factor computation")

    factor_values = df["alpha_factor"].to_numpy()
    forward_returns = df["fwd_return"].to_numpy()

    # confounders를 팩터 데이터와 정렬
    factor_dates = df["dt"].to_list()
    factor_symbols = df["symbol"].to_list() if "symbol" in df.columns else [None] * len(factor_dates)

    aligned = pd.DataFrame({"dt": factor_dates})
    aligned["dt"] = pd.to_datetime(aligned["dt"]).dt.date
    aligned = aligned.merge(confounders_df, on="dt", how="left")
    aligned["sector_id"] = [sector_map.get(s, 0) for s in factor_symbols]

    for col in ["market_return", "market_volatility", "base_rate"]:
        if col in aligned.columns:
            aligned[col] = aligned[col].ffill().bfill()

    # DoWhy 인과 검증
    causal_filter = FactorMirageFilter(
        placebo_threshold=settings.CAUSAL_PLACEBO_THRESHOLD,
        random_cause_threshold=settings.CAUSAL_RANDOM_CAUSE_THRESHOLD,
        num_simulations=settings.CAUSAL_NUM_SIMULATIONS,
        use_fast_engine=settings.CAUSAL_USE_FAST_ENGINE,
    )

    return causal_filter.validate(factor_values, forward_returns, aligned)


async def validate_single_factor(
    factor_id: uuid.UUID,
    db: AsyncSession,
    confounders_cache: dict | None = None,
    candles_cache: dict | None = None,
) -> CausalValidationResult:
    """개별 팩터에 대해 인과 검증을 수행하고 DB를 업데이트한다.

    Parameters
    ----------
    factor_id : 검증할 팩터 UUID
    db : DB 세션
    confounders_cache : 교란 변수 캐시 (배치 호출 시 재사용)
    candles_cache : 캔들 데이터 캐시 (배치 호출 시 재사용)

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

    # 1b. IC 조기 필터: IC가 threshold 미만이면 DoWhy 스킵
    if factor.ic_mean is not None and factor.ic_mean < settings.ALPHA_IC_THRESHOLD_PASS:
        await db.execute(
            update(AlphaFactor)
            .where(AlphaFactor.id == factor_id)
            .values(causal_robust=False, status="mirage", causal_failure_type="LOW_IC")
        )
        await db.commit()
        logger.info(
            "Factor %s skipped (IC %.4f < %.4f)",
            str(factor_id)[:8], factor.ic_mean, settings.ALPHA_IC_THRESHOLD_PASS,
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

    # 2. 마이닝 run에서 날짜 범위/종목 추출
    run_result = await db.execute(
        select(AlphaMiningRun).where(AlphaMiningRun.id == factor.mining_run_id)
    )
    run = run_result.scalar_one_or_none()

    config = run.config if run and run.config else {}
    context = run.context if run and run.context else {}
    start_date_str = config.get("start_date")
    end_date_str = config.get("end_date")
    interval = config.get("interval", "1d")
    universe_code = config.get("universe", "") or context.get("universe", "")
    symbols = config.get("symbols", [])  # 이전 호환성

    if not start_date_str or not end_date_str:
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

    # 유니버스 리졸브
    if universe_code:
        from app.alpha.universe import Universe, resolve_universe
        resolved_symbols = await resolve_universe(Universe(universe_code))
    elif symbols:
        resolved_symbols = symbols
    else:
        resolved_symbols = None

    # 3. 캔들 데이터 (캐시 우선) — enriched로 보조 피처 포함
    _symbols_for_cache = resolved_symbols or []
    cache_key = f"{start_date}_{end_date}_{interval}_{','.join(sorted(_symbols_for_cache))}"

    if candles_cache is not None and candles_cache.get("key") == cache_key:
        base_df = candles_cache["base_df"]
    else:
        candles = await load_enriched_candles(
            symbols=resolved_symbols,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
        )
        if candles.height == 0:
            raise ValueError("No candle data for factor validation")

        base_df = ensure_alpha_features(candles)
        base_df = compute_forward_returns(base_df, periods=1)

        if candles_cache is not None:
            candles_cache["key"] = cache_key
            candles_cache["base_df"] = base_df

    # 4. 교란 변수 (캐시 우선)
    if confounders_cache is not None and confounders_cache.get("key") == cache_key:
        confounders_df = confounders_cache["df"]
        sector_map = confounders_cache.get("sector_map", {})
    else:
        confounders_df = await load_confounders(start_date, end_date, resolved_symbols)
        try:
            sector_map = await load_sector_mapping(resolved_symbols)
        except Exception as e:
            logger.warning("Failed to load sector mapping: %s", e)
            sector_map = {}
        if confounders_cache is not None:
            confounders_cache["key"] = cache_key
            confounders_cache["df"] = confounders_df
            confounders_cache["sector_map"] = sector_map

    # confounders_df의 dt를 date 타입으로 정규화 (1회만)
    if not confounders_cache or not confounders_cache.get("_dt_normalized"):
        confounders_df = confounders_df.copy()
        confounders_df["dt"] = pd.to_datetime(confounders_df["dt"]).apply(
            lambda x: x.date() if hasattr(x, "date") else x
        )
        if confounders_cache is not None:
            confounders_cache["df"] = confounders_df
            confounders_cache["_dt_normalized"] = True

    # 5. CPU-heavy 팩터 계산 + DoWhy를 스레드에서 실행
    causal_result = await asyncio.to_thread(
        _prepare_factor_and_validate_sync,
        base_df,
        factor.expression_str,
        confounders_df,
        sector_map,
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
            causal_failure_type=causal_result.failure_type,
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
    max_concurrent: int = 1,
) -> int:
    """마이닝 run의 모든 discovered 팩터를 순차 검증한다.

    캔들/교란 변수를 1회 로드 후 캐시 공유.
    CPU-heavy 작업은 스레드에서 실행하여 이벤트 루프 블로킹 방지.

    Returns
    -------
    int
        검증된 팩터 수
    """
    from app.core.database import async_session

    result = await db.execute(
        select(AlphaFactor).where(
            AlphaFactor.mining_run_id == run_id,
            AlphaFactor.status == "discovered",
            AlphaFactor.causal_robust.is_(None),
        )
    )
    factors = result.scalars().all()

    if not factors:
        logger.info("No discovered factors to validate for run %s", run_id)
        return 0

    factor_ids = [f.id for f in factors]
    confounders_cache: dict = {}
    candles_cache: dict = {}

    logger.info(
        "Starting causal validation for run %s: %d factors",
        run_id, len(factor_ids),
    )

    sem = asyncio.Semaphore(max_concurrent)

    async def _validate_one(idx: int, factor_id: uuid.UUID) -> bool:
        """개별 팩터 검증 (Semaphore 제한). 성공 시 True."""
        async with sem:
            try:
                async with async_session() as factor_db:
                    await validate_single_factor(
                        factor_id, factor_db,
                        confounders_cache=confounders_cache,
                        candles_cache=candles_cache,
                    )
                logger.info(
                    "Causal validation [%d/%d] factor %s: OK",
                    idx + 1, len(factor_ids), str(factor_id)[:8],
                )
                return True
            except Exception as e:
                err_msg = str(e)
                logger.error(
                    "Causal validation [%d/%d] factor %s FAILED: %s",
                    idx + 1, len(factor_ids), str(factor_id)[:8], err_msg[:200],
                )
                try:
                    async with async_session() as err_db:
                        await err_db.execute(
                            update(AlphaFactor)
                            .where(AlphaFactor.id == factor_id)
                            .values(
                                causal_robust=False,
                                status="causal_failed",
                            )
                        )
                        await err_db.commit()
                except Exception:
                    pass
                return False

    results = await asyncio.gather(
        *[_validate_one(i, fid) for i, fid in enumerate(factor_ids)]
    )
    validated_count = sum(1 for r in results if r)
    failed_count = sum(1 for r in results if not r)

    logger.info(
        "Batch causal validation for run %s: %d/%d validated, %d failed",
        run_id, validated_count, len(factor_ids), failed_count,
    )

    return validated_count


async def validate_factors_by_ids(
    factor_ids: list[uuid.UUID],
    job_id: str,
    max_concurrent: int = 3,
) -> dict:
    """ID 목록으로 인과 검증 실행 (진행률 추적 포함).

    프론트엔드/MCP 배치 검증에서 호출. start_validation_job()이 먼저 호출되어야 한다.

    Returns
    -------
    dict
        {"validated": int, "failed": int, "robust": int, "mirage": int}
    """
    from app.core.database import async_session

    confounders_cache: dict = {}
    candles_cache: dict = {}
    validated = 0
    failed = 0
    robust = 0
    mirage = 0

    sem = asyncio.Semaphore(max_concurrent)

    async def _validate_one(fid: uuid.UUID) -> bool | None:
        nonlocal validated, failed, robust, mirage
        async with sem:
            try:
                async with async_session() as db:
                    result = await validate_single_factor(
                        fid, db,
                        confounders_cache=confounders_cache,
                        candles_cache=candles_cache,
                    )
                validated += 1
                if result.is_causally_robust:
                    robust += 1
                else:
                    mirage += 1
                update_validation_job(
                    job_id, completed=validated, failed=failed,
                    robust=robust, mirage=mirage,
                )
                return True
            except Exception as e:
                failed += 1
                update_validation_job(
                    job_id, completed=validated, failed=failed,
                    robust=robust, mirage=mirage,
                )
                logger.error("Validation %s failed: %s", str(fid)[:8], str(e)[:200])
                try:
                    async with async_session() as err_db:
                        await err_db.execute(
                            update(AlphaFactor)
                            .where(AlphaFactor.id == fid)
                            .values(causal_robust=False, status="causal_failed")
                        )
                        await err_db.commit()
                except Exception:
                    pass
                return False

    await asyncio.gather(*[_validate_one(fid) for fid in factor_ids])
    finish_validation_job(job_id)

    return {"validated": validated, "failed": failed, "robust": robust, "mirage": mirage}
