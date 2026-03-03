"""알파 마이닝 REST API 라우터."""

from __future__ import annotations

import asyncio
import uuid
from datetime import date

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.alpha.models import AlphaFactor, AlphaMiningRun
from app.alpha.runner import execute_alpha_mining
from app.alpha.schemas import (
    AlphaFactorBacktestRequest,
    AlphaFactorResponse,
    AlphaFactoryStartRequest,
    AlphaFactoryStatusResponse,
    AlphaMineRequest,
    AlphaMineResponse,
    AlphaMiningRunResponse,
    AlphaMiningRunSummary,
    CompositeFactorBuildRequest,
    CompositeFactorResponse,
    CorrelationRequest,
    CorrelationMatrixResponse,
    MiningIterationLogs,
)
from app.alpha.universe import Universe, get_universe_info
from app.core.database import get_db

router = APIRouter(prefix="/alpha", tags=["alpha"])


# ── 유니버스 ──


@router.get("/universes")
async def list_universes():
    """사용 가능한 유니버스 목록과 종목 수 반환."""
    return await get_universe_info()


# ── 마이닝 실행 ──

@router.post("/mine", status_code=202, response_model=AlphaMineResponse)
async def start_mining(data: AlphaMineRequest, db: AsyncSession = Depends(get_db)):
    """알파 마이닝 비동기 실행. 즉시 run_id 반환."""
    start = date.fromisoformat(data.start_date)
    end = date.fromisoformat(data.end_date)

    if start >= end:
        raise HTTPException(400, "시작일이 종료일보다 이전이어야 합니다.")

    run = AlphaMiningRun(
        name=data.name,
        context={"text": data.context, "universe": data.universe},
        config={
            "start_date": data.start_date,
            "end_date": data.end_date,
            "universe": data.universe,
            "max_iterations": data.max_iterations,
            "ic_threshold": data.ic_threshold,
            "orthogonality_threshold": data.orthogonality_threshold,
            "use_pysr": data.use_pysr,
            "pysr_max_size": data.pysr_max_size,
            "pysr_parsimony": data.pysr_parsimony,
        },
        status="PENDING",
        progress=0,
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)

    asyncio.create_task(
        execute_alpha_mining(
            run_id=run.id,
            name=data.name,
            context=data.context,
            universe=data.universe,
            start_date=start,
            end_date=end,
            max_iterations=data.max_iterations,
            ic_threshold=data.ic_threshold,
            orthogonality_threshold=data.orthogonality_threshold,
            use_pysr=data.use_pysr,
        )
    )

    return AlphaMineResponse(
        id=str(run.id),
        status=run.status,
        created_at=run.created_at.isoformat(),
    )


@router.get("/mine/{run_id}", response_model=AlphaMiningRunResponse)
async def get_mining_run(run_id: str, db: AsyncSession = Depends(get_db)):
    """마이닝 실행 상태/진행률 조회."""
    result = await db.execute(
        select(AlphaMiningRun).where(AlphaMiningRun.id == uuid.UUID(run_id))
    )
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(404, "Mining run not found")

    return AlphaMiningRunResponse(
        id=str(run.id),
        name=run.name,
        context=run.context,
        config=run.config,
        status=run.status,
        progress=run.progress,
        factors_found=run.factors_found,
        total_evaluated=run.total_evaluated,
        error_message=run.error_message,
        has_logs=run.iteration_logs is not None,
        created_at=run.created_at.isoformat(),
        completed_at=run.completed_at.isoformat() if run.completed_at else None,
    )


@router.get("/mine/{run_id}/logs", response_model=MiningIterationLogs)
async def get_mining_logs(run_id: str, db: AsyncSession = Depends(get_db)):
    """마이닝 실행의 상세 iteration 로그 조회."""
    result = await db.execute(
        select(AlphaMiningRun).where(AlphaMiningRun.id == uuid.UUID(run_id))
    )
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(404, "Mining run not found")

    logs = run.iteration_logs or {}
    return MiningIterationLogs(
        run_id=str(run.id),
        iterations=logs.get("iterations", []),
        summary=logs.get("summary", {}),
    )


@router.get("/mines", response_model=list[AlphaMiningRunSummary])
async def list_mining_runs(db: AsyncSession = Depends(get_db)):
    """마이닝 실행 목록."""
    result = await db.execute(
        select(AlphaMiningRun).order_by(AlphaMiningRun.created_at.desc())
    )
    runs = result.scalars().all()

    return [
        AlphaMiningRunSummary(
            id=str(r.id),
            name=r.name,
            status=r.status,
            progress=r.progress,
            factors_found=r.factors_found,
            total_evaluated=r.total_evaluated,
            created_at=r.created_at.isoformat(),
        )
        for r in runs
    ]


@router.delete("/mine/{run_id}", status_code=204)
async def delete_mining_run(run_id: str, db: AsyncSession = Depends(get_db)):
    """마이닝 실행 삭제 (cascade로 팩터도 삭제)."""
    await db.execute(
        delete(AlphaMiningRun).where(AlphaMiningRun.id == uuid.UUID(run_id))
    )
    await db.commit()


# ── 팩터 조회 ──

@router.get("/factors", response_model=list[AlphaFactorResponse])
async def list_factors(
    status: str | None = None,
    min_ic: float | None = None,
    db: AsyncSession = Depends(get_db),
):
    """팩터 목록 (status/min_ic 필터)."""
    query = select(AlphaFactor).order_by(AlphaFactor.ic_mean.desc().nulls_last())

    if status:
        query = query.where(AlphaFactor.status == status)
    if min_ic is not None:
        query = query.where(AlphaFactor.ic_mean >= min_ic)

    result = await db.execute(query)
    factors = result.scalars().all()

    return [_factor_to_response(f) for f in factors]


@router.get("/factor/{factor_id}", response_model=AlphaFactorResponse)
async def get_factor(factor_id: str, db: AsyncSession = Depends(get_db)):
    """팩터 상세."""
    result = await db.execute(
        select(AlphaFactor).where(AlphaFactor.id == uuid.UUID(factor_id))
    )
    factor = result.scalar_one_or_none()
    if not factor:
        raise HTTPException(404, "Factor not found")

    return _factor_to_response(factor)


@router.delete("/factor/{factor_id}", status_code=204)
async def delete_factor(factor_id: str, db: AsyncSession = Depends(get_db)):
    """팩터 삭제."""
    await db.execute(
        delete(AlphaFactor).where(AlphaFactor.id == uuid.UUID(factor_id))
    )
    await db.commit()


@router.post("/factor/{factor_id}/backtest", status_code=202)
async def backtest_with_factor(
    factor_id: str,
    data: AlphaFactorBacktestRequest,
    db: AsyncSession = Depends(get_db),
):
    """팩터로 백테스트 실행.

    팩터를 지표로 등록한 뒤 기존 백테스트 엔진에 위임한다.
    """
    result = await db.execute(
        select(AlphaFactor).where(AlphaFactor.id == uuid.UUID(factor_id))
    )
    factor = result.scalar_one_or_none()
    if not factor:
        raise HTTPException(404, "Factor not found")

    # 팩터를 지표로 등록
    from app.alpha.backtest_bridge import register_alpha_factor

    indicator_name = register_alpha_factor(
        factor_id=str(factor.id),
        expression_str=factor.expression_str,
    )

    # 백테스트 전략 구성
    from app.backtest.schemas import BacktestRunCreate, StrategySchema, ConditionSchema

    strategy = StrategySchema(
        name=f"Alpha: {factor.name}",
        description=f"Alpha factor: {factor.expression_str}",
        timeframe="1d",
        buy_conditions=[
            ConditionSchema(
                indicator=indicator_name,
                op=">",
                value=data.buy_threshold,
            )
        ],
        sell_conditions=[
            ConditionSchema(
                indicator=indicator_name,
                op="<",
                value=data.sell_threshold,
            )
        ],
    )

    backtest_data = BacktestRunCreate(
        strategy=strategy,
        start_date=data.start_date,
        end_date=data.end_date,
        symbols=data.symbols if data.symbols else None,
        initial_capital=data.initial_capital,
        position_size_pct=data.position_size_pct,
        max_positions=data.max_positions,
    )

    # 기존 백테스트 라우터 로직 재사용
    from app.backtest.cost_model import CostConfig
    from app.backtest.models import BacktestRun
    from app.backtest.runner import execute_backtest

    start = date.fromisoformat(data.start_date)
    end = date.fromisoformat(data.end_date)

    run = BacktestRun(
        strategy_name=strategy.name,
        strategy_json=strategy.model_dump(),
        start_date=start,
        end_date=end,
        initial_capital=float(data.initial_capital),
        cost_config=CostConfig().model_dump(),
        status="PENDING",
        progress=0,
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)

    asyncio.create_task(
        execute_backtest(
            run_id=run.id,
            strategy_json=strategy.model_dump(),
            start_date=start,
            end_date=end,
            initial_capital=data.initial_capital,
            symbols=data.symbols if data.symbols else None,
            position_size_pct=data.position_size_pct,
            max_positions=data.max_positions,
            cost_config=CostConfig(),
        )
    )

    return {"backtest_run_id": str(run.id)}


@router.post("/factor/{factor_id}/validate")
async def validate_factor(factor_id: str, db: AsyncSession = Depends(get_db)):
    """인과 검증 트리거 — DoWhy 4단계 인과 검증 실행."""
    from app.alpha.causal_runner import validate_single_factor
    from app.alpha.schemas import CausalValidationResponse

    result = await db.execute(
        select(AlphaFactor).where(AlphaFactor.id == uuid.UUID(factor_id))
    )
    factor = result.scalar_one_or_none()
    if not factor:
        raise HTTPException(404, "Factor not found")

    try:
        causal_result = await validate_single_factor(uuid.UUID(factor_id), db)
    except Exception as e:
        raise HTTPException(500, f"Causal validation failed: {str(e)[:200]}")

    return CausalValidationResponse(
        factor_id=str(factor.id),
        is_causally_robust=causal_result.is_causally_robust,
        causal_effect_size=causal_result.causal_effect_size,
        p_value=causal_result.p_value,
        placebo_passed=causal_result.placebo_passed,
        placebo_effect=causal_result.placebo_effect,
        random_cause_passed=causal_result.random_cause_passed,
        random_cause_delta=causal_result.random_cause_delta,
        regime_shift_passed=getattr(causal_result, "regime_shift_passed", False),
        regime_ate_first_half=getattr(causal_result, "regime_ate_first_half", 0.0),
        regime_ate_second_half=getattr(causal_result, "regime_ate_second_half", 0.0),
        dag_edges=causal_result.dag_edges,
    )


# ── Phase 3: 알파 팩토리 ──


@router.post("/factory/start", response_model=AlphaFactoryStatusResponse)
async def start_factory(data: AlphaFactoryStartRequest):
    """알파 팩토리 시작."""
    from app.alpha.scheduler import get_scheduler

    scheduler = get_scheduler()
    started = await scheduler.start(
        context=data.context,
        universe=data.universe,
        start_date=data.start_date,
        end_date=data.end_date,
        interval_minutes=data.interval_minutes,
        max_iterations=data.max_iterations_per_cycle,
        ic_threshold=data.ic_threshold,
        orthogonality_threshold=data.orthogonality_threshold,
        enable_crossover=data.enable_crossover,
        enable_causal=data.enable_causal,
    )

    if not started:
        raise HTTPException(409, "팩토리가 이미 실행 중입니다")

    status = scheduler.get_status()
    return AlphaFactoryStatusResponse(**status)


@router.post("/factory/stop", response_model=AlphaFactoryStatusResponse)
async def stop_factory():
    """알파 팩토리 중지."""
    from app.alpha.scheduler import get_scheduler

    scheduler = get_scheduler()
    stopped = await scheduler.stop()

    if not stopped:
        raise HTTPException(409, "팩토리가 실행 중이 아닙니다")

    status = scheduler.get_status()
    return AlphaFactoryStatusResponse(**status)


@router.get("/factory/status", response_model=AlphaFactoryStatusResponse)
async def get_factory_status():
    """알파 팩토리 상태 조회."""
    from app.alpha.scheduler import get_scheduler

    scheduler = get_scheduler()
    status = scheduler.get_status()
    return AlphaFactoryStatusResponse(**status)


# ── Phase 3: 팩터 포트폴리오 ──


@router.post("/portfolio/build", response_model=CompositeFactorResponse)
async def build_composite(
    data: CompositeFactorBuildRequest,
    db: AsyncSession = Depends(get_db),
):
    """복합 팩터 생성."""
    from app.alpha.portfolio import build_composite_factor

    try:
        result = await build_composite_factor(
            db=db,
            factor_ids=data.factor_ids,
            method=data.method,
            name=data.name,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    # 복합 팩터 DB 저장
    composite = AlphaFactor(
        name=data.name,
        expression_str=result.composite_expression,
        factor_type="composite",
        component_ids=result.component_ids,
        ic_mean=result.metrics.ic_mean,
        ic_std=result.metrics.ic_std,
        icir=result.metrics.icir,
        turnover=result.metrics.turnover,
        sharpe=result.metrics.sharpe,
        max_drawdown=result.metrics.max_drawdown,
        status="discovered",
    )
    db.add(composite)
    await db.commit()
    await db.refresh(composite)

    return CompositeFactorResponse(
        id=str(composite.id),
        name=composite.name,
        factor_type=composite.factor_type,
        expression_str=composite.expression_str,
        component_ids=result.component_ids,
        ic_mean=result.metrics.ic_mean,
        created_at=composite.created_at.isoformat(),
    )


@router.post("/portfolio/correlation", response_model=CorrelationMatrixResponse)
async def get_correlation(
    data: CorrelationRequest,
    db: AsyncSession = Depends(get_db),
):
    """팩터 간 상관행렬 조회."""
    from app.alpha.portfolio import compute_correlation_matrix

    try:
        result = await compute_correlation_matrix(db=db, factor_ids=data.factor_ids)
    except ValueError as e:
        raise HTTPException(400, str(e))

    return CorrelationMatrixResponse(**result)


# ── 내부 헬퍼 ──

def _factor_to_response(f: AlphaFactor) -> AlphaFactorResponse:
    return AlphaFactorResponse(
        id=str(f.id),
        mining_run_id=str(f.mining_run_id) if f.mining_run_id else None,
        name=f.name,
        expression_str=f.expression_str,
        expression_sympy=f.expression_sympy,
        polars_code=f.polars_code,
        hypothesis=f.hypothesis,
        generation=f.generation,
        ic_mean=f.ic_mean,
        ic_std=f.ic_std,
        icir=f.icir,
        turnover=f.turnover,
        sharpe=f.sharpe,
        max_drawdown=f.max_drawdown,
        status=f.status,
        causal_robust=f.causal_robust,
        causal_effect_size=f.causal_effect_size,
        causal_p_value=f.causal_p_value,
        parent_ids=f.parent_ids,
        factor_type=f.factor_type,
        component_ids=f.component_ids,
        fitness_composite=getattr(f, "fitness_composite", None),
        tree_depth=getattr(f, "tree_depth", None),
        tree_size=getattr(f, "tree_size", None),
        expression_hash=getattr(f, "expression_hash", None),
        operator_origin=getattr(f, "operator_origin", None),
        is_elite=getattr(f, "is_elite", None),
        population_active=getattr(f, "population_active", None),
        birth_generation=getattr(f, "birth_generation", None),
        created_at=f.created_at.isoformat(),
        updated_at=f.updated_at.isoformat(),
    )
