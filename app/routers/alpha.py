"""알파 마이닝 REST API 라우터."""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import date

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.alpha.models import AlphaFactor, AlphaMiningRun
from app.alpha.runner import execute_alpha_mining
from app.alpha.schemas import (
    AlphaFactorBacktestRequest,
    AlphaFactorPageResponse,
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
    FactorChatCreateResponse,
    FactorChatMessageRequest,
    FactorChatMessageResponse,
    FactorChatSessionResponse,
    MiningIterationLogs,
)
from app.alpha.universe import Universe, get_universe_info
from app.core.database import get_db

logger = logging.getLogger(__name__)

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
            "interval": data.interval,
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
            interval=data.interval,
            seed_factor_ids=data.seed_factor_ids or None,
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

_ALLOWED_SORT_COLUMNS = {
    "ic_mean", "icir", "sharpe", "max_drawdown",
    "generation", "fitness_composite", "created_at",
}


@router.get("/factors", response_model=AlphaFactorPageResponse)
async def list_factors(
    status: str | None = None,
    min_ic: float | None = None,
    causal_robust: bool | None = None,
    interval: str | None = None,
    sort_by: str = "ic_mean",
    order: str = "desc",
    offset: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
):
    """팩터 목록 (status/min_ic/causal_robust 필터, 멀티 정렬, 페이지네이션).

    sort_by/order는 쉼표 구분 문자열로 멀티 정렬 지원.
    예: sort_by=ic_mean,sharpe&order=desc,asc
    """
    sort_cols = [s.strip() for s in sort_by.split(",") if s.strip()]
    sort_orders = [s.strip() for s in order.split(",") if s.strip()]

    order_clauses = []
    for i, sc in enumerate(sort_cols):
        if sc not in _ALLOWED_SORT_COLUMNS:
            continue
        col = getattr(AlphaFactor, sc)
        od = sort_orders[i] if i < len(sort_orders) else "desc"
        order_clauses.append(
            col.asc().nulls_last() if od == "asc" else col.desc().nulls_last()
        )
    if not order_clauses:
        order_clauses.append(AlphaFactor.ic_mean.desc().nulls_last())

    # WHERE 조건 구성
    filters = []
    if status:
        filters.append(AlphaFactor.status == status)
    if min_ic is not None:
        filters.append(AlphaFactor.ic_mean >= min_ic)
    if causal_robust is not None:
        filters.append(AlphaFactor.causal_robust == causal_robust)
    if interval:
        filters.append(AlphaFactor.interval == interval)

    # 전체 개수 (ORDER BY 없이 — 정렬은 COUNT에 불필요)
    count_q = select(func.count()).select_from(AlphaFactor)
    for f in filters:
        count_q = count_q.where(f)
    total = await db.scalar(count_q)

    # 페이지 데이터 (WHERE + ORDER BY + OFFSET/LIMIT)
    data_q = select(AlphaFactor)
    for f in filters:
        data_q = data_q.where(f)
    data_q = data_q.order_by(*order_clauses).offset(offset).limit(limit)

    result = await db.execute(data_q)
    factors = result.scalars().all()

    return AlphaFactorPageResponse(
        items=[_factor_to_response(f) for f in factors],
        total=total or 0,
    )


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


@router.post("/factors/delete-batch", status_code=204)
async def delete_factors_batch(
    body: dict,
    db: AsyncSession = Depends(get_db),
):
    """팩터 일괄 삭제."""
    ids = body.get("factor_ids", [])
    if not ids:
        return
    uuids = [uuid.UUID(fid) for fid in ids]
    await db.execute(
        delete(AlphaFactor).where(AlphaFactor.id.in_(uuids))
    )
    await db.commit()


@router.post("/factor/{factor_id}/backtest", status_code=202)
async def backtest_with_factor(
    factor_id: str,
    data: AlphaFactorBacktestRequest,
    db: AsyncSession = Depends(get_db),
):
    """횡단면 포트폴리오 기반 팩터 백테스트.

    매일 전체 종목을 팩터 값으로 랭킹하여 상위 top_pct% 종목을 매수한다.
    symbols가 비어 있으면 마이닝 유니버스를 기본값으로 사용한다.
    """
    result = await db.execute(
        select(AlphaFactor).where(AlphaFactor.id == uuid.UUID(factor_id))
    )
    factor = result.scalar_one_or_none()
    if not factor:
        raise HTTPException(404, "Factor not found")

    # 마이닝 run config 조회 (symbols/날짜 폴백 공용)
    run_config: dict | None = None
    need_run_config = not data.symbols or not data.start_date or not data.end_date
    if need_run_config and factor.mining_run_id:
        run_result = await db.execute(
            select(AlphaMiningRun.config).where(
                AlphaMiningRun.id == factor.mining_run_id
            )
        )
        run_config = run_result.scalar_one_or_none()

    # symbols가 비어 있으면 마이닝 유니버스에서 가져옴
    symbols = data.symbols if data.symbols else None
    if not symbols:
        if not run_config or not run_config.get("universe"):
            raise HTTPException(
                400,
                "종목 리스트가 비어 있고, 마이닝 run에 유니버스 설정이 없습니다. "
                "symbols를 직접 지정해 주세요.",
            )

        from app.alpha.universe import Universe, resolve_universe

        symbols = await resolve_universe(Universe(run_config["universe"]))
        if not symbols:
            raise HTTPException(
                500,
                f"유니버스 '{run_config['universe']}' 리졸브 결과가 비어 있습니다.",
            )
        logger.info(
            "팩터 백테스트: 마이닝 유니버스 '%s' 사용 (%d종목)",
            run_config["universe"],
            len(symbols),
        )

    # 날짜 범위: 요청값 → 마이닝 config 폴백
    start_str = data.start_date or (run_config.get("start_date") if run_config else None)
    end_str = data.end_date or (run_config.get("end_date") if run_config else None)
    if not start_str or not end_str:
        raise HTTPException(
            400,
            "백테스트 날짜 범위가 지정되지 않았고 마이닝 run에도 날짜 설정이 없습니다.",
        )

    from app.backtest.cost_model import CostConfig, default_cost_config
    from app.backtest.models import BacktestRun
    from app.alpha.factor_backtest import execute_factor_backtest

    start = date.fromisoformat(start_str)
    end = date.fromisoformat(end_str)

    # 인터벌 검증: 팩터의 원래 인터벌과 불일치 방지
    bt_interval = data.interval
    if factor.interval and bt_interval != factor.interval:
        raise HTTPException(
            400,
            f"팩터 인터벌({factor.interval})과 요청 인터벌({bt_interval})이 다릅니다. "
            f"팩터에 맞는 인터벌을 사용하세요.",
        )
    if not factor.interval:
        bt_interval = "1d"  # interval 컬럼 추가 이전 레거시 팩터

    cost_cfg = default_cost_config(bt_interval)

    run = BacktestRun(
        strategy_name=f"Alpha: {factor.name}",
        strategy_json={
            "name": f"Alpha: {factor.name}",
            "expression": factor.expression_str,
            "mode": "cross_sectional_portfolio",
            "interval": bt_interval,
            "top_pct": data.top_pct,
            "max_positions": data.max_positions,
            "rebalance_freq": data.rebalance_freq,
            "band_threshold": data.band_threshold,
        },
        start_date=start,
        end_date=end,
        initial_capital=float(data.initial_capital),
        cost_config=cost_cfg.model_dump(),
        status="PENDING",
        progress=0,
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)

    asyncio.create_task(
        execute_factor_backtest(
            run_id=run.id,
            expression_str=factor.expression_str,
            symbols=symbols,
            start_date=start,
            end_date=end,
            initial_capital=data.initial_capital,
            top_pct=data.top_pct,
            max_positions=data.max_positions,
            rebalance_freq=data.rebalance_freq,
            band_threshold=data.band_threshold,
            cost_config=cost_cfg,
            interval=bt_interval,
            stop_loss_pct=data.stop_loss_pct,
            max_drawdown_pct=data.max_drawdown_pct,
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


@router.post("/factors/validate-batch", status_code=202)
async def validate_factors_batch_endpoint(
    body: dict,
    db: AsyncSession = Depends(get_db),
):
    """선택된 팩터 일괄 인과 검증 (비동기). job_id를 반환하고 백그라운드에서 실행."""
    from app.alpha.factory_client import get_factory_client

    ids = body.get("factor_ids", [])
    if not ids:
        return {"job_id": None, "total": 0, "skipped": 0}

    uuids = [uuid.UUID(fid) for fid in ids]
    result = await db.execute(
        select(AlphaFactor).where(AlphaFactor.id.in_(uuids))
    )
    factors = result.scalars().all()

    to_validate = []
    skipped = 0

    for factor in factors:
        if factor.causal_robust is not None:
            skipped += 1
        else:
            to_validate.append(factor)

    if not to_validate:
        return {"job_id": None, "total": 0, "skipped": skipped}

    job_id = uuid.uuid4().hex[:12]
    factor_ids = [f.id for f in to_validate]

    client = get_factory_client()
    await client.start_validation_batch(factor_ids, job_id, len(factor_ids))

    return {"job_id": job_id, "total": len(factor_ids), "skipped": skipped}


@router.get("/validate/{job_id}/status")
async def get_validation_status(job_id: str):
    """인과 검증 잡 진행 상황 조회."""
    from app.alpha.factory_client import get_factory_client

    client = get_factory_client()
    progress = await client.get_validation_progress(job_id)
    if progress is None:
        raise HTTPException(404, "Validation job not found")

    return progress


# ── Phase 3: 알파 팩토리 ──


@router.post("/factory/start", response_model=AlphaFactoryStatusResponse)
async def start_factory(data: AlphaFactoryStartRequest):
    """알파 팩토리 시작."""
    from app.alpha.factory_client import get_factory_client

    client = get_factory_client()
    result = await client.start(
        context=data.context,
        universe=data.universe,
        start_date=data.start_date,
        end_date=data.end_date,
        data_interval=data.data_interval,
        interval_minutes=data.interval_minutes,
        max_iterations=data.max_iterations_per_cycle,
        ic_threshold=data.ic_threshold,
        orthogonality_threshold=data.orthogonality_threshold,
        enable_crossover=data.enable_crossover,
        enable_causal=data.enable_causal,
        max_cycles=data.max_cycles,
    )

    if not result["started"]:
        raise HTTPException(409, "팩토리가 이미 실행 중입니다")

    return AlphaFactoryStatusResponse(**result["status"])


@router.post("/factory/stop", response_model=AlphaFactoryStatusResponse)
async def stop_factory():
    """알파 팩토리 중지."""
    from app.alpha.factory_client import get_factory_client

    client = get_factory_client()
    result = await client.stop()

    if not result["stopped"]:
        raise HTTPException(409, "팩토리가 실행 중이 아닙니다")

    return AlphaFactoryStatusResponse(**result["status"])


@router.get("/factory/status", response_model=AlphaFactoryStatusResponse)
async def get_factory_status():
    """알파 팩토리 상태 조회."""
    from app.alpha.factory_client import get_factory_client

    client = get_factory_client()
    status = await client.get_status()
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


# ── 팩터 AI 채팅 ──


@router.post(
    "/factor/{factor_id}/chat",
    response_model=FactorChatCreateResponse,
)
async def create_factor_chat(
    factor_id: str,
    db: AsyncSession = Depends(get_db),
):
    """팩터 기반 AI 채팅 세션 생성."""
    result = await db.execute(
        select(AlphaFactor).where(AlphaFactor.id == uuid.UUID(factor_id))
    )
    factor = result.scalar_one_or_none()
    if not factor:
        raise HTTPException(404, "Factor not found")

    # 마이닝 run에서 universe/dates/interval 추출
    universe = "KOSPI200"
    start_date = "2025-06-01"
    end_date = "2025-12-31"
    interval = getattr(factor, "interval", "1d") or "1d"

    if factor.mining_run_id:
        run_result = await db.execute(
            select(AlphaMiningRun.config).where(
                AlphaMiningRun.id == factor.mining_run_id
            )
        )
        run_config = run_result.scalar_one_or_none()
        if run_config:
            universe = run_config.get("universe", universe)
            start_date = run_config.get("start_date", start_date)
            end_date = run_config.get("end_date", end_date)
            interval = run_config.get("interval", interval)

    from app.alpha.factor_chat import factor_chat_store

    session = factor_chat_store.create(
        source_factor_id=str(factor.id),
        source_expression=factor.expression_str,
        source_hypothesis=factor.hypothesis or "",
        source_metrics={
            "ic_mean": factor.ic_mean or 0,
            "ic_std": factor.ic_std or 0,
            "icir": factor.icir or 0,
            "turnover": factor.turnover or 0,
            "sharpe": factor.sharpe or 0,
            "max_drawdown": factor.max_drawdown or 0,
        },
        current_expression=factor.expression_str,
        current_metrics={
            "ic_mean": factor.ic_mean or 0,
            "ic_std": factor.ic_std or 0,
            "icir": factor.icir or 0,
            "turnover": factor.turnover or 0,
            "sharpe": factor.sharpe or 0,
            "max_drawdown": factor.max_drawdown or 0,
        },
        universe=universe,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
    )

    return FactorChatCreateResponse(
        session_id=session.id,
        source_factor_id=str(factor.id),
        source_expression=factor.expression_str,
        universe=universe,
        interval=interval,
        status=session.status,
    )


@router.post(
    "/factor/chat/{session_id}/message",
    response_model=FactorChatMessageResponse,
)
async def send_factor_chat_message(
    session_id: str,
    req: FactorChatMessageRequest,
):
    """팩터 채팅 메시지 전송."""
    from app.alpha.factor_chat import factor_chat_store, process_message

    session = factor_chat_store.get(session_id)
    if not session:
        raise HTTPException(404, "Chat session not found or expired")

    try:
        assistant_msg = await process_message(session, req.message)
    except Exception as e:
        logger.exception("Factor chat error for session %s", session_id)
        raise HTTPException(500, f"채팅 처리 실패: {str(e)[:200]}")

    return FactorChatMessageResponse(
        role=assistant_msg.role,
        content=assistant_msg.content,
        timestamp=assistant_msg.timestamp,
        factor_draft=assistant_msg.factor_draft,
        current_expression=session.current_expression,
        current_metrics=session.current_metrics,
    )


@router.get(
    "/factor/chat/{session_id}",
    response_model=FactorChatSessionResponse,
)
async def get_factor_chat_session(session_id: str):
    """팩터 채팅 세션 조회."""
    from app.alpha.factor_chat import factor_chat_store

    session = factor_chat_store.get(session_id)
    if not session:
        raise HTTPException(404, "Chat session not found or expired")

    return FactorChatSessionResponse(**session.to_dict())


@router.post("/factor/chat/{session_id}/save", response_model=AlphaFactorResponse)
async def save_factor_from_chat(
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    """채팅에서 수정한 팩터를 새 AlphaFactor로 DB 저장."""
    from app.alpha.factor_chat import factor_chat_store

    session = factor_chat_store.get(session_id)
    if not session:
        raise HTTPException(404, "Chat session not found or expired")

    if not session.current_expression:
        raise HTTPException(400, "저장할 수식이 없습니다. 먼저 대화로 수식을 수정하세요.")

    # SymPy 문자열 생성
    expression_sympy = None
    polars_code = None
    try:
        from app.alpha.ast_converter import (
            parse_expression,
            sympy_to_code_string,
            sympy_to_polars,
        )
        import sympy

        parsed = parse_expression(session.current_expression)
        expression_sympy = sympy.srepr(parsed)
        polars_code = sympy_to_code_string(parsed)
    except Exception:
        pass

    metrics = session.current_metrics or {}

    new_factor = AlphaFactor(
        mining_run_id=None,
        name=f"Custom: {session.current_expression[:50]}",
        expression_str=session.current_expression,
        expression_sympy=expression_sympy,
        polars_code=polars_code,
        hypothesis=session.source_hypothesis,
        generation=0,
        ic_mean=metrics.get("ic_mean"),
        ic_std=metrics.get("ic_std"),
        icir=metrics.get("icir"),
        turnover=metrics.get("turnover"),
        sharpe=metrics.get("sharpe"),
        max_drawdown=metrics.get("max_drawdown"),
        status="discovered",
        operator_origin="manual",
        parent_ids=[session.source_factor_id],
        interval=session.interval,
    )
    db.add(new_factor)
    await db.commit()
    await db.refresh(new_factor)

    session.status = "saved"
    session.touch()

    return _factor_to_response(new_factor)


@router.delete("/factor/chat/{session_id}", status_code=204)
async def delete_factor_chat_session(session_id: str):
    """팩터 채팅 세션 삭제."""
    from app.alpha.factor_chat import factor_chat_store

    if not factor_chat_store.delete(session_id):
        raise HTTPException(404, "Chat session not found")


# ── 내부 헬퍼 ──

def _factor_to_response(f: AlphaFactor) -> AlphaFactorResponse:
    return AlphaFactorResponse(
        id=str(f.id),
        mining_run_id=str(f.mining_run_id) if f.mining_run_id else None,
        name=f.name,
        expression_str=f.expression_str,
        interval=getattr(f, "interval", "1d"),
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
