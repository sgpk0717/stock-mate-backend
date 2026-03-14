"""워크플로우 REST API — 상태 조회, 수동 트리거, 히스토리."""

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.workflow.orchestrator import get_orchestrator
from app.workflow.schemas import (
    BestFactorOut,
    TradingFeedbackSubmit,
    WorkflowEventOut,
    WorkflowRunOut,
    WorkflowStatusOut,
    WorkflowTriggerRequest,
    WorkflowTriggerResponse,
)

router = APIRouter(prefix="/workflow", tags=["workflow"])


@router.get("/status")
async def get_workflow_status() -> dict:
    """현재 워크플로우 상태 (MCP/OpenClaw 겸용)."""
    orchestrator = get_orchestrator()
    return await orchestrator.get_status()


@router.post("/trigger", response_model=WorkflowTriggerResponse)
async def trigger_workflow_phase(req: WorkflowTriggerRequest):
    """워크플로우 페이즈 수동 트리거."""
    orchestrator = get_orchestrator()

    handlers = {
        "pre_market": orchestrator.handle_pre_market,
        "market_open": orchestrator.handle_market_open,
        "market_close": orchestrator.handle_market_close,
        "review": orchestrator.handle_review,
        "mining": orchestrator.handle_mining,
        "emergency_stop": orchestrator.handle_emergency_stop,
        "resume": orchestrator.handle_resume,
        "reset": orchestrator.handle_reset,
    }

    handler = handlers.get(req.phase)
    if handler is None:
        raise HTTPException(
            status_code=400,
            detail=f"유효하지 않은 페이즈: {req.phase}. 허용: {list(handlers.keys())}",
        )

    result = await handler()
    return WorkflowTriggerResponse(
        success=result.get("success", False),
        phase=result.get("phase", req.phase),
        message=result.get("message", "처리 완료"),
    )


@router.get("/history", response_model=list[WorkflowRunOut])
async def get_workflow_history(
    limit: int = 30,
    session: AsyncSession = Depends(get_db),
):
    """워크플로우 실행 히스토리."""
    orchestrator = get_orchestrator()
    runs = await orchestrator.get_history(session, limit=limit)
    return [
        WorkflowRunOut(
            id=str(r.id),
            date=r.date,
            phase=r.phase,
            status=r.status,
            config=r.config,
            mining_run_id=str(r.mining_run_id) if r.mining_run_id else None,
            selected_factor_id=str(r.selected_factor_id) if r.selected_factor_id else None,
            trading_context_id=str(r.trading_context_id) if r.trading_context_id else None,
            review_summary=r.review_summary,
            trade_count=r.trade_count,
            pnl_amount=float(r.pnl_amount) if r.pnl_amount else None,
            pnl_pct=r.pnl_pct,
            mining_context=r.mining_context,
            started_at=r.started_at,
            completed_at=r.completed_at,
            error_message=r.error_message,
            created_at=r.created_at,
        )
        for r in runs
    ]


@router.get("/events/{run_id}", response_model=list[WorkflowEventOut])
async def get_workflow_events(
    run_id: str,
    limit: int = 100,
    session: AsyncSession = Depends(get_db),
):
    """특정 워크플로우 실행의 이벤트 로그."""
    orchestrator = get_orchestrator()
    events = await orchestrator.get_events(session, uuid.UUID(run_id), limit=limit)
    return [
        WorkflowEventOut(
            id=str(e.id),
            workflow_run_id=str(e.workflow_run_id) if e.workflow_run_id else None,
            phase=e.phase,
            event_type=e.event_type,
            message=e.message,
            data=e.data,
            created_at=e.created_at,
        )
        for e in events
    ]


@router.get("/best-factors", response_model=list[BestFactorOut])
async def get_best_factors(
    limit: int = 5,
    min_ic: float | None = None,
    min_sharpe: float | None = None,
    require_causal: bool | None = None,
    interval: str | None = None,
    session: AsyncSession = Depends(get_db),
):
    """매매 가능 최상위 팩터 조회."""
    from app.workflow.auto_selector import select_best_factors

    results = await select_best_factors(
        session,
        limit=limit,
        min_ic=min_ic,
        min_sharpe=min_sharpe,
        require_causal=require_causal,
        interval=interval,
    )
    return [
        BestFactorOut(
            factor_id=str(r["factor"].id),
            factor_name=r["factor"].name,
            expression_str=r["factor"].expression_str,
            ic_mean=r["factor"].ic_mean,
            icir=r["factor"].icir,
            sharpe=r["factor"].sharpe,
            causal_robust=r["factor"].causal_robust,
            interval=r["factor"].interval,
            composite_score=r["score"],
            score_breakdown=r["breakdown"],
        )
        for r in results
    ]
