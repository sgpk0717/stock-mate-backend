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
    """현재 워크플로우 상태 (MCP/OpenClaw 겸용).

    Phase 2: Redis 캐시 우선, 실패 시 기존 방식 폴백.
    """
    # Redis에서 읽기 시도
    try:
        import json
        from app.core.redis import hgetall
        cached = await hgetall("workflow:status")
        # 날짜 체크: 캐시가 오늘 것이 아니면 무시
        from datetime import date
        cached_date = cached.get("date", "") if cached else ""
        if cached and cached.get("phase") and cached_date == str(date.today()):
            result = {}
            for k, v in cached.items():
                if k in ("step_status",) and v:
                    try:
                        result[k] = json.loads(v)
                    except (json.JSONDecodeError, TypeError):
                        result[k] = v
                elif k in ("trade_count", "mining_cycles", "mining_factors"):
                    try:
                        result[k] = int(v) if v else 0
                    except (ValueError, TypeError):
                        result[k] = 0
                elif k in ("pnl_pct", "pnl_amount"):
                    try:
                        result[k] = float(v) if v else None
                    except (ValueError, TypeError):
                        result[k] = None
                elif k == "mining_running":
                    result[k] = v.lower() == "true" if v else False
                elif v == "None" or v == "":
                    result[k] = None
                else:
                    result[k] = v
            return result
    except Exception:
        pass

    # 폴백: 기존 방식 (DB 직접 조회)
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


@router.post("/test-review")
async def test_review_telegram():
    """리뷰 텔레그램 메시지 테스트 발송.

    상태 전이 없이, 현재 DB 데이터로 리뷰를 생성하여 텔레그램으로 발송.
    """
    import json as _json
    from app.core.database import async_session
    from app.workflow.models import WorkflowRun
    from sqlalchemy import select
    from datetime import date

    async with async_session() as session:
        stmt = select(WorkflowRun).where(WorkflowRun.date == date.today())
        result = await session.execute(stmt)
        run = result.scalar_one_or_none()

    if not run:
        raise HTTPException(404, "오늘 워크플로우 실행 기록 없음")

    # TradeReviewer로 세션별 데이터 생성
    async with async_session() as session:
        from app.workflow.trade_reviewer import TradeReviewer
        reviewer = TradeReviewer()
        review = await reviewer.generate_review(session, run)

    review_summary = review.to_dict()

    # Claude LLM 리뷰 생성
    from app.core.config import settings
    review_data = {
        "date": str(run.date),
        "initial_capital": int(settings.WORKFLOW_INITIAL_CAPITAL),
        "num_sessions": len((run.config or {}).get("trading_context_ids", [])),
        "total_trades": review.trade_count,
        "total_pnl_amount": review.total_pnl,
        "total_pnl_pct": round(review.total_pnl_pct, 2),
        "win_rate": round(review.win_rate, 1),
        "per_session": review.per_session,
        "improvements": review.improvements,
        "time_breakdown": review_summary.get("time_breakdown", {}),
    }

    try:
        from app.core.llm._anthropic import chat_simple
        llm_response = await chat_simple(
            system=(
                "당신은 퀀트 트레이딩 일일 리뷰어입니다. "
                "세션별(팩터별) 매매 결과를 분석하여 텔레그램 리포트를 작성하세요.\n"
                "- 각 팩터 수식이 무엇을 의미하는지 1줄 설명\n"
                "- 세션별 성과 비교 (거래수, 승률, 손익)\n"
                "- 전체 포트폴리오 수익률 (자본금 기준)\n"
                "- 어떤 팩터가 가장 잘/못했는지 분석\n"
                "- 개선 방향 1-2줄\n"
                "HTML 태그(<b>, <i>, <code>)를 사용하세요. "
                "이모지 적절히 사용. 800자 이내. 한국어로 작성."
            ),
            messages=[{
                "role": "user",
                "content": _json.dumps(review_data, ensure_ascii=False, default=str),
            }],
            max_tokens=1200,
            caller="workflow.review_api",
        )
        tg_msg = llm_response.text
    except Exception as e:
        tg_msg = (
            f"\U0001f4cb <b>일일 리뷰 (테스트)</b> ({run.date})\n"
            f"거래 {review.trade_count}건 | 손익 {review.total_pnl:+,.0f}원 ({review.total_pnl_pct:+.2f}%)\n"
            f"LLM 실패: {e}"
        )

    # 텔레그램 발송
    from app.telegram.bot import send_message
    await send_message(tg_msg, category="review_report", caller="test-review")

    return {"success": True, "message": "리뷰 발송됨", "review_data": review_data}


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
            step_status=r.step_status,
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
