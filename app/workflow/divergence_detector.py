"""팩터 라이브-백테스트 다이버전스 감지 + 자동 정지.

LiveFeedback 테이블에서 누적 PnL을 조회하여,
실매매 성능이 임계치 이하로 떨어진 팩터를 자동으로 halt/warn 처리한다.

halted 팩터는 select_best_factors()에서 자동 제외된다.
(AlphaFactor.status IN ('discovered', 'validated')만 필터)
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings

logger = logging.getLogger(__name__)


async def check_divergence(
    session: AsyncSession,
    factor_id: str,
) -> dict:
    """특정 팩터의 라이브-백테스트 다이버전스를 감지.

    Returns:
        {"action": "none" | "warn" | "halt", "cumulative_pnl_pct": float, "days": int}
    """
    if not settings.WORKFLOW_DIVERGENCE_CHECK_ENABLED:
        return {"action": "none", "reason": "divergence check disabled"}

    from app.workflow.models import LiveFeedback

    stmt = (
        select(
            func.sum(LiveFeedback.realized_pnl_pct).label("cum_pnl"),
            func.count(LiveFeedback.id).label("day_count"),
        )
        .where(LiveFeedback.factor_id == factor_id)
        .where(LiveFeedback.realized_pnl_pct.is_not(None))
    )
    result = await session.execute(stmt)
    row = result.one_or_none()

    if row is None or row.day_count is None or row.day_count == 0:
        return {"action": "none", "reason": "no live feedback data"}

    cum_pnl = float(row.cum_pnl or 0)
    days = int(row.day_count)

    result_data = {
        "cumulative_pnl_pct": round(cum_pnl, 4),
        "days": days,
        "factor_id": str(factor_id),
    }

    # 최소 일수 미만이면 판단 보류
    if days < settings.WORKFLOW_DIVERGENCE_MIN_DAYS:
        return {**result_data, "action": "none", "reason": f"min_days 미달 ({days}/{settings.WORKFLOW_DIVERGENCE_MIN_DAYS})"}

    # halt: 누적 PnL이 halt 임계값 이하
    if cum_pnl <= settings.WORKFLOW_DIVERGENCE_HALT_THRESHOLD:
        await _apply_halt(session, factor_id, cum_pnl, days)
        return {**result_data, "action": "halt"}

    # warn: 누적 PnL이 warn 임계값 이하
    if cum_pnl <= settings.WORKFLOW_DIVERGENCE_WARN_THRESHOLD:
        await _apply_warn(session, factor_id, cum_pnl, days)
        return {**result_data, "action": "warn"}

    return {**result_data, "action": "none"}


async def check_all_active_factors(session: AsyncSession) -> list[dict]:
    """활성 상태 팩터 전체에 대해 다이버전스 체크.

    Returns:
        list of divergence check results (action != 'none')
    """
    if not settings.WORKFLOW_DIVERGENCE_CHECK_ENABLED:
        return []

    from app.alpha.models import AlphaFactor

    stmt = select(AlphaFactor.id).where(
        AlphaFactor.status.in_(["discovered", "validated"])
    )
    result = await session.execute(stmt)
    factor_ids = [str(row[0]) for row in result.all()]

    actions: list[dict] = []
    for fid in factor_ids:
        check = await check_divergence(session, fid)
        if check.get("action") not in ("none",):
            actions.append(check)

    return actions


async def _apply_halt(
    session: AsyncSession,
    factor_id: str,
    cum_pnl: float,
    days: int,
) -> None:
    """팩터를 halted로 전환 + 감사 이벤트 기록."""
    from app.alpha.models import AlphaFactor
    from app.workflow.models import WorkflowEvent, WorkflowRun

    # AlphaFactor.status = "halted"
    stmt = (
        select(AlphaFactor).where(AlphaFactor.id == factor_id)
    )
    result = await session.execute(stmt)
    factor = result.scalar_one_or_none()
    if factor and factor.status not in ("halted", "retired"):
        factor.status = "halted"
        factor.staleness_warning = True
        logger.warning(
            "팩터 다이버전스 HALT: %s (누적 %.2f%%, %d일)",
            factor_id, cum_pnl, days,
        )

    # WorkflowEvent 감사 로그
    today_run_stmt = select(WorkflowRun).where(WorkflowRun.date == date.today())
    run_result = await session.execute(today_run_stmt)
    run = run_result.scalar_one_or_none()
    if run:
        event = WorkflowEvent(
            workflow_run_id=run.id,
            event_type="divergence_halt",
            message=(
                f"팩터 {factor_id} 자동 정지: "
                f"누적 PnL {cum_pnl:+.2f}% ({days}일)"
            ),
        )
        session.add(event)


async def _apply_warn(
    session: AsyncSession,
    factor_id: str,
    cum_pnl: float,
    days: int,
) -> None:
    """팩터에 staleness_warning 설정 + 감사 이벤트."""
    from app.alpha.models import AlphaFactor
    from app.workflow.models import WorkflowEvent, WorkflowRun

    stmt = select(AlphaFactor).where(AlphaFactor.id == factor_id)
    result = await session.execute(stmt)
    factor = result.scalar_one_or_none()
    if factor and not factor.staleness_warning:
        factor.staleness_warning = True
        logger.info(
            "팩터 다이버전스 WARNING: %s (누적 %.2f%%, %d일)",
            factor_id, cum_pnl, days,
        )

    today_run_stmt = select(WorkflowRun).where(WorkflowRun.date == date.today())
    run_result = await session.execute(today_run_stmt)
    run = run_result.scalar_one_or_none()
    if run:
        event = WorkflowEvent(
            workflow_run_id=run.id,
            event_type="divergence_warning",
            message=(
                f"팩터 {factor_id} 경고: "
                f"누적 PnL {cum_pnl:+.2f}% ({days}일)"
            ),
        )
        session.add(event)
