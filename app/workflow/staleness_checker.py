"""팩터 스탈니스 체커 — 7일/14일/30일 노화 관리.

설계서 §12.3: 활성 팩터의 실매매 IC를 재계산하고,
성과가 떨어지면 staleness_warning → stale → retired 순으로 전환.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.alpha.models import AlphaFactor
from app.workflow.models import LiveFeedback

logger = logging.getLogger(__name__)


async def check_staleness(session: AsyncSession) -> dict:
    """활성 팩터의 스탈니스 검사.

    Returns:
        {"warned": int, "stale": int, "retired": int}
    """
    now = datetime.now()
    warned = 0
    stale_count = 0
    retired_count = 0

    # 활성 팩터 조회 (discovered, validated)
    stmt = select(AlphaFactor).where(
        AlphaFactor.status.in_(["discovered", "validated"]),
    )
    result = await session.execute(stmt)
    factors = result.scalars().all()

    for factor in factors:
        # 최근 7일 실매매 피드백에서 IC/수익률 집계
        week_ago = now - timedelta(days=7)
        fb_stmt = select(
            func.avg(LiveFeedback.realized_pnl_pct).label("avg_pnl"),
            func.count(LiveFeedback.id).label("cnt"),
        ).where(
            LiveFeedback.factor_id == factor.id,
            LiveFeedback.date >= week_ago.date(),
        )
        fb_result = await session.execute(fb_stmt)
        fb_row = fb_result.one_or_none()

        if not fb_row or not fb_row.cnt or fb_row.cnt == 0:
            continue

        avg_pnl = float(fb_row.avg_pnl or 0)

        # live_ic_7d 업데이트 (실매매 PnL을 IC 대리 지표로 사용)
        factor.live_ic_7d = avg_pnl
        factor.last_evaluated_at = now

        # 백테스트 IC 대비 50% 이하면 경고
        backtest_ic = factor.ic_mean or 0.05
        if backtest_ic > 0 and avg_pnl < backtest_ic * 0.5:
            if not factor.staleness_warning:
                factor.staleness_warning = True
                warned += 1
                logger.info(
                    "스탈니스 경고: 팩터 %s (live_ic=%.4f vs backtest=%.4f)",
                    factor.id, avg_pnl, backtest_ic,
                )

    await session.flush()

    # 14일 연속 IC 하락 → stale
    two_weeks_ago = now - timedelta(days=14)
    stale_stmt = select(AlphaFactor).where(
        AlphaFactor.status.in_(["discovered", "validated"]),
        AlphaFactor.staleness_warning.is_(True),
        AlphaFactor.last_evaluated_at <= two_weeks_ago,
    )
    stale_result = await session.execute(stale_stmt)
    stale_factors = stale_result.scalars().all()

    for f in stale_factors:
        f.status = "stale"
        stale_count += 1
        logger.info("팩터 stale 전환: %s", f.id)

    # 30일 이상 stale → retired
    month_ago = now - timedelta(days=30)
    retired_stmt = (
        update(AlphaFactor)
        .where(
            AlphaFactor.status == "stale",
            AlphaFactor.last_evaluated_at <= month_ago,
        )
        .values(status="retired")
        .returning(AlphaFactor.id)
    )
    retired_result = await session.execute(retired_stmt)
    retired_ids = retired_result.scalars().all()
    retired_count = len(retired_ids)
    if retired_count:
        logger.info("팩터 retired 전환: %d개", retired_count)

    # 자동 퍼지: retired 90일 경과 → hard delete
    purge_retired_cutoff = now - timedelta(days=90)
    purge_retired_stmt = select(AlphaFactor.id).where(
        AlphaFactor.status == "retired",
        AlphaFactor.updated_at <= purge_retired_cutoff,
    )
    purge_retired_ids = (await session.execute(purge_retired_stmt)).scalars().all()
    purged_retired = 0
    if purge_retired_ids:
        from sqlalchemy import delete as sa_delete
        await session.execute(
            sa_delete(AlphaFactor).where(AlphaFactor.id.in_(purge_retired_ids))
        )
        purged_retired = len(purge_retired_ids)
        logger.info("retired 90일 퍼지: %d개 삭제", purged_retired)

    # 자동 퍼지: mirage 7일 경과 → hard delete
    purge_mirage_cutoff = now - timedelta(days=7)
    purge_mirage_stmt = select(AlphaFactor.id).where(
        AlphaFactor.status == "mirage",
        AlphaFactor.updated_at <= purge_mirage_cutoff,
    )
    purge_mirage_ids = (await session.execute(purge_mirage_stmt)).scalars().all()
    purged_mirage = 0
    if purge_mirage_ids:
        from sqlalchemy import delete as sa_delete
        await session.execute(
            sa_delete(AlphaFactor).where(AlphaFactor.id.in_(purge_mirage_ids))
        )
        purged_mirage = len(purge_mirage_ids)
        logger.info("mirage 7일 퍼지: %d개 삭제", purged_mirage)

    await session.commit()

    return {
        "warned": warned,
        "stale": stale_count,
        "retired": retired_count,
        "purged_retired": purged_retired,
        "purged_mirage": purged_mirage,
    }
