"""Phase 4: 시뮬레이션 (ABM) 라우터."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.simulation.models import StressTestRun
from app.simulation.runner import execute_stress_test
from app.simulation.scenarios import PRESET_SCENARIOS, generate_custom_scenario
from app.simulation.schemas import (
    CustomScenarioRequest,
    CustomScenarioResponse,
    ScenarioPreset,
    StressTestRequest,
    StressTestResponse,
    StressTestRunResponse,
    StressTestRunSummary,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/simulation", tags=["simulation"])


@router.post("/stress-test", status_code=202, response_model=StressTestResponse)
async def start_stress_test(
    data: StressTestRequest, db: AsyncSession = Depends(get_db)
):
    """Start async stress test. Returns run_id immediately."""
    run = StressTestRun(
        name=data.name,
        strategy_json=data.strategy_json,
        scenario_type=data.scenario.type,
        scenario_config=data.scenario.model_dump(),
        agent_config=data.agent_config.model_dump(),
        exchange_config=data.exchange_config.model_dump(),
        status="PENDING",
        progress=0,
        total_steps=data.exchange_config.total_steps,
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)

    asyncio.create_task(
        execute_stress_test(
            run_id=run.id,
            strategy_json=run.strategy_json,
            scenario_config=run.scenario_config,
            agent_config=run.agent_config,
            exchange_config=run.exchange_config,
        )
    )

    return StressTestResponse(
        id=str(run.id),
        status=run.status,
        created_at=run.created_at.isoformat(),
    )


@router.get("/stress-test/{run_id}", response_model=StressTestRunResponse)
async def get_stress_test(run_id: str, db: AsyncSession = Depends(get_db)):
    """Get stress test result."""
    result = await db.execute(
        select(StressTestRun).where(StressTestRun.id == run_id)
    )
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="Stress test not found")

    return _run_to_response(run)


@router.get("/stress-tests", response_model=list[StressTestRunSummary])
async def list_stress_tests(
    limit: int = 20, db: AsyncSession = Depends(get_db)
):
    """List recent stress tests."""
    result = await db.execute(
        select(StressTestRun)
        .order_by(StressTestRun.created_at.desc())
        .limit(limit)
    )
    runs = result.scalars().all()
    return [_run_to_summary(r) for r in runs]


@router.delete("/stress-test/{run_id}", status_code=204)
async def delete_stress_test(run_id: str, db: AsyncSession = Depends(get_db)):
    """Delete stress test run."""
    result = await db.execute(
        select(StressTestRun).where(StressTestRun.id == run_id)
    )
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="Stress test not found")
    await db.delete(run)
    await db.commit()


@router.get("/scenarios", response_model=list[ScenarioPreset])
async def list_scenarios():
    """List preset scenarios."""
    return PRESET_SCENARIOS


@router.post("/scenario/generate", response_model=CustomScenarioResponse)
async def generate_scenario(req: CustomScenarioRequest):
    """Generate custom scenario via Claude."""
    return await generate_custom_scenario(req.prompt)


# ── Helpers ───────────────────────────────────────────────


def _run_to_response(run: StressTestRun) -> StressTestRunResponse:
    return StressTestRunResponse(
        id=str(run.id),
        name=run.name,
        strategy_json=run.strategy_json,
        scenario_type=run.scenario_type,
        scenario_config=run.scenario_config,
        agent_config=run.agent_config,
        exchange_config=run.exchange_config,
        status=run.status,
        progress=run.progress,
        total_steps=run.total_steps,
        results=run.results,
        metrics=run.metrics,
        error_message=run.error_message,
        created_at=run.created_at.isoformat(),
        completed_at=run.completed_at.isoformat() if run.completed_at else None,
    )


def _run_to_summary(run: StressTestRun) -> StressTestRunSummary:
    metrics = run.metrics or {}
    return StressTestRunSummary(
        id=str(run.id),
        name=run.name,
        scenario_type=run.scenario_type,
        status=run.status,
        progress=run.progress,
        strategy_pnl=metrics.get("strategy_pnl"),
        crash_depth=metrics.get("crash_depth"),
        created_at=run.created_at.isoformat(),
    )
