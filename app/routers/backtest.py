"""백테스트 REST API 라우터."""

from __future__ import annotations

import asyncio
import uuid
from datetime import date, datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import delete, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.backtest.ai_strategy import generate_strategy_from_prompt
from app.backtest.cost_model import CostConfig
from app.backtest.models import BacktestRun
from app.backtest.presets import PRESETS, PRESET_MAP
from app.backtest.runner import execute_backtest
from app.backtest.schemas import (
    AIStrategyRequest,
    AIStrategyResponse,
    BacktestRunCreate,
    BacktestRunPageResponse,
    BacktestRunResponse,
    BacktestRunSummary,
    StrategyInfo,
    StrategySchema,
)
from app.core.database import get_db

router = APIRouter(prefix="/backtest", tags=["backtest"])


# ── 전략 관련 ──

@router.get("/strategies", response_model=list[StrategyInfo])
async def list_strategies():
    """내장 프리셋 전략 목록."""
    return PRESETS


@router.post("/ai-strategy", response_model=AIStrategyResponse)
async def create_ai_strategy(req: AIStrategyRequest):
    """자연어 → Claude API → 구조화된 전략 JSON."""
    try:
        result = await generate_strategy_from_prompt(req.prompt)
        strategy = StrategySchema(**result["strategy"])
        return AIStrategyResponse(
            strategy=strategy,
            explanation=result.get("explanation", ""),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI 전략 생성 실패: {e}")


# ── 백테스트 실행 ──

@router.post("/run", status_code=202, response_model=BacktestRunResponse)
async def start_backtest(data: BacktestRunCreate, db: AsyncSession = Depends(get_db)):
    """백테스트 비동기 실행. 즉시 run_id 반환."""
    start = date.fromisoformat(data.start_date)
    end = date.fromisoformat(data.end_date)

    if start >= end:
        raise HTTPException(400, "시작일이 종료일보다 이전이어야 합니다.")

    cost_cfg = CostConfig()
    if data.cost_config:
        cost_cfg = CostConfig(**data.cost_config.model_dump())

    run = BacktestRun(
        strategy_name=data.strategy.name or "Custom",
        strategy_json=data.strategy.model_dump(),
        start_date=start,
        end_date=end,
        initial_capital=float(data.initial_capital),
        cost_config=cost_cfg.model_dump(),
        status="PENDING",
        progress=0,
    )
    db.add(run)
    await db.flush()
    run_id = run.id

    # 백그라운드 실행
    asyncio.create_task(
        execute_backtest(
            run_id=run_id,
            strategy_json=data.strategy.model_dump(),
            start_date=start,
            end_date=end,
            initial_capital=float(data.initial_capital),
            symbols=data.symbols,
            position_size_pct=data.position_size_pct,
            max_positions=data.max_positions,
            cost_config=cost_cfg,
        )
    )

    return _run_to_response(run)


@router.get("/run/{run_id}", response_model=BacktestRunResponse)
async def get_backtest_run(run_id: str, db: AsyncSession = Depends(get_db)):
    """백테스트 실행 결과 조회."""
    result = await db.execute(
        select(BacktestRun).where(BacktestRun.id == uuid.UUID(run_id))
    )
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(404, "백테스트를 찾을 수 없습니다.")
    return _run_to_response(run)


_ALLOWED_SORT_COLUMNS = {
    "created_at", "strategy_name",
}
_METRIC_SORT_COLUMNS = {
    "total_return", "sharpe_ratio", "mdd", "win_rate", "total_trades",
}


@router.get("/runs", response_model=BacktestRunPageResponse)
async def list_backtest_runs(
    offset: int = 0,
    limit: int = 20,
    sort_by: str = "created_at",
    order: str = "desc",
    status: str | None = None,
    search: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    """백테스트 실행 목록 (페이징, 정렬, 필터, 검색)."""
    from sqlalchemy.orm import defer

    # WHERE 조건
    filters = []
    if status:
        filters.append(BacktestRun.status == status)
    if search:
        filters.append(BacktestRun.strategy_name.ilike(f"%{search}%"))

    # 전체 개수
    count_q = select(func.count()).select_from(BacktestRun)
    for f in filters:
        count_q = count_q.where(f)
    total = await db.scalar(count_q) or 0

    # ORDER BY
    if sort_by in _ALLOWED_SORT_COLUMNS:
        col = getattr(BacktestRun, sort_by)
        order_clause = col.asc().nulls_last() if order == "asc" else col.desc().nulls_last()
    elif sort_by in _METRIC_SORT_COLUMNS:
        # JSON 필드에서 추출 (PostgreSQL ->> 연산자)
        json_expr = BacktestRun.metrics[sort_by].as_float()
        order_clause = json_expr.asc().nulls_last() if order == "asc" else json_expr.desc().nulls_last()
    else:
        order_clause = BacktestRun.created_at.desc().nulls_last()

    # 데이터 쿼리
    data_q = (
        select(BacktestRun)
        .options(
            defer(BacktestRun.equity_curve),
            defer(BacktestRun.trades_summary),
            defer(BacktestRun.strategy_json),
        )
    )
    for f in filters:
        data_q = data_q.where(f)
    data_q = data_q.order_by(order_clause).offset(offset).limit(limit)

    result = await db.execute(data_q)
    runs = result.scalars().all()

    return BacktestRunPageResponse(
        items=[_run_to_summary(r) for r in runs],
        total=total,
    )


@router.delete("/run/{run_id}", status_code=204)
async def delete_backtest_run(run_id: str, db: AsyncSession = Depends(get_db)):
    """백테스트 실행 기록 삭제."""
    await db.execute(
        delete(BacktestRun).where(BacktestRun.id == uuid.UUID(run_id))
    )


# ── 헬퍼 ──

def _run_to_response(run: BacktestRun) -> BacktestRunResponse:
    return BacktestRunResponse(
        id=str(run.id),
        strategy_name=run.strategy_name,
        strategy_json=run.strategy_json or {},
        start_date=str(run.start_date),
        end_date=str(run.end_date),
        initial_capital=float(run.initial_capital),
        symbol_count=run.symbol_count,
        status=run.status,
        progress=run.progress,
        metrics=run.metrics,
        equity_curve=run.equity_curve,
        trades_summary=run.trades_summary,
        error_message=run.error_message,
        created_at=run.created_at.isoformat() if run.created_at else "",
        completed_at=run.completed_at.isoformat() if run.completed_at else None,
    )


def _run_to_summary(run: BacktestRun) -> BacktestRunSummary:
    metrics = run.metrics or {}
    return BacktestRunSummary(
        id=str(run.id),
        strategy_name=run.strategy_name,
        start_date=str(run.start_date),
        end_date=str(run.end_date),
        status=run.status,
        progress=run.progress,
        total_return=metrics.get("total_return"),
        sharpe_ratio=metrics.get("sharpe_ratio"),
        mdd=metrics.get("mdd"),
        win_rate=metrics.get("win_rate"),
        total_trades=metrics.get("total_trades"),
        created_at=run.created_at.isoformat() if run.created_at else "",
    )
