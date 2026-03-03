"""비동기 백테스트 실행 오케스트레이터."""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import asdict
from datetime import date, datetime

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.backtest.cost_model import CostConfig
from app.backtest.engine import run_backtest
from app.backtest.models import BacktestRun
from app.core.database import async_session
from app.services.ws_manager import manager

logger = logging.getLogger(__name__)


def _sanitize_for_json(obj):
    """NaN/Infinity를 None으로 변환하여 JSON 호환성 확보."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


async def execute_backtest(
    run_id: uuid.UUID,
    strategy_json: dict,
    start_date: date,
    end_date: date,
    initial_capital: float,
    symbols: list[str] | None,
    position_size_pct: float,
    max_positions: int,
    cost_config: CostConfig,
) -> None:
    """백그라운드에서 백테스트를 실행하고 DB에 결과를 저장한다."""
    channel = f"backtest:{run_id}"

    async def progress_cb(current: int, total: int, msg: str) -> None:
        # DB progress 업데이트
        async with async_session() as db:
            await db.execute(
                update(BacktestRun)
                .where(BacktestRun.id == run_id)
                .values(progress=current)
            )
            await db.commit()

        # WebSocket broadcast
        await manager.broadcast(channel, {
            "type": "progress",
            "current": current,
            "total": total,
            "percent": current,
            "message": msg,
        })

    try:
        # 상태 → RUNNING
        async with async_session() as db:
            await db.execute(
                update(BacktestRun)
                .where(BacktestRun.id == run_id)
                .values(status="RUNNING", progress=0)
            )
            await db.commit()

        # 엔진 실행
        result = await run_backtest(
            strategy=strategy_json,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            max_positions=max_positions,
            position_size_pct=position_size_pct,
            cost_config=cost_config,
            progress_cb=progress_cb,
        )

        # 엔진이 에러를 반환한 경우 (데이터 없음 등) → FAILED 처리
        if "error" in result.metrics:
            async with async_session() as db:
                await db.execute(
                    update(BacktestRun)
                    .where(BacktestRun.id == run_id)
                    .values(
                        status="FAILED",
                        error_message=str(result.metrics["error"])[:500],
                        completed_at=datetime.utcnow(),
                    )
                )
                await db.commit()

            await manager.broadcast(channel, {
                "type": "failed",
                "error": str(result.metrics["error"])[:200],
            })
            logger.warning("Backtest %s failed: %s", run_id, result.metrics["error"])
            return

        # 결과 저장
        trades_list = [asdict(t) for t in result.trades]

        async with async_session() as db:
            await db.execute(
                update(BacktestRun)
                .where(BacktestRun.id == run_id)
                .values(
                    status="COMPLETED",
                    progress=100,
                    metrics=_sanitize_for_json(result.metrics),
                    equity_curve=_sanitize_for_json(result.equity_curve),
                    trades_summary=_sanitize_for_json(trades_list),
                    symbol_count=len(set(t.symbol for t in result.trades)),
                    completed_at=datetime.utcnow(),
                )
            )
            await db.commit()

        # 완료 broadcast
        await manager.broadcast(channel, {
            "type": "completed",
            "metrics": result.metrics,
        })

        logger.info("Backtest %s completed: %s", run_id, result.metrics.get("total_return"))

    except Exception as e:
        logger.exception("Backtest %s failed", run_id)
        async with async_session() as db:
            await db.execute(
                update(BacktestRun)
                .where(BacktestRun.id == run_id)
                .values(
                    status="FAILED",
                    error_message=str(e)[:500],
                    completed_at=datetime.utcnow(),
                )
            )
            await db.commit()

        await manager.broadcast(channel, {
            "type": "failed",
            "error": str(e)[:200],
        })
