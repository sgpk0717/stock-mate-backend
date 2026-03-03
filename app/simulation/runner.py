"""스트레스 테스트 비동기 실행기.

backtest/runner.py 패턴 기반:
asyncio.create_task + progress_cb + DB 상태 추적 + WebSocket 브로드캐스트.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import asdict
from datetime import datetime

from sqlalchemy import update

from app.core.database import async_session
from app.services.ws_manager import manager
from app.simulation.agents import StrategyAgent, create_agents
from app.simulation.exchange import VirtualExchange
from app.simulation.models import StressTestRun

logger = logging.getLogger(__name__)


async def execute_stress_test(
    run_id: uuid.UUID,
    strategy_json: dict,
    scenario_config: dict,
    agent_config: dict,
    exchange_config: dict,
) -> None:
    """Background stress test execution."""
    channel = f"simulation:{run_id}"

    async def progress_cb(current: int, total: int, msg: str) -> None:
        pct = int(current / total * 100) if total > 0 else 0
        async with async_session() as db:
            await db.execute(
                update(StressTestRun)
                .where(StressTestRun.id == run_id)
                .values(progress=pct)
            )
            await db.commit()
        await manager.broadcast(
            channel,
            {
                "type": "progress",
                "current": current,
                "total": total,
                "percent": pct,
                "message": msg,
            },
        )

    try:
        # 1. DB 상태 → RUNNING
        async with async_session() as db:
            await db.execute(
                update(StressTestRun)
                .where(StressTestRun.id == run_id)
                .values(status="RUNNING", progress=0)
            )
            await db.commit()

        # 2. 에이전트 생성
        seed = exchange_config.get("seed")
        agents = create_agents(agent_config, seed=seed)

        # 3. StrategyAgent 추가
        strategy_agent = StrategyAgent(
            agent_id="strategy_0",
            strategy=strategy_json,
        )
        agents.append(strategy_agent)

        # 4. VirtualExchange 생성
        exchange = VirtualExchange(
            agents=agents,
            initial_price=exchange_config.get("initial_price", 50000.0),
            tick_size=exchange_config.get("tick_size", 10.0),
            seed=seed,
        )

        # 5. 시나리오 스케줄링
        scenario_type = scenario_config.get("type", "flash_crash")
        inject_at = scenario_config.get("inject_at_step", 500)
        params = scenario_config.get("params", {})
        exchange.schedule_event(inject_at, scenario_type, params)

        # 6. 시뮬레이션 실행
        total_steps = exchange_config.get("total_steps", 1000)
        result = await exchange.run_steps(total_steps, progress_cb)

        # 7. Metrics 계산
        metrics = exchange.compute_metrics()

        # 8. DB 결과 저장
        async with async_session() as db:
            await db.execute(
                update(StressTestRun)
                .where(StressTestRun.id == run_id)
                .values(
                    status="COMPLETED",
                    progress=100,
                    results={
                        "price_series": result.price_series,
                        "volume_series": result.volume_series,
                        "spread_series": result.spread_series,
                        "depth_series": result.depth_series,
                        "agent_metrics": result.agent_metrics,
                        "strategy_pnl": result.strategy_pnl,
                        "events_injected": result.events_injected,
                    },
                    metrics=asdict(metrics),
                    completed_at=datetime.utcnow(),
                )
            )
            await db.commit()

        # 9. WebSocket 완료 브로드캐스트
        await manager.broadcast(
            channel,
            {
                "type": "completed",
                "metrics": asdict(metrics),
            },
        )

        logger.info(
            "Stress test %s completed: strategy_pnl=%.2f, mdd=%.2f%%",
            run_id,
            metrics.strategy_pnl,
            metrics.max_drawdown,
        )

    except Exception as e:
        logger.exception("Stress test %s failed: %s", run_id, e)

        async with async_session() as db:
            await db.execute(
                update(StressTestRun)
                .where(StressTestRun.id == run_id)
                .values(
                    status="FAILED",
                    error_message=str(e)[:500],
                    completed_at=datetime.utcnow(),
                )
            )
            await db.commit()

        await manager.broadcast(
            channel,
            {
                "type": "failed",
                "error": str(e)[:200],
            },
        )
