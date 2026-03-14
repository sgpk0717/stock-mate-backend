"""비동기 알파 마이닝 실행기.

backtest/runner.py 패턴 동일: DB 상태 관리 + WebSocket 진행률.
"""

from __future__ import annotations

import logging
import uuid
from datetime import date, datetime

import sympy
from sqlalchemy import update

from app.alpha.miner import EvolutionaryAlphaMiner, DiscoveredFactor
from app.alpha.models import AlphaFactor, AlphaMiningRun
from app.alpha.universe import Universe, resolve_universe
from app.backtest.data_loader import load_enriched_candles
from app.core.config import settings
from app.core.database import async_session
from app.services.ws_manager import manager

logger = logging.getLogger(__name__)


async def execute_alpha_mining(
    run_id: uuid.UUID,
    name: str,
    context: str,
    universe: str,
    start_date: date,
    end_date: date,
    max_iterations: int,
    ic_threshold: float,
    orthogonality_threshold: float = 0.7,
    use_pysr: bool = False,
    interval: str = "1d",
    seed_factor_ids: list[str] | None = None,
) -> None:
    """백그라운드에서 알파 마이닝을 실행하고 DB에 결과를 저장한다."""
    channel = f"alpha:{run_id}"

    async def progress_cb(current: int, total: int, msg: str) -> None:
        async with async_session() as db:
            await db.execute(
                update(AlphaMiningRun)
                .where(AlphaMiningRun.id == run_id)
                .values(progress=current)
            )
            await db.commit()

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
                update(AlphaMiningRun)
                .where(AlphaMiningRun.id == run_id)
                .values(status="RUNNING", progress=0)
            )
            await db.commit()

        # 유니버스 리졸브 → 데이터 로드
        symbols = await resolve_universe(Universe(universe))
        data = await load_enriched_candles(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
        )

        if data.height == 0:
            raise ValueError("No candle data found for the given symbols and date range")

        # 마이너 실행
        miner = EvolutionaryAlphaMiner(
            data=data,
            context=context,
            max_iterations=max_iterations,
            ic_threshold=ic_threshold,
            orthogonality_threshold=orthogonality_threshold,
            use_pysr=use_pysr,
            interval=interval,
        )

        # 시드 팩터 주입
        if seed_factor_ids:
            from sqlalchemy import select as sa_select
            from app.alpha.evaluator import FactorMetrics

            async with async_session() as seed_db:
                result = await seed_db.execute(
                    sa_select(AlphaFactor).where(
                        AlphaFactor.id.in_(
                            [uuid.UUID(sid) for sid in seed_factor_ids]
                        )
                    )
                )
                seed_factors = result.scalars().all()

            seeds = []
            for sf in seed_factors:
                seeds.append(DiscoveredFactor(
                    name=sf.name,
                    expression_str=sf.expression_str,
                    expression_sympy=sf.expression_sympy or "",
                    polars_code=sf.polars_code or "",
                    hypothesis=sf.hypothesis or "",
                    generation=-1,
                    metrics=FactorMetrics(
                        ic_mean=sf.ic_mean or 0,
                        ic_std=sf.ic_std or 0,
                        icir=sf.icir or 0,
                        turnover=sf.turnover or 0,
                        sharpe=sf.sharpe or 0,
                        max_drawdown=sf.max_drawdown or 0,
                        ic_series=[],
                    ),
                    parent_ids=[str(sf.id)],
                ))

            if seeds:
                miner.inject_seeds(seeds)
                logger.info(
                    "Injected %d seed factors for run %s", len(seeds), run_id
                )

        # iteration_cb: 실시간 이벤트 브로드캐스트
        async def iteration_cb(event: dict) -> None:
            await manager.broadcast(channel, event)

        discovered = await miner.run(
            progress_cb=progress_cb,
            iteration_cb=iteration_cb,
        )

        # 팩터 DB 저장 + iteration_logs 저장 (시드 팩터 제외)
        new_factors = [f for f in discovered if f.generation >= 0]
        async with async_session() as db:
            for factor in new_factors:
                alpha_factor = AlphaFactor(
                    mining_run_id=run_id,
                    name=factor.name,
                    expression_str=factor.expression_str,
                    expression_sympy=factor.expression_sympy,
                    polars_code=factor.polars_code,
                    hypothesis=factor.hypothesis,
                    generation=factor.generation,
                    ic_mean=factor.metrics.ic_mean,
                    ic_std=factor.metrics.ic_std,
                    icir=factor.metrics.icir,
                    turnover=factor.metrics.turnover,
                    sharpe=factor.metrics.sharpe,
                    max_drawdown=factor.metrics.max_drawdown,
                    status="discovered",
                    parent_ids=factor.parent_ids,
                    interval=interval,
                )
                db.add(alpha_factor)

            # iteration_logs + summary를 DB에 저장
            logs_data = {
                "iterations": miner.iteration_logs,
                "summary": miner.build_summary(),
            }

            await db.execute(
                update(AlphaMiningRun)
                .where(AlphaMiningRun.id == run_id)
                .values(
                    status="COMPLETED",
                    progress=100,
                    factors_found=len(new_factors),
                    total_evaluated=max_iterations,
                    iteration_logs=logs_data,
                    completed_at=datetime.utcnow(),
                )
            )
            await db.commit()

        # Phase 2: 자동 인과 검증 (항상 실행)
        if len(new_factors) > 0:
            await manager.broadcast(channel, {
                "type": "progress",
                "current": 90,
                "total": 100,
                "percent": 90,
                "message": f"인과 검증 시작 ({len(new_factors)}개 팩터)...",
            })
            try:
                from app.alpha.causal_runner import validate_factors_batch
                async with async_session() as causal_db:
                    validated_count = await validate_factors_batch(run_id, causal_db)
                logger.info(
                    "Auto-validated %d/%d factors for run %s",
                    validated_count, len(new_factors), run_id,
                )
            except Exception as e:
                logger.error(
                    "Auto causal validation failed for run %s: %s",
                    run_id, e,
                )
                validated_count = 0

        await manager.broadcast(channel, {
            "type": "completed",
            "factors_found": len(new_factors),
        })

        logger.info(
            "Alpha mining %s completed: %d factors found (+ %d seeds)",
            run_id, len(new_factors), len(discovered) - len(new_factors),
        )

    except Exception as e:
        logger.exception("Alpha mining %s failed", run_id)
        async with async_session() as db:
            await db.execute(
                update(AlphaMiningRun)
                .where(AlphaMiningRun.id == run_id)
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
