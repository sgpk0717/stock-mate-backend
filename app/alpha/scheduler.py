"""자율 알파 팩토리 스케줄러.

주기적으로 알파 마이닝 사이클을 실행하는 백그라운드 태스크.
싱글턴 패턴으로 이중 시작을 방지한다.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime

from sqlalchemy import func, select, update

from app.alpha.evolution_engine import EvolutionEngine
from app.alpha.memory import ExperienceVectorMemory
from app.alpha.miner import DiscoveredFactor
from app.alpha.models import AlphaFactor, AlphaMiningRun
from app.alpha.operators import OperatorRegistry
from app.alpha.universe import Universe, resolve_universe
from app.backtest.data_loader import load_candles
from app.core.config import settings
from app.core.database import async_session
from app.services.ws_manager import manager

logger = logging.getLogger(__name__)


@dataclass
class _FactoryState:
    """스케줄러 내부 상태."""

    running: bool = False
    cycles_completed: int = 0
    factors_discovered_total: int = 0
    current_cycle_progress: int = 0
    last_cycle_at: str | None = None
    config: dict = field(default_factory=dict)
    population_size: int = 0
    elite_count: int = 0
    generation: int = 0
    operator_stats: dict = field(default_factory=dict)


class AlphaFactoryScheduler:
    """자율 알파 팩토리.

    interval_minutes 간격으로 마이닝 사이클을 반복한다.
    각 사이클: 데이터 로드 → RAG 메모리 초기화 → 마이너 실행 → DB 저장 → 브로드캐스트.
    """

    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._state = _FactoryState()
        self._vector_memory: ExperienceVectorMemory | None = None
        self._evolution_engine: EvolutionEngine | None = None
        self._operator_registry = OperatorRegistry(
            llm_ratio=settings.ALPHA_LLM_MUTATION_RATIO,
        )

    async def start(
        self,
        context: str = "",
        universe: str = "KOSPI200",
        start_date: str = "",
        end_date: str = "",
        interval_minutes: int | None = None,
        max_iterations: int | None = None,
        ic_threshold: float | None = None,
        orthogonality_threshold: float = 0.7,
        enable_crossover: bool | None = None,
        enable_causal: bool = False,
    ) -> bool:
        """스케줄러 시작. 이미 실행 중이면 False 반환."""
        async with self._lock:
            if self._state.running:
                return False

            interval = interval_minutes or settings.ALPHA_FACTORY_INTERVAL_MINUTES
            iterations = max_iterations or settings.ALPHA_FACTORY_MAX_ITERATIONS
            threshold = ic_threshold if ic_threshold is not None else settings.ALPHA_IC_THRESHOLD_PASS
            crossover = enable_crossover if enable_crossover is not None else settings.ALPHA_FACTORY_CROSSOVER_ENABLED

            self._state = _FactoryState(
                running=True,
                config={
                    "context": context,
                    "universe": universe,
                    "start_date": start_date,
                    "end_date": end_date,
                    "interval_minutes": interval,
                    "max_iterations": iterations,
                    "ic_threshold": threshold,
                    "orthogonality_threshold": orthogonality_threshold,
                    "enable_crossover": crossover,
                    "enable_causal": enable_causal,
                },
            )

            # 이전 엔진 리셋 (DB에서 세대 복원 후 새 엔진 생성)
            self._evolution_engine = None

            # 벡터 메모리 초기화
            self._vector_memory = ExperienceVectorMemory()
            try:
                async with async_session() as db:
                    await self._vector_memory.load_cache(db)
                    # DB에서 마지막 세대 번호 복원
                    max_gen = await db.execute(
                        select(func.max(AlphaFactor.birth_generation))
                        .where(AlphaFactor.population_active == True)  # noqa: E712
                    )
                    last_gen = max_gen.scalar() or 0
                    self._state.generation = last_gen
                    logger.info("Restored generation from DB: %d", last_gen)
            except Exception as e:
                logger.warning("Vector memory cache load failed: %s", e)

            self._task = asyncio.create_task(self._loop())
            logger.info(
                "Alpha factory started: interval=%dmin, iterations=%d",
                interval, iterations,
            )
            return True

    async def stop(self) -> bool:
        """스케줄러 중지."""
        async with self._lock:
            if not self._state.running:
                return False

            self._state.running = False
            if self._task and not self._task.done():
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
            self._task = None
            logger.info("Alpha factory stopped")
            return True

    def get_status(self) -> dict:
        """현재 상태 반환."""
        return {
            "running": self._state.running,
            "cycles_completed": self._state.cycles_completed,
            "factors_discovered_total": self._state.factors_discovered_total,
            "current_cycle_progress": self._state.current_cycle_progress,
            "last_cycle_at": self._state.last_cycle_at,
            "config": self._state.config,
            "population_size": self._state.population_size,
            "elite_count": self._state.elite_count,
            "generation": self._state.generation,
            "operator_stats": self._state.operator_stats,
        }

    async def _loop(self) -> None:
        """메인 스케줄러 루프."""
        config = self._state.config
        interval_seconds = config["interval_minutes"] * 60

        while self._state.running:
            try:
                await self._run_cycle()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("Alpha factory cycle failed: %s", e)
                await manager.broadcast("alpha:factory", {
                    "type": "cycle_error",
                    "error": str(e)[:200],
                })

            if not self._state.running:
                break

            try:
                await asyncio.sleep(interval_seconds)
            except asyncio.CancelledError:
                break

    async def _run_cycle(self) -> None:
        """단일 마이닝 사이클 실행."""
        config = self._state.config
        self._state.current_cycle_progress = 0
        cycle_num = self._state.cycles_completed + 1

        await manager.broadcast("alpha:factory", {
            "type": "cycle_start",
            "cycle": cycle_num,
        })

        # 유니버스 리졸브 → 데이터 로드
        universe_code = config.get("universe", "KOSPI200")
        symbols = await resolve_universe(Universe(universe_code))
        start_str = config.get("start_date") or ""
        end_str = config.get("end_date") or ""
        start = date.fromisoformat(start_str) if start_str else None
        end = date.fromisoformat(end_str) if end_str else None

        data = await load_candles(
            symbols=symbols,
            start_date=start,
            end_date=end,
        )

        if data.height == 0:
            logger.warning("Alpha factory cycle %d: no candle data", cycle_num)
            self._state.cycles_completed = cycle_num
            self._state.last_cycle_at = datetime.utcnow().isoformat()
            self._state.current_cycle_progress = 100
            await manager.broadcast("alpha:factory", {
                "type": "cycle_complete",
                "cycle": cycle_num,
                "factors_found": 0,
                "total_factors": self._state.factors_discovered_total,
                "message": "데이터 없음 — 건너뜀",
            })
            return

        # 진화 엔진 기반 실행
        async with async_session() as db:
            # 벡터 메모리 캐시 갱신
            if self._vector_memory:
                await self._vector_memory.load_cache(db)

            async def progress_cb(current: int, total: int, msg: str) -> None:
                self._state.current_cycle_progress = current
                await manager.broadcast("alpha:factory", {
                    "type": "progress",
                    "cycle": cycle_num,
                    "current": current,
                    "total": total,
                    "message": msg,
                })

            async def iteration_cb(event: dict) -> None:
                if event.get("type") == "generation_complete":
                    self._state.population_size = event.get("population_size", 0)
                    self._state.elite_count = event.get("elite_count", 0)
                await manager.broadcast("alpha:factory", {
                    **event,
                    "cycle": cycle_num,
                })

            # LLM 장애 카운터 리셋 (새 사이클)
            self._operator_registry.reset_llm_failures()

            # EvolutionEngine 생성 또는 데이터 갱신
            if self._evolution_engine is None:
                self._evolution_engine = EvolutionEngine(
                    data=data,
                    db=db,
                    operator_registry=self._operator_registry,
                    population_size=settings.ALPHA_POPULATION_SIZE,
                    elite_pct=settings.ALPHA_ELITE_PCT,
                    context=config.get("context", ""),
                    ic_threshold=config["ic_threshold"],
                    orthogonality_threshold=config.get("orthogonality_threshold", 0.7),
                    vector_memory=self._vector_memory,
                    generation=self._state.generation,
                )
            else:
                self._evolution_engine.update_data(data)
                self._evolution_engine._db = db

            discovered = await self._evolution_engine.run_generation(
                progress_cb=progress_cb,
                iteration_cb=iteration_cb,
            )

            # 마이닝 런 DB 저장 (discovered 팩터용 — population은 엔진이 직접 관리)
            run_id = uuid.uuid4()
            mining_run = AlphaMiningRun(
                id=run_id,
                name=f"Factory Cycle {cycle_num} (Gen {self._evolution_engine.generation})",
                context={"text": config.get("context", "")},
                config=config,
                status="COMPLETED",
                progress=100,
                factors_found=len(discovered),
                total_evaluated=settings.ALPHA_POPULATION_SIZE,
                iteration_logs={"operator_stats": self._operator_registry.to_dict()},
                completed_at=datetime.utcnow(),
            )
            db.add(mining_run)
            await db.flush()  # FK 제약: mining_run이 먼저 존재해야 함

            for factor in discovered:
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
                    population_active=False,
                    parent_ids=factor.parent_ids,
                )
                db.add(alpha_factor)

            await db.commit()

            # 상태 업데이트
            self._state.generation = self._evolution_engine.generation
            self._state.operator_stats = self._operator_registry.to_dict()

        # Phase 2: 선택적 인과 검증
        if config.get("enable_causal") and len(discovered) > 0:
            try:
                from app.alpha.causal_runner import validate_factors_batch

                async with async_session() as causal_db:
                    await validate_factors_batch(run_id, causal_db)
            except Exception as e:
                logger.warning("Factory causal validation failed: %s", e)

        # 상태 업데이트 + 브로드캐스트
        self._state.cycles_completed = cycle_num
        self._state.factors_discovered_total += len(discovered)
        self._state.last_cycle_at = datetime.utcnow().isoformat()
        self._state.current_cycle_progress = 100

        await manager.broadcast("alpha:factory", {
            "type": "cycle_complete",
            "cycle": cycle_num,
            "factors_found": len(discovered),
            "total_factors": self._state.factors_discovered_total,
        })

        logger.info(
            "Alpha factory cycle %d completed: %d factors found (total: %d)",
            cycle_num, len(discovered), self._state.factors_discovered_total,
        )


# ── 모듈 레벨 싱글턴 ──

_scheduler: AlphaFactoryScheduler | None = None


def get_scheduler() -> AlphaFactoryScheduler:
    """싱글턴 스케줄러 인스턴스 반환."""
    global _scheduler
    if _scheduler is None:
        _scheduler = AlphaFactoryScheduler()
    return _scheduler
