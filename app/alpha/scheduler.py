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

import polars as pl
from sqlalchemy import func, select, update

from app.alpha.evolution_engine import EvolutionEngine
from app.alpha.memory import ExperienceVectorMemory
from app.alpha.miner import DiscoveredFactor
from app.alpha.models import AlphaFactor, AlphaMiningRun
from app.alpha.operators import OperatorRegistry
from app.alpha.universe import Universe, resolve_universe
from app.backtest.data_loader import load_candles, load_enriched_candles
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
    current_cycle_message: str = ""
    last_cycle_at: str | None = None
    started_at: str | None = None
    config: dict = field(default_factory=dict)
    population_size: int = 0
    elite_count: int = 0
    generation: int = 0
    operator_stats: dict = field(default_factory=dict)
    last_funnel: dict = field(default_factory=dict)


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
        # 사이클 간 데이터 캐시 (780만 행 재로드 방지)
        self._cached_data: pl.DataFrame | None = None
        self._data_cache_key: str = ""

    async def start(
        self,
        context: str = "",
        universe: str = "KOSPI200",
        start_date: str = "",
        end_date: str = "",
        data_interval: str = "1d",
        interval_minutes: int | None = None,
        max_iterations: int | None = None,
        ic_threshold: float | None = None,
        orthogonality_threshold: float = 0.7,
        enable_crossover: bool | None = None,
        max_cycles: int | None = None,
    ) -> bool:
        """스케줄러 시작. 이미 실행 중이면 False 반환."""
        async with self._lock:
            if self._state.running:
                return False

            interval = interval_minutes if interval_minutes is not None else settings.ALPHA_FACTORY_INTERVAL_MINUTES
            iterations = max_iterations or settings.ALPHA_FACTORY_MAX_ITERATIONS
            threshold = ic_threshold if ic_threshold is not None else settings.ALPHA_IC_THRESHOLD_PASS
            crossover = enable_crossover if enable_crossover is not None else settings.ALPHA_FACTORY_CROSSOVER_ENABLED

            # 이전 상태에서 누적값 보존 (컨테이너 재시작/watchdog 복구 시)
            prev_cycles = self._state.cycles_completed
            prev_total = self._state.factors_discovered_total

            self._state = _FactoryState(
                running=True,
                cycles_completed=prev_cycles,
                factors_discovered_total=prev_total,
                started_at=datetime.utcnow().isoformat(),
                config={
                    "context": context,
                    "universe": universe,
                    "start_date": start_date,
                    "end_date": end_date,
                    "data_interval": data_interval,
                    "interval_minutes": interval,
                    "max_iterations": iterations,
                    "ic_threshold": threshold,
                    "orthogonality_threshold": orthogonality_threshold,
                    "enable_crossover": crossover,
                    "max_cycles": max_cycles,
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
        # task가 죽었는데 running 플래그만 True인 좀비 상태 감지
        if self._state.running and self._task and self._task.done():
            self._state.running = False
            logger.warning("Alpha factory task dead but running=True detected, resetting")
        return {
            "running": self._state.running,
            "cycles_completed": self._state.cycles_completed,
            "factors_discovered_total": self._state.factors_discovered_total,
            "current_cycle_progress": self._state.current_cycle_progress,
            "current_cycle_message": self._state.current_cycle_message,
            "last_cycle_at": self._state.last_cycle_at,
            "started_at": self._state.started_at,
            "config": self._state.config,
            "population_size": self._state.population_size,
            "elite_count": self._state.elite_count,
            "generation": self._state.generation,
            "operator_stats": self._state.operator_stats,
            "last_funnel": self._state.last_funnel,
        }

    async def _loop(self) -> None:
        """메인 스케줄러 루프."""
        config = self._state.config
        interval_seconds = config["interval_minutes"] * 60
        max_cycles = config.get("max_cycles")

        try:
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

                # max_cycles 도달 시 자동 중지
                if max_cycles and self._state.cycles_completed >= max_cycles:
                    logger.info("Alpha factory reached max_cycles=%d, stopping", max_cycles)
                    self._state.running = False
                    break

                if interval_seconds > 0:
                    try:
                        await asyncio.sleep(interval_seconds)
                    except asyncio.CancelledError:
                        break
        finally:
            # 좀비 방지: 루프 종료 시 반드시 running=False
            self._state.running = False
            # 루프 종료 → 프론트에 알림
            try:
                await manager.broadcast("alpha:factory", {"type": "factory_stopped"})
            except Exception:
                pass

    async def _run_cycle(self) -> None:
        """단일 마이닝 사이클 실행."""
        config = self._state.config
        self._state.current_cycle_progress = 0
        cycle_num = self._state.cycles_completed + 1
        discovered: list[DiscoveredFactor] = []
        run_id: uuid.UUID | None = None
        _last_funnel: dict = {}  # 퍼널 데이터 (텔레그램 보고용)
        _cycle_start = datetime.utcnow()  # 소요시간 측정용

        await manager.broadcast("alpha:factory", {
            "type": "cycle_start",
            "cycle": cycle_num,
        })

        try:
            logger.info("Cycle %d: starting (config=%s)", cycle_num, {k: v for k, v in config.items() if k != "context"})

            # 유니버스 리졸브 → 데이터 로드
            universe_code = config.get("universe", "KOSPI200")
            symbols = await resolve_universe(Universe(universe_code))
            logger.info("Cycle %d: resolved %d symbols for %s", cycle_num, len(symbols), universe_code)

            start_str = config.get("start_date") or ""
            end_str = config.get("end_date") or ""
            start = date.fromisoformat(start_str) if start_str else None
            end = date.fromisoformat(end_str) if end_str else None

            data_interval = config.get("data_interval", "1d")
            cache_key = f"{universe_code}_{start_str}_{end_str}_{data_interval}"
            if self._cached_data is not None and self._data_cache_key == cache_key:
                data = self._cached_data
                logger.info("Cycle %d: using cached candles %d rows x %d cols", cycle_num, data.height, data.width)
            else:
                data = await load_enriched_candles(
                    symbols=symbols,
                    start_date=start,
                    end_date=end,
                    interval=data_interval,
                )
                self._cached_data = data
                self._data_cache_key = cache_key
                logger.info("Cycle %d: loaded enriched candles %d rows x %d cols", cycle_num, data.height, data.width)

            if data.height == 0:
                logger.warning("Alpha factory cycle %d: no candle data", cycle_num)
                return

            # 진화 엔진 기반 실행
            async with async_session() as db:
                if self._vector_memory:
                    await self._vector_memory.load_cache(db)

                async def progress_cb(current: int, total: int, msg: str) -> None:
                    self._state.current_cycle_progress = current
                    self._state.current_cycle_message = msg
                    await manager.broadcast("alpha:factory", {
                        "type": "progress",
                        "cycle": cycle_num,
                        "current": current,
                        "total": total,
                        "message": msg,
                    })

                async def iteration_cb(event: dict) -> None:
                    nonlocal _last_funnel  # 메서드 레벨 변수 참조
                    etype = event.get("type")
                    if etype == "generation_start":
                        self._state.population_size = event.get("population_size", 0)
                        self._state.generation = event.get("generation", self._state.generation)
                    elif etype == "generation_complete":
                        self._state.population_size = event.get("population_size", 0)
                        self._state.elite_count = event.get("elite_count", 0)
                        _last_funnel = event.get("funnel", {})
                        self._state.last_funnel = _last_funnel
                    await manager.broadcast("alpha:factory", {
                        **event,
                        "cycle": cycle_num,
                    })

                self._operator_registry.reset_llm_failures()

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
                        interval=data_interval,
                    )
                else:
                    self._evolution_engine.update_data(data)
                    self._evolution_engine._db = db

                logger.info("Cycle %d: starting run_generation (gen=%d)", cycle_num, self._evolution_engine.generation)
                discovered = await self._evolution_engine.run_generation(
                    progress_cb=progress_cb,
                    iteration_cb=iteration_cb,
                )
                logger.info("Cycle %d: run_generation done, discovered=%d", cycle_num, len(discovered))

                self._state.generation = self._evolution_engine.generation
                self._state.operator_stats = self._operator_registry.to_dict()

            # DB 저장 (실패해도 카운터에 영향 없음)
            try:
                async with async_session() as save_db:
                    run_id = uuid.uuid4()
                    mining_run = AlphaMiningRun(
                        id=run_id,
                        name=f"Factory Cycle {cycle_num} (Gen {self._state.generation})",
                        context={"text": config.get("context", "")},
                        config=config,
                        status="COMPLETED",
                        progress=100,
                        factors_found=len(discovered),
                        total_evaluated=settings.ALPHA_POPULATION_SIZE,
                        iteration_logs={"operator_stats": self._operator_registry.to_dict()},
                        completed_at=datetime.utcnow(),
                    )
                    save_db.add(mining_run)
                    await save_db.flush()

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
                            interval=data_interval,
                        )
                        save_db.add(alpha_factor)

                    await save_db.commit()
            except Exception as e:
                logger.warning("Factory DB save failed for cycle %d: %s", cycle_num, e)
                run_id = None

            # 인과 검증: 항상 실행 (팩터 발견 시)
            if len(discovered) > 0 and run_id:
                await self._run_causal_validation(run_id, cycle_num)

        finally:
            # 성공이든 실패든 카운터 반영 + 브로드캐스트
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

            # 텔레그램 진행 보고 (처음 3사이클, 5의 배수, 팩터 발견 시)
            try:
                from app.telegram.bot import send_message as tg_send

                should_report = (
                    len(discovered) > 0
                    or cycle_num <= 3
                    or cycle_num % 5 == 0
                )
                if should_report:
                    f = _last_funnel
                    total = self._state.factors_discovered_total
                    gen = self._state.generation

                    # 소요시간 계산
                    elapsed = datetime.utcnow() - _cycle_start
                    elapsed_min = int(elapsed.total_seconds() // 60)
                    elapsed_sec = int(elapsed.total_seconds() % 60)
                    elapsed_str = f"{elapsed_min}분 {elapsed_sec}초" if elapsed_min > 0 else f"{elapsed_sec}초"

                    # 상태 이모지
                    if len(discovered) >= 5:
                        status_emoji = "\U0001f525"  # fire
                    elif len(discovered) > 0:
                        status_emoji = "\u2705"  # check
                    else:
                        status_emoji = "\U0001f50d"  # magnifier

                    # --- 헤더 ---
                    msg = (
                        f"{status_emoji} <b>마이닝 사이클 {cycle_num} 완료</b> ({elapsed_str} 소요)\n\n"
                    )

                    # --- 결과 ---
                    msg += (
                        f"\U0001f4ca <b>결과</b>\n"
                        f"  이번 사이클: <b>{len(discovered)}개</b> 팩터 발견\n"
                        f"  누적 발견: <b>{total}개</b>\n"
                    )

                    # --- 발견 팩터 상세 (있으면) ---
                    if discovered:
                        msg += f"\n\U0001f3c6 <b>발견 팩터 (상위 3개)</b>\n"
                        top = sorted(discovered, key=lambda d: d.metrics.ic_mean if d.metrics else 0, reverse=True)[:3]
                        for i, d in enumerate(top, 1):
                            ic = d.metrics.ic_mean if d.metrics else 0
                            sh = d.metrics.sharpe if d.metrics else 0
                            msg += (
                                f"  {i}. <code>{d.expression_str[:40]}</code>\n"
                                f"     IC {ic:.3f} (예측력) / Sharpe {sh:.1f} (수익 안정성)\n"
                            )

                    # --- 진화 퍼널 상세 ---
                    if f:
                        attempted = f.get('attempted', 0)
                        eval_ok = f.get('eval_ok', 0)
                        ic_pass = f.get('ic_pass', 0)
                        wf_overfit = f.get('wf_overfit', 0)
                        sharpe_fail = f.get('sharpe_fail', 0)
                        cpcv_cands = f.get('cpcv_candidates', 0)

                        eval_pct = int(eval_ok / attempted * 100) if attempted else 0
                        ic_pct = int(ic_pass / max(eval_ok, 1) * 100)

                        msg += (
                            f"\n\U0001f52c <b>진화 과정</b>\n"
                            f"  후보 수식 생성: {attempted}개\n"
                            f"  \u251c 계산 가능: {eval_ok}개 ({eval_pct}%)\n"
                            f"  \u251c 수익 예측력(IC) 통과: {ic_pass}개 ({ic_pct}%)\n"
                        )
                        if wf_overfit > 0:
                            msg += f"  \u251c 과적합 탈락: {wf_overfit}개\n"
                        if sharpe_fail > 0:
                            msg += f"  \u251c Sharpe 미달: {sharpe_fail}개\n"
                        msg += f"  \u2514 최종 검증 통과: {cpcv_cands}개\n"
                    else:
                        # 퍼널 데이터가 없어도 기본 정보 표시
                        pop_size = self._state.population_size or settings.ALPHA_POPULATION_SIZE
                        msg += (
                            f"\n\U0001f52c <b>진화 과정</b>\n"
                            f"  모집단 크기: {pop_size}개 수식\n"
                            f"  (상세 퍼널 데이터 미수신)\n"
                        )

                    # --- 해석 ---
                    if len(discovered) == 0:
                        ic_thr = config.get("ic_threshold", 0.03)
                        msg += (
                            f"\n\U0001f4a1 <b>해석</b>\n"
                            f"  수학 공식들로 주가 예측을 시도했지만,\n"
                            f"  기준(IC>{ic_thr})을 넘는 공식이 없었습니다.\n"
                            f"  세대가 쌓일수록 더 좋은 공식이 진화합니다.\n"
                        )
                    else:
                        msg += (
                            f"\n\U0001f4a1 <b>해석</b>\n"
                            f"  IC = 주가 예측력 (0.03 이상이면 유의미)\n"
                            f"  Sharpe = 수익 대비 위험 (1.0 이상이면 우수)\n"
                        )

                    # --- 설정 정보 ---
                    universe_code = config.get("universe", "KOSPI200")
                    data_interval = config.get("data_interval", "1d")
                    msg += f"\n\u2699\ufe0f {universe_code} / {data_interval} / 세대 {gen}"

                    await tg_send(msg)
            except Exception as e:
                logger.debug("Telegram report failed: %s", e)

            # WorkflowEvent에 사이클 결과 기록 (OpenClaw 조회용)
            try:
                async with async_session() as evt_db:
                    from app.workflow.models import WorkflowEvent, WorkflowRun
                    today_run_stmt = select(WorkflowRun).where(
                        WorkflowRun.date == date.today()
                    )
                    today_run = await evt_db.execute(today_run_stmt)
                    wf_run = today_run.scalar_one_or_none()
                    if wf_run:
                        evt = WorkflowEvent(
                            workflow_run_id=wf_run.id,
                            event_type="mining_cycle_complete",
                            message=(
                                f"사이클 {cycle_num}: {len(discovered)}개 발견 "
                                f"(총 {self._state.factors_discovered_total}개)"
                            ),
                        )
                        evt_db.add(evt)
                        await evt_db.commit()
            except Exception:
                pass  # 이벤트 기록 실패가 마이닝을 방해하면 안 됨

    async def _run_causal_validation(self, run_id: uuid.UUID, cycle_num: int) -> None:
        """인과 검증을 동기적으로 실행. 검증 완료 후 다음 사이클로 진행."""
        try:
            from app.alpha.causal_runner import validate_factors_batch

            logger.info("Cycle %d: starting causal validation (run=%s)", cycle_num, run_id)
            async with async_session() as causal_db:
                count = await validate_factors_batch(run_id, causal_db)
            logger.info("Cycle %d: causal validation complete (%d factors)", cycle_num, count)

            await manager.broadcast("alpha:factory", {
                "type": "causal_complete",
                "cycle": cycle_num,
                "validated_count": count,
            })
        except Exception as e:
            logger.error("Cycle %d: causal validation failed: %s", cycle_num, e)


# ── 모듈 레벨 싱글턴 ──

_scheduler: AlphaFactoryScheduler | None = None


def get_scheduler() -> AlphaFactoryScheduler:
    """싱글턴 스케줄러 인스턴스 반환."""
    global _scheduler
    if _scheduler is None:
        _scheduler = AlphaFactoryScheduler()
    return _scheduler
