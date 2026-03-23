"""자율 알파 팩토리 스케줄러.

주기적으로 알파 마이닝 사이클을 실행하는 백그라운드 태스크.
싱글턴 패턴으로 이중 시작을 방지한다.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

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
    generation_ic_history: list = field(default_factory=list)  # 세대별 IC 추이 (최근 20개)
    user_stopped: bool = False  # 사용자가 의도적으로 중지 (watchdog 재시작 방지)


class AlphaFactoryScheduler:
    """자율 알파 팩토리.

    interval_minutes 간격으로 마이닝 사이클을 반복한다.
    각 사이클: 데이터 로드 → RAG 메모리 초기화 → 마이너 실행 → DB 저장 → 브로드캐스트.
    """

    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self._task_ref: asyncio.Task | None = None  # GC 방지 백업 참조
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
            # Redis user_stopped 플래그 체크 — 프론트/API에서 중지 시 모든 경로에서 시작 차단
            try:
                from app.core.redis import get_client as get_redis
                _redis = get_redis()
                _flag = await _redis.get("alpha:factory:user_stopped")
                if _flag and str(_flag) == "true":
                    logger.info("Alpha factory start 차단 — user_stopped 플래그 활성")
                    return False
            except Exception:
                pass

            # task가 살아있으면 실행 중으로 간주 (get_status의 running과 무관)
            if self._task and not self._task.done():
                return False
            if self._state.running:
                return False

            interval = interval_minutes if interval_minutes is not None else settings.ALPHA_FACTORY_INTERVAL_MINUTES
            iterations = max_iterations or settings.ALPHA_FACTORY_MAX_ITERATIONS
            threshold = ic_threshold if ic_threshold is not None else settings.ALPHA_IC_THRESHOLD_PASS
            crossover = enable_crossover if enable_crossover is not None else settings.ALPHA_FACTORY_CROSSOVER_ENABLED

            # 수동 시작 → user_stopped 해제
            self._state.user_stopped = False

            # 이전 상태에서 누적값 보존 (컨테이너 재시작/watchdog 복구 시)
            prev_cycles = self._state.cycles_completed
            prev_total = self._state.factors_discovered_total

            self._state = _FactoryState(
                running=True,
                cycles_completed=prev_cycles,
                factors_discovered_total=prev_total,
                started_at=datetime.now(timezone.utc).isoformat(),
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
            self._task_ref = self._task  # GC 방지 강한 참조 유지
            logger.info(
                "Alpha factory started: interval=%dmin, iterations=%d, task=%s",
                interval, iterations, self._task,
            )
            return True

    async def stop(self) -> bool:
        """스케줄러 중지."""
        async with self._lock:
            if not self._state.running:
                return False

            self._state.running = False
            self._state.user_stopped = True
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
        """현재 상태 반환 (부작용 없음 — 상태 보고만)."""
        # task가 죽었는데 running 플래그만 True면 실제로는 미실행
        is_running = self._state.running
        if is_running and self._task and self._task.done():
            try:
                _exc = self._task.exception() if not self._task.cancelled() else "Cancelled"
            except Exception:
                _exc = "unknown"
            logger.warning(
                "상태 불일치: state.running=True but task.done()=True (exception=%s)", _exc,
            )
            is_running = False
        return {
            "running": is_running,
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
            "user_stopped": self._state.user_stopped,
        }

    async def _loop(self) -> None:
        """메인 스케줄러 루프."""
        _task_id = id(asyncio.current_task())
        logger.info("Alpha factory _loop ENTER (task_id=%s)", _task_id)
        my_state = self._state  # 이 루프가 소유하는 state (start() 교체 시 오염 방지)
        config = my_state.config
        interval_seconds = config["interval_minutes"] * 60
        max_cycles = config.get("max_cycles")

        try:
            while my_state.running:
                try:
                    await self._run_cycle()
                except asyncio.CancelledError:
                    logger.info("Alpha factory loop: CancelledError — 종료")
                    break
                except Exception as e:
                    logger.exception("Alpha factory cycle failed: %s", e)
                    try:
                        await manager.broadcast("alpha:factory", {
                            "type": "cycle_error",
                            "error": str(e)[:200],
                        })
                    except Exception:
                        pass

                if not my_state.running:
                    logger.info("Alpha factory loop: running=False — 종료")
                    break

                # max_cycles 도달 시 자동 중지
                if max_cycles and my_state.cycles_completed >= max_cycles:
                    logger.info("Alpha factory reached max_cycles=%d, stopping", max_cycles)
                    my_state.running = False
                    break

                if interval_seconds > 0:
                    try:
                        await asyncio.sleep(interval_seconds)
                    except asyncio.CancelledError:
                        logger.info("Alpha factory loop: sleep cancelled — 종료")
                        break
        except Exception as e:
            logger.exception("Alpha factory loop UNEXPECTED exit: %s", e)
        finally:
            logger.warning("Alpha factory loop exiting — running=False 설정 (task_id=%s)", _task_id)
            # 좀비 방지: 자신의 state만 False로 설정 (새 start()의 state 오염 방지)
            my_state.running = False
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

        # ★ LLM 장애 리셋을 최상단에서 보장 (데이터 로드 실패 시에도 리셋)
        self._operator_registry.reset_llm_failures()
        discovered: list[DiscoveredFactor] = []
        run_id: uuid.UUID | None = None
        _last_funnel: dict = {}  # 퍼널 데이터 (텔레그램 보고용)
        _last_eval: dict = {}  # eval_complete 이벤트 (IC 샘플)
        _last_candidates: dict = {}  # candidates_ready 이벤트 (연산자 분포)
        _cycle_start = datetime.now(timezone.utc)  # 소요시간 측정용

        await manager.broadcast("alpha:factory", {
            "type": "cycle_start",
            "cycle": cycle_num,
        })

        try:
            logger.info("Cycle %d: starting (config=%s)", cycle_num, {k: v for k, v in config.items() if k != "context"})

            # 유니버스 리졸브 → 데이터 로드
            universe_code = config.get("universe", "KOSPI200")
            data_interval = config.get("data_interval", "1d")
            symbols = await resolve_universe(Universe(universe_code))

            # 인터벌별 종목 수 동적 제한 (OOM 방지, 분봉일수록 종목 적게)
            from app.alpha.interval import max_symbols_for_mining
            max_sym = max_symbols_for_mining(data_interval)
            if len(symbols) > max_sym:
                orig_count = len(symbols)
                symbols = symbols[:max_sym]
                logger.info("Cycle %d: %s — 종목 수 %d → %d 제한 (OOM 방지)", cycle_num, data_interval, orig_count, max_sym)

            logger.info("Cycle %d: resolved %d symbols for %s", cycle_num, len(symbols), universe_code)

            start_str = config.get("start_date") or ""
            end_str = config.get("end_date") or ""
            start = date.fromisoformat(start_str) if start_str else None
            end = date.fromisoformat(end_str) if end_str else None
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
                    nonlocal _last_funnel, _last_eval, _last_candidates
                    etype = event.get("type")
                    if etype == "generation_start":
                        self._state.population_size = event.get("population_size", 0)
                        self._state.generation = event.get("generation", self._state.generation)
                    elif etype == "generation_complete":
                        self._state.population_size = event.get("population_size", 0)
                        self._state.elite_count = event.get("elite_count", 0)
                        _last_funnel = event.get("funnel", {})
                        self._state.last_funnel = _last_funnel
                        # IC 히스토리 기록 (세대별 추이 — LLM 리포트용)
                        _best_ic = 0.0
                        if _last_eval:
                            _all_s = _last_eval.get("top_samples", []) + _last_eval.get("fail_samples", [])
                            if _all_s:
                                _best_ic = max(s.get("ic", 0) for s in _all_s)
                        self._state.generation_ic_history.append({
                            "gen": event.get("generation", 0),
                            "best_ic": round(_best_ic, 4),
                            "discovered": event.get("new_discovered", 0),
                            "eval_ok": _last_funnel.get("eval_ok", 0),
                        })
                        self._state.generation_ic_history = self._state.generation_ic_history[-20:]
                    elif etype == "candidates_ready":
                        _last_candidates = event
                    elif etype == "eval_complete":
                        _last_eval = event
                    await manager.broadcast("alpha:factory", {
                        **event,
                        "cycle": cycle_num,
                    })

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

                # Phase 3 메모리 확보: cached_data 임시 해제 (train/val에 복사 완료)
                self._cached_data = None
                self._data_cache_key = ""
                import gc as _gc; _gc.collect()

                # alpha_mining_runs를 먼저 INSERT (FK 대상: _persist_population에서 참조)
                run_id = uuid.uuid4()
                mining_run = AlphaMiningRun(
                    id=run_id,
                    name=f"Factory Cycle {cycle_num} (Gen {self._state.generation})",
                    context={"text": config.get("context", "")},
                    config=config,
                    status="RUNNING",
                    progress=0,
                    factors_found=0,
                    total_evaluated=settings.ALPHA_POPULATION_SIZE,
                )
                db.add(mining_run)
                await db.flush()  # DB에 레코드 생성 → FK 참조 가능

                self._evolution_engine._current_run_id = str(run_id)

                logger.info("Cycle %d: starting run_generation (gen=%d)", cycle_num, self._evolution_engine.generation)
                discovered = await self._evolution_engine.run_generation(
                    progress_cb=progress_cb,
                    iteration_cb=iteration_cb,
                )
                logger.info("Cycle %d: run_generation done, discovered=%d", cycle_num, len(discovered))

                self._state.generation = self._evolution_engine.generation
                self._state.operator_stats = self._operator_registry.to_dict()

                # mining_run 상태 업데이트 (같은 세션 #1에서)
                mining_run.status = "COMPLETED"
                mining_run.progress = 100
                mining_run.factors_found = len(discovered)
                mining_run.iteration_logs = {"operator_stats": self._operator_registry.to_dict()}
                mining_run.completed_at = datetime.now(timezone.utc)
                await db.commit()

            # discovered 팩터 DB 저장 (별도 세션 — 실패해도 카운터에 영향 없음)
            try:
                if discovered:
                    async with async_session() as save_db:
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
            # ★ DB 기반 누적 계산 (메모리 리셋에 강건)
            if run_id:
                try:
                    async with async_session() as _count_db:
                        _cnt_result = await _count_db.execute(
                            select(func.count(AlphaFactor.id)).where(
                                AlphaFactor.mining_run_id == str(run_id)
                            )
                        )
                        _cycle_found = _cnt_result.scalar() or 0
                    self._state.factors_discovered_total += _cycle_found
                except Exception:
                    self._state.factors_discovered_total += len(discovered)
            else:
                self._state.factors_discovered_total += len(discovered)
            self._state.last_cycle_at = datetime.now(timezone.utc).isoformat()
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

                # 시간 기반 throttle: 최소 5분 간격 (사이클이 빠르게 도는 경우 폭주 방지)
                _now = datetime.now(timezone.utc)
                _last_report = getattr(self, "_last_tg_report_at", None)
                _min_interval = 300  # 5분
                _time_ok = _last_report is None or (_now - _last_report).total_seconds() >= _min_interval

                should_report = (
                    len(discovered) > 0
                    or (cycle_num <= 3 and _time_ok)
                    or (cycle_num % 5 == 0 and _time_ok)
                )
                if should_report:
                    # 소요시간 계산
                    elapsed = datetime.now(timezone.utc) - _cycle_start
                    elapsed_min = int(elapsed.total_seconds() // 60)
                    elapsed_sec = int(elapsed.total_seconds() % 60)
                    elapsed_str = f"{elapsed_min}분 {elapsed_sec}초" if elapsed_min > 0 else f"{elapsed_sec}초"

                    report_data = self._build_report_data(
                        cycle_num, discovered, config,
                        _last_funnel, _last_eval, _last_candidates, elapsed_str,
                    )

                    # LLM 리포트 생성 (Gemini), 실패 시 기존 f-string 폴백
                    try:
                        msg = await self._generate_llm_report(report_data)
                    except Exception as llm_err:
                        logger.warning("LLM report generation failed: %s — using fallback", llm_err)
                        msg = self._build_fallback_report(report_data)

                    await tg_send(msg, category="mining_report", caller="alpha.scheduler")
                    self._last_tg_report_at = datetime.now(timezone.utc)
            except Exception as e:
                logger.warning("Telegram report failed: %s", e, exc_info=True)

            # WorkflowEvent에 사이클 결과 기록 (내부 로직 관측용 — 텔레그램과 독립)
            try:
                async with async_session() as evt_db:
                    from app.workflow.models import WorkflowEvent, WorkflowRun
                    today_run_stmt = select(WorkflowRun).where(
                        WorkflowRun.date == date.today()
                    )
                    today_run = await evt_db.execute(today_run_stmt)
                    wf_run = today_run.scalar_one_or_none()
                    if wf_run:
                        disc_count = len(discovered)
                        best_ic = max((d.ic_mean for d in discovered), default=0)
                        if disc_count > 0:
                            evt_msg = f"[MINING] {gen}번째 탐색 완료 — {disc_count}개 발견 (최고 IC {best_ic:.4f})"
                        else:
                            evt_msg = f"[MINING] {gen}번째 탐색 완료 — 기준 통과 전략 없음"
                        evt = WorkflowEvent(
                            workflow_run_id=wf_run.id,
                            phase=wf_run.phase,
                            event_type="mining_cycle",
                            message=evt_msg,
                            data={
                                "level": "info",
                                "gen": gen,
                                "cycle": cycle_num,
                                "discovered": disc_count,
                                "best_ic": round(best_ic, 4),
                                "total": self._state.factors_discovered_total,
                            },
                        )
                        evt_db.add(evt)
                        await evt_db.commit()
            except Exception:
                pass  # 이벤트 기록 실패가 마이닝을 방해하면 안 됨

    # ── 텔레그램 리포트 생성 ──

    def _build_report_data(
        self,
        cycle_num: int,
        discovered: list[DiscoveredFactor],
        config: dict,
        funnel: dict,
        eval_data: dict,
        candidates_data: dict,
        elapsed_str: str,
    ) -> dict:
        """콜백 데이터를 LLM 입력용 dict로 구조화."""
        ic_thr = config.get("ic_threshold", 0.03)

        # 발견 팩터 상세
        discovered_factors = []
        for d in sorted(discovered, key=lambda x: x.metrics.ic_mean if x.metrics else 0, reverse=True):
            entry: dict = {"expression": d.expression_str[:60]}
            if d.metrics:
                entry.update({
                    "ic_mean": round(d.metrics.ic_mean, 4),
                    "icir": round(d.metrics.icir, 2) if hasattr(d.metrics, "icir") and d.metrics.icir else 0,
                    "sharpe": round(d.metrics.sharpe, 2) if d.metrics.sharpe else 0,
                    "max_drawdown": round(d.metrics.max_drawdown, 3) if hasattr(d.metrics, "max_drawdown") and d.metrics.max_drawdown else 0,
                    "turnover": round(d.metrics.turnover, 3) if hasattr(d.metrics, "turnover") and d.metrics.turnover else 0,
                })
            if d.hypothesis:
                entry["hypothesis"] = d.hypothesis[:100]
            discovered_factors.append(entry)

        # 연산자 분포
        op_breakdown = {}
        if candidates_data:
            op_breakdown = candidates_data.get("operator_breakdown", {})

        # 연산자 성능 통계 (UCB1)
        op_stats = self._state.operator_stats or {}

        return {
            "generation": self._state.generation,
            "cycle_num": cycle_num,
            "elapsed": elapsed_str,
            "universe": config.get("universe", "KOSPI200"),
            "data_interval": config.get("data_interval", "1d"),
            "ic_threshold": ic_thr,
            "total_discovered": self._state.factors_discovered_total,
            "funnel": {
                "attempted": funnel.get("attempted", 0),
                "eval_ok": funnel.get("eval_ok", 0),
                "ic_pass": funnel.get("ic_pass", 0),
                "wf_overfit": funnel.get("wf_overfit", 0),
                "sharpe_fail": funnel.get("sharpe_fail", 0),
                "cpcv_candidates": funnel.get("cpcv_candidates", 0),
            },
            "operator_breakdown": op_breakdown,
            "operator_stats": {
                k: {
                    "calls": v.get("calls", 0),
                    "avg_fitness_delta": round(v.get("avg_fitness_delta", 0), 4),
                }
                for k, v in (op_stats.get("operators", {}) if isinstance(op_stats, dict) else {}).items()
                if isinstance(v, dict)
            },
            "discovered_factors": discovered_factors,
            "top_samples": eval_data.get("top_samples", []) if eval_data else [],
            "fail_samples": eval_data.get("fail_samples", []) if eval_data else [],
            "generation_ic_trend": self._state.generation_ic_history,
        }

    async def _generate_llm_report(self, report_data: dict) -> str:
        """Gemini로 마이닝 리포트 생성."""
        import json
        from app.core.llm import chat_gemini

        system_prompt = (
            "당신은 알파 팩터 마이닝 시스템의 리포트 분석가입니다.\n"
            "진화적 알파 팩터 탐색 사이클의 결과 데이터를 받아 텔레그램 리포트를 작성합니다.\n\n"
            "## 리포트 구성 (순서대로)\n\n"
            "1. **헤더**: 상태이모지 + 세대 번호 + 소요시간\n"
            "   - 🔥 팩터 5개 이상 발견\n"
            "   - ✅ 팩터 1~4개 발견\n"
            "   - 🔬 팩터 미발견\n\n"
            "2. **결과 요약**: 이번 사이클 발견 팩터 수, 누적 발견 수\n\n"
            "3. **발견 팩터 상세** (있으면 상위 3개만):\n"
            "   - 수식 (<code>태그), IC, Sharpe\n"
            "   - 각 팩터의 경제적 의미 1줄 해석 (hypothesis 참고)\n\n"
            "4. **진화 퍼널**: attempted → eval_ok → IC통과 → 최종 (퍼센트 포함)\n\n"
            "5. **진화 방향성 분석** (generation_ic_trend 데이터 기반):\n"
            "   - IC 추이 해석: 수렴 중인지, 발산 중인지, 정체인지\n"
            "   - 어떤 연산자(operator_breakdown/operator_stats)가 효과적인지\n"
            "   - 탐색 공간 포화도 판단 (eval_ok 비율 추이 등)\n\n"
            "6. **전략적 제안** (1~2줄):\n"
            "   - 다음 사이클 방향 권고 (연산자 비율, 유니버스, 인터벌 등)\n"
            "   - 팩터 미발견 시 원인 진단 + 개선 방향\n\n"
            "7. **하단 설정**: universe / interval / cycle_num 한 줄\n\n"
            "## 형식 제약\n"
            "- Telegram HTML만 사용: <b>, <i>, <code> 태그만 허용\n"
            "- <br>, <p>, <div>, <span>, <ul>, <li> 등은 절대 사용 금지 (텔레그램 미지원)\n"
            "- 줄바꿈은 반드시 \\n 문자 사용\n"
            "- 총 길이 3500자 이내 (한국어 기준)\n"
            "- 한국어로 작성\n"
            "- 이모지는 섹션 구분용으로 적절히 사용 (과하지 않게)\n"
            "- 트리 구조 표현 시 ├, └ 유니코드 문자 사용 가능\n"
            "- 숫자 반올림: IC 소수점 4자리, Sharpe 2자리, 퍼센트 정수\n"
        )

        user_message = json.dumps(report_data, ensure_ascii=False, default=str)

        response = await chat_gemini(
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            max_tokens=2000,
            temperature=0.3,
            caller="alpha.scheduler",
        )

        msg = response.text.strip()

        # 4000자 제한 (KST 타임스탬프 ~30자 + 여유)
        if len(msg) > 3800:
            cut = msg[:3800].rfind("\n")
            if cut > 2000:
                msg = msg[:cut] + "\n\n... (전문 생략)"
            else:
                msg = msg[:3800] + "\n\n... (전문 생략)"

        logger.info(
            "LLM mining report generated: %d chars, %d input_tokens, %d output_tokens",
            len(msg), response.input_tokens, response.output_tokens,
        )
        return msg

    def _build_fallback_report(self, report_data: dict) -> str:
        """LLM 실패 시 기존 f-string 폴백 리포트."""
        gen = report_data["generation"]
        elapsed_str = report_data["elapsed"]
        discovered_count = len(report_data["discovered_factors"])
        total = report_data["total_discovered"]
        ic_thr = report_data["ic_threshold"]
        f = report_data["funnel"]
        universe_code = report_data["universe"]
        data_interval = report_data["data_interval"]
        cycle_num = report_data["cycle_num"]
        op_breakdown = report_data.get("operator_breakdown", {})

        # 상태 이모지
        if discovered_count >= 5:
            status_emoji = "\U0001f525"  # 🔥
        elif discovered_count > 0:
            status_emoji = "\u2705"  # ✅
        else:
            status_emoji = "\U0001f52c"  # 🔬

        msg = f"{status_emoji} <b>{gen}번째 탐색 완료</b> ({elapsed_str})\n\n"

        # 결과 — 쉬운 말로
        if discovered_count > 0:
            msg += f"이번에 쓸 만한 전략을 <b>{discovered_count}개</b> 찾았어요! (지금까지 총 {total}개)\n"
        else:
            msg += f"이번엔 기준을 통과한 전략이 없었어요. (지금까지 총 {total}개)\n"

        # 발견 팩터 상세 — 쉽게 설명
        if report_data["discovered_factors"]:
            msg += f"\n\U0001f3c6 <b>찾아낸 전략 (상위 3개)</b>\n"
            for i, d in enumerate(report_data["discovered_factors"][:3], 1):
                ic = d.get("ic_mean", 0)
                sh = d.get("sharpe", 0)
                # IC/Sharpe를 별점으로 직관 표현
                ic_stars = "\u2b50" * min(5, max(1, int(ic / 0.03)))
                msg += (
                    f"  {i}. 예측력 {ic_stars} ({ic:.3f})"
                    f" · 안정성 {sh:.1f}점\n"
                )

        # 진화 과정 — 비유로 설명
        if f.get("attempted"):
            attempted = f["attempted"]
            eval_ok = f.get("eval_ok", 0)
            ic_pass = f.get("ic_pass", 0)
            eval_pct = int(eval_ok / attempted * 100) if attempted else 0

            msg += (
                f"\n\U0001f9ec <b>이번 탐색 과정</b>\n"
                f"  {attempted}개 후보를 만들어서 테스트했어요\n"
                f"  \u251c 제대로 계산된 것: {eval_ok}개 ({eval_pct}%)\n"
                f"  \u251c 예측력 기준 통과: {ic_pass}개\n"
            )
            wf_overfit = f.get("wf_overfit", 0)
            sharpe_fail = f.get("sharpe_fail", 0)
            if wf_overfit > 0:
                msg += f"  \u251c 과거에만 잘 맞는 것 제외: {wf_overfit}개\n"
            if sharpe_fail > 0:
                msg += f"  \u251c 수익이 불안정한 것 제외: {sharpe_fail}개\n"
            msg += f"  \u2514 최종 합격: <b>{discovered_count}개</b>\n"

        # 미발견 시 최고 성적
        if not report_data["discovered_factors"]:
            samples = report_data.get("top_samples") or report_data.get("fail_samples", [])
            if samples:
                best_ic = max(s.get("ic", 0) for s in samples)
                msg += f"\n\U0001f4c8 가장 유망했던 후보의 예측력: {best_ic:.4f}\n"

        # 쉬운 해석
        if discovered_count == 0:
            best_ic = 0.0
            all_samples = report_data.get("top_samples", []) + report_data.get("fail_samples", [])
            if all_samples:
                best_ic = max(s.get("ic", 0) for s in all_samples)
            if best_ic >= ic_thr * 0.8:
                msg += f"\n\U0001f4a1 기준에 거의 근접했어요. 조금만 더 진화하면 찾을 수 있어요.\n"
            elif f.get("eval_ok", 0) == 0:
                msg += f"\n\u26a0\ufe0f 계산 실패가 많아요. 데이터를 점검해볼 필요가 있어요.\n"
            else:
                msg += f"\n\U0001f4a1 아직 좋은 전략을 못 찾았지만, 탐색을 계속하면 점점 나아져요.\n"
        else:
            msg += (
                f"\n\U0001f4a1 <b>용어 설명</b>\n"
                f"  예측력(IC): 내일 주가를 얼마나 잘 맞추는지 (높을수록 좋음)\n"
                f"  안정성(Sharpe): 수익이 꾸준한지 (1.0 이상이면 좋음)\n"
            )

        msg += f"\n\u2699\ufe0f 탐색 범위: {universe_code} / {data_interval} / {cycle_num}번째 사이클"
        return msg

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


# ── 인터벌별 스케줄러 인스턴스 ──

_schedulers: dict[str, AlphaFactoryScheduler] = {}


def get_scheduler(interval: str = "5m") -> AlphaFactoryScheduler:
    """인터벌별 스케줄러 인스턴스 반환. 같은 인터벌이면 같은 인스턴스."""
    if interval not in _schedulers:
        _schedulers[interval] = AlphaFactoryScheduler()
    return _schedulers[interval]


def get_all_schedulers() -> dict[str, AlphaFactoryScheduler]:
    """실행 중인 모든 스케줄러 반환."""
    return _schedulers
