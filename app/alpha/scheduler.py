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
from app.alpha.report_metrics import (
    compute_coverage_health,
    compute_derived_feature_usage,
    compute_family_delta,
    compute_family_distribution,
    compute_ic_trend,
)
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
    log_lines: list = field(default_factory=list)  # 실시간 로그 (최근 500줄 링버퍼)
    prev_family_distribution: dict[str, float] = field(default_factory=dict)


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

    def _append_log(self, msg: str) -> None:
        """실시간 로그에 타임스탬프 포함 메시지 추가 (최근 500줄)."""
        if not self._state.running:
            return
        from app.core.timezone import now_kst
        ts = now_kst().strftime("%H:%M:%S")
        self._state.log_lines.append(f"[{ts}] {msg}")
        if len(self._state.log_lines) > 500:
            self._state.log_lines = self._state.log_lines[-500:]
        self._state.current_cycle_message = msg

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
        seed_factor_ids: list[str] | None = None,
        population_size: int | None = None,
        cpcv_n_groups: int | None = None,
        cpcv_n_test: int | None = None,
        cpcv_embargo_days: int | None = None,
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
                    "seed_factor_ids": seed_factor_ids,
                    "population_size": population_size,
                    "cpcv_n_groups": cpcv_n_groups,
                    "cpcv_n_test": cpcv_n_test,
                    "cpcv_embargo_days": cpcv_embargo_days,
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
                        .where(AlphaFactor.interval == data_interval)
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

            # WebSocket 즉시 알림 — 프론트 폴링 대기 없이 UI 갱신
            try:
                from app.services.ws_manager import manager
                await manager.broadcast("alpha:factory", {
                    "type": "factory_started",
                    "interval": data_interval,
                    "config": self._state.config,
                })
            except Exception:
                pass

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

            # 메모리 해제: 대용량 캐시 참조 제거
            self._cached_data = None
            self._data_cache_key = ""
            if self._evolution_engine is not None:
                self._evolution_engine._data = None
                self._evolution_engine._population = []
                self._evolution_engine._last_population = []
                if hasattr(self._evolution_engine, '_FEATURE_LIST'):
                    self._evolution_engine._FEATURE_LIST = None
            if self._vector_memory is not None:
                self._vector_memory._cache = {}

            # DB 연결 풀 정리: 유휴 연결의 work_mem 해제
            try:
                from app.core.database import engine
                await engine.dispose()
                logger.info("DB 연결 풀 해제 완료")
            except Exception as e:
                logger.warning("DB 연결 풀 해제 실패: %s", e)

            # PostgreSQL에 CHECKPOINT + 캐시 정리 요청
            # (DB가 페이지 캐시를 해제하도록 유도 — WSL2 vmmem 반환)
            try:
                from sqlalchemy import text as sa_text
                from app.core.database import async_session
                async with async_session() as _db:
                    await _db.execute(sa_text("CHECKPOINT"))
                    await _db.execute(sa_text("DISCARD ALL"))
                    await _db.commit()
                logger.info("DB CHECKPOINT + DISCARD 완료 (캐시 정리)")
            except Exception as e:
                logger.debug("DB 캐시 정리 실패: %s", e)

            import gc
            gc.collect()
            logger.info("Alpha factory stopped (메모리 해제 완료)")

            # WebSocket 즉시 알림
            try:
                from app.services.ws_manager import manager
                await manager.broadcast("alpha:factory", {
                    "type": "factory_stopped",
                    "reason": "user",
                })
            except Exception:
                pass

            return True

    def get_last_config(self) -> dict:
        """마지막으로 사용한 팩토리 설정 반환 (자동 재시작용)."""
        return self._state.config if self._state else {}

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
            "log_lines": list(self._state.log_lines[-500:]),
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

        # ── 텔레그램 피드백 RAG 주입 (딥리서치 권고) ──
        try:
            from app.core.database import async_session
            from app.telegram.models import TelegramMessageLog
            import re as _re

            async with async_session() as _db:
                _result = await _db.execute(
                    select(TelegramMessageLog.text)
                    .where(TelegramMessageLog.category == "mining_report")
                    .order_by(TelegramMessageLog.created_at.desc())
                    .limit(1)
                )
                _last_report = _result.scalar()
                if _last_report:
                    _match = _re.search(
                        r"(?:전략적 제안|전략적 제언|Strategic|제언|Suggestion)(.*?)(?:\n\n|\Z)",
                        _last_report, _re.DOTALL,
                    )
                    if _match:
                        _suggestion = _match.group(1).strip()[:500]
                        _base_ctx = config.get("context", "")
                        config["context"] = f"{_base_ctx}\n\n[이전 사이클 관찰]\n{_suggestion}"
                        logger.info("텔레그램 피드백 RAG 주입: %d자", len(_suggestion))
        except Exception:
            pass  # 실패해도 마이닝은 계속

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
            self._append_log(f"사이클 {cycle_num} 시작")

            # 유니버스 리졸브 → 데이터 로드
            universe_code = config.get("universe", "KOSPI200")
            data_interval = config.get("data_interval", "1d")
            symbols = await resolve_universe(Universe(universe_code))
            self._append_log(f"유니버스 {universe_code}: {len(symbols)}개 종목 리졸브")

            # 인터벌별 종목 수 동적 제한 (OOM 방지, 분봉일수록 종목 적게)
            from app.alpha.interval import max_symbols_for_mining
            max_sym = max_symbols_for_mining(data_interval)
            if len(symbols) > max_sym:
                orig_count = len(symbols)
                symbols = symbols[:max_sym]
                logger.info("Cycle %d: %s — 종목 수 %d → %d 제한 (OOM 방지)", cycle_num, data_interval, orig_count, max_sym)
                self._append_log(f"종목 수 제한: {orig_count} → {max_sym} (OOM 방지)")

            logger.info("Cycle %d: resolved %d symbols for %s", cycle_num, len(symbols), universe_code)

            start_str = config.get("start_date") or ""
            end_str = config.get("end_date") or ""
            start = date.fromisoformat(start_str) if start_str else None
            end = date.fromisoformat(end_str) if end_str else None
            cache_key = f"{universe_code}_{start_str}_{end_str}_{data_interval}"
            if self._cached_data is not None and self._data_cache_key == cache_key:
                data = self._cached_data
                logger.info("Cycle %d: using cached candles %d rows x %d cols", cycle_num, data.height, data.width)
                self._append_log(f"캐시 데이터 사용 ({data.height:,}행 × {data.width}피처)")
                await manager.broadcast("alpha:factory", {
                    "type": "progress",
                    "cycle": cycle_num,
                    "phase": "data_cached",
                    "message": f"캐시 데이터 사용 ({data.height:,}행 × {data.width}피처)",
                    "current": 0, "total": 100,
                })
            else:
                self._append_log(f"Enriched 캔들 로드 중... ({data_interval}, {len(symbols)}종목)")
                data = await load_enriched_candles(
                    symbols=symbols,
                    start_date=start,
                    end_date=end,
                    interval=data_interval,
                )
                self._cached_data = data
                self._data_cache_key = cache_key
                _size_mb = data.estimated_size() / 1024 / 1024
                logger.info("Cycle %d: loaded enriched candles %d rows x %d cols (%.0fMB)", cycle_num, data.height, data.width, _size_mb)
                self._append_log(f"데이터 로드 완료: {data.height:,}행 × {data.width}피처 ({_size_mb:.0f}MB)")
                # 팩트 기반 로그: 실제 로드 결과
                await manager.broadcast("alpha:factory", {
                    "type": "progress",
                    "cycle": cycle_num,
                    "phase": "data_loaded",
                    "message": f"{data.height:,}행 × {data.width}피처 로드 ({_size_mb:.0f}MB)",
                    "current": 0, "total": 100,
                })

            if data.height == 0:
                logger.warning("Alpha factory cycle %d: no candle data", cycle_num)
                return

            # 진화 엔진 기반 실행
            self._append_log("진화 엔진 초기화 중...")
            async with async_session() as db:
                if self._vector_memory:
                    await self._vector_memory.load_cache(db)
                    self._append_log("벡터 메모리 캐시 로드 완료")

                async def progress_cb(current: int, total: int, msg: str) -> None:
                    self._state.current_cycle_progress = current
                    self._append_log(msg)
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
                        # [2026-04-07] 전체 offspring 통계 사용 (sampling 기반 → 전수 통계)
                        _best_ic = 0.0
                        _avg_ic = 0.0
                        _best_sharpe = 0.0
                        _avg_sharpe = 0.0
                        if _last_eval:
                            _best_ic = _last_eval.get("population_best_ic", 0)
                            _avg_ic = _last_eval.get("population_avg_ic", 0)
                            _best_sharpe = _last_eval.get("population_best_sharpe", 0)
                            _avg_sharpe = _last_eval.get("population_avg_sharpe", 0)
                            # 폴백: 이전 버전 호환 (population_best_ic 없으면 sampling 방식)
                            if not _best_ic and not _avg_ic:
                                _all_s = _last_eval.get("top_samples", []) + _last_eval.get("fail_samples", [])
                                if _all_s:
                                    _best_ic = max(s.get("ic", 0) for s in _all_s)
                        self._state.generation_ic_history.append({
                            "gen": event.get("generation", 0),
                            "best_ic": round(_best_ic, 4),
                            "avg_ic": round(_avg_ic, 4),
                            "best_sharpe": round(_best_sharpe, 2),
                            "avg_sharpe": round(_avg_sharpe, 2),
                            "discovered": event.get("new_discovered", 0),
                            "eval_ok": _last_funnel.get("eval_ok", 0),
                            "cross_gen_dup": _last_eval.get("cross_gen_dup", 0) if _last_eval else 0,
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
                    self._append_log(f"진화 엔진 신규 생성 (세대 {self._state.generation}, 모집단 {config.get('population_size') or settings.ALPHA_POPULATION_SIZE})")
                    self._evolution_engine = EvolutionEngine(
                        data=data,
                        db=db,
                        operator_registry=self._operator_registry,
                        population_size=config.get("population_size") or settings.ALPHA_POPULATION_SIZE,
                        elite_pct=settings.ALPHA_ELITE_PCT,
                        context=config.get("context", ""),
                        ic_threshold=config["ic_threshold"],
                        orthogonality_threshold=config.get("orthogonality_threshold", 0.7),
                        vector_memory=self._vector_memory,
                        generation=self._state.generation,
                        interval=data_interval,
                        cpcv_n_groups=config.get("cpcv_n_groups") or 10,
                        cpcv_n_test=config.get("cpcv_n_test") or 3,
                        cpcv_embargo_days=config.get("cpcv_embargo_days") or 10,
                    )
                    # 취소 체크 콜백 — to_thread 내부에서 running=False 감지 시 즉시 종료
                    self._evolution_engine._is_cancelled = lambda: not self._state.running
                else:
                    self._append_log(f"진화 엔진 데이터 갱신 (세대 {self._state.generation})")
                    self._evolution_engine.update_data(data)
                    self._evolution_engine._db = db
                    self._evolution_engine._is_cancelled = lambda: not self._state.running

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
                    total_evaluated=config.get("population_size") or settings.ALPHA_POPULATION_SIZE,
                )
                db.add(mining_run)
                await db.flush()  # DB에 레코드 생성 → FK 참조 가능

                self._evolution_engine._current_run_id = str(run_id)

                # seed_factor_ids: 첫 사이클에서만 주입, 이후 사이클에서는 무시
                _seed_ids = config.get("seed_factor_ids") if cycle_num == 1 else None

                def log_cb(msg: str) -> None:
                    self._append_log(msg)

                self._append_log(f"세대 {self._evolution_engine.generation + 1} run_generation 시작")
                logger.info("Cycle %d: starting run_generation (gen=%d, seed_factor_ids=%s)", cycle_num, self._evolution_engine.generation, _seed_ids)
                discovered = await self._evolution_engine.run_generation(
                    progress_cb=progress_cb,
                    iteration_cb=iteration_cb,
                    log_cb=log_cb,
                    seed_factor_ids=_seed_ids,
                )
                logger.info("Cycle %d: run_generation done, discovered=%d", cycle_num, len(discovered))
                self._append_log(f"세대 완료: {len(discovered)}개 팩터 발견")

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
                self._append_log(f"인과 검증 시작: {len(discovered)}개 팩터")
                await self._run_causal_validation(run_id, cycle_num)
                self._append_log("인과 검증 완료")

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
            self._append_log(f"사이클 {cycle_num} 완료: {len(discovered)}개 발견 (누적 {self._state.factors_discovered_total})")

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

                    # IC 트렌드는 DB 조회이므로 report_data 구성 직후 비동기로
                    report_data["ic_trend"] = await compute_ic_trend(
                        interval=config.get("data_interval", "1d"), limit=10,
                    )

                    # Redis 캐싱 (프론트엔드 API용, 24시간 TTL)
                    try:
                        import json as _json
                        from app.core.redis import get_client as get_redis
                        _redis = get_redis()
                        _interval = config.get("data_interval", "1d")
                        await _redis.set(
                            f"alpha:mining_report:{_interval}",
                            _json.dumps(report_data, default=str),
                            ex=86400,
                        )
                    except Exception:
                        pass

                    # DB 영속화
                    try:
                        from app.alpha.models import AlphaGenerationReport
                        async with async_session() as _persist_db:
                            _persist_db.add(AlphaGenerationReport(
                                generation=report_data["generation"],
                                data_interval=report_data.get("data_interval", "1d"),
                                cycle_num=report_data.get("cycle_num", 0),
                                report_data=report_data,
                            ))
                            await _persist_db.commit()
                    except Exception as e:
                        logger.warning("Mining report DB persist failed: %s", e)

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

        report = {
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
                    "avg_fitness_delta": round(v.get("recent_avg_reward", 0) or v.get("avg_fitness_delta", 0), 4),
                }
                for k, v in (op_stats.get("operators", {}) if isinstance(op_stats, dict) else {}).items()
                if isinstance(v, dict)
            },
            "discovered_factors": discovered_factors,
            "top_samples": eval_data.get("top_samples", []) if eval_data else [],
            "fail_samples": eval_data.get("fail_samples", []) if eval_data else [],
            "generation_ic_trend": self._state.generation_ic_history,
        }

        # ── 신규 메트릭 (마이닝 리포트 고도화) ──
        population = getattr(self._evolution_engine, "_last_population", []) if self._evolution_engine else []
        offspring = candidates_data.get("offspring", []) if candidates_data else []

        family_dist = compute_family_distribution(population)
        report["family_distribution"] = family_dist
        report["family_delta"] = compute_family_delta(
            family_dist,
            self._state.prev_family_distribution,
        )
        self._state.prev_family_distribution = family_dist

        report["derived_feature_usage"] = compute_derived_feature_usage(offspring)
        report["coverage_health"] = compute_coverage_health(population)

        # ── 다양성 메트릭 (세대간 수렴/정체 진단용) ──
        report["population_unique_hashes"] = len(set(
            f.expression_hash for f in population if hasattr(f, "expression_hash") and f.expression_hash
        ))
        report["cross_gen_dup"] = eval_data.get("cross_gen_dup", 0) if eval_data else 0

        return report

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
            "8. **피처 다양성 분석**: family_distribution 비율과 delta를 해석. 특정 패밀리가 과도하거나 과소한 경우 원인 분석.\n\n"
            "9. **커버리지 건강**: coverage_health의 Tier 분포를 해석. 데이터 부족으로 탈락한 팩터가 많으면 데이터 수집 강화 권고.\n\n"
            "10. **IC 트렌드**: ic_trend 시계열 추세를 해석. 정체/하락/상승 패턴 식별.\n\n"
            "11. **신규 피처**: derived_feature_usage 중 활발히 사용되는 피처와 미사용 피처를 언급.\n\n"
            "12. **다양성 진단**: population_unique_hashes(고유 수식 수)와 cross_gen_dup(교차세대 중복 수)를 확인.\n"
            "    - 고유 수식 수가 모집단의 50% 미만이면 수렴 경고.\n"
            "    - cross_gen_dup > 0이면 이전 세대와 동일한 팩터가 반복 발견되고 있음.\n"
            "    - generation_ic_trend의 avg_ic/best_ic/avg_sharpe를 세대간 비교하여 정체 여부 진단.\n\n"
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

    def _build_fallback_report(self, data: dict) -> str:
        """텔레그램 폴백 리포트 (LLM 실패 시)."""
        gen = data.get("generation", "?")
        elapsed = data.get("elapsed", "?")
        discovered = data.get("discovered_factors", [])
        funnel = data.get("funnel", {})
        total = data.get("total_discovered", 0)

        # ── Executive Summary ──
        best_ic = max((f.get("ic_mean", 0) for f in discovered), default=0)
        n_found = len(discovered)
        if n_found > 0:
            emoji = "\U0001f525" if best_ic >= 0.05 else "\u2705"
            summary = f"{emoji} <b>Gen {gen}</b>: {n_found}개 발견 (최고 IC {best_ic:.4f}) [{elapsed}]"
        else:
            emoji = "\U0001f52c"
            summary = f"{emoji} <b>Gen {gen}</b>: 탐색 중 [{elapsed}]"

        lines = [summary, ""]

        # ── 핵심 수치 ──
        attempted = funnel.get("attempted", 0)
        rate = (n_found / attempted * 100) if attempted > 0 else 0
        lines.append(
            f"\U0001f4ca 발견 {n_found}개 / 평가 {attempted}개 / "
            f"통과율 {rate:.1f}% / 총 {total}개"
        )
        lines.append("")

        # ── 상위 팩터 ──
        if discovered:
            lines.append("\U0001f3c6 <b>상위 전략</b>")
            for i, f in enumerate(discovered[:3], 1):
                ic = f.get("ic_mean", 0)
                sh = f.get("sharpe", 0)
                expr = f.get("expression", "?")[:70]
                lines.append(f"  {i}. IC {ic:.4f} | Sharpe {sh:.2f}")
                lines.append(f"     <code>{expr}</code>")
            lines.append("")

        # ── 퍼널 ──
        eval_ok = funnel.get("eval_ok", 0)
        ic_pass = funnel.get("ic_pass", 0)
        cpcv = funnel.get("cpcv_candidates", 0)
        lines.append("\U0001f52c <b>파이프라인</b>")
        lines.append(f"  {attempted} \u2192 {eval_ok} \u2192 {ic_pass} \u2192 {cpcv} \u2192 {n_found}")
        lines.append("")

        # ── 패밀리 분포 ──
        family_dist = data.get("family_distribution", {})
        family_delta = data.get("family_delta", {})
        if family_dist:
            lines.append("\U0001f4ca <b>패밀리 분포</b>")
            for fam in sorted(family_dist, key=family_dist.get, reverse=True):
                pct = family_dist[fam] * 100
                bar_len = int(pct / 5)
                bar = "\u2588" * bar_len + "\u2591" * (10 - bar_len)
                delta = family_delta.get(fam, 0) * 100
                delta_str = f" ({delta:+.0f}pp)" if abs(delta) >= 1 else ""
                lines.append(f"  {fam:10s} {bar} {pct:4.0f}%{delta_str}")
            lines.append("")

        # ── 커버리지 ──
        cov = data.get("coverage_health", {})
        if cov:
            ta = cov.get("tier_a", {})
            tb = cov.get("tier_b", {})
            lines.append(
                f"\U0001f4c8 커버리지  A(>80%): {ta.get('count', 0)}개 | "
                f"B(50-80%): {tb.get('count', 0)}개"
            )
            lines.append("")

        # ── IC 트렌드 ──
        ic_trend = data.get("ic_trend", [])
        if ic_trend and len(ic_trend) >= 2:
            lines.append("\U0001f4c9 <b>IC 추이</b>")
            max_ic = max(t["avg_ic"] for t in ic_trend) or 0.01
            for t in ic_trend[-5:]:
                bar_len = int(t["avg_ic"] / max_ic * 8) if max_ic > 0 else 0
                bar = "\u25a0" * bar_len
                mark = " \u2605" if t["avg_ic"] == max_ic else ""
                lines.append(f"  Gen{t['gen']:>3d} \u2524{bar:<8s} {t['avg_ic']:.4f}{mark}")
            lines.append("")

        # ── 연산자 Top 3 ──
        op_stats = data.get("operator_stats", {})
        if op_stats:
            sorted_ops = sorted(
                op_stats.items(),
                key=lambda x: x[1].get("avg_fitness_delta", 0),
                reverse=True,
            )[:3]
            lines.append("\u2699\ufe0f <b>연산자 Top 3</b>")
            for op, stats in sorted_ops:
                calls = stats.get("calls", 0)
                delta = stats.get("avg_fitness_delta", 0)
                lines.append(f"  {op}: {calls}회 (avg {delta:+.4f})")

        return "\n".join(lines)

    async def _run_causal_validation(self, run_id: uuid.UUID, cycle_num: int) -> None:
        """인과 검증을 동기적으로 실행. 검증 완료 후 다음 사이클로 진행."""
        try:
            from app.alpha.causal_runner import validate_factors_batch

            logger.info("Cycle %d: starting causal validation (run=%s)", cycle_num, run_id)
            async with async_session() as causal_db:
                count = await validate_factors_batch(
                    run_id, causal_db, log_cb=self._append_log,
                )
            logger.info("Cycle %d: causal validation complete (%d factors)", cycle_num, count)
            self._append_log(f"인과 검증 완료: {count}개 팩터 검증됨")

            await manager.broadcast("alpha:factory", {
                "type": "causal_complete",
                "cycle": cycle_num,
                "validated_count": count,
            })
        except Exception as e:
            logger.error("Cycle %d: causal validation failed: %s", cycle_num, e)
            self._append_log(f"인과 검증 실패: {e}")


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
