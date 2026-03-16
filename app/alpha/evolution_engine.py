"""DB 영속 모집단 기반 진화 엔진.

팩토리 전용. miner.py의 EvolutionaryAlphaMiner와 병렬 존재.
사이클 간 모집단을 DB에 유지하여 연속적 진화를 수행한다.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import math
import random
import uuid
from dataclasses import asdict

import polars as pl
import sympy
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.alpha.ast_converter import (
    ASTConversionError,
    ensure_alpha_features,
    expression_hash,
    parse_expression,
    sympy_to_code_string,
    sympy_to_polars,
    tree_depth as calc_tree_depth,
    tree_size as calc_tree_size,
)
from app.alpha.evaluator import (
    FactorMetrics,
    compute_factor_metrics,
    compute_forward_returns,
    compute_ic_series,
    compute_quantile_returns,
)
from app.alpha.evolution import (
    ScoredFactor,
    crossover,
    ephemeral_constant_mutation,
    hoist_mutation,
    mutate,
    tournament_select,
)
from app.alpha.expression_translator import generate_hypothesis_korean
from app.alpha.fitness import compute_composite_fitness
from app.alpha.memory import ExperienceVectorMemory
from app.alpha.miner import DiscoveredFactor
from app.alpha.models import AlphaFactor
from app.alpha.operators import OperatorRegistry
from app.core.config import settings

logger = logging.getLogger(__name__)


class EvolutionEngine:
    """DB 영속 모집단 기반 진화 엔진.

    miner.py의 EvolutionaryAlphaMiner와 달리:
    - 모집단이 DB에 영속 → 사이클 간 연속성
    - 90%+ AST 연산자 + ~8% Claude API (LASR "doping")
    - UCB1로 연산자 적응 선택
    - 구조적 해시로 중복 감지
    - 복합 적합도 (IC + ICIR - turnover - complexity)
    """

    def __init__(
        self,
        data: pl.DataFrame,
        db: AsyncSession,
        operator_registry: OperatorRegistry,
        population_size: int | None = None,
        elite_pct: float | None = None,
        context: str = "",
        ic_threshold: float | None = None,
        orthogonality_threshold: float = 0.7,
        vector_memory: ExperienceVectorMemory | None = None,
        generation: int = 0,
        train_ratio: float = 0.7,
        interval: str = "1d",
    ) -> None:
        self._data = data
        self._db = db
        self._operator_registry = operator_registry
        self._population_size = population_size or settings.ALPHA_POPULATION_SIZE
        self._elite_pct = elite_pct or settings.ALPHA_ELITE_PCT
        self._context = context
        self._ic_threshold = ic_threshold or settings.ALPHA_IC_THRESHOLD_PASS
        self._orthogonality_threshold = orthogonality_threshold
        self._vector_memory = vector_memory
        self._generation = generation
        self._train_ratio = train_ratio
        self._interval = interval
        self._current_run_id: str | None = None  # scheduler에서 설정

        # 데이터 전처리
        self._data = ensure_alpha_features(self._data)

    def update_data(self, data: pl.DataFrame) -> None:
        """새 사이클 데이터 반영."""
        self._data = ensure_alpha_features(data)

    async def load_population(self) -> list[ScoredFactor]:
        """DB에서 population_active=True인 팩터 로드 + 엘리트 아카이브 주입."""
        result = await self._db.execute(
            select(AlphaFactor).where(
                AlphaFactor.population_active == True,  # noqa: E712
            ).order_by(AlphaFactor.fitness_composite.desc().nullslast())
        )
        factors = list(result.scalars().all())

        # 엘리트 아카이브: validated 팩터 중 비활성 상태인 것을 랜덤 샘플링
        archive_size = max(5, int(self._population_size * 0.15))
        archive_result = await self._db.execute(
            select(AlphaFactor).where(
                AlphaFactor.status == "validated",
                AlphaFactor.population_active == False,  # noqa: E712
            ).order_by(func.random()).limit(archive_size)
        )
        archive_factors = list(archive_result.scalars().all())

        existing_ids = {f.id for f in factors}
        archive_injected = [f for f in archive_factors if f.id not in existing_ids]
        if archive_injected:
            logger.info(
                "Elite archive: injecting %d validated factors into population",
                len(archive_injected),
            )

        population: list[ScoredFactor] = []
        for f in factors:
            try:
                expr = parse_expression(f.expression_str)
                population.append(ScoredFactor(
                    expression=expr,
                    expression_str=f.expression_str,
                    hypothesis=f.hypothesis or "",
                    ic_mean=f.ic_mean or 0.0,
                    ic_std=f.ic_std or 0.0,
                    icir=f.icir or 0.0,
                    turnover=f.turnover or 0.0,
                    sharpe=f.sharpe or 0.0,
                    max_drawdown=f.max_drawdown or 0.0,
                    generation=f.birth_generation or 0,
                    factor_id=str(f.id),
                    fitness_composite=f.fitness_composite or 0.0,
                    tree_depth=f.tree_depth or 0,
                    tree_size=f.tree_size or 0,
                    expression_hash=f.expression_hash or "",
                    operator_origin=f.operator_origin or "",
                ))
            except (ASTConversionError, Exception) as e:
                logger.debug("Skipping unparseable factor %s: %s", f.id, e)

        # 아카이브 팩터: fitness를 모집단 중앙값으로 리셋 (다른 유니버스 stale fitness 방지)
        median_fitness = 0.0
        if population:
            sorted_fitness = sorted(f.fitness_composite for f in population)
            median_fitness = sorted_fitness[len(sorted_fitness) // 2]

        for f in archive_injected:
            try:
                expr = parse_expression(f.expression_str)
                population.append(ScoredFactor(
                    expression=expr,
                    expression_str=f.expression_str,
                    hypothesis=f.hypothesis or "",
                    ic_mean=f.ic_mean or 0.0,
                    ic_std=f.ic_std or 0.0,
                    icir=f.icir or 0.0,
                    turnover=f.turnover or 0.0,
                    sharpe=f.sharpe or 0.0,
                    max_drawdown=f.max_drawdown or 0.0,
                    generation=f.birth_generation or 0,
                    factor_id=str(f.id),
                    fitness_composite=median_fitness,  # 중앙값으로 리셋
                    tree_depth=f.tree_depth or 0,
                    tree_size=f.tree_size or 0,
                    expression_hash=f.expression_hash or "",
                    operator_origin=f.operator_origin or "archive",
                ))
            except (ASTConversionError, Exception) as e:
                logger.debug("Skipping archive factor %s: %s", f.id, e)

        return population

    async def run_generation(
        self,
        progress_cb=None,
        iteration_cb=None,
    ) -> list[DiscoveredFactor]:
        """한 세대 실행.

        Returns
        -------
        list[DiscoveredFactor]
            이번 세대에서 새로 IC 기준을 통과한 팩터들.
        """
        self._generation += 1
        self._eval_fail_logged = 0  # 세대별 리셋

        # 1. 모집단 로드
        population = await self.load_population()

        if not population:
            # 초기 시드 필요
            logger.info("No active population found, seeding initial population")
            population = await self._seed_population(
                min(self._population_size, 20)
            )

        if progress_cb:
            await progress_cb(10, 100, f"세대 {self._generation}: 모집단 {len(population)}개 로드")

        # 1.5. 세대 시작 시 상태 즉시 보고 (프론트엔드 대시보드 표시용)
        if iteration_cb:
            await iteration_cb({
                "type": "generation_start",
                "generation": self._generation,
                "population_size": len(population),
            })

        # 2. 엘리트 보존
        elites = self._select_elites(population)
        logger.info("세대 %d: 엘리트 %d개 선택, Train/Val 분할 시작", self._generation, len(elites))

        # 3. Train/Val 분할
        train_data, val_data = self._split_train_val()
        logger.info("세대 %d: Train/Val 분할 완료 (train=%d×%d, val=%s)", self._generation, train_data.height, train_data.width, val_data.height if val_data is not None else "None")

        # CPCV 검증에 전체 데이터가 필요하므로 참조 보존
        full_data = self._data

        # train/val 복사 완료 → 원본 해제 (scheduler._cached_data도 None이므로 참조 0 → GC 대상)
        self._data = None  # type: ignore[assignment]
        gc.collect()

        # 3.5. fwd_return 사전 계산 (팩터와 무관, 1회만 계산)
        train_data = compute_forward_returns(train_data, periods=1)
        if val_data is not None:
            val_data = compute_forward_returns(val_data, periods=1)

        # 4. 진화 3-Phase 파이프라인
        import time as _time

        offspring_target = self._population_size - len(elites)

        funnel = {
            "operator_null": 0,
            "eval_success": 0,
            "ic_pass": 0,
            "wf_tested": 0,
            "wf_overfit": 0,
            "sharpe_fail": 0,
        }

        _t_phase0 = _time.perf_counter()
        logger.info("세대 %d: Phase 1 시작 (offspring_target=%d)", self._generation, offspring_target)

        # ── Phase 1: 식 생성 (AST 즉시, LLM 지연) ──
        _GeneratedChild = tuple  # (expr, op_name, parent, parent_ids)
        ast_children: list[tuple] = []
        llm_coros: list[tuple] = []  # (op_name, parent, parent_ids, coro)

        for i in range(offspring_target):
            op_name = self._operator_registry.select()
            parents = tournament_select(
                population,
                k=min(settings.ALPHA_TOURNAMENT_K, len(population)),
                n_select=2,
                parsimony=True,
            )
            if not parents:
                funnel["operator_null"] += 1
                continue
            parent = parents[0]
            parent_ids = [parent.factor_id] if parent.factor_id else []

            if op_name.startswith("llm_"):
                coro = self._llm_seed() if op_name == "llm_seed" else self._llm_mutate(parent)
                llm_coros.append((op_name, parent, parent_ids, coro))
            else:
                child_expr = self._apply_ast_op(parents, op_name)
                if child_expr is not None:
                    if op_name == "ast_crossover" and len(parents) >= 2 and parents[1].factor_id:
                        parent_ids = parent_ids + [parents[1].factor_id]
                    ast_children.append((child_expr, op_name, parent, parent_ids))
                else:
                    self._operator_registry.update(op_name, delta_fitness=0.0)
                    funnel["operator_null"] += 1

        _t_phase1 = _time.perf_counter()
        logger.info("세대 %d: Phase 1 완료 (%.1fs) — AST %d개, LLM %d개", self._generation, _t_phase1 - _t_phase0, len(ast_children), len(llm_coros))

        if progress_cb:
            await progress_cb(
                20, 100,
                f"세대 {self._generation}: {len(ast_children)} AST + {len(llm_coros)} LLM 생성 완료"
            )

        # ── Phase 2: LLM 호출 동시 실행 (retry + 폴백) ──
        llm_success_count = 0
        if llm_coros:
            sem = asyncio.Semaphore(settings.ALPHA_LLM_MAX_CONCURRENT)

            async def _fire_llm(op_name, parent, parent_ids, coro):
                async with sem:
                    last_err = None
                    active_coro = coro
                    for attempt in range(settings.ALPHA_LLM_RETRY_MAX + 1):
                        try:
                            expr = await active_coro
                            if expr is None:
                                self._operator_registry.update(op_name, delta_fitness=0.0)
                                return None
                            sympy_to_polars(expr)  # 변환 검증
                            return (expr, op_name, parent, parent_ids)
                        except Exception as e:
                            err_name = type(e).__name__
                            is_rate_limit = "RateLimit" in err_name or "rate_limit" in str(e).lower()
                            is_timeout = "Timeout" in err_name or "timeout" in str(e).lower()

                            if is_rate_limit or is_timeout:
                                # 429 또는 타임아웃: 재시도 가치 있음
                                retry_after = None
                                if hasattr(e, "response") and hasattr(e.response, "headers"):
                                    retry_after = e.response.headers.get("retry-after")
                                wait = (
                                    float(retry_after) if retry_after
                                    else settings.ALPHA_LLM_RETRY_BASE_DELAY * (2 ** attempt)
                                )
                                logger.warning(
                                    "LLM %s %s (attempt %d/%d): waiting %.1fs",
                                    op_name, err_name, attempt + 1,
                                    settings.ALPHA_LLM_RETRY_MAX + 1, wait,
                                )
                                await asyncio.sleep(wait)
                                # 코루틴 재생성 (await된 코루틴은 재사용 불가)
                                active_coro = (
                                    self._llm_seed() if op_name == "llm_seed"
                                    else self._llm_mutate(parent)
                                )
                                last_err = e
                            else:
                                # 파싱/변환 에러: 재시도 의미 없음
                                self._operator_registry.update(op_name, delta_fitness=0.0)
                                self._operator_registry.record_llm_failure()
                                return None

                    # 모든 재시도 실패
                    logger.warning(
                        "LLM %s failed after %d attempts: %s",
                        op_name, settings.ALPHA_LLM_RETRY_MAX + 1, last_err,
                    )
                    self._operator_registry.update(op_name, delta_fitness=0.0)
                    self._operator_registry.record_llm_failure()
                    return None

            llm_results = await asyncio.gather(
                *[_fire_llm(*t) for t in llm_coros],
                return_exceptions=True,
            )
            for r in llm_results:
                if r is not None and not isinstance(r, Exception):
                    ast_children.append(r)
                    llm_success_count += 1
                else:
                    funnel["operator_null"] += 1

            # LLM 전멸 폴백: AST 연산자로 부족분 보충
            if llm_success_count == 0 and len(llm_coros) > 0:
                shortfall = len(llm_coros)
                logger.warning(
                    "All %d LLM calls failed → generating %d extra AST children as fallback",
                    shortfall, shortfall,
                )
                _fallback_ops = ["ast_mutate_feature", "ast_mutate_operator", "ast_crossover"]
                for _ in range(shortfall):
                    fb_op = random.choice(_fallback_ops)
                    fb_parents = tournament_select(
                        population,
                        k=min(settings.ALPHA_TOURNAMENT_K, len(population)),
                        n_select=2, parsimony=True,
                    )
                    if fb_parents:
                        fb_expr = self._apply_ast_op(fb_parents, fb_op)
                        if fb_expr is not None:
                            fb_parent = fb_parents[0]
                            fb_pids = [fb_parent.factor_id] if fb_parent.factor_id else []
                            ast_children.append((fb_expr, fb_op, fb_parent, fb_pids))

        _t_phase2 = _time.perf_counter()
        all_children = ast_children
        logger.info("세대 %d: Phase 2 완료 (%.1fs) — 총 %d개 자식, to_thread 시작", self._generation, _t_phase2 - _t_phase1, len(all_children))

        if progress_cb:
            await progress_cb(
                30, 100,
                f"세대 {self._generation}: LLM 완료, {len(all_children)}개 평가 시작"
            )

        # 생성된 후보 수식 샘플을 프론트엔드에 전송 (Phase 3 전, OOM 크래시 전)
        if iteration_cb and all_children:
            import random as _rand
            sample_indices = _rand.sample(range(len(all_children)), min(8, len(all_children)))
            candidate_samples = []
            for idx in sample_indices:
                child_tuple = all_children[idx]
                expr = child_tuple[0]       # sympy expression
                op_name = child_tuple[1]    # operator name
                candidate_samples.append({
                    "expression": str(expr)[:80],
                    "operator": op_name,
                })
            # 연산자별 집계
            op_counts: dict[str, int] = {}
            for child_tuple in all_children:
                op = child_tuple[1]
                op_counts[op] = op_counts.get(op, 0) + 1

            await iteration_cb({
                "type": "candidates_ready",
                "generation": self._generation,
                "total": len(all_children),
                "samples": candidate_samples,
                "operator_breakdown": op_counts,
            })

        # ── Phase 3: CPU-bound 평가를 별도 스레드에서 실행 (이벤트 루프 해방) ──
        offspring, new_discovered, funnel = await asyncio.to_thread(
            self._evaluate_batch_sync,
            all_children, train_data, val_data,
            offspring_target, funnel,
            _t_phase0, _t_phase1, _t_phase2,
            full_data,
        )

        logger.info("세대 %d: Phase 3 to_thread 완료 — offspring %d개", self._generation, len(offspring))

        if progress_cb:
            await progress_cb(90, 100, f"세대 {self._generation}: 평가 완료")

        # Phase 3 결과 상세 이벤트 (프론트엔드 라이브 피드용)
        if iteration_cb:
            # IC 통과 상위 팩터 샘플 (최대 5개)
            ic_passed = [
                o for o in offspring
                if hasattr(o, "ic_mean") and o.ic_mean >= self._ic_threshold
            ]
            ic_passed.sort(key=lambda o: o.ic_mean, reverse=True)
            top_samples = [
                {
                    "expression": o.expression_str[:60],
                    "ic": round(o.ic_mean, 4),
                    "sharpe": round(o.sharpe, 2) if hasattr(o, "sharpe") else 0,
                    "operator": getattr(o, "operator_origin", ""),
                }
                for o in ic_passed[:5]
            ]
            # IC 미달 샘플 (최근 3개)
            ic_failed = [
                o for o in offspring
                if hasattr(o, "ic_mean") and o.ic_mean < self._ic_threshold
            ]
            fail_samples = [
                {
                    "expression": o.expression_str[:60],
                    "ic": round(o.ic_mean, 4),
                }
                for o in ic_failed[-3:]
            ]
            await iteration_cb({
                "type": "eval_complete",
                "generation": self._generation,
                "total_evaluated": len(offspring),
                "ic_pass_count": len(ic_passed),
                "ic_threshold": self._ic_threshold,
                "cpcv_candidates": funnel.get("cpcv_candidates", 0),
                "discovered_count": len(new_discovered),
                "top_samples": top_samples,
                "fail_samples": fail_samples,
                "discovered": [
                    {
                        "name": d.name,
                        "expression": d.expression_str[:60],
                        "ic": round(d.metrics.ic_mean, 4) if d.metrics else 0,
                        "sharpe": round(d.metrics.sharpe, 2) if d.metrics else 0,
                    }
                    for d in new_discovered
                ],
            })

        # 5. 중복 제거
        all_new = elites + offspring
        all_new = self._deduplicate(all_new)

        # 6. 모집단 크기 제한
        all_new.sort(key=lambda f: f.fitness_composite, reverse=True)
        final_population = all_new[:self._population_size]

        # 6.5. 벡터 메모리에 경험 저장
        if self._vector_memory and new_discovered:
            for disc in new_discovered:
                try:
                    await self._vector_memory.add(
                        db=self._db,
                        expression_str=disc.expression_str,
                        hypothesis=disc.hypothesis or "",
                        ic_mean=disc.metrics.ic_mean if disc.metrics else 0.0,
                        generation=self._generation,
                        success=True,
                    )
                except Exception as e:
                    logger.debug("Failed to save experience: %s", e)

        # 7. DB 업데이트
        await self._persist_population(final_population)

        if progress_cb:
            await progress_cb(100, 100, f"세대 {self._generation} 완료: {len(new_discovered)}개 발견")

        if iteration_cb:
            await iteration_cb({
                "type": "generation_complete",
                "generation": self._generation,
                "population_size": len(final_population),
                "elite_count": len(elites),
                "new_discovered": len(new_discovered),
                "operator_stats": self._operator_registry.to_dict(),
                "funnel": {
                    "attempted": offspring_target,
                    "eval_ok": funnel["eval_success"],
                    "ic_pass": funnel["ic_pass"],
                    "wf_overfit": funnel["wf_overfit"],
                    "sharpe_fail": funnel["sharpe_fail"],
                    "cpcv_candidates": funnel.get("cpcv_candidates", 0),
                },
            })

        return new_discovered

    def _evaluate_batch_sync(
        self,
        all_children: list[tuple],
        train_data: pl.DataFrame,
        val_data: pl.DataFrame | None,
        offspring_target: int,
        funnel: dict,
        t_phase0: float,
        t_phase1: float,
        t_phase2: float,
        full_data: pl.DataFrame | None = None,
    ) -> tuple[list, list, dict]:
        """Phase 3: 배치 평가 + Walk-Forward + CPCV (CPU-bound, 별도 스레드에서 실행).

        Returns (offspring, new_discovered, funnel)
        """
        import time as _time
        from app.alpha.evaluator import compute_ic_series_batch
        from app.alpha.interval import default_round_trip_cost, is_intraday

        offspring: list[ScoredFactor] = []
        new_discovered: list[DiscoveredFactor] = []
        cpcv_candidates: list[ScoredFactor] = []

        _BATCH_SIZE = settings.ALPHA_EVAL_BATCH_SIZE
        _is_intraday = is_intraday(self._interval)
        _rtc = default_round_trip_cost(self._interval)

        for batch_start in range(0, len(all_children), _BATCH_SIZE):
            batch = all_children[batch_start:batch_start + _BATCH_SIZE]

            # Step 1: 배치 with_columns (유효 식만)
            valid_batch: list[tuple[int, tuple]] = []
            factor_exprs: list[pl.Expr] = []
            for i, child_tuple in enumerate(batch):
                child_expr = child_tuple[0]
                try:
                    pe = sympy_to_polars(child_expr)
                    factor_exprs.append(pe.alias(f"_bf_{i}"))
                    valid_batch.append((i, child_tuple))
                except (ASTConversionError, Exception):
                    op_name = child_tuple[1]
                    self._operator_registry.update(op_name, delta_fitness=0.0)
                    funnel["operator_null"] += 1

            if not valid_batch:
                continue

            # Step 2+3: 분봉/일봉 분기
            if _is_intraday:
                # 분봉: per-factor 개별 평가 (배치 with_columns 불필요 — 메모리 절약)
                # 필요 컬럼만 선택해서 메모리 압축 (32컬럼 → 최소 컬럼)
                _eval_cols = {"symbol", "dt", "open", "high", "low", "close", "volume", "fwd_return"}
                for idx, (i, child_tuple) in enumerate(valid_batch):
                    child_expr, op_name, parent, parent_ids = child_tuple
                    # 수식에 필요한 피처 컬럼 추출 (SymPy 변수명 → Polars 컬럼명 변환)
                    try:
                        from app.alpha.ast_converter import _ALL_VARIABLES
                        sym_names = {str(s) for s in child_expr.free_symbols}
                        # SymPy 변수명을 Polars 컬럼명으로 변환 (e.g. earnings_per_share → eps)
                        needed = set()
                        for name in sym_names:
                            col = _ALL_VARIABLES.get(name, name)
                            needed.add(col)
                        needed &= set(train_data.columns)
                    except Exception:
                        needed = set()
                    select_cols = list((_eval_cols | needed) & set(train_data.columns))
                    slim_data = train_data.select(select_cols)
                    metrics = self._evaluate_on_data_full(child_expr, slim_data)
                    del slim_data
                    scored = self._build_scored_factor(
                        child_expr, op_name, parent, parent_ids, metrics, funnel,
                    )
                    if scored is not None:
                        offspring.append(scored)
                    # 메모리 회수: GC + OS에 회수 시간 부여
                    gc.collect()
                    if idx % 3 == 2:
                        _time.sleep(0.2)
                continue

            # 일봉: 배치 with_columns + 배치 IC (실패 시 per-factor 폴백)
            try:
                df_batch = train_data.with_columns(factor_exprs)
            except Exception as e:
                logger.warning(
                    "Batch with_columns failed (size=%d): %s → per-factor fallback",
                    len(factor_exprs), e,
                )
                for i, child_tuple in valid_batch:
                    child_expr, op_name, parent, parent_ids = child_tuple
                    metrics = self._evaluate_on_data_full(child_expr, train_data)
                    scored = self._build_scored_factor(
                        child_expr, op_name, parent, parent_ids, metrics, funnel,
                    )
                    if scored is not None:
                        offspring.append(scored)
                continue

            # 배치 IC 계산 (실패 시 per-factor 폴백)
            factor_cols = [f"_bf_{i}" for i, _ in valid_batch]
            try:
                ic_dict = compute_ic_series_batch(df_batch, factor_cols)
            except Exception as e:
                logger.warning("Batch IC failed: %s → per-factor fallback", e)
                ic_dict = {}
                for col in factor_cols:
                    try:
                        ic_dict[col] = compute_ic_series(df_batch, factor_col=col)
                    except Exception:
                        ic_dict[col] = []

            for i, child_tuple in valid_batch:
                child_expr, op_name, parent, parent_ids = child_tuple
                col = f"_bf_{i}"
                ic_series = ic_dict.get(col, [])
                ls_returns = compute_quantile_returns(df_batch, factor_col=col)
                metrics = compute_factor_metrics(
                    ic_series, ls_returns=ls_returns,
                    annualize=252.0,
                    round_trip_cost=_rtc,
                )
                scored = self._build_scored_factor(
                    child_expr, op_name, parent, parent_ids, metrics, funnel,
                )
                if scored is not None:
                    offspring.append(scored)

            # 배치 완료 후 임시 DataFrame 해제 + GC
            del df_batch
            gc.collect()

        _t_phase3 = _time.perf_counter()

        # Phase 3b: IC 통과 → Walk-Forward + CPCV 후보 수집
        for scored in offspring:
            if scored.ic_mean >= self._ic_threshold:
                funnel["ic_pass"] += 1

                if val_data is not None and val_data.height > 30:
                    funnel["wf_tested"] += 1
                    val_ic = self._evaluate_on_data(scored.expression, val_data)
                    if self._is_overfit(scored.ic_mean, val_ic):
                        funnel["wf_overfit"] += 1
                        if funnel["wf_overfit"] <= 3:
                            logger.info(
                                "WF overfit sample: expr=%s, train_ic=%.4f, val_ic=%.4f (threshold=%.4f)",
                                str(scored.expression)[:60],
                                scored.ic_mean, val_ic, self._ic_threshold * 0.5,
                            )
                        continue

                if scored.sharpe < settings.ALPHA_SHARPE_THRESHOLD:
                    funnel["sharpe_fail"] += 1
                    continue

                cpcv_candidates.append(scored)

        # CPCV 2단계 검증: top-K 후보만
        if cpcv_candidates:
            from app.alpha.cpcv import cpcv_validate

            cpcv_candidates.sort(key=lambda c: c.fitness_composite, reverse=True)
            cpcv_top_k = min(20, len(cpcv_candidates))

            for child in cpcv_candidates[:cpcv_top_k]:
                cpcv_result = cpcv_validate(
                    full_data, child.expression,
                    ic_threshold=self._ic_threshold,
                )
                if cpcv_result.passed:
                    disc_metrics = FactorMetrics(
                        ic_mean=child.ic_mean,
                        ic_std=child.ic_std,
                        icir=child.icir,
                        turnover=child.turnover,
                        sharpe=child.sharpe,
                        max_drawdown=child.max_drawdown,
                        ic_series=[],
                    )
                    df = DiscoveredFactor(
                        name=f"evo_g{self._generation}_{len(new_discovered)}",
                        expression_str=child.expression_str,
                        expression_sympy=str(child.expression),
                        polars_code=self._safe_code_string(child.expression),
                        hypothesis=child.hypothesis,
                        generation=self._generation,
                        metrics=disc_metrics,
                        parent_ids=child.parent_ids,
                    )
                    new_discovered.append(df)
                    logger.info(
                        "CPCV passed: %s (mean_ic=%.4f, pbo=%.2f)",
                        child.expression_str[:50], cpcv_result.mean_ic, cpcv_result.pbo,
                    )
                else:
                    logger.debug(
                        "CPCV rejected: %s (reason=%s, mean_ic=%.4f, pbo=%.2f)",
                        child.expression_str[:50], cpcv_result.reason,
                        cpcv_result.mean_ic, cpcv_result.pbo,
                    )

        # cpcv_candidates 수를 funnel에 기록 (run_generation의 iteration_cb에서 참조)
        funnel["cpcv_candidates"] = len(cpcv_candidates)

        # 퍼널 + 타이밍 로그
        sharpe_pass = funnel["ic_pass"] - funnel["wf_overfit"] - funnel["sharpe_fail"]
        logger.warning(
            "세대 %d 퍼널: attempted=%d → eval_ok=%d → IC≥%.2f=%d "
            "→ WF_tested=%d (overfit=%d) → Sharpe≥%.1f=%d → CPCV=%d → discovered=%d",
            self._generation,
            offspring_target, funnel["eval_success"],
            self._ic_threshold, funnel["ic_pass"],
            funnel["wf_tested"], funnel["wf_overfit"],
            settings.ALPHA_SHARPE_THRESHOLD, max(sharpe_pass, 0),
            len(cpcv_candidates), len(new_discovered),
        )
        logger.warning(
            "세대 %d 타이밍: 생성=%.1fs, LLM=%.1fs, 평가=%.1fs, 총=%.1fs",
            self._generation,
            t_phase1 - t_phase0, t_phase2 - t_phase1,
            _t_phase3 - t_phase2, _t_phase3 - t_phase0,
        )

        del full_data
        return offspring, new_discovered, funnel

    def _select_elites(self, population: list[ScoredFactor]) -> list[ScoredFactor]:
        """상위 elite_pct% 개체를 엘리트로 보존."""
        if not population:
            return []
        n_elite = max(1, math.ceil(len(population) * self._elite_pct))
        sorted_pop = sorted(population, key=lambda f: f.fitness_composite, reverse=True)
        return sorted_pop[:n_elite]

    def _split_train_val(self) -> tuple[pl.DataFrame, pl.DataFrame | None]:
        """데이터를 Train(70%) / Validation(30%) 분할."""
        dates = self._data.select("dt").unique().sort("dt")
        date_list = dates["dt"].to_list()

        if len(date_list) < 20:
            return self._data, None

        split_idx = int(len(date_list) * self._train_ratio)
        split_date = date_list[split_idx]

        train = self._data.filter(pl.col("dt") <= split_date).rechunk().shrink_to_fit()
        val = self._data.filter(pl.col("dt") > split_date).rechunk().shrink_to_fit()

        return train, val if val.height > 0 else None

    def _is_overfit(self, train_ic: float, val_ic: float) -> bool:
        """과적합 판단: Val IC < threshold × 0.5면 과적합."""
        return val_ic < self._ic_threshold * 0.5

    async def _seed_population(self, count: int) -> list[ScoredFactor]:
        """초기 모집단 생성: 간단한 수식 시드."""
        seeds: list[ScoredFactor] = []
        # 기본 수식 템플릿 (확장 피처 포함)
        templates = [
            # 기존
            "rsi",
            "volume_ratio",
            "macd_hist",
            "close / sma_20 - 1",
            "(close - bb_lower) / (bb_upper - bb_lower)",
            "log(volume_ratio) * rsi",
            "price_change_pct * volume_ratio",
            "(close - sma_20) / atr_14",
            "macd_hist * volume_ratio",
            "rsi * (close / sma_20 - 1)",
            "log(volume_ratio) * (30 - rsi) / atr_14",
            "ema_20 / sma_20 - 1",
            "price_change_pct / atr_14",
            "abs(close - sma_20) / sma_20",
            "(high - low) / close",
            "volume_ratio * price_change_pct * rsi",
            "(bb_upper - bb_lower) / sma_20",
            "macd_hist / atr_14",
            "log(volume_ratio) * macd_hist",
            "rsi / 100 * volume_ratio",
            # 멀티 윈도우 추세
            "sma_5 / sma_60 - 1",
            "ema_5 / ema_20 - 1",
            "(sma_5 - sma_20) / atr_14",
            # 멀티 RSI
            "rsi_7 - rsi_21",
            "(100 - rsi_7) * volume_ratio",
            # 모멘텀 + 변동성
            "return_5d / atr_7",
            "return_20d * volume_ratio",
            "return_5d - return_20d",
            # 시차 기반 평균회귀
            "close / close_lag_5 - 1",
            "(close - close_lag_20) / atr_21",
            "log(volume / volume_lag_5)",
            # 밴드 위치
            "bb_position * rsi / 100",
            "bb_position * volume_ratio",
            # 복합
            "(ema_10 - sma_60) / atr_14 * volume_ratio",
            "abs(return_5d) / atr_7 * rsi_7 / 100",
        ]

        train_data, _ = self._split_train_val()

        for i, tmpl in enumerate(templates[:count]):
            try:
                expr = parse_expression(tmpl)
                metrics = self._evaluate_on_data_full(expr, train_data)
                if metrics is None:
                    continue

                depth = calc_tree_depth(expr)
                size = calc_tree_size(expr)
                fitness = compute_composite_fitness(
                    ic_mean=metrics.ic_mean,
                    icir=metrics.icir,
                    turnover=metrics.turnover,
                    tree_depth=depth,
                    tree_size=size,
                    sharpe=metrics.sharpe,
                    max_drawdown=metrics.max_drawdown,
                )

                seeds.append(ScoredFactor(
                    expression=expr,
                    expression_str=tmpl,
                    hypothesis=generate_hypothesis_korean(expr, "initial"),
                    ic_mean=metrics.ic_mean,
                    ic_std=metrics.ic_std,
                    icir=metrics.icir,
                    turnover=metrics.turnover,
                    sharpe=metrics.sharpe,
                    max_drawdown=metrics.max_drawdown,
                    generation=self._generation,
                    fitness_composite=fitness,
                    tree_depth=depth,
                    tree_size=size,
                    expression_hash=expression_hash(expr),
                    operator_origin="seed",
                ))
            except (ASTConversionError, Exception) as e:
                logger.debug("Seed %s failed: %s", tmpl, e)

        # 시드 결과 로깅
        logger.warning(
            "시드 모집단: %d/%d 성공 (실패=%d)",
            len(seeds), count, count - len(seeds),
        )
        if seeds:
            ic_values = [s.ic_mean for s in seeds]
            logger.warning(
                "시드 IC 분포: min=%.4f, max=%.4f, mean=%.4f, IC≥%.2f=%d개",
                min(ic_values), max(ic_values),
                sum(ic_values) / len(ic_values),
                self._ic_threshold,
                sum(1 for ic in ic_values if ic >= self._ic_threshold),
            )

        return seeds

    async def _apply_operator(
        self,
        population: list[ScoredFactor],
        operator_name: str,
        train_data: pl.DataFrame,
    ) -> ScoredFactor | None:
        """연산자 적용 → 평가 → ScoredFactor 반환."""
        try:
            # 부모 선택
            parents = tournament_select(
                population,
                k=min(settings.ALPHA_TOURNAMENT_K, len(population)),
                n_select=2,
                parsimony=True,
            )
            if not parents:
                return None
            parent = parents[0]

            # 연산자 실행
            child_expr = None
            parent_ids = [parent.factor_id] if parent.factor_id else []

            if operator_name == "ast_crossover":
                if len(parents) >= 2:
                    children = crossover(parent.expression, parents[1].expression)
                    if children:
                        child_expr = children[0]
                        if parents[1].factor_id:
                            parent_ids.append(parents[1].factor_id)
            elif operator_name == "ast_hoist":
                child_expr = hoist_mutation(parent.expression)
            elif operator_name == "ast_ephemeral_constant":
                child_expr = ephemeral_constant_mutation(parent.expression)
            elif operator_name == "ast_mutate_operator":
                from app.alpha.evolution import _mutate_operator
                child_expr = _mutate_operator(parent.expression)
            elif operator_name == "ast_mutate_constant":
                from app.alpha.evolution import _mutate_constant
                child_expr = _mutate_constant(parent.expression)
            elif operator_name == "ast_mutate_feature":
                from app.alpha.evolution import _mutate_feature
                child_expr = _mutate_feature(parent.expression)
            elif operator_name == "ast_mutate_function":
                from app.alpha.evolution import _mutate_function
                child_expr = _mutate_function(parent.expression)
            elif operator_name == "llm_seed":
                child_expr = await self._llm_seed()
            elif operator_name == "llm_mutate":
                child_expr = await self._llm_mutate(parent)
            else:
                child_expr = mutate(parent.expression)

            if child_expr is None:
                self._operator_registry.update(operator_name, delta_fitness=0.0)
                return None

            # Polars 변환 확인
            sympy_to_polars(child_expr)

            # 평가
            metrics = self._evaluate_on_data_full(child_expr, train_data)
            if metrics is None:
                if self._eval_fail_logged < 3:
                    self._eval_fail_logged += 1
                    logger.warning(
                        "eval_full → None (sample %d/3): op=%s, expr=%s",
                        self._eval_fail_logged, operator_name,
                        str(child_expr)[:80],
                    )
                self._operator_registry.update(operator_name, delta_fitness=0.0)
                return None

            depth = calc_tree_depth(child_expr)
            size = calc_tree_size(child_expr)
            fitness = compute_composite_fitness(
                ic_mean=metrics.ic_mean,
                icir=metrics.icir,
                turnover=metrics.turnover,
                tree_depth=depth,
                tree_size=size,
                sharpe=metrics.sharpe,
                max_drawdown=metrics.max_drawdown,
            )

            # UCB1 업데이트
            delta = fitness - parent.fitness_composite
            self._operator_registry.update(operator_name, delta_fitness=delta)

            return ScoredFactor(
                expression=child_expr,
                expression_str=str(child_expr),
                hypothesis=generate_hypothesis_korean(child_expr, operator_name),
                ic_mean=metrics.ic_mean,
                ic_std=metrics.ic_std,
                icir=metrics.icir,
                turnover=metrics.turnover,
                sharpe=metrics.sharpe,
                max_drawdown=metrics.max_drawdown,
                generation=self._generation,
                parent_ids=parent_ids,
                fitness_composite=fitness,
                tree_depth=depth,
                tree_size=size,
                expression_hash=expression_hash(child_expr),
                operator_origin=operator_name,
            )

        except (ASTConversionError, Exception) as e:
            logger.debug("Operator %s failed: %s", operator_name, e)
            self._operator_registry.update(operator_name, delta_fitness=0.0)
            if operator_name.startswith("llm_"):
                self._operator_registry.record_llm_failure()
            return None

    def _apply_ast_op(
        self, parents: list[ScoredFactor], operator_name: str,
    ) -> sympy.Basic | None:
        """AST 연산자 적용 (동기, ~1ms). LLM 연산자는 처리하지 않는다."""
        try:
            parent = parents[0]
            if operator_name == "ast_crossover":
                if len(parents) >= 2:
                    children = crossover(parent.expression, parents[1].expression)
                    return children[0] if children else None
                return None
            elif operator_name == "ast_hoist":
                return hoist_mutation(parent.expression)
            elif operator_name == "ast_ephemeral_constant":
                return ephemeral_constant_mutation(parent.expression)
            elif operator_name == "ast_mutate_operator":
                from app.alpha.evolution import _mutate_operator
                return _mutate_operator(parent.expression)
            elif operator_name == "ast_mutate_constant":
                from app.alpha.evolution import _mutate_constant
                return _mutate_constant(parent.expression)
            elif operator_name == "ast_mutate_feature":
                from app.alpha.evolution import _mutate_feature
                return _mutate_feature(parent.expression)
            elif operator_name == "ast_mutate_function":
                from app.alpha.evolution import _mutate_function
                return _mutate_function(parent.expression)
            else:
                return mutate(parent.expression)
        except Exception as e:
            logger.debug("AST operator %s failed: %s", operator_name, e)
            return None

    def _build_scored_factor(
        self,
        child_expr: sympy.Basic,
        op_name: str,
        parent: ScoredFactor,
        parent_ids: list[str],
        metrics: FactorMetrics | None,
        funnel: dict,
    ) -> ScoredFactor | None:
        """메트릭으로 ScoredFactor 생성 + UCB1 업데이트. 실패 시 None."""
        if metrics is None or not metrics.ic_series:
            if self._eval_fail_logged < 3:
                self._eval_fail_logged += 1
                logger.warning(
                    "eval_full → None (sample %d/3): op=%s, expr=%s",
                    self._eval_fail_logged, op_name, str(child_expr)[:80],
                )
            self._operator_registry.update(op_name, delta_fitness=0.0)
            return None

        depth = calc_tree_depth(child_expr)
        size = calc_tree_size(child_expr)
        fitness = compute_composite_fitness(
            ic_mean=metrics.ic_mean,
            icir=metrics.icir,
            turnover=metrics.turnover,
            tree_depth=depth,
            tree_size=size,
            sharpe=metrics.sharpe,
            max_drawdown=metrics.max_drawdown,
        )

        delta = fitness - parent.fitness_composite
        self._operator_registry.update(op_name, delta_fitness=delta)
        funnel["eval_success"] += 1

        return ScoredFactor(
            expression=child_expr,
            expression_str=str(child_expr),
            hypothesis=generate_hypothesis_korean(child_expr, op_name),
            ic_mean=metrics.ic_mean,
            ic_std=metrics.ic_std,
            icir=metrics.icir,
            turnover=metrics.turnover,
            sharpe=metrics.sharpe,
            max_drawdown=metrics.max_drawdown,
            generation=self._generation,
            parent_ids=parent_ids,
            fitness_composite=fitness,
            tree_depth=depth,
            tree_size=size,
            expression_hash=expression_hash(child_expr),
            operator_origin=op_name,
        )

    # eval 실패 샘플 로깅 카운터 (세대당 리셋)
    _eval_fail_logged: int = 0

    # 확장된 피처 목록 (LLM 프롬프트용)
    _FEATURE_LIST = (
        "close, open, high, low, volume, "
        "sma_5, sma_10, sma_20, sma_60, ema_5, ema_10, ema_20, ema_60, "
        "rsi, rsi_7, rsi_21, "
        "volume_ratio, atr_7, atr_14, atr_21, "
        "macd_hist, bb_upper, bb_lower, bb_width, bb_position, "
        "price_change_pct, "
        "close_lag_1, close_lag_5, close_lag_20, "
        "volume_lag_1, volume_lag_5, "
        "return_5d, return_20d, "
        "rank_close, rank_volume, zscore_close, zscore_volume, "
        "foreign_net_norm, inst_net_norm, retail_net_norm, "
        "eps, bps, operating_margin, debt_to_equity, earnings_yield, book_yield"
    )

    async def _llm_seed(self) -> sympy.Basic | None:
        """Claude API로 새 수식 생성."""
        try:
            from anthropic import AsyncAnthropic

            client = AsyncAnthropic()
            prompt = (
                "Generate a single alpha factor formula using these variables: "
                f"{self._FEATURE_LIST}.\n"
                "Use only: +, -, *, /, log(), sqrt(), abs().\n"
                "Return ONLY the formula, nothing else."
            )

            # RAG 경험 추가
            if self._vector_memory:
                rag = self._vector_memory.format_rag_context(
                    self._context or "alpha factor for Korean equities"
                )
                if rag and rag != "아직 탐색 이력이 없습니다.":
                    prompt += f"\n\nPast experience:\n{rag}"

            # 구조화된 피드백이 context에 포함된 경우 RAG 뒤에 배치
            if self._context:
                prompt += f"\n\n{self._context}"

            response = await client.messages.create(
                model=settings.AGENT_MODEL,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            formula = response.content[0].text.strip()
            return parse_expression(formula)
        except Exception as e:
            logger.debug("LLM seed failed: %s", e)
            return None

    async def _llm_mutate(self, parent: ScoredFactor) -> sympy.Basic | None:
        """Claude API로 기존 수식 변이 (다차원 메트릭 피드백)."""
        try:
            from anthropic import AsyncAnthropic

            client = AsyncAnthropic()
            prompt = (
                f"Mutate this alpha factor formula to improve it:\n"
                f"{parent.expression_str}\n"
                f"Current metrics:\n"
                f"  IC: {parent.ic_mean:.4f} (target: >= {self._ic_threshold})\n"
                f"  Sharpe: {parent.sharpe:.3f} (target: >= {settings.ALPHA_SHARPE_THRESHOLD})\n"
                f"  Turnover: {parent.turnover:.3f} (lower is better)\n"
                f"  Complexity: depth={parent.tree_depth}, size={parent.tree_size}\n"
                f"Use only these variables: {self._FEATURE_LIST}.\n"
                "Use only: +, -, *, /, log(), sqrt(), abs().\n"
                "Return ONLY the new formula, nothing else."
            )

            # RAG: 유사한 과거 시도
            if self._vector_memory:
                rag = self._vector_memory.format_rag_context(
                    parent.expression_str
                )
                if rag and rag != "아직 탐색 이력이 없습니다.":
                    prompt += f"\n\nSimilar past attempts:\n{rag}"

            response = await client.messages.create(
                model=settings.AGENT_MODEL,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            formula = response.content[0].text.strip()
            return parse_expression(formula)
        except Exception as e:
            logger.debug("LLM mutate failed: %s", e)
            return None

    def _evaluate_on_data(self, expr: sympy.Basic, data: pl.DataFrame) -> float:
        """수식의 IC mean을 계산 (간략 버전 — Walk-Forward 검증용)."""
        try:
            polars_expr = sympy_to_polars(expr)
            df = data.with_columns(polars_expr.alias("alpha_factor"))

            from app.alpha.interval import is_intraday

            if is_intraday(self._interval):
                from app.alpha.evaluator import _collapse_to_daily
                df = _collapse_to_daily(df, factor_col="alpha_factor")
                df = df.drop_nulls(subset=["alpha_factor", "fwd_return"])
            else:
                if "fwd_return" not in df.columns:
                    df = compute_forward_returns(df, periods=1)
                df = df.filter(
                    pl.col("alpha_factor").is_not_null()
                    & pl.col("alpha_factor").is_not_nan()
                    & pl.col("fwd_return").is_not_null()
                    & pl.col("fwd_return").is_not_nan()
                )

            ic_series = compute_ic_series(df, factor_col="alpha_factor")
            if not ic_series:
                return 0.0
            import numpy as np
            return float(np.mean(ic_series))
        except Exception:
            return 0.0

    def _evaluate_on_data_full(
        self, expr: sympy.Basic, data: pl.DataFrame,
    ) -> FactorMetrics | None:
        """수식의 전체 메트릭 계산 (evaluate_factor()와 동일한 방법론)."""
        try:
            polars_expr = sympy_to_polars(expr)

            from app.alpha.interval import default_round_trip_cost, is_intraday

            if is_intraday(self._interval):
                from app.alpha.evaluator import _collapse_to_daily
                # with_columns + select 체이닝: 큰 중간 DataFrame 방지
                df = (
                    data.with_columns(polars_expr.alias("alpha_factor"))
                    .select(["symbol", "dt", "close", "alpha_factor"])
                )
                del data  # 호출자의 slim_data 참조 해제 촉진
                df = _collapse_to_daily(df, factor_col="alpha_factor")
                df = df.drop_nulls(subset=["alpha_factor", "fwd_return"])
            else:
                df = data.with_columns(polars_expr.alias("alpha_factor"))
                if "fwd_return" not in df.columns:
                    df = compute_forward_returns(df, periods=1)
                df = df.filter(
                    pl.col("alpha_factor").is_not_null()
                    & pl.col("alpha_factor").is_not_nan()
                    & pl.col("fwd_return").is_not_null()
                    & pl.col("fwd_return").is_not_nan()
                )

            if df.height < 10:
                return None

            ic_series = compute_ic_series(df, factor_col="alpha_factor")
            ls_returns = compute_quantile_returns(df, factor_col="alpha_factor")
            return compute_factor_metrics(
                ic_series, ls_returns=ls_returns,
                annualize=252.0,  # 항상 일별 기준 (분봉은 _collapse_to_daily로 일별화)
                round_trip_cost=default_round_trip_cost(self._interval),
            )
        except Exception as e:
            if self._eval_fail_logged < 5:
                logger.warning("Evaluation failed (sample %d/5): %s: %s", self._eval_fail_logged + 1, type(e).__name__, e)
                self._eval_fail_logged += 1
            return None

    def _deduplicate(self, population: list[ScoredFactor]) -> list[ScoredFactor]:
        """expression_hash 기반 중복 제거. 동일 구조 중 fitness 최고만 유지."""
        seen: dict[str, ScoredFactor] = {}
        for factor in population:
            h = factor.expression_hash
            if not h:
                h = expression_hash(factor.expression)
                factor.expression_hash = h
            if h not in seen or factor.fitness_composite > seen[h].fitness_composite:
                seen[h] = factor
        return list(seen.values())

    async def _persist_population(self, population: list[ScoredFactor]) -> None:
        """모집단을 DB에 영속화."""
        # 기존 활성 모집단 비활성화
        await self._db.execute(
            update(AlphaFactor)
            .where(AlphaFactor.population_active == True)  # noqa: E712
            .values(population_active=False, is_elite=False)
        )

        # 엘리트 집합 미리 계산 (id() 기반 O(1) 조회)
        elite_list = self._select_elites(population)
        elite_obj_ids = {id(f) for f in elite_list}

        # 새 모집단 저장/업데이트
        for factor in population:
            is_elite_now = id(factor) in elite_obj_ids
            if factor.factor_id:
                # 기존 팩터 업데이트
                await self._db.execute(
                    update(AlphaFactor)
                    .where(AlphaFactor.id == uuid.UUID(factor.factor_id))
                    .values(
                        population_active=True,
                        fitness_composite=factor.fitness_composite,
                        ic_mean=factor.ic_mean,
                        ic_std=factor.ic_std,
                        icir=factor.icir,
                        turnover=factor.turnover,
                        sharpe=factor.sharpe,
                        max_drawdown=factor.max_drawdown,
                        is_elite=is_elite_now,
                        birth_generation=factor.generation,
                        interval=self._interval,
                    )
                )
            else:
                # 신규 팩터 삽입
                new_factor = AlphaFactor(
                    mining_run_id=uuid.UUID(self._current_run_id) if self._current_run_id else None,
                    name=f"evo_g{self._generation}",
                    expression_str=factor.expression_str,
                    expression_sympy=str(factor.expression),
                    polars_code=self._safe_code_string(factor.expression),
                    hypothesis=factor.hypothesis,
                    generation=factor.generation,
                    ic_mean=factor.ic_mean,
                    ic_std=factor.ic_std,
                    icir=factor.icir,
                    turnover=factor.turnover,
                    sharpe=factor.sharpe,
                    max_drawdown=factor.max_drawdown,
                    fitness_composite=factor.fitness_composite,
                    tree_depth=factor.tree_depth,
                    tree_size=factor.tree_size,
                    expression_hash=factor.expression_hash,
                    operator_origin=factor.operator_origin,
                    population_active=True,
                    is_elite=is_elite_now,
                    birth_generation=self._generation,
                    status="population",
                    interval=self._interval,
                    parent_ids=factor.parent_ids if factor.parent_ids else None,
                )
                self._db.add(new_factor)
                await self._db.flush()
                factor.factor_id = str(new_factor.id)

        await self._db.commit()

    @staticmethod
    def _safe_code_string(expr: sympy.Basic) -> str | None:
        """안전한 코드 문자열 변환."""
        try:
            return sympy_to_code_string(expr)
        except Exception:
            return None

    @property
    def generation(self) -> int:
        """현재 세대 번호."""
        return self._generation
