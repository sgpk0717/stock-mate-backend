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
from sqlalchemy import func, or_ as db_or, select, update
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
    compute_long_only_returns,
    compute_position_turnover,
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

        # MAP-Elites 행동 아카이브: 8 family × 4 complexity = 32셀 그리드
        # 각 셀의 최고 fitness 팩터를 로드하고, 활성 모집단에서 과소 대표된 패밀리 우선 선택
        archive_injected = await self._load_map_elites_archive(
            existing_ids={f.id for f in factors},
            active_factors=factors,
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
                    genotypic_age=f.genotypic_age if hasattr(f, "genotypic_age") else 0,
                ))
            except (ASTConversionError, Exception) as e:
                logger.debug("Skipping unparseable factor %s: %s", f.id, e)

        # 아카이브 팩터: 감쇄 적합도 공식 (기존 중앙값 리셋 대체)
        # f_reinjected = f_archived × exp(-λ × age_gens), floor = 25th percentile
        fitness_floor = 0.0
        if population:
            sorted_fitness = sorted(f.fitness_composite for f in population)
            fitness_floor = sorted_fitness[len(sorted_fitness) // 4]  # 25th percentile

        for f in archive_injected:
            try:
                expr = parse_expression(f.expression_str)
                # 감쇄 적합도: 원래 fitness를 세대 경과에 따라 감쇄
                original_fitness = f.fitness_composite or 0.0
                age_gens = max(0, self._generation - (f.birth_generation or 0))
                decay = math.exp(-0.05 * age_gens)  # λ=0.05
                decayed_fitness = max(fitness_floor, original_fitness * decay)
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
                    fitness_composite=decayed_fitness,
                    tree_depth=f.tree_depth or 0,
                    tree_size=f.tree_size or 0,
                    expression_hash=f.expression_hash or "",
                    operator_origin=f.operator_origin or "archive",
                    genotypic_age=0,  # AFPO: 아카이브 재주입 = 새 유전자 취급
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

        # 2b. 수렴 감지 → LLM 재시드 주입
        self._diversity_injection_needed = False  # Phase 3 후 시드 주입 플래그
        if len(population) >= 10:
            from collections import Counter as _Counter

            # 조건 1 (기존): 단일 피처가 top 10 중 8개 이상
            _feat_counts = _Counter()
            _top10 = sorted(population, key=lambda x: x.fitness_composite, reverse=True)[:10]
            for _f in _top10:
                for _sym in _f.expression.free_symbols:
                    _feat_counts[str(_sym)] += 1
            _dominant = _feat_counts.most_common(1)
            _feat_converged = _dominant and _dominant[0][1] >= 8

            # 조건 2 (신규): 니치 분포에서 하나의 패밀리가 60% 이상
            from app.alpha.ast_converter import classify_niche as _classify
            _niche_counts = _Counter(_classify(f.expression) for f in population)
            _max_niche_pct = max(_niche_counts.values()) / len(population) if population else 0
            _niche_converged = _max_niche_pct >= 0.6

            if _feat_converged or _niche_converged:
                _reason = []
                if _feat_converged:
                    _reason.append(f"피처 '{_dominant[0][0]}' top10 중 {_dominant[0][1]}개")
                if _niche_converged:
                    _top_niche = _niche_counts.most_common(1)[0]
                    _reason.append(f"니치 '{_top_niche[0]}' {_max_niche_pct:.0%}")
                logger.warning(
                    "세대 %d: 수렴 감지 — %s. LLM 시드 + 다양성 시드 주입.",
                    self._generation, " / ".join(_reason),
                )
                self._diversity_injection_needed = True

                # 기존 LLM 시드 주입 (5개)
                _injected = 0
                for _ in range(5):
                    try:
                        _seed_expr = await self._llm_seed()
                        if _seed_expr is not None:
                            _injected += 1
                    except Exception:
                        pass
                if _injected > 0:
                    logger.info("세대 %d: LLM 시드 %d개 주입 완료", self._generation, _injected)

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
                if op_name == "llm_seed":
                    coro = self._llm_seed()
                elif op_name == "llm_crossover":
                    # 시맨틱 교차: 서로 다른 패밀리의 부모 2개 사용
                    if len(parents) >= 2:
                        parent_ids = parent_ids + ([parents[1].factor_id] if parents[1].factor_id else [])
                    coro = self._llm_semantic_crossover(parent, parents[1] if len(parents) >= 2 else parent)
                else:
                    coro = self._llm_mutate(parent)
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
                                if op_name == "llm_seed":
                                    active_coro = self._llm_seed()
                                elif op_name == "llm_crossover":
                                    active_coro = self._llm_semantic_crossover(parent, parent)
                                else:
                                    active_coro = self._llm_mutate(parent)
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

        # ★ 자식 0개 시 Phase 3 스킵 (0초 빈 실행 방지)
        if not all_children:
            logger.warning(
                "세대 %d: 후보 자식 0개 (operator_null=%d, LLM disabled=%s). Phase 3 스킵.",
                self._generation,
                funnel.get("operator_null", 0),
                self._operator_registry.is_llm_disabled(),
            )
            if progress_cb:
                await progress_cb(100, 100, f"세대 {self._generation}: 자식 생성 실패")
            return []

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

        # 4.5. 다양성 시드 주입 (수렴 감지 시에만)
        if getattr(self, '_diversity_injection_needed', False):
            diversity_seeds = self._make_diversity_seeds(offspring)
            if diversity_seeds:
                offspring.extend(diversity_seeds)
                logger.warning(
                    "세대 %d: 다양성 시드 %d개 offspring에 주입",
                    self._generation, len(diversity_seeds),
                )

        # 5. 중복 제거
        all_new = elites + offspring
        all_new = self._deduplicate(all_new)

        # 5.5. 표현형 클러스터 갱신 (10세대마다)
        if self._generation % 10 == 0 and train_data is not None:
            try:
                self._update_phenotypic_clusters(all_new, train_data)
            except Exception as e:
                logger.debug("Phenotypic cluster update failed: %s", e)

        # 6. 모집단 크기 제한 (niche-based diversity cap)
        final_population = self._niche_cap_trim(all_new, self._population_size)

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

        _BATCH_SIZE = getattr(self, "_dynamic_batch_size", settings.ALPHA_EVAL_BATCH_SIZE)
        _is_intraday = is_intraday(self._interval)
        _rtc = default_round_trip_cost(self._interval)

        for batch_start in range(0, len(all_children), _BATCH_SIZE):
            # 동적 배치 사이즈: OOM 발생 시에만 축소, 여유 시 복구
            try:
                import psutil
                mem_pct = psutil.virtual_memory().percent
                if mem_pct < 60 and _BATCH_SIZE < settings.ALPHA_EVAL_BATCH_SIZE:
                    old = _BATCH_SIZE
                    _BATCH_SIZE = min(settings.ALPHA_EVAL_BATCH_SIZE, _BATCH_SIZE + 1)
                    self._dynamic_batch_size = _BATCH_SIZE
                    logger.info("메모리 %.0f%% — 배치 크기 %d→%d 복구", mem_pct, old, _BATCH_SIZE)
            except Exception:
                pass
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
                lo_returns = compute_long_only_returns(df_batch, factor_col=col)
                avg_to, to_series = compute_position_turnover(df_batch, factor_col=col)
                metrics = compute_factor_metrics(
                    ic_series, ls_returns=ls_returns,
                    long_only_returns=lo_returns,
                    position_turnover=avg_to,
                    turnover_series=to_series,
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
            cpcv_top_k = min(50, len(cpcv_candidates))

            # ★ 해시 기반 클론 중복 제거
            _discovered_hashes: set[str] = set()

            for child in cpcv_candidates[:cpcv_top_k]:
                cpcv_result = cpcv_validate(
                    full_data, child.expression,
                    ic_threshold=self._ic_threshold,
                )
                if cpcv_result.passed:
                    # ★ 구조적 해시로 클론 제거 (상수만 다른 동일 수식 방지)
                    _hash = expression_hash(child.expression)
                    if _hash in _discovered_hashes:
                        logger.debug("Clone skipped: %s", child.expression_str[:50])
                        continue
                    _discovered_hashes.add(_hash)

                    # CPCV 통과 후 최종 Sharpe 검증
                    if child.sharpe < settings.ALPHA_SHARPE_THRESHOLD:
                        logger.info(
                            "CPCV 통과했지만 Sharpe %.2f < %.1f → 탈락: %s",
                            child.sharpe, settings.ALPHA_SHARPE_THRESHOLD,
                            child.expression_str[:50],
                        )
                        continue
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

    # 표현형 클러스터 캐시 (10세대마다 갱신)
    _phenotypic_cluster_cache: dict[str, str] | None = None  # expression_hash → cluster_id
    _phenotypic_cluster_gen: int = -1  # 마지막 갱신 세대

    def _niche_cap_trim(
        self, population: list[ScoredFactor], target_size: int,
    ) -> list[ScoredFactor]:
        """니치(피처 패밀리)별 최대 점유율을 적용하여 모집단 크기 제한.

        Algorithm:
        0. 글로벌 상위 엘리트를 사전 분리 — 니치캡 면제
        1. 니치 분류: 표현형 클러스터 (10세대마다) 또는 구조적 패밀리 (폴백)
        2. fitness 내림차순 정렬
        3. Pass 1: 니치별 max_per_niche까지만 수용, 초과 시 overflow로
        4. Pass 2: 미달 시 overflow에서 보충 (항상 target_size 유지)
        """
        from app.alpha.ast_converter import classify_niche

        if len(population) <= target_size:
            return population

        max_pct = settings.ALPHA_NICHE_MAX_PCT
        if max_pct >= 1.0:
            population.sort(key=lambda f: f.fitness_composite, reverse=True)
            return population[:target_size]

        # Step 0: 글로벌 상위 엘리트를 니치캡 면제
        n_protected = max(1, math.ceil(target_size * self._elite_pct))
        sorted_all = sorted(population, key=lambda f: f.fitness_composite, reverse=True)
        protected_elites = sorted_all[:n_protected]
        protected_ids = {id(f) for f in protected_elites}
        non_protected = [f for f in population if id(f) not in protected_ids]
        remaining_target = target_size - len(protected_elites)

        if remaining_target <= 0:
            return protected_elites[:target_size]

        max_per_niche = max(2, int(remaining_target * max_pct))

        # Step 1: 니치 분류 — 표현형 클러스터 (10세대마다) 또는 구조적 패밀리
        use_phenotypic = (
            self._phenotypic_cluster_cache is not None
            and len(self._phenotypic_cluster_cache) > 0
        )

        def _get_niche(factor: ScoredFactor) -> str:
            if use_phenotypic and factor.expression_hash in self._phenotypic_cluster_cache:
                return self._phenotypic_cluster_cache[factor.expression_hash]
            return classify_niche(factor.expression)

        labeled = [(f, _get_niche(f)) for f in non_protected]
        labeled.sort(key=lambda x: x[0].fitness_composite, reverse=True)

        # Pass 1: 니치별 캡 적용
        accepted: list[ScoredFactor] = []
        overflow: list[ScoredFactor] = []
        niche_counts: dict[str, int] = {}

        for factor, niche in labeled:
            count = niche_counts.get(niche, 0)
            if count < max_per_niche:
                accepted.append(factor)
                niche_counts[niche] = count + 1
                if len(accepted) >= remaining_target:
                    break
            else:
                overflow.append(factor)

        # Pass 2: 미달분 overflow에서 보충
        if len(accepted) < remaining_target:
            remaining = remaining_target - len(accepted)
            accepted.extend(overflow[:remaining])

        final_population = protected_elites + accepted

        # 니치 분포 로그
        final_dist: dict[str, int] = {}
        for factor in final_population:
            final_dist[_get_niche(factor)] = final_dist.get(_get_niche(factor), 0) + 1

        niche_mode = "phenotypic" if use_phenotypic else "structural"
        logger.warning(
            "Niche cap trim (%s): %d → %d (protected=%d, max_per_niche=%d). Distribution: %s",
            niche_mode,
            len(population), len(final_population), len(protected_elites), max_per_niche,
            {k: v for k, v in sorted(final_dist.items(), key=lambda x: -x[1])},
        )

        return final_population

    def _update_phenotypic_clusters(
        self, population: list[ScoredFactor], data: pl.DataFrame,
    ) -> None:
        """10세대마다 호출: 팩터 output 상관 기반 표현형 클러스터링.

        각 팩터의 Polars expression을 데이터에 적용하여 output 벡터를 생성하고,
        pairwise Pearson 상관으로 거리 행렬을 구성한 뒤 계층적 클러스터링 수행.
        결과를 _phenotypic_cluster_cache에 캐시.
        """
        import numpy as np

        if len(population) < 5:
            return

        # 1. 각 팩터의 output 벡터 계산 (일별 평균)
        outputs: list[tuple[ScoredFactor, list[float]]] = []
        sample_data = data
        # 메모리 절약: 최근 100일만 샘플링
        if "dt" in data.columns:
            dates = data.select("dt").unique().sort("dt")["dt"].to_list()
            if len(dates) > 100:
                cutoff = dates[-100]
                sample_data = data.filter(pl.col("dt") >= cutoff)

        for factor in population:
            try:
                pe = sympy_to_polars(factor.expression)
                col_name = "_pheno_tmp"
                df = sample_data.with_columns(pe.alias(col_name))
                # 행별 값 추출 (횡단면 또는 시계열)
                vals = df[col_name].to_list()
                # NaN/Inf 제거
                clean = [float(v) for v in vals if v is not None and not math.isinf(float(v)) and not math.isnan(float(v))]
                if len(clean) >= 10:
                    outputs.append((factor, clean))
            except Exception:
                continue

        if len(outputs) < 5:
            logger.debug("Phenotypic clustering skipped: only %d valid outputs", len(outputs))
            return

        # 2. 길이 통일 (최소 공통 길이)
        min_len = min(len(o[1]) for o in outputs)
        matrix = np.array([o[1][:min_len] for o in outputs], dtype=np.float64)

        # 3. Pearson 상관 행렬 → 거리 행렬
        try:
            corr = np.corrcoef(matrix)
            # NaN 처리
            corr = np.nan_to_num(corr, nan=0.0)
            dist = 1.0 - np.abs(corr)  # 상관이 높을수록(양/음) 거리 작음
            np.fill_diagonal(dist, 0.0)
        except Exception as e:
            logger.debug("Phenotypic correlation failed: %s", e)
            return

        # 4. 계층적 클러스터링
        try:
            from scipy.cluster.hierarchy import fcluster, linkage
            from scipy.spatial.distance import squareform

            condensed = squareform(dist, checks=False)
            Z = linkage(condensed, method="average")
            # 상관 0.7 이상이면 같은 클러스터 (거리 0.3 이하)
            labels = fcluster(Z, t=0.3, criterion="distance")
        except Exception as e:
            logger.debug("Phenotypic clustering failed: %s", e)
            return

        # 5. 캐시 저장: expression_hash → "pheno_N"
        cache: dict[str, str] = {}
        for i, (factor, _) in enumerate(outputs):
            cluster_id = f"pheno_{labels[i]}"
            if factor.expression_hash:
                cache[factor.expression_hash] = cluster_id

        self._phenotypic_cluster_cache = cache
        self._phenotypic_cluster_gen = self._generation

        n_clusters = len(set(labels))
        logger.warning(
            "Phenotypic clustering: %d factors → %d clusters (gen %d)",
            len(outputs), n_clusters, self._generation,
        )

    # ── MAP-Elites 행동 아카이브 ──

    # complexity bin 경계: depth 1-2, 3, 4, 5-6
    _COMPLEXITY_BINS = [3, 4, 5, 7]  # upper bounds (exclusive)

    async def _load_map_elites_archive(
        self,
        existing_ids: set,
        active_factors: list,
        max_inject: int = 15,
    ) -> list:
        """MAP-Elites 그리드 기반 아카이브 로드.

        8 family × 4 complexity bin = 32셀 그리드.
        각 셀에 최고 fitness 팩터 1개만 보존.
        활성 모집단에서 과소 대표된 패밀리의 셀을 우선 선택하여 재주입.
        """
        from app.alpha.ast_converter import classify_niche

        # 1. validated + 비활성 팩터 전부 로드
        archive_result = await self._db.execute(
            select(AlphaFactor).where(
                AlphaFactor.status == "validated",
                AlphaFactor.population_active == False,  # noqa: E712
            )
        )
        all_archived = list(archive_result.scalars().all())

        if not all_archived:
            return []

        # 2. 그리드 구축: (family, complexity_bin) → 최고 fitness 팩터
        grid: dict[tuple[str, int], AlphaFactor] = {}
        for f in all_archived:
            if f.id in existing_ids:
                continue
            # 패밀리 분류 (expression 파싱 없이 DB 필드 활용)
            try:
                expr = parse_expression(f.expression_str)
                family = classify_niche(expr)
            except Exception:
                family = "unknown"

            depth = f.tree_depth or 1
            c_bin = 0
            for i, upper in enumerate(self._COMPLEXITY_BINS):
                if depth < upper:
                    c_bin = i
                    break
            else:
                c_bin = len(self._COMPLEXITY_BINS) - 1

            cell_key = (family, c_bin)
            current = grid.get(cell_key)
            if current is None or (f.fitness_composite or 0) > (current.fitness_composite or 0):
                grid[cell_key] = f

        if not grid:
            return []

        # 3. 활성 모집단의 패밀리 분포 계산 → 과소 대표 패밀리 우선
        active_family_counts: dict[str, int] = {}
        for af in active_factors:
            try:
                expr = parse_expression(af.expression_str)
                fam = classify_niche(expr)
            except Exception:
                fam = "unknown"
            active_family_counts[fam] = active_family_counts.get(fam, 0) + 1
        total_active = max(len(active_factors), 1)

        # 4. 셀별 가중치: (1 - family_share) × staleness_decay × fitness
        weighted_cells: list[tuple[float, tuple, AlphaFactor]] = []
        for cell_key, factor in grid.items():
            family = cell_key[0]
            family_share = active_family_counts.get(family, 0) / total_active
            age_gens = max(0, self._generation - (factor.birth_generation or 0))
            staleness = math.exp(-0.03 * age_gens)
            fitness = factor.fitness_composite or 0.0
            weight = (1.0 - family_share) * staleness * max(fitness, 0.001)
            weighted_cells.append((weight, cell_key, factor))

        # 가중치 내림차순 정렬 → 상위 max_inject개 선택
        weighted_cells.sort(key=lambda x: x[0], reverse=True)
        selected = weighted_cells[:max_inject]

        injected = [f for _, _, f in selected]

        # 5. 로깅: 그리드 상태
        occupied = len(grid)
        total_cells = 8 * len(self._COMPLEXITY_BINS)
        injected_families = {}
        for _, cell_key, _ in selected:
            fam = cell_key[0]
            injected_families[fam] = injected_families.get(fam, 0) + 1

        logger.info(
            "MAP-Elites archive: grid %d/%d occupied, injecting %d (families: %s)",
            occupied, total_cells, len(injected),
            {k: v for k, v in sorted(injected_families.items(), key=lambda x: -x[1])},
        )

        return injected

    # ── 비volume 다양성 시드 (OHLCV + 기술적 지표 기반만 사용) ──
    _DIVERSITY_TEMPLATES: dict[str, list[str]] = {
        "price": [
            "close / sma_20 - 1",
            "(close - bb_lower) / (bb_upper - bb_lower + 0.001)",
            "ema_5 / ema_60 - 1",
            "(high - low) / close",
        ],
        "momentum": [
            "rsi_7 - rsi_21",
            "return_5d / (atr_7 + 0.001)",
            "macd_hist * sign(return_20d)",
            "return_5d - return_20d",
        ],
        "volatility": [
            "atr_7 / (atr_21 + 0.001)",
            "atr_14 / (close + 0.001) * 100",
            "(bb_upper - bb_lower) / (sma_20 + 0.001)",
        ],
        "mixed_no_volume": [
            "rsi / 100 * (close / sma_20 - 1)",
            "(close - sma_20) / (atr_14 + 0.001)",
            "price_change_pct / (atr_14 + 0.001)",
            "macd_hist / (atr_14 + 0.001)",
            "abs(return_5d) / (atr_7 + 0.001) * rsi_7 / 100",
        ],
    }

    def _make_diversity_seeds(
        self, offspring: list[ScoredFactor],
    ) -> list[ScoredFactor]:
        """비volume 다양성 시드 생성. offspring의 중앙 fitness로 삽입.

        OHLCV + 기술적 지표 기반만 사용 (외부 데이터 피처 제외).
        수렴 감지 시에만 호출됨.
        """
        import random as _rand

        # offspring 중앙 fitness
        if offspring:
            _sorted = sorted(o.fitness_composite for o in offspring)
            median_fitness = _sorted[len(_sorted) // 2]
        else:
            median_fitness = 0.0

        seeds: list[ScoredFactor] = []
        all_templates = []
        for family, tmpls in self._DIVERSITY_TEMPLATES.items():
            for t in tmpls:
                all_templates.append((family, t))

        # 셔플 → 최대 15개 시도
        _rand.shuffle(all_templates)
        for family, tmpl in all_templates[:15]:
            try:
                expr = parse_expression(tmpl)
                code_str = sympy_to_code_string(expr)
                depth = calc_tree_depth(expr)
                size = calc_tree_size(expr)

                seeds.append(ScoredFactor(
                    expression=expr,
                    expression_str=code_str,
                    hypothesis=f"[diversity-seed:{family}]",
                    ic_mean=0.0,
                    ic_std=0.0,
                    icir=0.0,
                    turnover=0.0,
                    sharpe=0.0,
                    max_drawdown=0.0,
                    generation=self._generation,
                    fitness_composite=median_fitness,
                    tree_depth=depth,
                    tree_size=size,
                    expression_hash=expression_hash(expr),
                    operator_origin=f"diversity_seed_{family}",
                    genotypic_age=0,  # AFPO: 다양성 시드 = 새 유전자
                ))
            except Exception as e:
                logger.debug("Diversity seed '%s' failed: %s", tmpl, e)

        return seeds

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
        """과적합 판단: Val IC 기준 미달 또는 Train→Val 하락 50% 초과."""
        # 조건 1: Val IC가 임계값의 80% 미만
        if val_ic < self._ic_threshold * 0.8:
            return True
        # 조건 2: Train→Val IC 하락이 50% 초과
        if train_ic > 0 and (train_ic - val_ic) / train_ic > 0.5:
            return True
        return False

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
                    w_ic=settings.ALPHA_FITNESS_W_IC,
                    w_icir=settings.ALPHA_FITNESS_W_ICIR,
                    w_sharpe=settings.ALPHA_FITNESS_W_SHARPE,
                    w_mdd=settings.ALPHA_FITNESS_W_MDD,
                    w_turnover=settings.ALPHA_FITNESS_W_TURNOVER,
                    w_complexity=settings.ALPHA_FITNESS_W_COMPLEXITY,
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
                    genotypic_age=0,  # AFPO: 시드 = 새 유전자
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
                w_ic=settings.ALPHA_FITNESS_W_IC,
                w_icir=settings.ALPHA_FITNESS_W_ICIR,
                w_sharpe=settings.ALPHA_FITNESS_W_SHARPE,
                w_mdd=settings.ALPHA_FITNESS_W_MDD,
                w_turnover=settings.ALPHA_FITNESS_W_TURNOVER,
                w_complexity=settings.ALPHA_FITNESS_W_COMPLEXITY,
            )

            # UCB1 업데이트
            delta = fitness - parent.fitness_composite
            self._operator_registry.update(operator_name, delta_fitness=delta)

            # AFPO: LLM 생성 = 새 유전자(age=0), AST 변이 = 부모 age + 1
            child_age = 0 if operator_name.startswith("llm_") else parent.genotypic_age + 1

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
                genotypic_age=child_age,
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
            w_ic=settings.ALPHA_FITNESS_W_IC,
            w_icir=settings.ALPHA_FITNESS_W_ICIR,
            w_sharpe=settings.ALPHA_FITNESS_W_SHARPE,
            w_mdd=settings.ALPHA_FITNESS_W_MDD,
            w_turnover=settings.ALPHA_FITNESS_W_TURNOVER,
            w_complexity=settings.ALPHA_FITNESS_W_COMPLEXITY,
        )

        delta = fitness - parent.fitness_composite
        self._operator_registry.update(op_name, delta_fitness=delta)
        funnel["eval_success"] += 1

        # AFPO: LLM 생성 = 새 유전자(age=0), AST 변이 = 부모 age + 1
        child_age = 0 if op_name.startswith("llm_") else parent.genotypic_age + 1

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
            genotypic_age=child_age,
        )

    # eval 실패 샘플 로깅 카운터 (세대당 리셋)
    _eval_fail_logged: int = 0

    # 피처 목록은 miner._build_available_features()로 동적 생성
    _FEATURE_LIST: str | None = None  # _init_feature_list()에서 초기화

    def _get_feature_list(self) -> str:
        """동적 피처 목록 생성 (miner와 통일)."""
        if self._FEATURE_LIST is None:
            from app.alpha.miner import _build_available_features
            data_cols = set(self._data.columns) if self._data is not None else set()
            self.__class__._FEATURE_LIST = _build_available_features(data_cols)
        return self._FEATURE_LIST

    def _get_underrepresented_families(self) -> list[str]:
        """현재 모집단에서 과소 대표된 피처 패밀리 식별."""
        from app.alpha.ast_converter import classify_niche

        if not self._population:
            return []

        family_counts: dict[str, int] = {}
        for f in self._population:
            try:
                expr = parse_expression(f.expression_str)
                fam = classify_niche(expr)
            except Exception:
                fam = "unknown"
            family_counts[fam] = family_counts.get(fam, 0) + 1

        total = max(len(self._population), 1)
        all_families = ["price", "volume", "momentum", "volatility", "supply", "fundamental", "sentiment", "market_micro"]
        # 점유율 5% 미만이거나 아예 없는 패밀리
        underrep = [f for f in all_families if family_counts.get(f, 0) / total < 0.05]
        return underrep

    async def _llm_call(self, prompt: str) -> str:
        """LLM 호출 통합 래퍼. ALPHA_LLM_PROVIDER 설정에 따라 Gemini/Anthropic 선택."""
        if settings.ALPHA_LLM_PROVIDER == "gemini":
            from app.core.llm import chat_gemini
            response = await chat_gemini(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                caller="alpha.evolution_engine",
            )
            return response.text.strip()
        else:
            from app.core.llm import chat_simple
            response = await chat_simple(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                caller="alpha.evolution_engine",
            )
            return response.text.strip()

    async def _llm_seed(self) -> sympy.Basic | None:
        """LLM으로 새 수식 생성. MAP-Elites 빈 셀 타겟팅."""
        try:
            import random as _random

            from app.alpha.miner import _CATEGORY_EXAMPLES, _CATEGORIES

            feature_list = self._get_feature_list()

            # MAP-Elites 타겟팅: 과소 대표 패밀리가 있으면 50% 확률로 해당 패밀리 강제
            underrep = self._get_underrepresented_families()
            if underrep and _random.random() < 0.5:
                category = _random.choice(underrep)
                family_features = {
                    "price": "close, open, high, low, sma_20, ema_20, bb_upper, bb_lower",
                    "momentum": "rsi, macd_hist, price_change_pct, return_5d, return_20d",
                    "volatility": "atr_14, atr_7, bb_width, true_range",
                    "supply": "foreign_net_norm, inst_net_norm, retail_net_norm",
                    "fundamental": "eps, bps, earnings_yield, book_yield, operating_margin",
                    "sentiment": "sentiment_score, event_score, article_count",
                    "market_micro": "margin_rate, short_balance_rate, pgm_net_norm",
                }
                target_features = family_features.get(category, "")
                example_strs = ""
                prompt = (
                    f"Generate a single alpha factor formula.\n\n"
                    f"{feature_list}\n\n"
                    f"Available functions: +, -, *, /, log(), exp(), sqrt(), abs(), "
                    f"sign(), step(), Max(), Min(), clip(x, lo, hi).\n"
                    f"IMPORTANT: You MUST primarily use features from the '{category}' category: {target_features}.\n"
                    f"Do NOT use volume or zscore_volume as the main component.\n"
                    f"Return ONLY the formula, nothing else."
                )
            else:
                # 기존 랜덤 카테고리 방식
                category = _random.choice(_CATEGORIES)
                examples = _CATEGORY_EXAMPLES[category]
                example_strs = "; ".join(e["formula"] for e in examples[:2])

                prompt = (
                    f"Generate a single alpha factor formula.\n\n"
                    f"{feature_list}\n\n"
                    f"Available functions: +, -, *, /, log(), exp(), sqrt(), abs(), "
                    f"sign(), step(), Max(), Min(), clip(x, lo, hi).\n"
                    f"Focus on '{category}' type factors. Examples: {example_strs}\n"
                    f"Create a DIFFERENT formula from the examples.\n"
                    f"Return ONLY the formula, nothing else."
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

            formula = await self._llm_call(prompt)
            return parse_expression(formula)
        except Exception as e:
            logger.debug("LLM seed failed: %s", e)
            return None

    async def _llm_mutate(self, parent: ScoredFactor) -> sympy.Basic | None:
        """LLM으로 기존 수식 변이 (다차원 메트릭 피드백)."""
        try:
            feature_list = self._get_feature_list()
            prompt = (
                f"Mutate this alpha factor formula to improve it:\n"
                f"{parent.expression_str}\n"
                f"Current metrics:\n"
                f"  IC: {parent.ic_mean:.4f} (target: >= {self._ic_threshold})\n"
                f"  Sharpe: {parent.sharpe:.3f} (target: >= {settings.ALPHA_SHARPE_THRESHOLD})\n"
                f"  Turnover: {parent.turnover:.3f} (lower is better)\n"
                f"  Complexity: depth={parent.tree_depth}, size={parent.tree_size}\n"
                f"{feature_list}\n"
                "Available functions: +, -, *, /, log(), exp(), sqrt(), abs(), "
                "sign(), step(), Max(), Min(), clip(x, lo, hi).\n"
                "Return ONLY the new formula, nothing else."
            )

            # RAG: 유사한 과거 시도
            if self._vector_memory:
                rag = self._vector_memory.format_rag_context(
                    parent.expression_str
                )
                if rag and rag != "아직 탐색 이력이 없습니다.":
                    prompt += f"\n\nSimilar past attempts:\n{rag}"

            formula = await self._llm_call(prompt)
            return parse_expression(formula)
        except Exception as e:
            logger.debug("LLM mutate failed: %s", e)
            return None

    async def _llm_semantic_crossover(
        self, parent_a: ScoredFactor, parent_b: ScoredFactor,
    ) -> sympy.Basic | None:
        """LLM 시맨틱 교차: 두 부모의 금융 논리를 보존하면서 결합.

        LMX (Meyerson et al., 2023) + QuantaAlpha (2025) 패러다임.
        무작위 서브트리 교환 대신 LLM이 의미론적으로 유의미한 결합을 수행.
        """
        try:
            feature_list = self._get_feature_list()

            prompt = (
                f"You are an expert quantitative researcher. "
                f"Combine the predictive insights of these two alpha factor formulas "
                f"into a single new formula.\n\n"
                f"Parent A (IC={parent_a.ic_mean:.4f}, Sharpe={parent_a.sharpe:.2f}):\n"
                f"  {parent_a.expression_str}\n\n"
                f"Parent B (IC={parent_b.ic_mean:.4f}, Sharpe={parent_b.sharpe:.2f}):\n"
                f"  {parent_b.expression_str}\n\n"
                f"{feature_list}\n"
                f"Available functions: +, -, *, /, log(), exp(), sqrt(), abs(), "
                f"sign(), step(), Max(), Min(), clip(x, lo, hi).\n"
                f"Maximum depth: 6. Preserve the strongest sub-expressions from each parent.\n"
                f"Return ONLY the combined formula, nothing else."
            )

            # RAG 경험 추가
            if self._vector_memory:
                query = f"{parent_a.expression_str} {parent_b.expression_str}"
                rag = self._vector_memory.format_rag_context(query)
                if rag and rag != "아직 탐색 이력이 없습니다.":
                    prompt += f"\n\nPast experience:\n{rag}"

            formula = await self._llm_call(prompt)
            return parse_expression(formula)
        except Exception as e:
            logger.debug("LLM semantic crossover failed: %s", e)
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
            lo_returns = compute_long_only_returns(df, factor_col="alpha_factor")
            avg_turnover, turnover_series = compute_position_turnover(df, factor_col="alpha_factor")
            return compute_factor_metrics(
                ic_series, ls_returns=ls_returns,
                long_only_returns=lo_returns,
                position_turnover=avg_turnover,
                turnover_series=turnover_series,
                annualize=252.0,
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
                        genotypic_age=factor.genotypic_age,
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
                    genotypic_age=factor.genotypic_age,
                    status="population",
                    interval=self._interval,
                    parent_ids=factor.parent_ids if factor.parent_ids else None,
                )
                self._db.add(new_factor)
                await self._db.flush()
                factor.factor_id = str(new_factor.id)

        await self._db.commit()

        # 자동 퍼지: 비활성(population_active=False) + population 상태 + 품질 미달 → hard delete
        try:
            from sqlalchemy import delete as sa_delete
            purge_result = await self._db.execute(
                sa_delete(AlphaFactor).where(
                    AlphaFactor.population_active == False,  # noqa: E712
                    AlphaFactor.status == "population",
                    db_or(
                        AlphaFactor.ic_mean < 0.03,
                        AlphaFactor.ic_mean.is_(None),
                        AlphaFactor.sharpe < 0.3,
                        AlphaFactor.sharpe.is_(None),
                    ),
                )
            )
            purged = purge_result.rowcount
            if purged:
                await self._db.commit()
                logger.info("진화 후 품질 미달 팩터 퍼지: %d개 삭제", purged)
        except Exception as e:
            logger.debug("진화 후 퍼지 실패 (무시): %s", e)

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
