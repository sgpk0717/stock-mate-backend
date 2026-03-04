"""DB 영속 모집단 기반 진화 엔진.

팩토리 전용. miner.py의 EvolutionaryAlphaMiner와 병렬 존재.
사이클 간 모집단을 DB에 유지하여 연속적 진화를 수행한다.
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
import uuid
from dataclasses import asdict

import polars as pl
import sympy
from sqlalchemy import select, update
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
)
from app.alpha.evolution import (
    ScoredFactor,
    crossover,
    ephemeral_constant_mutation,
    hoist_mutation,
    mutate,
    tournament_select,
)
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

        # 데이터 전처리
        self._data = ensure_alpha_features(self._data)

    def update_data(self, data: pl.DataFrame) -> None:
        """새 사이클 데이터 반영."""
        self._data = ensure_alpha_features(data)

    async def load_population(self) -> list[ScoredFactor]:
        """DB에서 population_active=True인 팩터 로드."""
        result = await self._db.execute(
            select(AlphaFactor).where(
                AlphaFactor.population_active == True,  # noqa: E712
            ).order_by(AlphaFactor.fitness_composite.desc().nullslast())
        )
        factors = result.scalars().all()

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

        # 2. 엘리트 보존
        elites = self._select_elites(population)

        # 3. Train/Val 분할
        train_data, val_data = self._split_train_val()

        # 4. 진화 루프: 나머지 슬롯 채우기
        offspring_target = self._population_size - len(elites)
        offspring: list[ScoredFactor] = []
        new_discovered: list[DiscoveredFactor] = []

        for i in range(offspring_target):
            if progress_cb and i % 10 == 0:
                pct = 10 + int(80 * i / max(offspring_target, 1))
                await progress_cb(pct, 100, f"세대 {self._generation}: {i}/{offspring_target} 개체 진화 중")

            # 4a. 연산자 선택
            op_name = self._operator_registry.select()

            # 4b. 부모 선택 + 연산자 적용
            child = await self._apply_operator(population, op_name, train_data)

            if child is not None:
                offspring.append(child)

                # IC 기준 통과 확인
                if child.ic_mean >= self._ic_threshold:
                    # Walk-Forward 검증
                    if val_data is not None and val_data.height > 30:
                        val_ic = self._evaluate_on_data(child.expression, val_data)
                        if self._is_overfit(child.ic_mean, val_ic):
                            continue  # 과적합 → 모집단에는 넣되 discovered에서 제외

                    # discovered 팩터로 추가 (ScoredFactor의 실제 메트릭 사용)
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

        # 5. 중복 제거
        all_new = elites + offspring
        all_new = self._deduplicate(all_new)

        # 6. 모집단 크기 제한
        all_new.sort(key=lambda f: f.fitness_composite, reverse=True)
        final_population = all_new[:self._population_size]

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
            })

        return new_discovered

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

        train = self._data.filter(pl.col("dt") <= split_date)
        val = self._data.filter(pl.col("dt") > split_date)

        return train, val if val.height > 0 else None

    def _is_overfit(self, train_ic: float, val_ic: float) -> bool:
        """과적합 판단: Val IC < threshold × 0.5면 과적합."""
        return val_ic < self._ic_threshold * 0.5

    async def _seed_population(self, count: int) -> list[ScoredFactor]:
        """초기 모집단 생성: 간단한 수식 시드."""
        seeds: list[ScoredFactor] = []
        # 기본 수식 템플릿
        templates = [
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
                )

                seeds.append(ScoredFactor(
                    expression=expr,
                    expression_str=tmpl,
                    hypothesis=f"Seed factor {i}",
                    ic_mean=metrics.ic_mean,
                    generation=self._generation,
                    fitness_composite=fitness,
                    tree_depth=depth,
                    tree_size=size,
                    expression_hash=expression_hash(expr),
                    operator_origin="seed",
                ))
            except (ASTConversionError, Exception) as e:
                logger.debug("Seed %s failed: %s", tmpl, e)

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
            )

            # UCB1 업데이트
            delta = fitness - parent.fitness_composite
            self._operator_registry.update(operator_name, delta_fitness=delta)

            return ScoredFactor(
                expression=child_expr,
                expression_str=str(child_expr),
                hypothesis=f"Evolved from {parent.expression_str[:50]}",
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

    async def _llm_seed(self) -> sympy.Basic | None:
        """Claude API로 새 수식 생성."""
        try:
            from anthropic import AsyncAnthropic

            client = AsyncAnthropic()
            prompt = (
                "Generate a single alpha factor formula using these variables: "
                "close, open, high, low, volume, sma_20, ema_20, rsi, "
                "volume_ratio, atr_14, macd_hist, bb_upper, bb_lower, price_change_pct.\n"
                "Use only: +, -, *, /, log(), sqrt(), abs().\n"
                "Return ONLY the formula, nothing else."
            )
            if self._context:
                prompt += f"\nContext: {self._context}"

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
        """Claude API로 기존 수식 변이."""
        try:
            from anthropic import AsyncAnthropic

            client = AsyncAnthropic()
            prompt = (
                f"Mutate this alpha factor formula to improve it:\n"
                f"{parent.expression_str}\n"
                f"Current IC: {parent.ic_mean:.4f}\n"
                "Use only these variables: close, open, high, low, volume, "
                "sma_20, ema_20, rsi, volume_ratio, atr_14, macd_hist, "
                "bb_upper, bb_lower, price_change_pct.\n"
                "Use only: +, -, *, /, log(), sqrt(), abs().\n"
                "Return ONLY the new formula, nothing else."
            )

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
        """수식의 IC mean을 계산 (간략 버전)."""
        try:
            polars_expr = sympy_to_polars(expr)
            df = data.with_columns(polars_expr.alias("alpha_factor"))
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
        """수식의 전체 메트릭 계산."""
        try:
            polars_expr = sympy_to_polars(expr)
            df = data.with_columns(polars_expr.alias("alpha_factor"))
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
            return compute_factor_metrics(ic_series)
        except Exception as e:
            logger.debug("Evaluation failed: %s", e)
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
                    )
                )
            else:
                # 신규 팩터 삽입
                new_factor = AlphaFactor(
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
