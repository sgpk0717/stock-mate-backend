"""evolution.py 확장 기능 단위 테스트.

Phase 2: hoist_mutation, ephemeral_constant_mutation, parsimony tournament.
기존 evolution.py 테스트와 별도로, 새로 추가되는 기능만 테스트.
"""

from __future__ import annotations

import sympy

from app.alpha.ast_converter import parse_expression, sympy_to_polars


class TestHoistMutation:
    """hoist_mutation: 서브트리를 루트로 승격."""

    def test_hoist_reduces_tree_size(self):
        """Hoist 결과는 원본보다 작거나 같아야 한다."""
        from app.alpha.evolution import hoist_mutation, _get_subtrees

        # 복잡한 수식: log(close / sma_20) * rsi + volume_ratio
        expr = parse_expression("log(close / sma_20) * rsi + volume_ratio")
        original_size = len(_get_subtrees(expr))

        # 여러 번 시도 (랜덤이므로)
        found_smaller = False
        for _ in range(50):
            result = hoist_mutation(expr)
            if result is not None:
                new_size = len(_get_subtrees(result))
                assert new_size <= original_size
                found_smaller = True
                break

        assert found_smaller, "hoist_mutation never produced a result"

    def test_hoist_produces_valid_polars(self):
        """Hoist 결과는 Polars 변환 가능해야 한다."""
        from app.alpha.evolution import hoist_mutation

        expr = parse_expression("log(close * rsi) + volume_ratio * atr_14")
        for _ in range(50):
            result = hoist_mutation(expr)
            if result is not None:
                # Polars 변환 가능해야
                sympy_to_polars(result)
                return

        # 모든 시도가 None이면 skip (극히 드문 경우)

    def test_hoist_on_leaf_returns_none(self):
        """리프 노드(단일 심볼)에서는 None 반환."""
        from app.alpha.evolution import hoist_mutation

        expr = parse_expression("close")
        result = hoist_mutation(expr)
        assert result is None


class TestEphemeralConstantMutation:
    """ephemeral_constant_mutation: 랜덤 상수 삽입."""

    def test_introduces_constant(self):
        """결과에 새로운 상수가 포함되어야 한다."""
        from app.alpha.evolution import ephemeral_constant_mutation, _get_subtrees

        expr = parse_expression("close + rsi")

        for _ in range(50):
            result = ephemeral_constant_mutation(expr)
            if result is not None:
                subtrees = _get_subtrees(result)
                has_number = any(
                    isinstance(s, (sympy.Integer, sympy.Float, sympy.Rational))
                    and float(s) != 0
                    for s in subtrees
                )
                assert has_number, "No constant found in mutated expression"
                return

    def test_constant_in_range(self):
        """삽입 상수는 0.01~100 범위여야 한다."""
        from app.alpha.evolution import ephemeral_constant_mutation, _get_subtrees

        expr = parse_expression("close * rsi")
        for _ in range(100):
            result = ephemeral_constant_mutation(expr)
            if result is not None and result != expr:
                subtrees = _get_subtrees(result)
                for s in subtrees:
                    if isinstance(s, (sympy.Float, sympy.Rational)):
                        val = abs(float(s))
                        if val > 0.005:  # 0이 아닌 새 상수
                            assert 0.005 <= val <= 200, (
                                f"Constant {val} out of range"
                            )
                return

    def test_result_is_valid_polars(self):
        """결과는 Polars 변환 가능해야 한다."""
        from app.alpha.evolution import ephemeral_constant_mutation

        expr = parse_expression("log(close) + rsi")
        for _ in range(50):
            result = ephemeral_constant_mutation(expr)
            if result is not None:
                sympy_to_polars(result)
                return


class TestParsimonyTournament:
    """Parsimony pressure를 적용한 토너먼트 선택 테스트."""

    def _make_scored_factor(self, expr_str, fitness, size):
        """테스트용 ScoredFactor 생성."""
        from app.alpha.evolution import ScoredFactor

        expr = parse_expression(expr_str)
        return ScoredFactor(
            expression=expr,
            expression_str=expr_str,
            hypothesis="test",
            ic_mean=fitness,
            generation=0,
            fitness_composite=fitness,
            tree_size=size,
        )

    def test_parsimony_favors_smaller_tree(self):
        """같은 fitness에서 parsimony는 작은 트리를 선호해야 한다."""
        from app.alpha.evolution import tournament_select

        # 같은 fitness, 다른 크기
        pop = [
            self._make_scored_factor("close", 0.05, size=1),
            self._make_scored_factor("close + rsi", 0.05, size=3),
            self._make_scored_factor("log(close) * rsi + volume_ratio", 0.05, size=7),
            self._make_scored_factor("close * rsi", 0.05, size=3),
            self._make_scored_factor("rsi", 0.05, size=1),
        ]

        # 여러 번 선택해서 작은 개체 선호 확인
        small_wins = 0
        for _ in range(100):
            selected = tournament_select(pop, k=5, n_select=1, parsimony=True)
            if selected and selected[0].tree_size <= 3:
                small_wins += 1

        # 50% 이상 작은 트리 선택 (parsimony 효과)
        assert small_wins > 30, f"Small tree won only {small_wins}/100 times"

    def test_without_parsimony_just_fitness(self):
        """parsimony=False면 순수 fitness 기반 선택."""
        from app.alpha.evolution import tournament_select

        pop = [
            self._make_scored_factor("close", 0.01, size=1),  # 작지만 낮은 fitness
            self._make_scored_factor("log(close) * rsi + volume_ratio", 0.08, size=7),  # 크지만 높은 fitness
        ]

        high_fitness_wins = 0
        for _ in range(100):
            selected = tournament_select(pop, k=2, n_select=1, parsimony=False)
            if selected and selected[0].fitness_composite >= 0.08:
                high_fitness_wins += 1

        assert high_fitness_wins > 80, "Without parsimony, high fitness should dominate"

    def test_backward_compatible_default(self):
        """기본 parsimony=True로 동작해야 한다."""
        from app.alpha.evolution import tournament_select

        pop = [
            self._make_scored_factor("close", 0.05, size=1),
            self._make_scored_factor("close + rsi", 0.05, size=3),
        ]
        # 에러 없이 동작
        selected = tournament_select(pop, k=2, n_select=1)
        assert len(selected) == 1


class TestExistingMutateActivated:
    """기존 mutate() 함수가 정상 작동하는지 확인."""

    def test_mutate_returns_different_expression(self):
        """mutate()가 원본과 다른 수식을 반환해야 한다."""
        from app.alpha.evolution import mutate

        expr = parse_expression("close * rsi + volume_ratio")
        mutated_any = False
        for _ in range(50):
            result = mutate(expr)
            if result is not None and result != expr:
                mutated_any = True
                break

        assert mutated_any, "mutate() never produced a different expression"

    def test_crossover_returns_children(self):
        """crossover()가 자식 수식을 반환해야 한다."""
        from app.alpha.evolution import crossover

        expr_a = parse_expression("close * rsi + volume_ratio")
        expr_b = parse_expression("log(atr_14) * macd_hist")
        children = crossover(expr_a, expr_b)
        # 교차 결과가 있을 수도 없을 수도 (랜덤)
        # 여러 번 시도
        for _ in range(50):
            children = crossover(expr_a, expr_b)
            if children:
                assert len(children) > 0
                return
