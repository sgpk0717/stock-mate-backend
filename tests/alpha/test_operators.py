"""연산자 레지스트리 + UCB1 단위 테스트.

Phase 2: operators.py의 OperatorRegistry, OperatorStats.
"""

from __future__ import annotations

import math


class TestOperatorStats:
    """OperatorStats 데이터 클래스 테스트."""

    def test_initial_values(self):
        """초기 통계값: calls=0, fitness_improvements=0."""
        from app.alpha.operators import OperatorStats

        stats = OperatorStats(name="ast_mutate_operator")
        assert stats.calls == 0
        assert stats.fitness_improvements == 0.0
        assert stats.is_llm is False

    def test_llm_flag(self):
        """LLM 연산자 플래그."""
        from app.alpha.operators import OperatorStats

        stats = OperatorStats(name="llm_seed", is_llm=True)
        assert stats.is_llm is True


class TestOperatorRegistry:
    """UCB1 기반 연산자 레지스트리 테스트."""

    def test_registry_has_all_operators(self):
        """레지스트리가 7+2=9개 연산자를 포함해야 한다."""
        from app.alpha.operators import OperatorRegistry

        registry = OperatorRegistry()
        stats = registry.get_stats()
        # AST 7개 + LLM 2개
        assert len(stats) >= 9

    def test_select_returns_valid_operator(self):
        """select()가 등록된 연산자 이름을 반환해야 한다."""
        from app.alpha.operators import OperatorRegistry

        registry = OperatorRegistry()
        name = registry.select()
        stats = registry.get_stats()
        valid_names = [s.name for s in stats]
        assert name in valid_names

    def test_select_respects_llm_ratio(self):
        """LLM 연산자 비율 제한을 준수해야 한다."""
        from app.alpha.operators import OperatorRegistry

        registry = OperatorRegistry(llm_ratio=0.1)
        llm_count = 0
        total = 100
        for _ in range(total):
            name = registry.select()
            registry.update(name, delta_fitness=0.001)
            if name.startswith("llm_"):
                llm_count += 1

        # 10% + 마진 (UCB 탐색으로 약간 초과 가능)
        assert llm_count <= total * 0.25, (
            f"LLM calls {llm_count}/{total} exceeds ratio"
        )

    def test_update_increments_calls(self):
        """update() 호출 시 calls 카운터 증가."""
        from app.alpha.operators import OperatorRegistry

        registry = OperatorRegistry()
        name = registry.select()
        registry.update(name, delta_fitness=0.01)
        stats = registry.get_stats()
        op_stats = next(s for s in stats if s.name == name)
        assert op_stats.calls == 1

    def test_ucb1_explores_untried_operators(self):
        """UCB1은 미시도 연산자를 먼저 탐색해야 한다."""
        from app.alpha.operators import OperatorRegistry

        registry = OperatorRegistry()
        # 첫 번째 연산자를 반복 업데이트하여 다른 연산자보다 데이터가 많게
        first = registry.select()
        for _ in range(20):
            registry.update(first, delta_fitness=0.001)

        # 이후 select는 아직 미시도인 연산자를 선택할 확률이 높아야
        seen = {first}
        for _ in range(30):
            name = registry.select()
            seen.add(name)
            registry.update(name, delta_fitness=0.001)

        # 최소 3가지 이상 연산자가 선택되어야 (탐색)
        assert len(seen) >= 3

    def test_ucb1_favors_high_fitness_operator(self):
        """UCB1은 적합도 개선이 큰 연산자를 선호해야 한다."""
        from app.alpha.operators import OperatorRegistry

        registry = OperatorRegistry(llm_ratio=0.0)  # AST만 사용

        all_stats = registry.get_stats()
        ast_ops = [s.name for s in all_stats if not s.is_llm]

        # 모든 연산자를 여러 번 사용하여 UCB1 탐색 해소
        for _ in range(5):
            for op_name in ast_ops:
                registry.update(op_name, delta_fitness=0.001)

        # 하나에만 높은 fitness 부여
        good_op = ast_ops[0]
        for _ in range(50):
            registry.update(good_op, delta_fitness=0.5)

        # 이후 선택에서 good_op이 많이 뽑혀야
        counts = {}
        for _ in range(100):
            name = registry.select()
            counts[name] = counts.get(name, 0) + 1
            registry.update(name, delta_fitness=0.001)

        # good_op이 상위권에 있어야
        max_op = max(counts, key=counts.get)
        assert max_op == good_op or counts.get(good_op, 0) >= 10

    def test_disable_llm_operators(self):
        """LLM 비활성화 시 LLM 연산자가 선택되지 않아야 한다."""
        from app.alpha.operators import OperatorRegistry

        registry = OperatorRegistry(llm_ratio=0.0)
        for _ in range(50):
            name = registry.select()
            assert not name.startswith("llm_"), f"LLM operator {name} selected with ratio 0"
            registry.update(name, delta_fitness=0.001)

    def test_get_stats_returns_list(self):
        """get_stats()가 OperatorStats 리스트를 반환."""
        from app.alpha.operators import OperatorRegistry, OperatorStats

        registry = OperatorRegistry()
        stats = registry.get_stats()
        assert isinstance(stats, list)
        assert all(isinstance(s, OperatorStats) for s in stats)

    def test_to_dict(self):
        """to_dict()가 직렬화 가능한 dict를 반환."""
        from app.alpha.operators import OperatorRegistry

        registry = OperatorRegistry()
        registry.update(registry.select(), delta_fitness=0.01)
        d = registry.to_dict()
        assert isinstance(d, dict)
        assert "operators" in d
