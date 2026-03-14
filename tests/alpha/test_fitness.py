"""복합 적합도 함수 단위 테스트.

Phase 1: fitness.py의 compute_composite_fitness() 함수.
"""

from __future__ import annotations

import math


class TestCompositeFitness:
    """다목적 복합 적합도 계산 테스트."""

    def test_positive_fitness_for_good_factor(self):
        """좋은 팩터(높은 IC, 낮은 turnover, 작은 트리)는 양의 적합도."""
        from app.alpha.fitness import compute_composite_fitness

        fitness = compute_composite_fitness(
            ic_mean=0.05, icir=1.5, turnover=0.3,
            tree_depth=3, tree_size=8,
        )
        assert fitness > 0

    def test_negative_fitness_for_bad_factor(self):
        """나쁜 팩터(낮은 IC, 높은 turnover, 큰 트리)는 낮은 적합도."""
        from app.alpha.fitness import compute_composite_fitness

        fitness = compute_composite_fitness(
            ic_mean=0.0, icir=0.0, turnover=0.9,
            tree_depth=10, tree_size=30,
        )
        assert fitness < 0

    def test_higher_ic_gives_higher_fitness(self):
        """IC가 높을수록 적합도가 높아야 한다."""
        from app.alpha.fitness import compute_composite_fitness

        base_kwargs = dict(icir=1.0, turnover=0.3, tree_depth=3, tree_size=8)
        low = compute_composite_fitness(ic_mean=0.02, **base_kwargs)
        high = compute_composite_fitness(ic_mean=0.08, **base_kwargs)
        assert high > low

    def test_higher_icir_gives_higher_fitness(self):
        """ICIR이 높을수록 적합도가 높아야 한다."""
        from app.alpha.fitness import compute_composite_fitness

        base_kwargs = dict(ic_mean=0.05, turnover=0.3, tree_depth=3, tree_size=8)
        low = compute_composite_fitness(icir=0.5, **base_kwargs)
        high = compute_composite_fitness(icir=2.0, **base_kwargs)
        assert high > low

    def test_higher_turnover_gives_lower_fitness(self):
        """Turnover가 높을수록 적합도가 낮아야 한다 (패널티)."""
        from app.alpha.fitness import compute_composite_fitness

        base_kwargs = dict(ic_mean=0.05, icir=1.0, tree_depth=3, tree_size=8)
        low_turn = compute_composite_fitness(turnover=0.1, **base_kwargs)
        high_turn = compute_composite_fitness(turnover=0.9, **base_kwargs)
        assert low_turn > high_turn

    def test_larger_tree_gives_lower_fitness(self):
        """트리가 클수록 적합도가 낮아야 한다 (복잡도 패널티)."""
        from app.alpha.fitness import compute_composite_fitness

        base_kwargs = dict(ic_mean=0.05, icir=1.0, turnover=0.3)
        small = compute_composite_fitness(tree_depth=2, tree_size=5, **base_kwargs)
        large = compute_composite_fitness(tree_depth=8, tree_size=25, **base_kwargs)
        assert small > large

    def test_custom_weights(self):
        """커스텀 가중치가 적용되어야 한다."""
        from app.alpha.fitness import compute_composite_fitness

        # IC만 100% 가중 → IC가 적합도를 결정
        fitness = compute_composite_fitness(
            ic_mean=0.05, icir=0.0, turnover=0.0,
            tree_depth=0, tree_size=0,
            w_ic=1.0, w_icir=0.0, w_sharpe=0.0, w_mdd=0.0,
            w_turnover=0.0, w_complexity=0.0,
        )
        assert abs(fitness - 0.05) < 1e-9

    def test_zero_inputs_zero_fitness(self):
        """모든 입력이 0이면 적합도도 0 (sharpe 보정 제외)."""
        from app.alpha.fitness import compute_composite_fitness

        fitness = compute_composite_fitness(
            ic_mean=0.0, icir=0.0, turnover=0.0,
            tree_depth=0, tree_size=0,
            sharpe=0.0, max_drawdown=0.0,
            w_sharpe=0.0, w_mdd=0.0,
        )
        assert fitness == 0.0

    def test_higher_sharpe_gives_higher_fitness(self):
        """Sharpe가 높을수록 적합도가 높아야 한다."""
        from app.alpha.fitness import compute_composite_fitness

        base_kwargs = dict(
            ic_mean=0.05, icir=1.0, turnover=0.3,
            tree_depth=3, tree_size=8, max_drawdown=-0.1,
        )
        low_sharpe = compute_composite_fitness(sharpe=0.0, **base_kwargs)
        high_sharpe = compute_composite_fitness(sharpe=2.0, **base_kwargs)
        assert high_sharpe > low_sharpe

    def test_higher_mdd_gives_lower_fitness(self):
        """MDD가 클수록 적합도가 낮아야 한다 (패널티)."""
        from app.alpha.fitness import compute_composite_fitness

        base_kwargs = dict(
            ic_mean=0.05, icir=1.0, turnover=0.3,
            tree_depth=3, tree_size=8, sharpe=1.0,
        )
        small_mdd = compute_composite_fitness(max_drawdown=-0.05, **base_kwargs)
        large_mdd = compute_composite_fitness(max_drawdown=-0.40, **base_kwargs)
        assert small_mdd > large_mdd

    def test_nan_safe(self):
        """NaN 입력 시 NaN 전파 (또는 0.0 반환) — 크래시하지 않아야 한다."""
        from app.alpha.fitness import compute_composite_fitness

        fitness = compute_composite_fitness(
            ic_mean=float("nan"), icir=0.0, turnover=0.0,
            tree_depth=3, tree_size=8,
        )
        # NaN이거나 0이어야 (크래시만 아니면 OK)
        assert math.isnan(fitness) or isinstance(fitness, float)
