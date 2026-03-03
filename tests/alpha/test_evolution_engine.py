"""EvolutionEngine 단위 테스트.

Phase 3: evolution_engine.py의 EvolutionEngine.
DB 의존성은 mock으로 처리.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import polars as pl
import pytest
import sympy

from app.alpha.ast_converter import parse_expression


@pytest.fixture
def sample_evolution_data() -> pl.DataFrame:
    """진화 엔진 테스트용 OHLCV 데이터 (500행, 2종목)."""
    rng = np.random.default_rng(42)
    n = 250
    symbols = ["005930", "000660"]
    frames = []

    for sym in symbols:
        base_price = 50000.0
        returns = rng.normal(0.0005, 0.02, n)
        prices = base_price * np.cumprod(1 + returns)
        close = prices
        open_ = close * (1 + rng.normal(0, 0.005, n))
        high = np.maximum(close, open_) * (1 + rng.uniform(0, 0.01, n))
        low = np.minimum(close, open_) * (1 - rng.uniform(0, 0.01, n))
        volume = rng.integers(100_000, 10_000_000, n).astype(float)

        dates = pl.date_range(
            pl.date(2024, 1, 2),
            pl.date(2024, 1, 2) + pl.duration(days=n - 1),
            eager=True,
        )
        frames.append(pl.DataFrame({
            "dt": dates,
            "symbol": [sym] * n,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }))

    return pl.concat(frames)


class TestEvolutionEngineDeduplicate:
    """expression_hash 기반 중복 제거 테스트."""

    def test_dedup_removes_structural_duplicates(self):
        """같은 구조(상수만 다른) 수식은 하나만 남아야 한다."""
        from app.alpha.evolution import ScoredFactor
        from app.alpha.ast_converter import expression_hash
        from app.alpha.evolution_engine import EvolutionEngine

        # 같은 구조, 다른 상수
        expr_a = parse_expression("close * 2.0 + rsi")
        expr_b = parse_expression("close * 5.0 + rsi")

        pop = [
            ScoredFactor(
                expression=expr_a,
                expression_str="close * 2.0 + rsi",
                hypothesis="test_a",
                ic_mean=0.04,
                generation=0,
                fitness_composite=0.04,
                expression_hash=expression_hash(expr_a),
            ),
            ScoredFactor(
                expression=expr_b,
                expression_str="close * 5.0 + rsi",
                hypothesis="test_b",
                ic_mean=0.06,
                generation=0,
                fitness_composite=0.06,
                expression_hash=expression_hash(expr_b),
            ),
        ]

        engine = EvolutionEngine.__new__(EvolutionEngine)
        result = engine._deduplicate(pop)
        assert len(result) == 1
        # 더 높은 fitness가 남아야
        assert result[0].fitness_composite == 0.06

    def test_dedup_keeps_different_structures(self):
        """다른 구조의 수식은 모두 유지해야 한다."""
        from app.alpha.evolution import ScoredFactor
        from app.alpha.ast_converter import expression_hash
        from app.alpha.evolution_engine import EvolutionEngine

        expr_a = parse_expression("close * rsi")
        expr_b = parse_expression("close + rsi")

        pop = [
            ScoredFactor(
                expression=expr_a,
                expression_str="close * rsi",
                hypothesis="test_a",
                ic_mean=0.04,
                generation=0,
                fitness_composite=0.04,
                expression_hash=expression_hash(expr_a),
            ),
            ScoredFactor(
                expression=expr_b,
                expression_str="close + rsi",
                hypothesis="test_b",
                ic_mean=0.05,
                generation=0,
                fitness_composite=0.05,
                expression_hash=expression_hash(expr_b),
            ),
        ]

        engine = EvolutionEngine.__new__(EvolutionEngine)
        result = engine._deduplicate(pop)
        assert len(result) == 2


class TestEvolutionEngineElitism:
    """엘리트 보존 테스트."""

    def test_top_percent_preserved(self):
        """상위 5% 개체가 다음 세대에 그대로 보존되어야 한다."""
        from app.alpha.evolution import ScoredFactor
        from app.alpha.evolution_engine import EvolutionEngine

        # 20개 모집단 → 상위 1개가 엘리트 (5%)
        pop = []
        for i in range(20):
            expr = parse_expression(f"close * {i + 1}.0 + rsi")
            pop.append(ScoredFactor(
                expression=expr,
                expression_str=f"close * {i + 1}.0 + rsi",
                hypothesis=f"test_{i}",
                ic_mean=0.01 * (i + 1),
                generation=0,
                fitness_composite=0.01 * (i + 1),
                expression_hash=f"hash_{i}",
            ))

        engine = EvolutionEngine.__new__(EvolutionEngine)
        engine._population_size = 20
        engine._elite_pct = 0.05

        elites = engine._select_elites(pop)
        assert len(elites) == 1  # ceil(20 * 0.05)
        # 가장 높은 fitness가 엘리트
        assert elites[0].fitness_composite == 0.20


class TestEvolutionEngineWalkForward:
    """Walk-Forward 데이터 분할 테스트."""

    def test_train_val_split_ratio(self, sample_evolution_data):
        """Train 70% / Validation 30% 비율 확인."""
        from app.alpha.evolution_engine import EvolutionEngine

        engine = EvolutionEngine.__new__(EvolutionEngine)
        engine._data = sample_evolution_data
        engine._train_ratio = 0.7

        train, val = engine._split_train_val()

        total_dates = sample_evolution_data.select("dt").unique().height
        train_dates = train.select("dt").unique().height
        val_dates = val.select("dt").unique().height

        # 70% ± 5% 허용
        ratio = train_dates / total_dates
        assert 0.65 <= ratio <= 0.75, f"Train ratio {ratio} not in expected range"
        assert val_dates > 0, "Validation set is empty"

    def test_train_val_no_overlap(self, sample_evolution_data):
        """Train과 Validation 날짜가 겹치지 않아야 한다."""
        from app.alpha.evolution_engine import EvolutionEngine

        engine = EvolutionEngine.__new__(EvolutionEngine)
        engine._data = sample_evolution_data
        engine._train_ratio = 0.7

        train, val = engine._split_train_val()

        train_dates = set(train["dt"].to_list())
        val_dates = set(val["dt"].to_list())
        overlap = train_dates & val_dates
        assert len(overlap) == 0, f"Found {len(overlap)} overlapping dates"

    def test_val_dates_after_train(self, sample_evolution_data):
        """Validation 날짜가 Train 날짜 이후여야 한다."""
        from app.alpha.evolution_engine import EvolutionEngine

        engine = EvolutionEngine.__new__(EvolutionEngine)
        engine._data = sample_evolution_data
        engine._train_ratio = 0.7

        train, val = engine._split_train_val()

        train_max = train["dt"].max()
        val_min = val["dt"].min()
        assert val_min > train_max, "Validation starts before train ends"


class TestEvolutionEngineOverfitFilter:
    """과적합 필터 테스트."""

    def test_overfit_factor_rejected(self):
        """Train IC 통과 + Val IC 미달 → 과적합으로 거부."""
        from app.alpha.evolution_engine import EvolutionEngine

        engine = EvolutionEngine.__new__(EvolutionEngine)
        engine._ic_threshold = 0.03

        # Train IC=0.05 (통과), Val IC=0.01 (미달: < 0.03 * 0.5 = 0.015)
        is_overfit = engine._is_overfit(train_ic=0.05, val_ic=0.01)
        assert is_overfit is True

    def test_robust_factor_accepted(self):
        """Train IC 통과 + Val IC도 통과 → 정상."""
        from app.alpha.evolution_engine import EvolutionEngine

        engine = EvolutionEngine.__new__(EvolutionEngine)
        engine._ic_threshold = 0.03

        # Train IC=0.05, Val IC=0.04 (> 0.03 * 0.5 = 0.015)
        is_overfit = engine._is_overfit(train_ic=0.05, val_ic=0.04)
        assert is_overfit is False
