"""팩토리 스케줄러 진화 통합 테스트.

Phase 3: scheduler.py가 EvolutionEngine을 사용하는지 검증.
Phase 5: Graceful Degradation (Claude 장애 시 AST fallback).
"""

from __future__ import annotations

import pytest

# pykrx가 설치되지 않은 환경에서는 scheduler import 불가
try:
    from app.alpha.scheduler import AlphaFactoryScheduler, _FactoryState
    HAS_PYKRX = True
except (ImportError, ModuleNotFoundError):
    HAS_PYKRX = False


_skip_no_pykrx = pytest.mark.skipif(
    not HAS_PYKRX, reason="pykrx not installed (Docker-only dependency)"
)


class TestSchedulerUsesEvolutionEngine:
    """스케줄러가 EvolutionEngine을 사용하는지 검증."""

    @_skip_no_pykrx
    def test_scheduler_has_evolution_engine_attr(self):
        """스케줄러에 _evolution_engine 속성이 있어야 한다."""
        scheduler = AlphaFactoryScheduler()
        assert hasattr(scheduler, "_evolution_engine")

    @_skip_no_pykrx
    def test_scheduler_has_operator_registry(self):
        """스케줄러에 _operator_registry 속성이 있어야 한다."""
        scheduler = AlphaFactoryScheduler()
        assert hasattr(scheduler, "_operator_registry")

    @_skip_no_pykrx
    def test_factory_state_has_generation(self):
        """_FactoryState에 generation 필드가 있어야 한다."""
        state = _FactoryState()
        assert hasattr(state, "generation")
        assert state.generation == 0

    @_skip_no_pykrx
    def test_factory_state_has_population_size(self):
        """_FactoryState에 population_size 필드가 있어야 한다."""
        state = _FactoryState()
        assert hasattr(state, "population_size")

    @_skip_no_pykrx
    def test_factory_status_includes_generation(self):
        """get_status()가 generation을 포함해야 한다."""
        scheduler = AlphaFactoryScheduler()
        status = scheduler.get_status()
        assert "generation" in status


class TestGracefulDegradation:
    """Claude API 장애 시 Graceful Degradation 테스트."""

    def test_operator_registry_fallback_on_llm_failure(self):
        """LLM 연산자 연속 실패 시 AST 연산자로 fallback."""
        from app.alpha.operators import OperatorRegistry

        registry = OperatorRegistry(llm_ratio=0.1)

        # LLM 연산자 3회 연속 실패 시뮬레이션
        registry.record_llm_failure()
        registry.record_llm_failure()
        registry.record_llm_failure()

        # LLM이 비활성화되어야
        assert registry.is_llm_disabled() is True

        # select는 AST만 반환해야
        for _ in range(20):
            name = registry.select()
            assert not name.startswith("llm_"), "LLM selected after 3 failures"

    def test_operator_registry_resets_after_cycle(self):
        """다음 사이클에서 LLM 장애 카운터 리셋."""
        from app.alpha.operators import OperatorRegistry

        registry = OperatorRegistry(llm_ratio=0.1)

        # 3회 실패 → 비활성화
        for _ in range(3):
            registry.record_llm_failure()
        assert registry.is_llm_disabled() is True

        # 리셋
        registry.reset_llm_failures()
        assert registry.is_llm_disabled() is False


class TestConfigEvolutionSettings:
    """config.py에 진화 엔진 설정이 추가되었는지 확인."""

    def test_population_size_setting(self):
        """ALPHA_POPULATION_SIZE 설정 존재."""
        from app.core.config import settings

        assert hasattr(settings, "ALPHA_POPULATION_SIZE")
        assert isinstance(settings.ALPHA_POPULATION_SIZE, int)
        assert settings.ALPHA_POPULATION_SIZE > 0

    def test_elite_pct_setting(self):
        """ALPHA_ELITE_PCT 설정 존재."""
        from app.core.config import settings

        assert hasattr(settings, "ALPHA_ELITE_PCT")
        assert 0.0 < settings.ALPHA_ELITE_PCT < 1.0

    def test_ast_mutation_ratio_setting(self):
        """ALPHA_AST_MUTATION_RATIO 설정 존재."""
        from app.core.config import settings

        assert hasattr(settings, "ALPHA_AST_MUTATION_RATIO")
        assert 0.0 < settings.ALPHA_AST_MUTATION_RATIO <= 1.0

    def test_llm_mutation_ratio_setting(self):
        """ALPHA_LLM_MUTATION_RATIO 설정 존재."""
        from app.core.config import settings

        assert hasattr(settings, "ALPHA_LLM_MUTATION_RATIO")
        assert 0.0 <= settings.ALPHA_LLM_MUTATION_RATIO < 1.0

    def test_ratios_sum_to_one(self):
        """AST + LLM 비율 합이 1.0이어야 한다."""
        from app.core.config import settings

        total = settings.ALPHA_AST_MUTATION_RATIO + settings.ALPHA_LLM_MUTATION_RATIO
        assert abs(total - 1.0) < 0.01

    def test_fitness_weights_sum_to_one(self):
        """적합도 가중치 합이 1.0이어야 한다."""
        from app.core.config import settings

        total = (
            settings.ALPHA_FITNESS_W_IC
            + settings.ALPHA_FITNESS_W_ICIR
            + settings.ALPHA_FITNESS_W_SHARPE
            + settings.ALPHA_FITNESS_W_MDD
            + settings.ALPHA_FITNESS_W_TURNOVER
            + settings.ALPHA_FITNESS_W_COMPLEXITY
        )
        assert abs(total - 1.0) < 0.01

    def test_tournament_k_setting(self):
        """ALPHA_TOURNAMENT_K 설정 존재 (5로 증가)."""
        from app.core.config import settings

        assert hasattr(settings, "ALPHA_TOURNAMENT_K")
        assert settings.ALPHA_TOURNAMENT_K >= 3


class TestSchemaExtensions:
    """스키마 확장 테스트."""

    def test_alpha_factor_response_has_fitness_fields(self):
        """AlphaFactorResponse에 진화 관련 필드 존재."""
        from app.alpha.schemas import AlphaFactorResponse

        fields = AlphaFactorResponse.model_fields
        for field_name in [
            "fitness_composite",
            "tree_depth",
            "tree_size",
            "expression_hash",
            "operator_origin",
            "is_elite",
            "population_active",
            "birth_generation",
        ]:
            assert field_name in fields, f"Missing field: {field_name}"

    def test_alpha_factory_status_has_generation(self):
        """AlphaFactoryStatusResponse에 generation 필드 존재."""
        from app.alpha.schemas import AlphaFactoryStatusResponse

        fields = AlphaFactoryStatusResponse.model_fields
        assert "generation" in fields
        assert "population_size" in fields
        assert "operator_stats" in fields


class TestModelColumns:
    """DB 모델 컬럼 테스트."""

    def test_alpha_factor_has_evolution_columns(self):
        """AlphaFactor 모델에 진화 컬럼이 추가되어야 한다."""
        from app.alpha.models import AlphaFactor

        for col_name in [
            "fitness_composite",
            "tree_depth",
            "tree_size",
            "expression_hash",
            "operator_origin",
            "is_elite",
            "population_active",
            "birth_generation",
        ]:
            assert hasattr(AlphaFactor, col_name), (
                f"AlphaFactor missing column: {col_name}"
            )
