"""알파 마이닝 Pydantic 스키마."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AlphaMineRequest(BaseModel):
    name: str = "Alpha Mining"
    context: str = ""  # Claude에 전달할 시장 맥락
    universe: str = "KOSPI200"  # KOSPI200, KOSDAQ150, KRX300, ALL
    start_date: str  # YYYY-MM-DD
    end_date: str
    max_iterations: int = Field(5, ge=1, le=50)
    ic_threshold: float = Field(0.03, ge=0.0, le=1.0)
    orthogonality_threshold: float = Field(0.7, ge=0.0, le=1.0)
    use_pysr: bool = False  # True=PySR+Julia, False=Claude-only
    pysr_max_size: int = Field(15, ge=5, le=50)
    pysr_parsimony: float = Field(0.03, ge=0.001, le=1.0)


class AlphaMineResponse(BaseModel):
    id: str
    status: str
    created_at: str


class AlphaFactorResponse(BaseModel):
    id: str
    mining_run_id: str | None = None
    name: str
    expression_str: str
    expression_sympy: str | None = None
    polars_code: str | None = None
    hypothesis: str | None = None
    generation: int = 0
    # IC 메트릭
    ic_mean: float | None = None
    ic_std: float | None = None
    icir: float | None = None
    turnover: float | None = None
    sharpe: float | None = None
    max_drawdown: float | None = None
    # 상태
    status: str = "discovered"
    # Phase 2: 인과 검증
    causal_robust: bool | None = None
    causal_effect_size: float | None = None
    causal_p_value: float | None = None
    # Phase 3: 계보 + 복합 팩터
    parent_ids: list[str] | None = None
    factor_type: str = "single"
    component_ids: list[str] | None = None
    # Evolution Engine
    fitness_composite: float | None = None
    tree_depth: int | None = None
    tree_size: int | None = None
    expression_hash: str | None = None
    operator_origin: str | None = None
    is_elite: bool | None = None
    population_active: bool | None = None
    birth_generation: int | None = None
    # 시간
    created_at: str
    updated_at: str


class AlphaMiningRunResponse(BaseModel):
    id: str
    name: str
    context: dict | None = None
    config: dict | None = None
    status: str
    progress: int
    factors_found: int
    total_evaluated: int
    error_message: str | None = None
    has_logs: bool = False
    created_at: str
    completed_at: str | None = None


class AlphaMiningRunSummary(BaseModel):
    id: str
    name: str
    status: str
    progress: int
    factors_found: int
    total_evaluated: int
    created_at: str


class AlphaFactorBacktestRequest(BaseModel):
    factor_id: str
    buy_threshold: float = 0.0  # 팩터 > threshold → BUY
    sell_threshold: float = 0.0  # 팩터 < threshold → SELL
    start_date: str
    end_date: str
    symbols: list[str] = Field(default_factory=list)
    initial_capital: float = 100_000_000
    position_size_pct: float = 0.1
    max_positions: int = 10


class CausalValidationResponse(BaseModel):
    factor_id: str
    is_causally_robust: bool
    causal_effect_size: float
    p_value: float
    placebo_passed: bool
    placebo_effect: float
    random_cause_passed: bool
    random_cause_delta: float
    regime_shift_passed: bool = False
    regime_ate_first_half: float = 0.0
    regime_ate_second_half: float = 0.0
    dag_edges: list[dict] = Field(default_factory=list)


# ── Phase 3: Alpha Factory ──


class AlphaFactoryStartRequest(BaseModel):
    context: str = ""
    universe: str = "KOSPI200"  # KOSPI200, KOSDAQ150, KRX300, ALL
    start_date: str  # YYYY-MM-DD
    end_date: str
    interval_minutes: int = Field(360, ge=1, le=1440)
    max_iterations_per_cycle: int = Field(5, ge=1, le=50)
    ic_threshold: float = Field(0.03, ge=0.0, le=1.0)
    orthogonality_threshold: float = Field(0.7, ge=0.0, le=1.0)
    enable_crossover: bool = True
    enable_causal: bool = False


class AlphaFactoryStatusResponse(BaseModel):
    running: bool
    cycles_completed: int = 0
    factors_discovered_total: int = 0
    current_cycle_progress: int = 0
    last_cycle_at: str | None = None
    config: dict | None = None
    population_size: int = 0
    elite_count: int = 0
    generation: int = 0
    operator_stats: dict | None = None


class CompositeFactorBuildRequest(BaseModel):
    factor_ids: list[str] = Field(..., min_length=2, max_length=20)
    method: str = Field("ic_weighted", pattern=r"^(equal_weight|ic_weighted)$")
    name: str = "Composite Alpha"


class CompositeFactorResponse(BaseModel):
    id: str
    name: str
    factor_type: str
    expression_str: str
    component_ids: list[str]
    ic_mean: float | None = None
    created_at: str


class CorrelationRequest(BaseModel):
    factor_ids: list[str] = Field(..., min_length=2, max_length=20)


class CorrelationMatrixResponse(BaseModel):
    factor_ids: list[str]
    factor_names: list[str]
    matrix: list[list[float]]


class AlphaExperienceResponse(BaseModel):
    id: str
    factor_id: str | None = None
    expression_str: str
    hypothesis: str | None = None
    ic_mean: float | None = None
    success: bool
    generation: int = 0
    parent_ids: list[str] | None = None
    created_at: str


# ── Iteration 로그 (탐색 과정 투명성) ──


class IterationAttempt(BaseModel):
    depth: int  # 0=원본, 1+=변이
    expression_str: str
    hypothesis: str
    ic_mean: float | None = None
    passed_ic: bool = False
    orthogonality_max_corr: float | None = None
    passed_orthogonality: bool | None = None
    outcome: str  # "discovered" | "ic_below_threshold" | "orthogonality_rejected" | "parse_error" | "eval_error"
    error_message: str | None = None


class IterationLog(BaseModel):
    iteration: int
    hypothesis: str
    attempts: list[IterationAttempt] = Field(default_factory=list)
    discovered_factor_name: str | None = None


class MiningLogSummary(BaseModel):
    total_iterations: int = 0
    total_attempts: int = 0
    total_discovered: int = 0
    total_ic_failures: int = 0
    total_parse_errors: int = 0
    total_orthogonality_rejections: int = 0
    avg_ic_all_attempts: float | None = None
    max_ic_failed: float | None = None
    failure_breakdown: dict[str, int] = Field(default_factory=dict)


class MiningIterationLogs(BaseModel):
    run_id: str
    iterations: list[IterationLog] = Field(default_factory=list)
    summary: MiningLogSummary = Field(default_factory=MiningLogSummary)
