"""알파 마이닝 Pydantic 스키마."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AlphaMineRequest(BaseModel):
    name: str = "Alpha Mining"
    context: str = ""  # Claude에 전달할 시장 맥락
    universe: str = "KOSPI200"  # KOSPI200, KOSDAQ150, KRX300, ALL
    start_date: str  # YYYY-MM-DD
    end_date: str
    interval: str = Field("1d", pattern=r"^(1m|3m|5m|15m|30m|1h|1d)$")
    max_iterations: int = Field(5, ge=1, le=100)
    ic_threshold: float = Field(0.03, ge=0.0, le=1.0)
    orthogonality_threshold: float = Field(0.7, ge=0.0, le=1.0)
    use_pysr: bool = False  # True=PySR+Julia, False=Claude-only
    pysr_max_size: int = Field(15, ge=5, le=50)
    pysr_parsimony: float = Field(0.03, ge=0.001, le=1.0)
    seed_factor_ids: list[str] = Field(default_factory=list)  # 시드 팩터 ID 목록


class AlphaMineResponse(BaseModel):
    id: str
    status: str
    created_at: str


class AlphaFactorResponse(BaseModel):
    id: str
    mining_run_id: str | None = None
    name: str
    expression_str: str
    interval: str | None = None
    expression_sympy: str | None = None
    polars_code: str | None = None
    hypothesis: str | None = None
    generation: int = 0
    # IC 메트릭
    ic_mean: float | None = None
    ic_std: float | None = None
    icir: float | None = None
    turnover: float | None = None       # 포지션 턴오버 (일별 포트폴리오 변경 비율)
    sharpe: float | None = None         # Long-only Sharpe (상위 분위 포트폴리오)
    max_drawdown: float | None = None   # Long-only MDD
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


class AlphaFactorPageResponse(BaseModel):
    items: list[AlphaFactorResponse]
    total: int


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
    start_date: str = ""  # 비어 있으면 마이닝 run config에서 자동 추출
    end_date: str = ""
    interval: str = Field("1d", pattern=r"^(1m|3m|5m|15m|30m|1h|1d)$")
    symbols: list[str] = Field(default_factory=list)
    initial_capital: float = 100_000_000
    top_pct: float = Field(0.2, ge=0.05, le=0.5)  # 상위 몇 % 매수
    max_positions: int = Field(20, ge=1, le=100)
    rebalance_freq: str = Field("weekly", pattern=r"^(every_bar|hourly|daily|weekly|monthly)$")
    band_threshold: float = Field(0.05, ge=0.0, le=0.2)  # 밴드 리밸런싱 임계값
    stop_loss_pct: float = Field(0.0, ge=0.0, le=0.5)  # 포지션 손절 (0=비활성, 0.05=5%)
    max_drawdown_pct: float = Field(0.0, ge=0.0, le=0.5)  # 포트폴리오 서킷 브레이커 (0=비활성)
    eod_liquidation: bool = Field(True, description="장중 단타: 매일 장 마감 시 전량 청산")
    skip_opening_minutes: int = Field(0, ge=0, le=120, description="장 시작 N분 회피 (0=비활성, 30=09:30부터 리밸런싱)")
    engine: str = Field("loop", pattern=r"^(loop|vectorbt)$", description="시뮬레이션 엔진 (loop=기본, vectorbt=Numba 가속)")


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
    data_interval: str = Field("1d", pattern=r"^(1m|3m|5m|15m|30m|1h|1d)$")
    interval_minutes: int = Field(0, ge=0, le=1440)
    max_iterations_per_cycle: int = Field(5, ge=1, le=50)
    ic_threshold: float = Field(0.03, ge=0.0, le=1.0)
    orthogonality_threshold: float = Field(0.7, ge=0.0, le=1.0)
    enable_crossover: bool = True
    max_cycles: int | None = Field(None, ge=1, le=1000)


class AlphaFactoryStatusResponse(BaseModel):
    running: bool
    cycles_completed: int = 0
    factors_discovered_total: int = 0
    current_cycle_progress: int = 0
    current_cycle_message: str = ""
    last_cycle_at: str | None = None
    started_at: str | None = None
    config: dict | None = None
    population_size: int = 0
    elite_count: int = 0
    generation: int = 0
    operator_stats: dict | None = None
    last_funnel: dict | None = None
    user_stopped: bool = False


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


class AutoOptimizeRequest(BaseModel):
    min_ic: float = Field(0.03, ge=0.01, le=0.5)
    min_turnover: float = Field(0.02, ge=0.0, le=1.0)
    max_k: int = Field(7, ge=3, le=15)
    lambda_decorr: float = Field(0.5, ge=0.0, le=2.0)
    shrinkage_delta: float = Field(0.5, ge=0.0, le=1.0)
    interval: str = Field("5m", pattern=r"^(1m|3m|5m|15m|30m|1h|1d)$")
    causal_only: bool = False


class SelectionStepResponse(BaseModel):
    step: int
    selected_id: str
    selected_name: str
    niche: str
    ic: float
    reason: str
    cumulative_ir2: float
    avg_correlation: float


class OptimizationResultResponse(BaseModel):
    k: int
    factor_ids: list[str]
    factor_names: list[str]
    weights: dict[str, float]
    composite_ic: float
    composite_icir: float
    composite_sharpe: float
    avg_correlation: float
    expression_str: str


class AutoOptimizeResponse(BaseModel):
    best_k: int
    results: list[OptimizationResultResponse]
    selection_log: list[SelectionStepResponse]
    correlation_matrix: CorrelationMatrixResponse
    candidate_count: int


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


# ── 팩터 AI 채팅 ──


class FactorChatCreateResponse(BaseModel):
    session_id: str
    source_factor_id: str
    source_expression: str
    universe: str
    interval: str
    status: str


class FactorChatMessageRequest(BaseModel):
    message: str


class FactorChatMessageResponse(BaseModel):
    role: str
    content: str
    timestamp: str
    factor_draft: dict | None = None
    current_expression: str | None = None
    current_metrics: dict | None = None


class FactorChatSessionResponse(BaseModel):
    session_id: str
    messages: list[dict]
    source_factor_id: str
    source_expression: str
    current_expression: str | None = None
    current_metrics: dict | None = None
    universe: str
    interval: str
    status: str
    created_at: str
    updated_at: str
