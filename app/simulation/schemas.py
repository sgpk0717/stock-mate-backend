"""Phase 4: 시뮬레이션 + MCP Pydantic 스키마."""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Agent Config ──────────────────────────────────────────


class AgentConfig(BaseModel):
    fundamental_count: int = Field(20, ge=0, le=100)
    chartist_count: int = Field(30, ge=0, le=100)
    noise_count: int = Field(100, ge=0, le=500)
    llm_count: int = Field(5, ge=0, le=20)
    llm_call_interval: int = Field(20, ge=5, le=100)


# ── Exchange Config ───────────────────────────────────────


class ExchangeConfig(BaseModel):
    initial_price: float = Field(50000.0, ge=1.0)
    tick_size: float = Field(10.0, ge=1.0)
    total_steps: int = Field(1000, ge=100, le=10000)
    seed: int | None = None


# ── Scenario ──────────────────────────────────────────────


class ScenarioConfig(BaseModel):
    type: str
    params: dict = Field(default_factory=dict)
    inject_at_step: int = Field(500, ge=0)


class ScenarioPreset(BaseModel):
    type: str
    name: str
    description: str
    default_params: dict


# ── Stress Test Request / Response ────────────────────────


class StressTestRequest(BaseModel):
    name: str = "Stress Test"
    strategy_json: dict
    scenario: ScenarioConfig
    agent_config: AgentConfig = Field(default_factory=AgentConfig)
    exchange_config: ExchangeConfig = Field(default_factory=ExchangeConfig)


class CustomScenarioRequest(BaseModel):
    prompt: str = Field(..., min_length=5, max_length=1000)


class StressTestResponse(BaseModel):
    id: str
    status: str
    created_at: str


class StressTestRunResponse(BaseModel):
    id: str
    name: str
    strategy_json: dict
    scenario_type: str
    scenario_config: dict | None = None
    agent_config: dict | None = None
    exchange_config: dict | None = None
    status: str
    progress: int
    total_steps: int
    results: dict | None = None
    metrics: dict | None = None
    error_message: str | None = None
    created_at: str
    completed_at: str | None = None


class StressTestRunSummary(BaseModel):
    id: str
    name: str
    scenario_type: str
    status: str
    progress: int
    strategy_pnl: float | None = None
    crash_depth: float | None = None
    created_at: str


class CustomScenarioResponse(BaseModel):
    scenario: ScenarioConfig
    explanation: str


# ── MCP ───────────────────────────────────────────────────


class McpStatusResponse(BaseModel):
    running: bool
    sse_port: int
    governance: dict


class McpToolResponse(BaseModel):
    name: str
    description: str


class McpAuditLogResponse(BaseModel):
    id: str
    tool_name: str
    input_params: dict | None = None
    output: dict | None = None
    status: str
    blocked_reason: str | None = None
    execution_ms: int | None = None
    created_at: str


class GovernanceRulesUpdate(BaseModel):
    max_order_qty: int | None = None
    allowed_actions: list[str] | None = None
    require_human_approval_real: bool | None = None
    enabled: bool | None = None
