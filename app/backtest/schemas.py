"""백테스트 Pydantic 스키마."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ConditionSchema(BaseModel):
    indicator: str
    params: dict = Field(default_factory=dict)
    op: str = ">="
    value: float = 0


class PositionSizingSchema(BaseModel):
    mode: str = "fixed"  # "fixed" | "conviction" | "atr_target" | "kelly"
    weights: dict[str, float] | None = None
    atr_period: int = 14
    target_risk_pct: float = 0.02
    kelly_fraction: float = 0.5


class ScalingSchema(BaseModel):
    enabled: bool = False
    initial_pct: float = 0.5
    scale_in_trigger: str = "price_drop"  # "price_drop" | "support_touch"
    scale_in_drop_pct: float = 3.0
    max_scale_in: int = 1
    partial_exit_pct: float = 0.5
    partial_exit_gain_pct: float = 5.0


class RiskManagementSchema(BaseModel):
    stop_loss_pct: float | None = None
    trailing_stop_pct: float | None = None
    atr_stop_multiplier: float | None = None


class StrategySchema(BaseModel):
    name: str = ""
    description: str = ""
    timeframe: str = "1d"
    buy_conditions: list[ConditionSchema] = Field(default_factory=list)
    buy_logic: str = "AND"
    sell_conditions: list[ConditionSchema] = Field(default_factory=list)
    sell_logic: str = "OR"
    position_sizing: PositionSizingSchema | None = None
    scaling: ScalingSchema | None = None
    risk_management: RiskManagementSchema | None = None


class CostConfigSchema(BaseModel):
    buy_commission: float = 0.00015
    sell_commission: float = 0.00215
    slippage_pct: float = 0.001


class BacktestRunCreate(BaseModel):
    strategy: StrategySchema
    start_date: str  # YYYY-MM-DD
    end_date: str
    initial_capital: float = 100_000_000
    symbols: list[str] | None = None
    position_size_pct: float = 0.1
    max_positions: int = 10
    cost_config: CostConfigSchema | None = None


class AIStrategyRequest(BaseModel):
    prompt: str


class AIStrategyResponse(BaseModel):
    strategy: StrategySchema
    explanation: str


class BacktestRunResponse(BaseModel):
    id: str
    strategy_name: str
    strategy_json: dict
    start_date: str
    end_date: str
    initial_capital: float
    symbol_count: int
    status: str
    progress: int
    metrics: dict | None = None
    equity_curve: list[dict] | None = None
    trades_summary: list[dict] | None = None
    error_message: str | None = None
    created_at: str
    completed_at: str | None = None


class BacktestRunSummary(BaseModel):
    id: str
    strategy_name: str
    start_date: str
    end_date: str
    status: str
    progress: int
    total_return: float | None = None
    mdd: float | None = None
    win_rate: float | None = None
    total_trades: int | None = None
    created_at: str


class StrategyInfo(BaseModel):
    name: str
    description: str
    strategy: StrategySchema
