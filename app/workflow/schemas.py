"""워크플로우 Pydantic 스키마."""

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, Field


# ── WorkflowRun ──

class WorkflowRunOut(BaseModel):
    id: str
    date: date
    phase: str
    status: str
    config: dict | None = None
    mining_run_id: str | None = None
    selected_factor_id: str | None = None
    trading_context_id: str | None = None
    review_summary: dict | None = None
    trade_count: int = 0
    pnl_amount: float | None = None
    pnl_pct: float | None = None
    mining_context: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class WorkflowStatusOut(BaseModel):
    """워크플로우 현재 상태 요약 (MCP/OpenClaw용)."""

    phase: str
    date: Optional[date] = None
    status: str = "NO_RUN"
    selected_factor_name: str | None = None
    selected_factor_ic: float | None = None
    selected_factor_sharpe: float | None = None
    trade_count: int = 0
    pnl_pct: float | None = None
    mining_cycles: int | None = None
    factors_discovered: int | None = None
    system_ok: bool = True
    message: str = ""


class WorkflowEventOut(BaseModel):
    id: str
    workflow_run_id: str | None = None
    phase: str | None = None
    event_type: str | None = None
    message: str | None = None
    data: dict | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class WorkflowTriggerRequest(BaseModel):
    phase: str = Field(..., description="트리거할 페이즈: pre_market, market_open, market_close, review, mining")


class WorkflowTriggerResponse(BaseModel):
    success: bool
    phase: str
    message: str


# ── TradingContext ──

class TradingContextOut(BaseModel):
    id: str
    mode: str
    status: str
    strategy: dict
    strategy_name: str
    position_sizing: dict | None = None
    scaling: dict | None = None
    risk_management: dict | None = None
    cost_config: dict | None = None
    initial_capital: float
    position_size_pct: float
    max_positions: int
    symbols: list[str] | None = None
    source_backtest_id: str | None = None
    source_factor_id: str | None = None
    auto_created: bool = False
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ── AutoSelector ──

class BestFactorOut(BaseModel):
    """팩터 자동 선택 결과 (설계서 §8)."""

    factor_id: str
    factor_name: str
    expression_str: str
    ic_mean: float | None = None
    icir: float | None = None
    sharpe: float | None = None
    max_drawdown: float | None = None
    causal_robust: bool | None = None
    interval: str = "1m"
    composite_score: float
    score_breakdown: dict


# ── LiveFeedback ──

class LiveFeedbackOut(BaseModel):
    id: str
    factor_id: str
    workflow_run_id: str | None = None
    date: date
    realized_pnl: float | None = None
    realized_pnl_pct: float | None = None
    realized_sharpe: float | None = None
    trade_count: int = 0
    win_rate: float | None = None
    avg_holding_minutes: float | None = None
    max_single_loss_pct: float | None = None
    gap_to_backtest_sharpe: float | None = None
    gap_to_ic_prediction: float | None = None
    market_regime: str | None = None
    feedback_context: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class TradingFeedbackSubmit(BaseModel):
    """OpenClaw이 제출하는 매매 피드백."""

    date: date
    factor_id: str
    review_text: str = ""
    improvement_suggestions: list[str] = []
    market_regime: str | None = None
