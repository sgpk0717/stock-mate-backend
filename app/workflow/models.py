"""워크플로우 DB 모델 — TradingContext 영속화 + 일일 워크플로우 추적."""

import uuid
from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class TradingContextModel(Base):
    """TradingContext DB 영속화 모델."""

    __tablename__ = "trading_contexts"
    __table_args__ = (
        Index("ix_tc_status", "status"),
        Index("ix_tc_mode", "mode"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    mode: Mapped[str] = mapped_column(
        String(10), nullable=False, server_default="paper"
    )
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default="active"
    )

    # 전략
    strategy: Mapped[dict] = mapped_column(JSON, nullable=False, server_default="{}")
    strategy_name: Mapped[str] = mapped_column(
        String(200), nullable=False, server_default=""
    )

    # 포지션/리스크
    position_sizing: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    scaling: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    risk_management: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    cost_config: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # 자본
    initial_capital: Mapped[float] = mapped_column(
        Numeric(18, 2), server_default="100000000"
    )
    position_size_pct: Mapped[float] = mapped_column(Float, server_default="0.1")
    max_positions: Mapped[int] = mapped_column(Integer, server_default="10")

    # 종목
    symbols: Mapped[list | None] = mapped_column(JSON, nullable=True)

    # 출처 추적
    source_backtest_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )
    source_factor_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )
    auto_created: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false"
    )

    # 세션 상태 (C2: 서버 재시작 시 LiveSession 복구용)
    session_state: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class LiveTrade(Base):
    """실시간 매매 로그 DB 영속화."""

    __tablename__ = "live_trades"
    __table_args__ = (
        Index("ix_lt_context_date", "context_id", "executed_at"),
        Index("ix_lt_symbol", "symbol", "executed_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    context_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("trading_contexts.id", ondelete="CASCADE"),
        nullable=False,
    )
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    side: Mapped[str] = mapped_column(String(4), nullable=False)  # BUY/SELL
    step: Mapped[str] = mapped_column(
        String(10), nullable=False, server_default=""
    )  # B1/B2/S-STOP/S-TRAIL/S-HALF
    qty: Mapped[int] = mapped_column(Integer, nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    expected_price: Mapped[float | None] = mapped_column(Float, nullable=True)  # TCA: 시그널 시점 기대가
    pnl_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    pnl_amount: Mapped[float | None] = mapped_column(Numeric(18, 2), nullable=True)
    holding_minutes: Mapped[float | None] = mapped_column(Float, nullable=True)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    order_id: Mapped[str] = mapped_column(String(50), nullable=False, server_default="")
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    snapshot: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    conditions: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    sizing: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    position_context: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    executed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class WorkflowRun(Base):
    """일일 워크플로우 실행 추적."""

    __tablename__ = "workflow_runs"
    __table_args__ = (
        UniqueConstraint("date", name="uq_wf_date"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    date: Mapped[date] = mapped_column(Date, nullable=False)
    phase: Mapped[str] = mapped_column(
        String(30), nullable=False, server_default="IDLE"
    )
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default="PENDING"
    )

    # 설정
    config: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # 연결 ID
    mining_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )
    selected_factor_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )
    trading_context_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )

    # 실적
    review_summary: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    trade_count: Mapped[int] = mapped_column(Integer, server_default="0")
    pnl_amount: Mapped[float | None] = mapped_column(Numeric(18, 2), nullable=True)
    pnl_pct: Mapped[float | None] = mapped_column(Float, nullable=True)

    # 단계별 수행 상태 (중복 실행/누락 감지)
    step_status: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # 마이닝 컨텍스트
    mining_context: Mapped[str | None] = mapped_column(Text, nullable=True)

    # 타임스탬프
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class WorkflowEvent(Base):
    """워크플로우 이벤트 감사 로그."""

    __tablename__ = "workflow_events"
    __table_args__ = (
        Index("ix_wfe_run", "workflow_run_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    workflow_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("workflow_runs.id", ondelete="CASCADE"),
        nullable=True,
    )
    phase: Mapped[str | None] = mapped_column(String(30), nullable=True)
    event_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    data: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class LiveFeedback(Base):
    """실매매 피드백 — 팩터별 실적 추적."""

    __tablename__ = "live_feedback"
    __table_args__ = (
        Index("ix_lf_factor", "factor_id", "date"),
        Index("ix_lf_date", "date"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    factor_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False
    )
    workflow_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )
    date: Mapped[date] = mapped_column(Date, nullable=False)

    # 실적
    realized_pnl: Mapped[float | None] = mapped_column(Numeric(18, 2), nullable=True)
    realized_pnl_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    realized_sharpe: Mapped[float | None] = mapped_column(Float, nullable=True)
    trade_count: Mapped[int] = mapped_column(Integer, server_default="0")
    win_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_holding_minutes: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_single_loss_pct: Mapped[float | None] = mapped_column(Float, nullable=True)

    # 갭 분석
    gap_to_backtest_sharpe: Mapped[float | None] = mapped_column(Float, nullable=True)
    gap_to_ic_prediction: Mapped[float | None] = mapped_column(Float, nullable=True)

    # 시장 컨텍스트
    market_regime: Mapped[str | None] = mapped_column(String(20), nullable=True)
    feedback_context: Mapped[str | None] = mapped_column(Text, nullable=True)

    # 파라미터 피드백 (통합 피드백 시스템)
    param_snapshot: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    param_adjustments: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
