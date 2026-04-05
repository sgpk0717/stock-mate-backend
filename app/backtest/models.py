"""백테스트 실행 기록 DB 모델."""

import uuid
from datetime import date, datetime

from sqlalchemy import (
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
from sqlalchemy.dialects.postgresql import JSON, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class BacktestRun(Base):
    __tablename__ = "backtest_runs"
    __table_args__ = (
        Index("ix_backtest_runs_created_at", "created_at"),
        Index("ix_backtest_runs_status", "status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    strategy_name: Mapped[str] = mapped_column(String(100), nullable=False)
    strategy_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date] = mapped_column(Date, nullable=False)
    initial_capital: Mapped[float] = mapped_column(
        Numeric(18, 2), nullable=False, server_default="100000000"
    )
    cost_config: Mapped[dict] = mapped_column(JSON, nullable=True)
    symbol_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0"
    )
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default="'PENDING'"
    )
    progress: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0"
    )

    # 결과
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    equity_curve: Mapped[list | None] = mapped_column(JSON, nullable=True)
    trades_summary: Mapped[list | None] = mapped_column(JSON, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class BacktestDailySnapshot(Base):
    """백테스트 거래별 일별 스냅샷 (보유기간 타임라인용)."""

    __tablename__ = "backtest_daily_snapshots"
    __table_args__ = (
        UniqueConstraint(
            "backtest_run_id", "trade_index", "snapshot_date",
            name="uq_bds_run_trade_date",
        ),
        Index("idx_bds_run_trade", "backtest_run_id", "trade_index"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    backtest_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("backtest_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    trade_index: Mapped[int] = mapped_column(Integer, nullable=False)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    snapshot_date: Mapped[date] = mapped_column(Date, nullable=False)
    close: Mapped[float | None] = mapped_column(Float, nullable=True)
    variables: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")
