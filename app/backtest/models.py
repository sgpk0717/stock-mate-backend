"""백테스트 실행 기록 DB 모델."""

import uuid
from datetime import date, datetime

from sqlalchemy import (
    Date,
    DateTime,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSON, UUID
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
