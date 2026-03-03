"""Phase 4: 시뮬레이션 + MCP DB 모델."""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class StressTestRun(Base):
    __tablename__ = "stress_test_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    strategy_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    scenario_type: Mapped[str] = mapped_column(String(50), nullable=False)
    scenario_config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    agent_config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    exchange_config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default="'PENDING'"
    )
    progress: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0"
    )
    total_steps: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="1000"
    )
    results: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class McpAuditLog(Base):
    __tablename__ = "mcp_audit_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tool_name: Mapped[str] = mapped_column(String(100), nullable=False)
    input_params: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    output: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default="'success'"
    )
    blocked_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    execution_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
