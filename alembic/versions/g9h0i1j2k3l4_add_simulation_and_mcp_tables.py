"""add simulation and mcp tables

Revision ID: g9h0i1j2k3l4
Revises: f8g9h0i1j2k3
Create Date: 2026-03-02

Phase 4: 금융 월드 모델 (ABM) + MCP 통합
- stress_test_runs 테이블 (스트레스 테스트 실행/결과)
- mcp_audit_logs 테이블 (MCP 도구 호출 감사 로그)
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSON, UUID

# revision identifiers, used by Alembic.
revision = "g9h0i1j2k3l4"
down_revision = "f8g9h0i1j2k3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. stress_test_runs 테이블
    op.create_table(
        "stress_test_runs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("strategy_json", JSON, nullable=False),
        sa.Column("scenario_type", sa.String(50), nullable=False),
        sa.Column("scenario_config", JSON, nullable=True),
        sa.Column("agent_config", JSON, nullable=True),
        sa.Column("exchange_config", JSON, nullable=True),
        sa.Column(
            "status", sa.String(20), nullable=False, server_default="'PENDING'"
        ),
        sa.Column("progress", sa.Integer, nullable=False, server_default="0"),
        sa.Column(
            "total_steps", sa.Integer, nullable=False, server_default="1000"
        ),
        sa.Column("results", JSON, nullable=True),
        sa.Column("metrics", JSON, nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_stress_test_runs_status", "stress_test_runs", ["status"]
    )

    # 2. mcp_audit_logs 테이블
    op.create_table(
        "mcp_audit_logs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("tool_name", sa.String(100), nullable=False),
        sa.Column("input_params", JSON, nullable=True),
        sa.Column("output", JSON, nullable=True),
        sa.Column(
            "status", sa.String(20), nullable=False, server_default="'success'"
        ),
        sa.Column("blocked_reason", sa.Text, nullable=True),
        sa.Column("execution_ms", sa.Integer, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )
    op.create_index(
        "ix_mcp_audit_logs_tool_name", "mcp_audit_logs", ["tool_name"]
    )
    op.create_index(
        "ix_mcp_audit_logs_created_at", "mcp_audit_logs", ["created_at"]
    )


def downgrade() -> None:
    op.drop_index("ix_mcp_audit_logs_created_at", table_name="mcp_audit_logs")
    op.drop_index("ix_mcp_audit_logs_tool_name", table_name="mcp_audit_logs")
    op.drop_table("mcp_audit_logs")
    op.drop_index("ix_stress_test_runs_status", table_name="stress_test_runs")
    op.drop_table("stress_test_runs")
