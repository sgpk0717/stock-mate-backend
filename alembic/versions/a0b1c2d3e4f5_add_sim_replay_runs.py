"""add sim_replay_runs table

Revision ID: a0b1c2d3e4f5
Revises: z8a9b0c1d2e3
Create Date: 2026-03-18

시뮬레이션 리플레이 히스토리 저장.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSON, UUID

revision = "a0b1c2d3e4f5"
down_revision = "z8a9b0c1d2e3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "sim_replay_runs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("interval", sa.String(10), nullable=False, server_default="5m"),
        sa.Column("strategy_preset", sa.String(50), nullable=False),
        sa.Column("factor_id", UUID(as_uuid=True), nullable=True),
        sa.Column("data_source", sa.String(20), nullable=False, server_default="synthetic"),
        sa.Column("config", JSON, nullable=False, server_default="{}"),
        sa.Column("total_bars", sa.Integer, nullable=False, server_default="0"),
        sa.Column("total_events", sa.Integer, nullable=False, server_default="0"),
        sa.Column("pnl", sa.Numeric, nullable=True),
        sa.Column("pnl_pct", sa.Float, nullable=True),
        sa.Column("events", JSON, nullable=False, server_default="[]"),
        sa.Column("final_state", JSON, nullable=True),
        sa.Column("analysis", sa.Text, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_sim_replay_runs_created_at", "sim_replay_runs", ["created_at"])
    op.create_index("ix_sim_replay_runs_symbol", "sim_replay_runs", ["symbol"])


def downgrade() -> None:
    op.drop_index("ix_sim_replay_runs_symbol")
    op.drop_index("ix_sim_replay_runs_created_at")
    op.drop_table("sim_replay_runs")
