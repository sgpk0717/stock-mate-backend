"""Add backtest_runs table

Revision ID: b7c8d9e0f1a2
Revises: a1b2c3d4e5f6
Create Date: 2026-02-28 12:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSON, UUID

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b7c8d9e0f1a2"
down_revision: Union[str, None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "backtest_runs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("strategy_name", sa.String(100), nullable=False),
        sa.Column("strategy_json", JSON, nullable=False),
        sa.Column("start_date", sa.Date, nullable=False),
        sa.Column("end_date", sa.Date, nullable=False),
        sa.Column(
            "initial_capital",
            sa.Numeric(18, 2),
            nullable=False,
            server_default="100000000",
        ),
        sa.Column("cost_config", JSON, nullable=True),
        sa.Column(
            "symbol_count", sa.Integer, nullable=False, server_default="0"
        ),
        sa.Column(
            "status", sa.String(20), nullable=False, server_default="'PENDING'"
        ),
        sa.Column("progress", sa.Integer, nullable=False, server_default="0"),
        sa.Column("metrics", JSON, nullable=True),
        sa.Column("equity_curve", JSON, nullable=True),
        sa.Column("trades_summary", JSON, nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("backtest_runs")
