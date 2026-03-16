"""Add indexes to backtest_runs for query performance.

Revision ID: x6y7z8a9b0c1
Revises: w5x6y7z8a9b0
Create Date: 2026-03-16
"""

from alembic import op

revision = "x6y7z8a9b0c1"
down_revision = "w5x6y7z8a9b0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index("ix_backtest_runs_created_at", "backtest_runs", ["created_at"])
    op.create_index("ix_backtest_runs_status", "backtest_runs", ["status"])


def downgrade() -> None:
    op.drop_index("ix_backtest_runs_status", table_name="backtest_runs")
    op.drop_index("ix_backtest_runs_created_at", table_name="backtest_runs")
