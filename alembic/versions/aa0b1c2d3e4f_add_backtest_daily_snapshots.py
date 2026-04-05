"""add backtest_daily_snapshots

Revision ID: aa0b1c2d3e4f
Revises: 63e4f792433a
Create Date: 2026-04-02

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


# revision identifiers, used by Alembic.
revision: str = "aa0b1c2d3e4f"
down_revision: Union[str, None] = "63e4f792433a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "backtest_daily_snapshots",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "backtest_run_id",
            sa.dialects.postgresql.UUID(as_uuid=True),
            sa.ForeignKey("backtest_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("trade_index", sa.Integer, nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("snapshot_date", sa.Date, nullable=False),
        sa.Column("close", sa.Float, nullable=True),
        sa.Column("variables", JSONB, nullable=False, server_default="{}"),
        sa.UniqueConstraint(
            "backtest_run_id", "trade_index", "snapshot_date",
            name="uq_bds_run_trade_date",
        ),
    )
    op.create_index(
        "idx_bds_run_trade",
        "backtest_daily_snapshots",
        ["backtest_run_id", "trade_index"],
    )


def downgrade() -> None:
    op.drop_table("backtest_daily_snapshots")
