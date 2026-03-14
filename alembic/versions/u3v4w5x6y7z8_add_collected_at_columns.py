"""Add collected_at columns to data collection tables.

Revision ID: u3v4w5x6y7z8
Revises: t2u3v4w5x6y7
Create Date: 2026-03-14
"""

from alembic import op
import sqlalchemy as sa

revision = "u3v4w5x6y7z8"
down_revision = "t2u3v4w5x6y7"
branch_labels = None
depends_on = None

TABLES = [
    "stock_candles",
    "investor_trading",
    "dart_financials",
    "program_trading",
    "margin_short_daily",
]


def upgrade() -> None:
    for table in TABLES:
        op.add_column(
            table,
            sa.Column(
                "collected_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=True,
            ),
        )


def downgrade() -> None:
    for table in TABLES:
        op.drop_column(table, "collected_at")
