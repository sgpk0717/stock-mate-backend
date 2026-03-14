"""Add interval column to alpha_factors.

Revision ID: l4m5n6o7p8q9
Revises: k3l4m5n6o7p8
Create Date: 2026-03-09 12:00:00.000000
"""

import sqlalchemy as sa
from alembic import op

revision = "l4m5n6o7p8q9"
down_revision = "k3l4m5n6o7p8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "alpha_factors",
        sa.Column("interval", sa.String(5), nullable=False, server_default="1d"),
    )
    op.create_index(
        "ix_alpha_factors_interval", "alpha_factors", ["interval"]
    )


def downgrade() -> None:
    op.drop_index("ix_alpha_factors_interval", table_name="alpha_factors")
    op.drop_column("alpha_factors", "interval")
