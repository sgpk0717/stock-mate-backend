"""add causal_failure_type to alpha_factors

Revision ID: r0s1t2u3v4w5
Revises: q9r0s1t2u3v4
Create Date: 2026-03-13
"""
from alembic import op
import sqlalchemy as sa

revision = "r0s1t2u3v4w5"
down_revision = "q9r0s1t2u3v4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "alpha_factors",
        sa.Column("causal_failure_type", sa.String(20), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("alpha_factors", "causal_failure_type")
