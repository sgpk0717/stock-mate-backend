"""add staleness columns to alpha_factors

Revision ID: p8q9r0s1t2u3
Revises: o7p8q9r0s1t2
Create Date: 2026-03-12
"""

from alembic import op
import sqlalchemy as sa

revision = "p8q9r0s1t2u3"
down_revision = "o7p8q9r0s1t2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "alpha_factors",
        sa.Column("last_evaluated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "alpha_factors",
        sa.Column("live_ic_7d", sa.Float, nullable=True),
    )
    op.add_column(
        "alpha_factors",
        sa.Column("live_sharpe_7d", sa.Float, nullable=True),
    )
    op.add_column(
        "alpha_factors",
        sa.Column(
            "staleness_warning",
            sa.Boolean,
            server_default="false",
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_column("alpha_factors", "staleness_warning")
    op.drop_column("alpha_factors", "live_sharpe_7d")
    op.drop_column("alpha_factors", "live_ic_7d")
    op.drop_column("alpha_factors", "last_evaluated_at")
