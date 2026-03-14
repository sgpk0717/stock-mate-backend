"""Add indexes on alpha_factors sort/filter columns.

Revision ID: j2k3l4m5n6o7
Revises: i1j2k3l4m5n6
Create Date: 2026-03-06 20:00:00.000000
"""

from alembic import op

revision = "j2k3l4m5n6o7"
down_revision = "i1j2k3l4m5n6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index("ix_alpha_factors_icir", "alpha_factors", ["icir"])
    op.create_index("ix_alpha_factors_sharpe", "alpha_factors", ["sharpe"])
    op.create_index("ix_alpha_factors_max_drawdown", "alpha_factors", ["max_drawdown"])
    op.create_index("ix_alpha_factors_causal_robust", "alpha_factors", ["causal_robust"])
    op.create_index("ix_alpha_factors_created_at", "alpha_factors", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_alpha_factors_created_at", table_name="alpha_factors")
    op.drop_index("ix_alpha_factors_causal_robust", table_name="alpha_factors")
    op.drop_index("ix_alpha_factors_max_drawdown", table_name="alpha_factors")
    op.drop_index("ix_alpha_factors_sharpe", table_name="alpha_factors")
    op.drop_index("ix_alpha_factors_icir", table_name="alpha_factors")
