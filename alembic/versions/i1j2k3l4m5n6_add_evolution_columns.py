"""Add evolution engine columns to alpha_factors.

Revision ID: i1j2k3l4m5n6
Revises: h0i1j2k3l4m5
Create Date: 2026-03-04 12:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

revision = "i1j2k3l4m5n6"
down_revision = "h0i1j2k3l4m5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "alpha_factors",
        sa.Column("fitness_composite", sa.Float(), nullable=True),
    )
    op.add_column(
        "alpha_factors",
        sa.Column("tree_depth", sa.Integer(), nullable=True),
    )
    op.add_column(
        "alpha_factors",
        sa.Column("tree_size", sa.Integer(), nullable=True),
    )
    op.add_column(
        "alpha_factors",
        sa.Column("expression_hash", sa.String(64), nullable=True),
    )
    op.add_column(
        "alpha_factors",
        sa.Column("operator_origin", sa.String(30), nullable=True),
    )
    op.add_column(
        "alpha_factors",
        sa.Column(
            "is_elite",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    op.add_column(
        "alpha_factors",
        sa.Column(
            "population_active",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("true"),
        ),
    )
    op.add_column(
        "alpha_factors",
        sa.Column(
            "birth_generation",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
    )
    op.create_index(
        "ix_alpha_factors_expression_hash",
        "alpha_factors",
        ["expression_hash"],
    )
    op.create_index(
        "ix_alpha_factors_population_active",
        "alpha_factors",
        ["population_active"],
    )
    op.create_index(
        "ix_alpha_factors_fitness_composite",
        "alpha_factors",
        ["fitness_composite"],
    )


def downgrade() -> None:
    op.drop_index("ix_alpha_factors_fitness_composite", table_name="alpha_factors")
    op.drop_index("ix_alpha_factors_population_active", table_name="alpha_factors")
    op.drop_index("ix_alpha_factors_expression_hash", table_name="alpha_factors")
    op.drop_column("alpha_factors", "birth_generation")
    op.drop_column("alpha_factors", "population_active")
    op.drop_column("alpha_factors", "is_elite")
    op.drop_column("alpha_factors", "operator_origin")
    op.drop_column("alpha_factors", "expression_hash")
    op.drop_column("alpha_factors", "tree_size")
    op.drop_column("alpha_factors", "tree_depth")
    op.drop_column("alpha_factors", "fitness_composite")
