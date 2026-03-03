"""Add alpha_mining_runs and alpha_factors tables

Revision ID: e7f8a9b0c1d2
Revises: d5e6f7a8b9c0
Create Date: 2026-03-02 12:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSON, UUID

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e7f8a9b0c1d2"
down_revision: Union[str, None] = "d5e6f7a8b9c0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "alpha_mining_runs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("context", JSON, nullable=True),
        sa.Column("config", JSON, nullable=True),
        sa.Column(
            "status", sa.String(20), nullable=False, server_default="'PENDING'"
        ),
        sa.Column("progress", sa.Integer, nullable=False, server_default="0"),
        sa.Column(
            "factors_found", sa.Integer, nullable=False, server_default="0"
        ),
        sa.Column(
            "total_evaluated", sa.Integer, nullable=False, server_default="0"
        ),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_table(
        "alpha_factors",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "mining_run_id",
            UUID(as_uuid=True),
            sa.ForeignKey("alpha_mining_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("expression_str", sa.Text, nullable=False),
        sa.Column("expression_sympy", sa.Text, nullable=True),
        sa.Column("polars_code", sa.Text, nullable=True),
        sa.Column("hypothesis", sa.Text, nullable=True),
        sa.Column("generation", sa.Integer, nullable=False, server_default="0"),
        # IC 메트릭
        sa.Column("ic_mean", sa.Float, nullable=True),
        sa.Column("ic_std", sa.Float, nullable=True),
        sa.Column("icir", sa.Float, nullable=True),
        sa.Column("turnover", sa.Float, nullable=True),
        sa.Column("sharpe", sa.Float, nullable=True),
        sa.Column("max_drawdown", sa.Float, nullable=True),
        # 상태
        sa.Column(
            "status",
            sa.String(20),
            nullable=False,
            server_default="'discovered'",
        ),
        # Phase 2: 인과 검증
        sa.Column("causal_robust", sa.Boolean, nullable=True),
        sa.Column("causal_effect_size", sa.Float, nullable=True),
        sa.Column("causal_p_value", sa.Float, nullable=True),
        # 시간
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )

    op.create_index(
        "ix_alpha_factors_mining_run_id",
        "alpha_factors",
        ["mining_run_id"],
    )
    op.create_index(
        "ix_alpha_factors_status", "alpha_factors", ["status"]
    )
    op.create_index(
        "ix_alpha_factors_ic_mean", "alpha_factors", ["ic_mean"]
    )


def downgrade() -> None:
    op.drop_table("alpha_factors")
    op.drop_table("alpha_mining_runs")
