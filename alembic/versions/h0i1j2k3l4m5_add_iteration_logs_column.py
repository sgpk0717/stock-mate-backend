"""add iteration_logs column to alpha_mining_runs

Revision ID: h0i1j2k3l4m5
Revises: g9h0i1j2k3l4
Create Date: 2026-03-03

알파 마이닝 탐색 과정 투명성: iteration_logs JSON 컬럼 추가
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSON

revision = "h0i1j2k3l4m5"
down_revision = "g9h0i1j2k3l4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "alpha_mining_runs",
        sa.Column("iteration_logs", JSON, nullable=True),
    )


def downgrade() -> None:
    op.drop_column("alpha_mining_runs", "iteration_logs")
