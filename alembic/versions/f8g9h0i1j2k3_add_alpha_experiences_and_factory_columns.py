"""add alpha_experiences and factory columns

Revision ID: f8g9h0i1j2k3
Revises: e7f8a9b0c1d2
Create Date: 2026-03-02

Phase 3: 자율 알파 팩토리
- alpha_experiences 테이블 생성 (벡터 경험 메모리)
- alpha_factors에 parent_ids, factor_type, component_ids 컬럼 추가
- alpha_factors.mining_run_id nullable 변경 (복합 팩터 지원)
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSON, UUID

# revision identifiers, used by Alembic.
revision = "f8g9h0i1j2k3"
down_revision = "e7f8a9b0c1d2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. alpha_experiences 테이블 생성
    op.create_table(
        "alpha_experiences",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "factor_id",
            UUID(as_uuid=True),
            sa.ForeignKey("alpha_factors.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("expression_str", sa.Text, nullable=False),
        sa.Column("hypothesis", sa.Text, nullable=True),
        sa.Column("embedding", JSON, nullable=True),
        sa.Column("ic_mean", sa.Float, nullable=True),
        sa.Column(
            "success", sa.Boolean, nullable=False, server_default="false"
        ),
        sa.Column(
            "generation", sa.Integer, nullable=False, server_default="0"
        ),
        sa.Column("parent_ids", JSON, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )
    op.create_index(
        "ix_alpha_experiences_success", "alpha_experiences", ["success"]
    )
    op.create_index(
        "ix_alpha_experiences_ic_mean", "alpha_experiences", ["ic_mean"]
    )

    # 2. alpha_factors에 Phase 3 컬럼 추가
    op.add_column(
        "alpha_factors", sa.Column("parent_ids", JSON, nullable=True)
    )
    op.add_column(
        "alpha_factors",
        sa.Column(
            "factor_type",
            sa.String(20),
            nullable=False,
            server_default="'single'",
        ),
    )
    op.add_column(
        "alpha_factors", sa.Column("component_ids", JSON, nullable=True)
    )

    # 3. mining_run_id를 nullable로 변경 (복합 팩터는 mining_run 없음)
    op.alter_column(
        "alpha_factors", "mining_run_id", existing_type=UUID(as_uuid=True),
        nullable=True,
    )


def downgrade() -> None:
    op.alter_column(
        "alpha_factors", "mining_run_id", existing_type=UUID(as_uuid=True),
        nullable=False,
    )
    op.drop_column("alpha_factors", "component_ids")
    op.drop_column("alpha_factors", "factor_type")
    op.drop_column("alpha_factors", "parent_ids")
    op.drop_index("ix_alpha_experiences_ic_mean", table_name="alpha_experiences")
    op.drop_index("ix_alpha_experiences_success", table_name="alpha_experiences")
    op.drop_table("alpha_experiences")
