"""alpha_generation_reports table

Revision ID: g1h2i3j4k5l6
Revises: f57a8905e56f
Create Date: 2026-03-28
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSON, UUID

revision = "g1h2i3j4k5l6"
down_revision = "f57a8905e56f"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "alpha_generation_reports",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("generation", sa.Integer, nullable=False),
        sa.Column("data_interval", sa.String(5), nullable=False, server_default="1d"),
        sa.Column("cycle_num", sa.Integer, nullable=False, server_default="0"),
        sa.Column("report_data", JSON, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index(
        "ix_gen_report_interval_gen",
        "alpha_generation_reports",
        ["data_interval", "generation"],
    )
    op.create_index(
        "ix_gen_report_interval_created",
        "alpha_generation_reports",
        ["data_interval", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_gen_report_interval_created", "alpha_generation_reports")
    op.drop_index("ix_gen_report_interval_gen", "alpha_generation_reports")
    op.drop_table("alpha_generation_reports")
