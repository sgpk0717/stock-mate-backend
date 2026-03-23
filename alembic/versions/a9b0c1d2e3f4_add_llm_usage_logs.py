"""add llm_usage_logs

Revision ID: a9b0c1d2e3f4
Revises: z8a9b0c1d2e3
Create Date: 2026-03-21
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = "a9b0c1d2e3f4"
down_revision = "z8a9b0c1d2e3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "llm_usage_logs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("caller", sa.String(100), nullable=False),
        sa.Column("provider", sa.String(20), nullable=False),
        sa.Column("model", sa.String(100), nullable=False),
        sa.Column("input_tokens", sa.Integer, nullable=False, server_default="0"),
        sa.Column("output_tokens", sa.Integer, nullable=False, server_default="0"),
        sa.Column("total_tokens", sa.Integer, nullable=False, server_default="0"),
        sa.Column("cost_usd", sa.Float, nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="'success'"),
        sa.Column("error", sa.Text, nullable=True),
        sa.Column("duration_ms", sa.Integer, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_llm_usage_caller", "llm_usage_logs", ["caller"])
    op.create_index("ix_llm_usage_provider", "llm_usage_logs", ["provider"])
    op.create_index("ix_llm_usage_created", "llm_usage_logs", ["created_at"])
    op.create_index(
        "ix_llm_usage_caller_date", "llm_usage_logs", ["caller", "created_at"]
    )


def downgrade() -> None:
    op.drop_index("ix_llm_usage_caller_date", "llm_usage_logs")
    op.drop_index("ix_llm_usage_created", "llm_usage_logs")
    op.drop_index("ix_llm_usage_provider", "llm_usage_logs")
    op.drop_index("ix_llm_usage_caller", "llm_usage_logs")
    op.drop_table("llm_usage_logs")
