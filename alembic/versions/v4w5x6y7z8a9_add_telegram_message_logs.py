"""Add telegram_message_logs table.

Revision ID: v4w5x6y7z8a9
Revises: u3v4w5x6y7z8
Create Date: 2026-03-15
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = "v4w5x6y7z8a9"
down_revision = "u3v4w5x6y7z8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "telegram_message_logs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("category", sa.String(50), nullable=False, server_default="'system'"),
        sa.Column("caller", sa.String(100), nullable=False, server_default="''"),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("chat_id", sa.String(50), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="'success'"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("telegram_message_id", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_telegram_logs_category", "telegram_message_logs", ["category"])
    op.create_index("ix_telegram_logs_created", "telegram_message_logs", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_telegram_logs_created", table_name="telegram_message_logs")
    op.drop_index("ix_telegram_logs_category", table_name="telegram_message_logs")
    op.drop_table("telegram_message_logs")
