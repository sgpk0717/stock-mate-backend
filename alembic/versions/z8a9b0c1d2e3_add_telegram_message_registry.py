"""Add telegram_message_registry table.

Revision ID: z8a9b0c1d2e3
Revises: y7z8a9b0c1d2
Create Date: 2026-03-17
"""

from alembic import op
import sqlalchemy as sa

revision = "z8a9b0c1d2e3"
down_revision = "y7z8a9b0c1d2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "telegram_message_registry",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("message_key", sa.String(50), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="sent"),
        sa.Column("sent_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("sender", sa.String(20), nullable=False),
        sa.UniqueConstraint("message_key", "date", name="uq_tmr_key_date"),
    )
    op.create_index("ix_tmr_key_date", "telegram_message_registry", ["message_key", "date"])


def downgrade() -> None:
    op.drop_index("ix_tmr_key_date", table_name="telegram_message_registry")
    op.drop_table("telegram_message_registry")
