"""add param_snapshot and param_adjustments to live_feedback

Revision ID: s1t2u3v4w5x6
Revises: r0s1t2u3v4w5
Create Date: 2026-03-13
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSON

revision = "s1t2u3v4w5x6"
down_revision = "r0s1t2u3v4w5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("live_feedback", sa.Column("param_snapshot", JSON, nullable=True))
    op.add_column("live_feedback", sa.Column("param_adjustments", JSON, nullable=True))


def downgrade() -> None:
    op.drop_column("live_feedback", "param_adjustments")
    op.drop_column("live_feedback", "param_snapshot")
