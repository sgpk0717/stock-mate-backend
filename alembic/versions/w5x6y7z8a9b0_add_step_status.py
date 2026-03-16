"""Add step_status column to workflow_runs.

Revision ID: w5x6y7z8a9b0
Revises: v4w5x6y7z8a9
Create Date: 2026-03-16
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSON

revision = "w5x6y7z8a9b0"
down_revision = "v4w5x6y7z8a9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "workflow_runs",
        sa.Column("step_status", JSON, nullable=True),
    )


def downgrade() -> None:
    op.drop_column("workflow_runs", "step_status")
