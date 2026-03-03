"""Add sector columns to stock_masters

Revision ID: d5e6f7a8b9c0
Revises: c3d4e5f6a7b8
Create Date: 2026-03-01 14:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSON

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d5e6f7a8b9c0"
down_revision: Union[str, None] = "c3d4e5f6a7b8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("stock_masters", sa.Column("sector", sa.String(50), nullable=True))
    op.add_column("stock_masters", sa.Column("sub_sector", sa.String(100), nullable=True))
    op.add_column("stock_masters", sa.Column("description", sa.String(500), nullable=True))
    op.add_column("stock_masters", sa.Column("embedding", JSON, nullable=True))


def downgrade() -> None:
    op.drop_column("stock_masters", "embedding")
    op.drop_column("stock_masters", "description")
    op.drop_column("stock_masters", "sub_sector")
    op.drop_column("stock_masters", "sector")
