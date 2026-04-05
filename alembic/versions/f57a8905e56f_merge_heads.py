"""merge heads

Revision ID: f57a8905e56f
Revises: a0b1c2d3e4f5, b0c1d2e3f4g5
Create Date: 2026-03-26 20:31:12.170263

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f57a8905e56f'
down_revision: Union[str, None] = ('a0b1c2d3e4f5', 'b0c1d2e3f4g5')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
