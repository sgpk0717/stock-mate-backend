"""extend investor_trading with buy/sell volume columns

Revision ID: n6o7p8q9r0s1
Revises: m5n6o7p8q9r0
Create Date: 2026-03-09
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "n6o7p8q9r0s1"
down_revision = "m5n6o7p8q9r0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("investor_trading", sa.Column("foreign_buy_vol", sa.BigInteger(), server_default="0"))
    op.add_column("investor_trading", sa.Column("foreign_sell_vol", sa.BigInteger(), server_default="0"))
    op.add_column("investor_trading", sa.Column("inst_buy_vol", sa.BigInteger(), server_default="0"))
    op.add_column("investor_trading", sa.Column("inst_sell_vol", sa.BigInteger(), server_default="0"))
    op.add_column("investor_trading", sa.Column("retail_buy_vol", sa.BigInteger(), server_default="0"))
    op.add_column("investor_trading", sa.Column("retail_sell_vol", sa.BigInteger(), server_default="0"))


def downgrade() -> None:
    op.drop_column("investor_trading", "retail_sell_vol")
    op.drop_column("investor_trading", "retail_buy_vol")
    op.drop_column("investor_trading", "inst_sell_vol")
    op.drop_column("investor_trading", "inst_buy_vol")
    op.drop_column("investor_trading", "foreign_sell_vol")
    op.drop_column("investor_trading", "foreign_buy_vol")
