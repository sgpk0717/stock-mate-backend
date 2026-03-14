"""add program_trading and margin_short_daily tables

Revision ID: m5n6o7p8q9r0
Revises: l4m5n6o7p8q9
Create Date: 2026-03-09
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "m5n6o7p8q9r0"
down_revision = "l4m5n6o7p8q9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # program_trading: KIS API 프로그램 매매 스냅샷
    op.create_table(
        "program_trading",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(length=20), nullable=False),
        sa.Column("dt", sa.DateTime(timezone=True), nullable=False),
        sa.Column("pgm_buy_qty", sa.BigInteger(), server_default="0", nullable=True),
        sa.Column("pgm_sell_qty", sa.BigInteger(), server_default="0", nullable=True),
        sa.Column("pgm_net_qty", sa.BigInteger(), server_default="0", nullable=True),
        sa.Column("pgm_buy_amount", sa.BigInteger(), server_default="0", nullable=True),
        sa.Column("pgm_sell_amount", sa.BigInteger(), server_default="0", nullable=True),
        sa.Column("pgm_net_amount", sa.BigInteger(), server_default="0", nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("symbol", "dt", name="uq_program_trading"),
    )
    op.create_index("ix_program_trading_lookup", "program_trading", ["symbol", "dt"])

    # margin_short_daily: pykrx 신용잔고/공매도 일별 데이터
    op.create_table(
        "margin_short_daily",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(length=20), nullable=False),
        sa.Column("dt", sa.Date(), nullable=False),
        sa.Column("margin_balance", sa.BigInteger(), server_default="0", nullable=True),
        sa.Column("margin_rate", sa.Float(), server_default="0", nullable=True),
        sa.Column("short_volume", sa.BigInteger(), server_default="0", nullable=True),
        sa.Column("short_balance", sa.BigInteger(), server_default="0", nullable=True),
        sa.Column("short_balance_rate", sa.Float(), server_default="0", nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("symbol", "dt", name="uq_margin_short"),
    )
    op.create_index("ix_margin_short_lookup", "margin_short_daily", ["symbol", "dt"])


def downgrade() -> None:
    op.drop_index("ix_margin_short_lookup", table_name="margin_short_daily")
    op.drop_table("margin_short_daily")
    op.drop_index("ix_program_trading_lookup", table_name="program_trading")
    op.drop_table("program_trading")
