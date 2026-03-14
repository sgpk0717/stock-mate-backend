"""Add investor_trading, dart_financials tables and failure_reason column.

Revision ID: k3l4m5n6o7p8
Revises: j2k3l4m5n6o7
Create Date: 2026-03-08 12:00:00.000000
"""

import sqlalchemy as sa
from alembic import op

revision = "k3l4m5n6o7p8"
down_revision = "j2k3l4m5n6o7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. investor_trading 테이블
    op.create_table(
        "investor_trading",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("dt", sa.Date, nullable=False),
        sa.Column("foreign_net", sa.BigInteger, server_default="0"),
        sa.Column("inst_net", sa.BigInteger, server_default="0"),
        sa.Column("retail_net", sa.BigInteger, server_default="0"),
        sa.UniqueConstraint("symbol", "dt", name="uq_investor_trading"),
    )
    op.create_index(
        "ix_investor_trading_lookup", "investor_trading", ["symbol", "dt"]
    )

    # 2. dart_financials 테이블
    op.create_table(
        "dart_financials",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("disclosure_date", sa.Date, nullable=False),
        sa.Column("fiscal_year", sa.String(4), nullable=False),
        sa.Column("fiscal_quarter", sa.String(2), nullable=False),
        sa.Column("fiscal_period_end", sa.Date, nullable=True),
        sa.Column("eps", sa.Float, nullable=True),
        sa.Column("bps", sa.Float, nullable=True),
        sa.Column("operating_margin", sa.Float, nullable=True),
        sa.Column("debt_to_equity", sa.Float, nullable=True),
        sa.UniqueConstraint(
            "symbol", "fiscal_year", "fiscal_quarter", name="uq_dart_financial"
        ),
    )
    op.create_index(
        "ix_dart_financials_lookup", "dart_financials", ["symbol", "disclosure_date"]
    )

    # 3. alpha_experiences 에 failure_reason 컬럼 추가
    op.add_column(
        "alpha_experiences",
        sa.Column("failure_reason", sa.Text, nullable=True),
    )


def downgrade() -> None:
    op.drop_column("alpha_experiences", "failure_reason")
    op.drop_index("ix_dart_financials_lookup", table_name="dart_financials")
    op.drop_table("dart_financials")
    op.drop_index("ix_investor_trading_lookup", table_name="investor_trading")
    op.drop_table("investor_trading")
