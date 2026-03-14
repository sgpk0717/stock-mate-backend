"""add live_trades table and session_state column

Revision ID: q9r0s1t2u3v4
Revises: p8q9r0s1t2u3
Create Date: 2026-03-13
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSON


# revision identifiers, used by Alembic.
revision = "q9r0s1t2u3v4"
down_revision = "p8q9r0s1t2u3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. live_trades 테이블
    op.create_table(
        "live_trades",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "context_id", UUID(as_uuid=True),
            sa.ForeignKey("trading_contexts.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("name", sa.String(100), nullable=True),
        sa.Column("side", sa.String(4), nullable=False),
        sa.Column("step", sa.String(10), nullable=False, server_default=""),
        sa.Column("qty", sa.Integer, nullable=False),
        sa.Column("price", sa.Float, nullable=False),
        sa.Column("pnl_pct", sa.Float, nullable=True),
        sa.Column("pnl_amount", sa.Numeric(18, 2), nullable=True),
        sa.Column("holding_minutes", sa.Float, nullable=True),
        sa.Column("success", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("order_id", sa.String(50), nullable=False, server_default=""),
        sa.Column("reason", sa.Text, nullable=True),
        sa.Column("snapshot", JSON, nullable=True),
        sa.Column("conditions", JSON, nullable=True),
        sa.Column("sizing", JSON, nullable=True),
        sa.Column("position_context", JSON, nullable=True),
        sa.Column("executed_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_lt_context_date", "live_trades", ["context_id", "executed_at"])
    op.create_index("ix_lt_symbol", "live_trades", ["symbol", "executed_at"])

    # 2. trading_contexts에 session_state JSONB 컬럼 추가 (C2: 세션 복구)
    op.add_column("trading_contexts", sa.Column("session_state", JSON, nullable=True))


def downgrade() -> None:
    op.drop_column("trading_contexts", "session_state")
    op.drop_table("live_trades")
