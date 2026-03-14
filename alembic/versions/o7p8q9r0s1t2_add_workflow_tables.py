"""add workflow tables (trading_contexts, workflow_runs, workflow_events, live_feedback)

Revision ID: o7p8q9r0s1t2
Revises: n6o7p8q9r0s1
Create Date: 2026-03-11
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSON


# revision identifiers, used by Alembic.
revision = "o7p8q9r0s1t2"
down_revision = "n6o7p8q9r0s1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. trading_contexts
    op.create_table(
        "trading_contexts",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("mode", sa.String(10), nullable=False, server_default="paper"),
        sa.Column("status", sa.String(20), nullable=False, server_default="active"),
        sa.Column("strategy", JSON, nullable=False, server_default="{}"),
        sa.Column("strategy_name", sa.String(200), nullable=False, server_default=""),
        sa.Column("position_sizing", JSON, nullable=True),
        sa.Column("scaling", JSON, nullable=True),
        sa.Column("risk_management", JSON, nullable=True),
        sa.Column("cost_config", JSON, nullable=True),
        sa.Column("initial_capital", sa.Numeric(18, 2), server_default="100000000"),
        sa.Column("position_size_pct", sa.Float, server_default="0.1"),
        sa.Column("max_positions", sa.Integer, server_default="10"),
        sa.Column("symbols", JSON, nullable=True),
        sa.Column("source_backtest_id", UUID(as_uuid=True), nullable=True),
        sa.Column("source_factor_id", UUID(as_uuid=True), nullable=True),
        sa.Column("auto_created", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_tc_status", "trading_contexts", ["status"])
    op.create_index("ix_tc_mode", "trading_contexts", ["mode"])

    # 2. workflow_runs
    op.create_table(
        "workflow_runs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("date", sa.Date, nullable=False),
        sa.Column("phase", sa.String(30), nullable=False, server_default="IDLE"),
        sa.Column("status", sa.String(20), nullable=False, server_default="PENDING"),
        sa.Column("config", JSON, nullable=True),
        sa.Column("mining_run_id", UUID(as_uuid=True), nullable=True),
        sa.Column("selected_factor_id", UUID(as_uuid=True), nullable=True),
        sa.Column("trading_context_id", UUID(as_uuid=True), nullable=True),
        sa.Column("review_summary", JSON, nullable=True),
        sa.Column("trade_count", sa.Integer, server_default="0"),
        sa.Column("pnl_amount", sa.Numeric(18, 2), nullable=True),
        sa.Column("pnl_pct", sa.Float, nullable=True),
        sa.Column("mining_context", sa.Text, nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("uq_wf_date", "workflow_runs", ["date"], unique=True)

    # 3. workflow_events
    op.create_table(
        "workflow_events",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "workflow_run_id", UUID(as_uuid=True),
            sa.ForeignKey("workflow_runs.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("phase", sa.String(30), nullable=True),
        sa.Column("event_type", sa.String(50), nullable=True),
        sa.Column("message", sa.Text, nullable=True),
        sa.Column("data", JSON, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_wfe_run", "workflow_events", ["workflow_run_id", "created_at"])

    # 4. live_feedback
    op.create_table(
        "live_feedback",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("factor_id", UUID(as_uuid=True), nullable=False),
        sa.Column("workflow_run_id", UUID(as_uuid=True), nullable=True),
        sa.Column("date", sa.Date, nullable=False),
        sa.Column("realized_pnl", sa.Numeric(18, 2), nullable=True),
        sa.Column("realized_pnl_pct", sa.Float, nullable=True),
        sa.Column("realized_sharpe", sa.Float, nullable=True),
        sa.Column("trade_count", sa.Integer, server_default="0"),
        sa.Column("win_rate", sa.Float, nullable=True),
        sa.Column("avg_holding_minutes", sa.Float, nullable=True),
        sa.Column("max_single_loss_pct", sa.Float, nullable=True),
        sa.Column("gap_to_backtest_sharpe", sa.Float, nullable=True),
        sa.Column("gap_to_ic_prediction", sa.Float, nullable=True),
        sa.Column("market_regime", sa.String(20), nullable=True),
        sa.Column("feedback_context", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_lf_factor", "live_feedback", ["factor_id", "date"])
    op.create_index("ix_lf_date", "live_feedback", ["date"])


def downgrade() -> None:
    op.drop_table("live_feedback")
    op.drop_table("workflow_events")
    op.drop_table("workflow_runs")
    op.drop_table("trading_contexts")
