"""add universe replay fields to sim_replay_runs

Revision ID: b0c1d2e3f4g5
Revises: a9b0c1d2e3f4
Create Date: 2026-03-20

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSON


revision = "b0c1d2e3f4g5"
down_revision = "a9b0c1d2e3f4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("sim_replay_runs", sa.Column("mode", sa.String(20), nullable=False, server_default="single"))
    op.add_column("sim_replay_runs", sa.Column("universe", sa.String(50), nullable=True))
    op.add_column("sim_replay_runs", sa.Column("tick_summaries", JSON, nullable=True))
    op.add_column("sim_replay_runs", sa.Column("trade_log_json", JSON, nullable=True))
    op.add_column("sim_replay_runs", sa.Column("decisions_json", JSON, nullable=True))
    op.add_column("sim_replay_runs", sa.Column("equity_curve", JSON, nullable=True))
    op.add_column("sim_replay_runs", sa.Column("total_trades", sa.Integer(), nullable=True))
    op.add_column("sim_replay_runs", sa.Column("final_equity", sa.Numeric(), nullable=True))
    op.create_index("ix_sim_replay_runs_mode", "sim_replay_runs", ["mode"])


def downgrade() -> None:
    op.drop_index("ix_sim_replay_runs_mode", "sim_replay_runs")
    op.drop_column("sim_replay_runs", "final_equity")
    op.drop_column("sim_replay_runs", "total_trades")
    op.drop_column("sim_replay_runs", "equity_curve")
    op.drop_column("sim_replay_runs", "decisions_json")
    op.drop_column("sim_replay_runs", "trade_log_json")
    op.drop_column("sim_replay_runs", "tick_summaries")
    op.drop_column("sim_replay_runs", "universe")
    op.drop_column("sim_replay_runs", "mode")
