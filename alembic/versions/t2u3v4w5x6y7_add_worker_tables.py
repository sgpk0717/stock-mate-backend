"""Add worker_state and worker_commands tables for worker separation.

Revision ID: t2u3v4w5x6y7
Revises: s1t2u3v4w5x6
Create Date: 2026-03-14
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "t2u3v4w5x6y7"
down_revision = "s1t2u3v4w5x6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # worker_state — 단일 행 singleton
    op.create_table(
        "worker_state",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("factory_status", JSONB, nullable=False, server_default="{}"),
        sa.Column("causal_jobs", JSONB, nullable=False, server_default="{}"),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint("id = 1", name="ck_worker_state_singleton"),
    )
    # 초기 행 삽입
    op.execute("INSERT INTO worker_state (id) VALUES (1)")

    # worker_commands — API→워커 명령 큐
    op.create_table(
        "worker_commands",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("command", sa.String(50), nullable=False),
        sa.Column("payload", JSONB, nullable=False, server_default="{}"),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("result", JSONB, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("picked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "idx_worker_commands_pending",
        "worker_commands",
        ["status"],
        postgresql_where=sa.text("status = 'pending'"),
    )


def downgrade() -> None:
    op.drop_table("worker_commands")
    op.drop_table("worker_state")
