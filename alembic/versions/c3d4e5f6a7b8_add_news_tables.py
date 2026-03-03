"""Add news_articles and news_sentiment_daily tables

Revision ID: c3d4e5f6a7b8
Revises: b7c8d9e0f1a2
Create Date: 2026-03-01 12:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSON, UUID

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c3d4e5f6a7b8"
down_revision: Union[str, None] = "b7c8d9e0f1a2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "news_articles",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("source", sa.String(20), nullable=False),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("content", sa.Text, nullable=True),
        sa.Column("url", sa.Text, nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("symbols", JSON, nullable=True),
        sa.Column("sentiment_score", sa.Float, nullable=True),
        sa.Column("sentiment_magnitude", sa.Float, nullable=True),
        sa.Column("market_impact", sa.Float, nullable=True),
        sa.Column("analyzed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint("url", name="uq_news_url"),
    )
    op.create_index("ix_news_published", "news_articles", ["published_at"])
    op.create_index("ix_news_source", "news_articles", ["source"])

    op.create_table(
        "news_sentiment_daily",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("date", sa.Date, nullable=False),
        sa.Column("avg_sentiment", sa.Float, nullable=False, server_default="0"),
        sa.Column("article_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("event_score", sa.Float, nullable=False, server_default="0"),
        sa.Column("top_headlines", JSON, nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )
    op.create_index(
        "ix_sentiment_daily_lookup",
        "news_sentiment_daily",
        ["symbol", "date"],
    )


def downgrade() -> None:
    op.drop_index("ix_sentiment_daily_lookup", table_name="news_sentiment_daily")
    op.drop_table("news_sentiment_daily")
    op.drop_index("ix_news_source", table_name="news_articles")
    op.drop_index("ix_news_published", table_name="news_articles")
    op.drop_table("news_articles")
