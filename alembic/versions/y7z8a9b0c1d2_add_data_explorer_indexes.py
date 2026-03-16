"""Add indexes for data explorer query performance.

Revision ID: y7z8a9b0c1d2
Revises: x6y7z8a9b0c1
Create Date: 2026-03-16
"""

from alembic import op

revision = "y7z8a9b0c1d2"
down_revision = "x6y7z8a9b0c1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # dart_financials: 라우터가 (symbol, fiscal_year) 필터링 사용
    op.create_index(
        "ix_dart_financials_sym_year", "dart_financials", ["symbol", "fiscal_year"]
    )
    # news_articles: symbols JSON → JSONB 캐스팅 후 GIN 인덱스
    op.execute(
        "CREATE INDEX ix_news_articles_symbols_gin ON news_articles USING GIN ((symbols::jsonb) jsonb_path_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_news_articles_symbols_gin")
    op.drop_index("ix_dart_financials_sym_year", table_name="dart_financials")
