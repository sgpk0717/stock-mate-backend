"""TimescaleDB setup: hypertable + continuous aggregates

Revision ID: a1b2c3d4e5f6
Revises: 64ea2ba3255f
Create Date: 2026-02-27 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = "64ea2ba3255f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. TimescaleDB 확장 활성화
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")

    # 2. stock_ticks PK 변경 (hypertable은 time 컬럼이 PK에 포함되어야 함)
    op.execute("ALTER TABLE stock_ticks DROP CONSTRAINT stock_ticks_pkey")
    op.execute("ALTER TABLE stock_ticks ADD PRIMARY KEY (ts, id)")

    # 3. hypertable 변환 (1일 청크)
    op.execute("""
        SELECT create_hypertable(
            'stock_ticks', 'ts',
            chunk_time_interval => INTERVAL '1 day',
            migrate_data => true
        )
    """)

    # 4. Continuous Aggregate: 1분봉 (raw tick → 1m candle)
    op.execute("""
        CREATE MATERIALIZED VIEW candles_1m
        WITH (timescaledb.continuous) AS
        SELECT
            time_bucket('1 minute', ts) AS bucket,
            symbol,
            first(price, ts) AS open,
            max(price) AS high,
            min(price) AS low,
            last(price, ts) AS close,
            sum(volume) AS volume
        FROM stock_ticks
        GROUP BY bucket, symbol
        WITH NO DATA
    """)

    # 5. Continuous Aggregate: 1시간봉 (1분봉 → 1h candle)
    op.execute("""
        CREATE MATERIALIZED VIEW candles_1h
        WITH (timescaledb.continuous) AS
        SELECT
            time_bucket('1 hour', bucket) AS bucket,
            symbol,
            first(open, bucket) AS open,
            max(high) AS high,
            min(low) AS low,
            last(close, bucket) AS close,
            sum(volume) AS volume
        FROM candles_1m
        GROUP BY time_bucket('1 hour', bucket), symbol
        WITH NO DATA
    """)

    # 6. Continuous Aggregate: 일봉 (1시간봉 → 1d candle)
    op.execute("""
        CREATE MATERIALIZED VIEW candles_1d
        WITH (timescaledb.continuous) AS
        SELECT
            time_bucket('1 day', bucket) AS bucket,
            symbol,
            first(open, bucket) AS open,
            max(high) AS high,
            min(low) AS low,
            last(close, bucket) AS close,
            sum(volume) AS volume
        FROM candles_1h
        GROUP BY time_bucket('1 day', bucket), symbol
        WITH NO DATA
    """)

    # 7. 자동 갱신 정책
    op.execute("""
        SELECT add_continuous_aggregate_policy('candles_1m',
            start_offset => INTERVAL '2 hours',
            end_offset => INTERVAL '1 minute',
            schedule_interval => INTERVAL '1 minute'
        )
    """)
    op.execute("""
        SELECT add_continuous_aggregate_policy('candles_1h',
            start_offset => INTERVAL '2 days',
            end_offset => INTERVAL '1 hour',
            schedule_interval => INTERVAL '1 hour'
        )
    """)
    op.execute("""
        SELECT add_continuous_aggregate_policy('candles_1d',
            start_offset => INTERVAL '7 days',
            end_offset => INTERVAL '1 day',
            schedule_interval => INTERVAL '1 day'
        )
    """)

    # 8. 압축 정책 (7일 이후 raw tick 압축)
    op.execute("""
        ALTER TABLE stock_ticks SET (
            timescaledb.compress,
            timescaledb.compress_segmentby = 'symbol',
            timescaledb.compress_orderby = 'ts DESC'
        )
    """)
    op.execute("""
        SELECT add_compression_policy('stock_ticks', INTERVAL '7 days')
    """)

    # 9. 보존 정책 (30일 이후 raw tick 삭제)
    op.execute("""
        SELECT add_retention_policy('stock_ticks', INTERVAL '30 days')
    """)


def downgrade() -> None:
    # 정책 제거
    op.execute("SELECT remove_retention_policy('stock_ticks', if_exists => true)")
    op.execute("SELECT remove_compression_policy('stock_ticks', if_exists => true)")
    op.execute("SELECT remove_continuous_aggregate_policy('candles_1d', if_not_exists => true)")
    op.execute("SELECT remove_continuous_aggregate_policy('candles_1h', if_not_exists => true)")
    op.execute("SELECT remove_continuous_aggregate_policy('candles_1m', if_not_exists => true)")

    # Continuous aggregates 삭제 (역순)
    op.execute("DROP MATERIALIZED VIEW IF EXISTS candles_1d CASCADE")
    op.execute("DROP MATERIALIZED VIEW IF EXISTS candles_1h CASCADE")
    op.execute("DROP MATERIALIZED VIEW IF EXISTS candles_1m CASCADE")

    # PK 복원
    op.execute("ALTER TABLE stock_ticks DROP CONSTRAINT stock_ticks_pkey")
    op.execute("ALTER TABLE stock_ticks ADD PRIMARY KEY (id)")

    op.execute("DROP EXTENSION IF EXISTS timescaledb CASCADE")
