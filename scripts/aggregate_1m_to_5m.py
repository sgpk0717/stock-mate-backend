"""1분봉 → 5분봉 집계 스크립트.

stock_candles 테이블의 interval='1m' 데이터를 5분 단위로 집계하여
interval='5m'으로 저장한다. TimescaleDB time_bucket + first/last 사용.

Usage:
    docker compose run --rm app python -m scripts.aggregate_1m_to_5m
    docker compose run --rm app python -m scripts.aggregate_1m_to_5m --dry-run
    docker compose run --rm app python -m scripts.aggregate_1m_to_5m --verify-only
"""

import argparse
import asyncio
import logging
import time

from sqlalchemy import text

from app.core.database import async_session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# 심볼 배치 크기 (한번에 처리할 종목 수 — DB 부하 분산)
SYMBOL_BATCH_SIZE = 50


async def get_1m_symbols() -> list[str]:
    """1분봉 데이터가 있는 심볼 목록 조회."""
    async with async_session() as db:
        result = await db.execute(
            text("SELECT DISTINCT symbol FROM stock_candles WHERE interval = '1m' ORDER BY symbol")
        )
        return [row[0] for row in result.fetchall()]


async def get_1m_stats() -> dict:
    """1분봉 데이터 통계."""
    async with async_session() as db:
        result = await db.execute(text("""
            SELECT COUNT(*) AS cnt, MIN(dt) AS min_dt, MAX(dt) AS max_dt
            FROM stock_candles WHERE interval = '1m'
        """))
        row = result.fetchone()
        return {"count": row[0], "min_dt": row[1], "max_dt": row[2]}


async def get_5m_stats() -> dict:
    """5분봉 데이터 통계."""
    async with async_session() as db:
        result = await db.execute(text("""
            SELECT COUNT(*) AS cnt, MIN(dt) AS min_dt, MAX(dt) AS max_dt
            FROM stock_candles WHERE interval = '5m'
        """))
        row = result.fetchone()
        return {"count": row[0], "min_dt": row[1], "max_dt": row[2]}


async def delete_existing_5m() -> int:
    """기존 5분봉 데이터 삭제."""
    async with async_session() as db:
        result = await db.execute(
            text("DELETE FROM stock_candles WHERE interval = '5m'")
        )
        await db.commit()
        return result.rowcount


async def aggregate_batch(symbols: list[str], dry_run: bool = False) -> int:
    """심볼 배치 단위로 1분봉→5분봉 집계 INSERT.

    TimescaleDB time_bucket + first/last 사용.
    ON CONFLICT UPSERT로 중복 안전.
    """
    if dry_run:
        async with async_session() as db:
            result = await db.execute(text("""
                SELECT COUNT(*) FROM (
                    SELECT symbol, time_bucket('5 minutes', dt) AS dt_bucket
                    FROM stock_candles
                    WHERE interval = '1m' AND symbol = ANY(:symbols)
                    GROUP BY symbol, time_bucket('5 minutes', dt)
                ) sub
            """), {"symbols": symbols})
            return result.scalar() or 0

    async with async_session() as db:
        result = await db.execute(text("""
            INSERT INTO stock_candles (symbol, dt, interval, open, high, low, close, volume)
            SELECT
                symbol,
                time_bucket('5 minutes', dt) AS dt_bucket,
                '5m',
                first(open, dt),
                max(high),
                min(low),
                last(close, dt),
                sum(volume)
            FROM stock_candles
            WHERE interval = '1m' AND symbol = ANY(:symbols)
            GROUP BY symbol, time_bucket('5 minutes', dt)
            ON CONFLICT (symbol, dt, interval)
            DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume
        """), {"symbols": symbols})
        await db.commit()
        return result.rowcount


async def verify_sample(n: int = 5) -> list[dict]:
    """랜덤 5분봉 샘플을 1분봉 원본과 대조 검증."""
    async with async_session() as db:
        # 랜덤 5분봉 5개 추출
        samples = await db.execute(text("""
            SELECT symbol, dt, open, high, low, close, volume
            FROM stock_candles
            WHERE interval = '5m'
            ORDER BY RANDOM()
            LIMIT :n
        """), {"n": n})
        results = []
        for row in samples.fetchall():
            symbol, dt_5m, o5, h5, l5, c5, v5 = row
            # 해당 5분 구간의 1분봉 원본 조회
            originals = await db.execute(text("""
                SELECT open, high, low, close, volume, dt
                FROM stock_candles
                WHERE interval = '1m'
                  AND symbol = :symbol
                  AND dt >= :dt_start
                  AND dt < :dt_start + INTERVAL '5 minutes'
                ORDER BY dt
            """), {"symbol": symbol, "dt_start": dt_5m})
            rows_1m = originals.fetchall()
            if not rows_1m:
                results.append({"symbol": symbol, "dt": str(dt_5m), "status": "NO_1M_DATA"})
                continue

            # 수동 집계
            expected_open = rows_1m[0][0]  # first open
            expected_high = max(r[1] for r in rows_1m)
            expected_low = min(r[2] for r in rows_1m)
            expected_close = rows_1m[-1][3]  # last close
            expected_volume = sum(r[4] for r in rows_1m)

            match = (
                float(o5) == float(expected_open)
                and float(h5) == float(expected_high)
                and float(l5) == float(expected_low)
                and float(c5) == float(expected_close)
                and int(v5) == int(expected_volume)
            )
            results.append({
                "symbol": symbol,
                "dt": str(dt_5m),
                "1m_count": len(rows_1m),
                "match": match,
                "status": "OK" if match else "MISMATCH",
            })
        return results


async def main(dry_run: bool = False, verify_only: bool = False) -> None:
    t0 = time.time()

    # 현재 상태 출력
    stats_1m = await get_1m_stats()
    stats_5m = await get_5m_stats()
    logger.info("=== 1분봉 현황: %s행, %s ~ %s", f"{stats_1m['count']:,}", stats_1m['min_dt'], stats_1m['max_dt'])
    logger.info("=== 5분봉 현황: %s행, %s ~ %s", f"{stats_5m['count']:,}", stats_5m['min_dt'], stats_5m['max_dt'])

    if verify_only:
        logger.info("=== 검증 모드 ===")
        results = await verify_sample(10)
        for r in results:
            logger.info("  %s %s: %s (1m봉 %s개)", r["symbol"], r["dt"], r["status"], r.get("1m_count", "?"))
        ok = sum(1 for r in results if r["status"] == "OK")
        logger.info("=== 검증 결과: %d/%d OK ===", ok, len(results))
        return

    # 심볼 목록
    symbols = await get_1m_symbols()
    logger.info("=== 1분봉 심볼: %d개 ===", len(symbols))

    if not dry_run:
        # 기존 5분봉 삭제
        deleted = await delete_existing_5m()
        logger.info("=== 기존 5분봉 %s행 삭제 완료 ===", f"{deleted:,}")

    # 배치 처리
    total_inserted = 0
    batches = [symbols[i:i + SYMBOL_BATCH_SIZE] for i in range(0, len(symbols), SYMBOL_BATCH_SIZE)]

    for idx, batch in enumerate(batches, 1):
        t_batch = time.time()
        count = await aggregate_batch(batch, dry_run=dry_run)
        total_inserted += count
        elapsed = time.time() - t_batch
        logger.info(
            "  배치 %d/%d: %s 종목, %s행 %s (%.1f초)",
            idx, len(batches), len(batch), f"{count:,}",
            "예상" if dry_run else "삽입",
            elapsed,
        )

    elapsed_total = time.time() - t0
    logger.info(
        "=== %s 완료: 총 %s행, %.1f분 소요 ===",
        "DRY-RUN" if dry_run else "집계",
        f"{total_inserted:,}",
        elapsed_total / 60,
    )

    if not dry_run:
        # 결과 검증
        stats_5m_after = await get_5m_stats()
        logger.info("=== 5분봉 결과: %s행, %s ~ %s", f"{stats_5m_after['count']:,}", stats_5m_after['min_dt'], stats_5m_after['max_dt'])

        logger.info("=== 샘플 검증 ===")
        results = await verify_sample(10)
        for r in results:
            logger.info("  %s %s: %s", r["symbol"], r["dt"], r["status"])
        ok = sum(1 for r in results if r["status"] == "OK")
        logger.info("=== 검증 결과: %d/%d OK ===", ok, len(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="1분봉 → 5분봉 집계")
    parser.add_argument("--dry-run", action="store_true", help="실제 삽입 없이 예상 행 수만 확인")
    parser.add_argument("--verify-only", action="store_true", help="기존 5분봉 데이터 검증만")
    args = parser.parse_args()
    asyncio.run(main(dry_run=args.dry_run, verify_only=args.verify_only))
