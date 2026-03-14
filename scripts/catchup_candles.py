"""최신 캔들 데이터 캐치업 수집.

수집 갭(마지막 DB 날짜 ~ 어제)을 채운다.
- Phase 1: 일봉 — pykrx 벌크 (전 종목, 날짜당 ~10초)
- Phase 2: 분봉 — KIS API (전 종목, 갭 기간만)

Usage:
    docker compose run --rm app python -m scripts.catchup_candles
    docker compose run --rm app python -m scripts.catchup_candles --dry-run
"""

import argparse
import asyncio
import fcntl
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone

from sqlalchemy import text

from app.core.database import async_session
from app.services.candle_writer import write_candles_bulk
from app.trading.kis_client import get_kis_client
from scripts.collect_minute_kis import collect_one

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9))
CATCHUP_PROGRESS_FILE = "kis_catchup_progress.json"
LOCK_FILE = "/tmp/catchup_candles.lock"


def _format_duration(seconds: float) -> str:
    d = int(seconds // 86400)
    h = int((seconds % 86400) // 3600)
    m = int((seconds % 3600) // 60)
    parts = []
    if d:
        parts.append(f"{d}d")
    if h:
        parts.append(f"{h}h")
    parts.append(f"{m}m")
    return " ".join(parts)


# ── DB 조회 ──────────────────────────────────────────────


async def get_last_dates() -> dict:
    """DB에서 일봉/분봉 각각의 대표 최신 날짜 조회.

    분봉은 글로벌 MAX가 아니라 종목별 MAX의 10th percentile을 사용한다.
    (대부분의 종목이 커버되도록)
    """
    async with async_session() as db:
        r1 = await db.execute(
            text("SELECT MAX(dt) FROM stock_candles WHERE interval = '1d'")
        )
        last_daily = r1.scalar()

        # 분봉: 종목별 MAX(dt)의 10th percentile → 90% 종목 커버
        r2 = await db.execute(
            text("""
                SELECT percentile_disc(0.10) WITHIN GROUP (ORDER BY latest)
                FROM (
                    SELECT MAX(dt) AS latest
                    FROM stock_candles WHERE interval = '1m'
                    GROUP BY symbol
                ) t
            """)
        )
        last_minute = r2.scalar()

    result = {}
    if last_daily:
        dt = last_daily.astimezone(KST) if hasattr(last_daily, "astimezone") else last_daily
        result["daily"] = dt.strftime("%Y%m%d")
    if last_minute:
        dt = last_minute.astimezone(KST) if hasattr(last_minute, "astimezone") else last_minute
        result["minute"] = dt.strftime("%Y%m%d")

    return result


async def get_all_symbols() -> list[str]:
    """stock_masters에서 전 종목 코드 조회."""
    async with async_session() as db:
        result = await db.execute(
            text("SELECT symbol FROM stock_masters ORDER BY symbol")
        )
        return [r[0] for r in result.fetchall()]


# ── Phase 1: 일봉 (pykrx 벌크) ──────────────────────────


def _generate_dates(from_date: str, to_date: str) -> list[str]:
    """from_date ~ to_date 사이 날짜 리스트 (YYYYMMDD)."""
    start = datetime.strptime(from_date, "%Y%m%d")
    end = datetime.strptime(to_date, "%Y%m%d")
    dates = []
    cur = start
    while cur <= end:
        dates.append(cur.strftime("%Y%m%d"))
        cur += timedelta(days=1)
    return dates


def _pykrx_fetch_range(symbol: str, from_date: str, to_date: str) -> list[dict]:
    """pykrx 종목별 날짜 범위 OHLCV 조회."""
    from pykrx import stock as krx

    df = krx.get_market_ohlcv_by_date(from_date, to_date, symbol)
    if df.empty:
        return []

    rows = []
    for idx, row in df.iterrows():
        c = float(row.get("종가", 0))
        if c <= 0:
            continue
        # idx는 Timestamp — YYYYMMDD로 변환
        dt_str = idx.strftime("%Y%m%d")
        rows.append({
            "dt": dt_str,
            "open": float(row.get("시가", 0)),
            "high": float(row.get("고가", 0)),
            "low": float(row.get("저가", 0)),
            "close": c,
            "volume": int(row.get("거래량", 0)),
        })
    return rows


async def catchup_daily(from_date: str, to_date: str, *, dry_run: bool = False):
    """Phase 1: pykrx로 일봉 갭 채움 (종목별 날짜 범위 일괄 조회)."""
    symbols = await get_all_symbols()

    logger.info("=== Phase 1: 일봉 캐치업 ===")
    logger.info("  기간: %s ~ %s, 종목 수: %d", from_date, to_date, len(symbols))

    if dry_run:
        logger.info("  [DRY-RUN] 실제 수집 없이 종료")
        return

    total_candles = 0
    failed = 0
    start_time = time.time()

    for i, sym in enumerate(symbols):
        try:
            rows = await asyncio.to_thread(_pykrx_fetch_range, sym, from_date, to_date)
            if rows:
                await write_candles_bulk(sym, rows, "1d")
                total_candles += len(rows)
        except Exception as e:
            failed += 1
            if failed <= 5:
                logger.warning("  %s 일봉 실패: %s", sym, e)

        if (i + 1) % 500 == 0:
            elapsed = time.time() - start_time
            logger.info(
                "  [%d/%d] %d건 저장, 실패 %d (elapsed %s)",
                i + 1, len(symbols), total_candles, failed,
                _format_duration(elapsed),
            )

        # pykrx IP 차단 방지 — 가벼운 throttle
        if (i + 1) % 100 == 0:
            await asyncio.sleep(1)

    elapsed = time.time() - start_time
    logger.info(
        "=== Phase 1 완료: %d건, 실패 %d, 소요 %s ===",
        total_candles, failed, _format_duration(elapsed),
    )


# ── Phase 2: 분봉 (KIS API) ─────────────────────────────


async def catchup_minute(gap_days: int, *, dry_run: bool = False):
    """Phase 2: KIS API로 분봉 갭 채움."""
    symbols = await get_all_symbols()
    today = datetime.now().strftime("%Y%m%d")

    logger.info("=== Phase 2: 분봉 캐치업 ===")
    logger.info("  종목 수: %d, 갭: %d일", len(symbols), gap_days)

    if dry_run:
        logger.info("  [DRY-RUN] 실제 수집 없이 종료")
        return

    # Progress 로딩
    progress: dict = {}
    if os.path.exists(CATCHUP_PROGRESS_FILE):
        with open(CATCHUP_PROGRESS_FILE, "r", encoding="utf-8") as f:
            progress = json.load(f)
    completed = set(progress.get("completed", []))
    total_candles = progress.get("total_candles", 0)

    remaining = [s for s in symbols if s not in completed]
    done = len(completed)
    total = len(symbols)

    logger.info("  완료: %d, 남은: %d", done, len(remaining))

    client = get_kis_client(is_mock=False)
    start_time = time.time()

    for i, sym in enumerate(remaining):
        elapsed = time.time() - start_time
        if i > 0:
            avg_per = elapsed / i
            eta = _format_duration(avg_per * (len(remaining) - i))
        else:
            eta = "..."

        current = done + i + 1
        if (current - 1) % 100 == 0 or i == 0:
            logger.info(
                "[%d/%d] %s (elapsed %s, ETA %s)",
                current, total, sym, _format_duration(elapsed), eta,
            )

        try:
            count = await collect_one(client, sym, today, gap_days)
            total_candles += count
        except Exception as e:
            logger.warning("  %s 실패: %s", sym, e)

        completed.add(sym)
        progress["completed"] = sorted(completed)
        progress["total"] = total
        progress["total_candles"] = total_candles
        progress["last_symbol"] = sym
        with open(CATCHUP_PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start_time
    logger.info("=== Phase 2 완료: %s건, 소요 %s ===", f"{total_candles:,}", _format_duration(elapsed))

    await client.close()


# ── 메인 ─────────────────────────────────────────────────


async def _run(args):
    """메인 수집 로직."""
    # 주말 빠른 스킵 (토=5, 일=6)
    now_kst = datetime.now(KST)
    weekday = now_kst.weekday()
    if weekday in (5, 6) and not args.dry_run:
        logger.info("주말 (%s) — 스킵", "토" if weekday == 5 else "일")
        return

    # 갭 확인
    last = await get_last_dates()
    today = now_kst.strftime("%Y%m%d")

    logger.info("=== 캔들 캐치업 수집 ===")
    logger.info("  DB 최신 일봉: %s", last.get("daily", "없음"))
    logger.info("  DB 최신 분봉: %s", last.get("minute", "없음"))
    logger.info("  오늘: %s (KST)", today)

    # Phase 1: 일봉
    if not args.minute_only:
        last_daily = last.get("daily")
        if last_daily:
            next_day = (datetime.strptime(last_daily, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")
            if next_day <= today:
                await catchup_daily(next_day, today, dry_run=args.dry_run)
            else:
                logger.info("[일봉] 이미 최신 (%s)", last_daily)
        else:
            logger.warning("[일봉] DB에 일봉 데이터 없음 — seed_candles.py 먼저 실행")

    # Phase 2: 분봉
    if not args.daily_only:
        last_minute = last.get("minute")
        if last_minute:
            gap_days = (datetime.strptime(today, "%Y%m%d") - datetime.strptime(last_minute, "%Y%m%d")).days + 2
            if gap_days > 2:
                # 새 캐치업이면 이전 progress 초기화
                if os.path.exists(CATCHUP_PROGRESS_FILE):
                    os.remove(CATCHUP_PROGRESS_FILE)
                await catchup_minute(gap_days, dry_run=args.dry_run)
            else:
                logger.info("[분봉] 이미 최신 (%s)", last_minute)
        else:
            logger.warning("[분봉] DB에 분봉 데이터 없음 — collect_minute_kis.py 먼저 실행")

    logger.info("=== 캐치업 완료 ===")


async def main():
    parser = argparse.ArgumentParser(description="최신 캔들 데이터 캐치업 수집")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="갭 확인만 (실제 수집 없이)",
    )
    parser.add_argument(
        "--daily-only", action="store_true",
        help="일봉만 수집",
    )
    parser.add_argument(
        "--minute-only", action="store_true",
        help="분봉만 수집",
    )
    args = parser.parse_args()

    # 파일 잠금 — 동시 실행 방지
    lock_fp = open(LOCK_FILE, "w")
    try:
        fcntl.flock(lock_fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, IOError):
        logger.warning("이미 다른 catchup 프로세스가 실행 중 — 종료")
        sys.exit(0)

    try:
        await _run(args)
    finally:
        fcntl.flock(lock_fp, fcntl.LOCK_UN)
        lock_fp.close()


if __name__ == "__main__":
    asyncio.run(main())
