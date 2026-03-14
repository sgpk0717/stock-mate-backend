"""외부 데이터 캐치업 수집 (투자자 수급 / 공매도·신용 / 뉴스).

캔들 캐치업(catchup_candles.py)과 분리하여 실행한다.
Windows Task Scheduler → daily_batch.ps1 에서 호출.

Usage:
    docker compose run --rm app python -m scripts.catchup_external
    docker compose run --rm app python -m scripts.catchup_external --jobs investor,margin_short
    docker compose run --rm app python -m scripts.catchup_external --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone

KST = timezone(timedelta(hours=9))
LOCK_FILE = "/tmp/catchup_external.lock"
ALL_JOBS = ("investor", "margin_short", "news")

# 로그 설정: SQLAlchemy echo 끄기, 간결한 포맷
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
# SQLAlchemy / httpx 로그 억제
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def _now_kst() -> datetime:
    return datetime.now(KST)


def _fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    parts = []
    if h:
        parts.append(f"{h}h")
    parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)


def _is_weekend() -> bool:
    return _now_kst().weekday() >= 5


async def _is_trading_day(date_str: str) -> bool:
    """pykrx로 거래일 여부 확인."""
    def _check():
        from pykrx import stock as krx
        tickers = krx.get_market_ticker_list(date_str, market="ALL")
        return len(tickers) > 0

    try:
        return await asyncio.to_thread(_check)
    except Exception as e:
        logger.warning("거래일 확인 실패 (기본 True): %s", e)
        return True


async def run_investor(date_str: str) -> dict:
    """투자자별 매매동향 수집."""
    from app.scheduler.circuit_breaker import CircuitBreaker
    from app.scheduler.collectors.investor import collect_investor

    cb = CircuitBreaker("kis-investor", failure_threshold=5, reset_timeout=60)
    t0 = time.monotonic()
    result = await collect_investor(date_str, cb=cb)
    elapsed = time.monotonic() - t0
    return {
        "job": "investor",
        "completed": result.completed,
        "failed": result.failed,
        "skipped": result.skipped,
        "error": result.error,
        "elapsed": elapsed,
    }


async def run_margin_short(date_str: str) -> dict:
    """공매도 + 신용잔고 수집."""
    from app.scheduler.circuit_breaker import CircuitBreaker
    from app.scheduler.collectors.margin_short import collect_margin_short

    cb = CircuitBreaker("kis-margin", failure_threshold=5, reset_timeout=60)
    t0 = time.monotonic()
    result = await collect_margin_short(date_str, cb=cb)
    elapsed = time.monotonic() - t0
    return {
        "job": "margin_short",
        "completed": result.completed,
        "failed": result.failed,
        "skipped": result.skipped,
        "error": result.error,
        "elapsed": elapsed,
    }


async def run_news(date_str: str) -> dict:
    """뉴스 수집 + 감성 분석."""
    from app.scheduler.circuit_breaker import CircuitBreaker
    from app.scheduler.collectors.news_batch import collect_news

    cb = CircuitBreaker("claude-news", failure_threshold=3, reset_timeout=120)
    t0 = time.monotonic()
    result = await collect_news(date_str, cb=cb)
    elapsed = time.monotonic() - t0
    return {
        "job": "news",
        "completed": result.completed,
        "failed": result.failed,
        "skipped": result.skipped,
        "error": result.error,
        "elapsed": elapsed,
    }


JOB_MAP = {
    "investor": run_investor,
    "margin_short": run_margin_short,
    "news": run_news,
}


async def main(jobs: tuple[str, ...], dry_run: bool = False) -> int:
    today = _now_kst().strftime("%Y%m%d")

    if _is_weekend():
        logger.info("주말 — 스킵")
        return 0

    if not await _is_trading_day(today):
        logger.info("비거래일 (%s) — 스킵", today)
        return 0

    logger.info("=== 외부 데이터 캐치업 시작 (date=%s, jobs=%s) ===", today, ",".join(jobs))

    if dry_run:
        logger.info("[DRY-RUN] 실제 수집 없이 종료")
        return 0

    exit_code = 0

    for job_name in jobs:
        fn = JOB_MAP.get(job_name)
        if not fn:
            logger.warning("알 수 없는 잡: %s — 스킵", job_name)
            continue

        logger.info("--- [%s] 시작 ---", job_name)
        try:
            result = await fn(today)
            err = result.get("error")
            elapsed = _fmt_duration(result.get("elapsed", 0))
            logger.info(
                "--- [%s] 완료: %d성공 / %d실패 / %d스킵, 소요 %s%s ---",
                job_name,
                result.get("completed", 0),
                result.get("failed", 0),
                result.get("skipped", 0),
                elapsed,
                f" (ERROR: {str(err)[:200]})" if err else "",
            )
            if result.get("failed", 0) > 0 or err:
                exit_code = 1
        except Exception as e:
            logger.error("--- [%s] 예외: %s ---", job_name, str(e)[:300])
            exit_code = 1

    logger.info("=== 외부 데이터 캐치업 종료 ===")
    return exit_code


def cli():
    parser = argparse.ArgumentParser(description="외부 데이터 캐치업 수집")
    parser.add_argument(
        "--jobs",
        default=",".join(ALL_JOBS),
        help=f"수집 잡 (쉼표 구분). 기본: {','.join(ALL_JOBS)}",
    )
    parser.add_argument("--dry-run", action="store_true", help="실행 없이 확인만")
    args = parser.parse_args()

    jobs = tuple(j.strip() for j in args.jobs.split(",") if j.strip())

    # 파일 락: 중복 실행 방지
    try:
        lock_fd = open(LOCK_FILE, "w")
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (IOError, OSError):
        logger.error("이미 실행 중 (lock=%s) — 종료", LOCK_FILE)
        sys.exit(1)

    try:
        code = asyncio.run(main(jobs, dry_run=args.dry_run))
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()
        try:
            os.remove(LOCK_FILE)
        except OSError:
            pass

    sys.exit(code)


if __name__ == "__main__":
    cli()
