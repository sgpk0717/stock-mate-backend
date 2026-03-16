"""통합 데이터 백필 스크립트.

수집 공백 기간의 데이터를 메꾸는 일시적 스크립트.
각 job별로 DB 기존 데이터 날짜를 조회하여 gap(누락 날짜)을 감지하고,
dry-run이 아니면 기존 collector를 호출하여 데이터를 채운다.

Usage:
    docker-compose run --rm app python -m scripts.backfill_all \
      --start 2024-01-01 --end 2026-03-14 \
      --jobs daily,minute,investor,margin_short,news,dart \
      --dry-run

    # 전체 잡, 특정 종목만
    docker-compose run --rm app python -m scripts.backfill_all \
      --start 2025-01-01 --symbols 005930,000660

    # 투자자 수급 + 공매도만
    docker-compose run --rm app python -m scripts.backfill_all \
      --start 2025-06-01 --jobs investor,margin_short
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from datetime import date, timedelta, timezone

import asyncpg

from app.core.config import settings

# 순환 임포트 방지: 모든 모델을 먼저 로드
import app.models.base  # noqa: F401

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
# 잡음 억제
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9))
ALL_JOBS = ("daily", "minute", "investor", "margin_short", "news", "dart")


# ── 유틸리티 ──────────────────────────────────────────────


def _dsn() -> str:
    s = settings
    return (
        f"postgresql://{s.POSTGRES_USER}:{s.POSTGRES_PASSWORD}"
        f"@{s.POSTGRES_HOST}:{s.POSTGRES_PORT}/{s.POSTGRES_DB}"
    )


def _fmt_duration(seconds: float) -> str:
    d = int(seconds // 86400)
    h = int((seconds % 86400) // 3600)
    m = int((seconds % 3600) // 60)
    sec = int(seconds % 60)
    parts = []
    if d:
        parts.append(f"{d}d")
    if h:
        parts.append(f"{h}h")
    parts.append(f"{m}m")
    parts.append(f"{sec}s")
    return " ".join(parts)


def _parse_date(s: str) -> date:
    """YYYY-MM-DD 또는 YYYYMMDD 파싱."""
    s = s.strip().replace("-", "")
    return date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def _date_to_str(d: date) -> str:
    """date → YYYYMMDD 문자열."""
    return d.strftime("%Y%m%d")


# ── 거래일 캘린더 ─────────────────────────────────────────


def _build_trading_calendar(start: date, end: date) -> list[date]:
    """pykrx 삼성전자 일봉으로 start~end 구간의 거래일 목록 생성.

    get_market_ohlcv_by_date 1회 호출로 전체 거래일을 추출한다.
    날짜별 ticker 조회(수백 회)보다 훨씬 빠르고 rate limit에 안전하다.
    """
    from pykrx import stock as krx

    logger.info("거래일 캘린더 생성 중 (%s ~ %s)...", start, end)

    start_str = _date_to_str(start)
    end_str = _date_to_str(end)

    try:
        # 삼성전자(005930) 일봉 — 1회 호출로 전체 거래일 추출
        df = krx.get_market_ohlcv_by_date(start_str, end_str, "005930")
        if df is not None and not df.empty:
            trading_days = sorted([d.date() for d in df.index])
        else:
            trading_days = []
    except Exception as e:
        logger.warning("pykrx 거래일 조회 실패: %s — 주말 제외 캘린더로 대체", e)
        # fallback: 주말만 제거
        trading_days = []
        cur = start
        while cur <= end:
            if cur.weekday() < 5:
                trading_days.append(cur)
            cur += timedelta(days=1)

    logger.info("거래일 캘린더 완료: %d일", len(trading_days))
    return trading_days


# ── Gap 감지 ──────────────────────────────────────────────


async def _detect_gaps_daily(
    conn: asyncpg.Connection,
    trading_days: list[date],
    symbols: list[str] | None,
) -> list[date]:
    """일봉 gap 감지: DB에 존재하는 날짜 vs 거래일 캘린더."""
    query = "SELECT DISTINCT dt::date FROM stock_candles WHERE interval = '1d'"
    rows = await conn.fetch(query)
    existing = {r[0] for r in rows}
    gaps = [d for d in trading_days if d not in existing]
    return sorted(gaps)


async def _detect_gaps_minute(
    conn: asyncpg.Connection,
    trading_days: list[date],
    symbols: list[str] | None,
) -> list[date]:
    """분봉 gap 감지: DB에 존재하는 날짜 vs 거래일 캘린더."""
    query = "SELECT DISTINCT dt::date FROM stock_candles WHERE interval = '1m'"
    rows = await conn.fetch(query)
    existing = {r[0] for r in rows}
    gaps = [d for d in trading_days if d not in existing]
    return sorted(gaps)


async def _detect_gaps_investor(
    conn: asyncpg.Connection,
    trading_days: list[date],
    symbols: list[str] | None,
) -> list[date]:
    """투자자 수급 gap 감지."""
    query = "SELECT DISTINCT dt FROM investor_trading"
    rows = await conn.fetch(query)
    existing = {r[0] for r in rows}
    gaps = [d for d in trading_days if d not in existing]
    return sorted(gaps)


async def _detect_gaps_margin_short(
    conn: asyncpg.Connection,
    trading_days: list[date],
    symbols: list[str] | None,
) -> list[date]:
    """공매도/신용잔고 gap 감지."""
    query = "SELECT DISTINCT dt FROM margin_short_daily"
    rows = await conn.fetch(query)
    existing = {r[0] for r in rows}
    gaps = [d for d in trading_days if d not in existing]
    return sorted(gaps)


async def _detect_gaps_news(
    conn: asyncpg.Connection,
    trading_days: list[date],
    symbols: list[str] | None,
) -> list[date]:
    """뉴스 gap 감지: 기사가 없는 날."""
    query = "SELECT DISTINCT published_at::date FROM news_articles"
    rows = await conn.fetch(query)
    existing = {r[0] for r in rows}
    gaps = [d for d in trading_days if d not in existing]
    return sorted(gaps)


async def _detect_gaps_dart(
    conn: asyncpg.Connection,
    trading_days: list[date],
    symbols: list[str] | None,
) -> list[tuple[int, str]]:
    """DART 재무 gap 감지: 연도x분기 중 데이터 없는 조합.

    Returns:
        [(year, quarter), ...] 예: [(2024, "1Q"), (2024, "2Q")]
    """
    start_year = trading_days[0].year if trading_days else date.today().year
    end_year = trading_days[-1].year if trading_days else date.today().year

    all_quarters = ["1Q", "2Q", "3Q", "4Q"]
    expected: set[tuple[int, str]] = set()
    for y in range(start_year, end_year + 1):
        for q in all_quarters:
            expected.add((y, q))

    query = "SELECT DISTINCT fiscal_year, fiscal_quarter FROM dart_financials"
    rows = await conn.fetch(query)
    existing = {(int(r[0]), r[1]) for r in rows}

    gaps = sorted(expected - existing)
    return gaps


# ── Job 실행 ──────────────────────────────────────────────


async def _get_all_symbols() -> list[str]:
    """stock_masters에서 전 종목 코드 조회."""
    from sqlalchemy import text

    from app.core.database import async_session

    async with async_session() as db:
        result = await db.execute(
            text("SELECT symbol FROM stock_masters ORDER BY symbol"),
        )
        return [r[0] for r in result.fetchall()]


async def _run_daily(
    gaps: list[date],
    symbols: list[str] | None,
) -> dict:
    """일봉 백필: pykrx 종목별 범위 조회."""
    from app.services.candle_writer import write_candles_bulk

    all_symbols = symbols or await _get_all_symbols()
    total_candles = 0
    failed = 0

    for gi, gap_date in enumerate(gaps):
        date_str = _date_to_str(gap_date)
        logger.info("[daily] %d/%d gaps — date=%s", gi + 1, len(gaps), date_str)

        def _fetch_all(d: str, syms: list[str]) -> list[tuple[str, list[dict]]]:
            from pykrx import stock as krx

            results = []
            for i, sym in enumerate(syms):
                try:
                    df = krx.get_market_ohlcv_by_date(d, d, sym)
                    if df.empty:
                        continue
                    row = df.iloc[0]
                    c = float(row.get("종가", 0))
                    if c <= 0:
                        continue
                    results.append((sym, [{
                        "dt": d,
                        "open": float(row.get("시가", 0)),
                        "high": float(row.get("고가", 0)),
                        "low": float(row.get("저가", 0)),
                        "close": c,
                        "volume": int(row.get("거래량", 0)),
                    }]))
                except Exception:
                    pass
                # IP 차단 방지
                if (i + 1) % 100 == 0:
                    time.sleep(1)
            return results

        try:
            results = await asyncio.to_thread(_fetch_all, date_str, all_symbols)
            for sym, rows in results:
                await write_candles_bulk(sym, rows, "1d")
                total_candles += len(rows)
        except Exception as e:
            failed += 1
            logger.warning("[daily] date=%s 실패: %s", date_str, e)

        # 쓰로틀링
        await asyncio.sleep(0.3)

    return {"filled": len(gaps) - failed, "failed": failed, "candles": total_candles}


async def _run_minute(
    gaps: list[date],
    symbols: list[str] | None,
) -> dict:
    """분봉 백필: KIS API 종목별 수집."""
    from app.trading.kis_client import get_kis_client
    from scripts.collect_minute_kis import collect_one

    all_symbols = symbols or await _get_all_symbols()
    client = get_kis_client(is_mock=False)

    # 토큰 warmup
    for attempt in range(3):
        try:
            await client._get_token()
            break
        except Exception as e:
            if "403" in str(e) and attempt < 2:
                logger.info("[minute] 토큰 1분 제한 — 65초 대기 (attempt %d)", attempt + 1)
                await asyncio.sleep(65)
            else:
                logger.error("[minute] 토큰 발급 실패: %s", e)
                return {"filled": 0, "failed": len(gaps), "candles": 0}

    total_candles = 0
    filled_dates = 0

    for gi, gap_date in enumerate(gaps):
        date_str = _date_to_str(gap_date)
        logger.info("[minute] %d/%d gaps — date=%s (%d symbols)",
                    gi + 1, len(gaps), date_str, len(all_symbols))

        date_candles = 0
        for si, sym in enumerate(all_symbols):
            try:
                count = await collect_one(client, sym, date_str, max_days=1)
                date_candles += count
            except Exception as e:
                if si < 3:
                    logger.warning("[minute] %s date=%s 실패: %s", sym, date_str, e)

            # KIS API 쓰로틀링
            await asyncio.sleep(0.12)

            if (si + 1) % 200 == 0:
                logger.info("[minute]   %d/%d symbols, %d candles",
                            si + 1, len(all_symbols), date_candles)

        total_candles += date_candles
        filled_dates += 1
        logger.info("[minute] date=%s 완료: %d candles", date_str, date_candles)

    await client.close()
    return {"filled": filled_dates, "failed": len(gaps) - filled_dates, "candles": total_candles}


async def _run_investor(
    gaps: list[date],
    symbols: list[str] | None,
) -> dict:
    """투자자 수급 백필."""
    from app.scheduler.circuit_breaker import CircuitBreaker
    from app.scheduler.collectors.investor import collect_investor

    cb = CircuitBreaker("kis-backfill-investor", failure_threshold=5, reset_timeout=60)
    filled = 0
    failed = 0

    for gi, gap_date in enumerate(gaps):
        date_str = _date_to_str(gap_date)
        logger.info("[investor] %d/%d gaps — date=%s", gi + 1, len(gaps), date_str)

        try:
            result = await collect_investor(date_str, cb=cb)
            if result.error:
                failed += 1
                logger.warning("[investor] date=%s error: %s", date_str, result.error)
            else:
                filled += 1
                logger.info("[investor] date=%s 완료: %d/%d",
                            date_str, result.completed, result.total)
        except Exception as e:
            failed += 1
            logger.warning("[investor] date=%s 예외: %s", date_str, e)

    return {"filled": filled, "failed": failed}


async def _run_margin_short(
    gaps: list[date],
    symbols: list[str] | None,
) -> dict:
    """공매도/신용잔고 백필."""
    from app.scheduler.circuit_breaker import CircuitBreaker
    from app.scheduler.collectors.margin_short import collect_margin_short

    cb = CircuitBreaker("kis-backfill-margin", failure_threshold=5, reset_timeout=60)
    filled = 0
    failed = 0

    for gi, gap_date in enumerate(gaps):
        date_str = _date_to_str(gap_date)
        logger.info("[margin_short] %d/%d gaps — date=%s", gi + 1, len(gaps), date_str)

        try:
            result = await collect_margin_short(date_str, cb=cb)
            if result.error:
                failed += 1
                logger.warning("[margin_short] date=%s error: %s", date_str, result.error)
            else:
                filled += 1
                logger.info("[margin_short] date=%s 완료: %d/%d",
                            date_str, result.completed, result.total)
        except Exception as e:
            failed += 1
            logger.warning("[margin_short] date=%s 예외: %s", date_str, e)

    return {"filled": filled, "failed": failed}


async def _run_news(
    gaps: list[date],
    symbols: list[str] | None,
) -> dict:
    """뉴스 백필 — collect_and_analyze 직접 호출."""
    all_symbols = symbols or await _get_all_symbols()
    # 상위 200개 종목만 (전 종목은 과도)
    top_symbols = all_symbols[:200]

    filled = 0
    failed = 0

    for gi, gap_date in enumerate(gaps):
        date_str = _date_to_str(gap_date)
        logger.info("[news] %d/%d gaps — date=%s", gi + 1, len(gaps), date_str)

        try:
            from app.core.database import async_session as get_session
            from app.news.scheduler import collect_and_analyze

            async with get_session() as session:
                result = await collect_and_analyze(
                    session, top_symbols, days=1,
                )
                await session.commit()
            filled += 1
            logger.info("[news] date=%s 완료: %s", date_str, result)
        except Exception as e:
            failed += 1
            logger.warning("[news] date=%s 예외: %s", date_str, e)

        # 크롤링 쓰로틀링
        await asyncio.sleep(1)

    return {"filled": filled, "failed": failed}


async def _run_dart(
    gaps: list[tuple[int, str]],
    symbols: list[str] | None,
) -> dict:
    """DART 재무 백필: seed_dart_financials.seed() 호출."""
    from scripts.seed_dart_financials import seed

    if not gaps:
        return {"filled": 0, "failed": 0}

    # 연도 목록 추출 (중복 제거)
    years = sorted({y for y, _q in gaps})

    logger.info("[dart] %d 연도 x 분기 조합 gap — years=%s", len(gaps), years)

    try:
        await seed(years, symbols)
        return {"filled": len(gaps), "failed": 0}
    except Exception as e:
        logger.error("[dart] 실패: %s", e)
        return {"filled": 0, "failed": len(gaps)}


# ── 메인 로직 ─────────────────────────────────────────────


async def run(
    start: date,
    end: date,
    jobs: tuple[str, ...],
    symbols: list[str] | None,
    dry_run: bool,
) -> None:
    """백필 메인 로직."""
    logger.info("=" * 60)
    logger.info("통합 백필 시작")
    logger.info("  기간: %s ~ %s", start, end)
    logger.info("  잡: %s", ", ".join(jobs))
    logger.info("  종목: %s", "전체" if symbols is None else f"{len(symbols)}개")
    logger.info("  dry-run: %s", dry_run)
    logger.info("=" * 60)

    # 1) 거래일 캘린더 생성
    trading_days = await asyncio.to_thread(_build_trading_calendar, start, end)
    if not trading_days:
        logger.warning("거래일이 없습니다. 기간을 확인하세요.")
        return

    # 2) DB 연결 (gap 감지용)
    conn = await asyncpg.connect(_dsn())

    # 3) 각 job별 gap 감지 + 실행
    summary: dict[str, dict] = {}
    t0_total = time.monotonic()

    for job_name in jobs:
        logger.info("")
        logger.info("--- [%s] Gap 분석 ---", job_name)
        t0_job = time.monotonic()

        try:
            if job_name == "daily":
                gaps = await _detect_gaps_daily(conn, trading_days, symbols)
                logger.info("[daily] gap: %d일 누락 (전체 거래일 %d)",
                            len(gaps), len(trading_days))
                if gaps:
                    logger.info("[daily] 첫 gap: %s, 마지막 gap: %s", gaps[0], gaps[-1])

                if dry_run or not gaps:
                    summary["daily"] = {"gaps": len(gaps), "filled": 0, "status": "dry-run" if dry_run else "no-gaps"}
                    continue

                result = await _run_daily(gaps, symbols)
                elapsed = time.monotonic() - t0_job
                summary["daily"] = {
                    "gaps": len(gaps), **result,
                    "elapsed": _fmt_duration(elapsed),
                }

            elif job_name == "minute":
                gaps = await _detect_gaps_minute(conn, trading_days, symbols)
                logger.info("[minute] gap: %d일 누락 (전체 거래일 %d)",
                            len(gaps), len(trading_days))
                if gaps:
                    logger.info("[minute] 첫 gap: %s, 마지막 gap: %s", gaps[0], gaps[-1])

                if dry_run or not gaps:
                    summary["minute"] = {"gaps": len(gaps), "filled": 0, "status": "dry-run" if dry_run else "no-gaps"}
                    continue

                result = await _run_minute(gaps, symbols)
                elapsed = time.monotonic() - t0_job
                summary["minute"] = {
                    "gaps": len(gaps), **result,
                    "elapsed": _fmt_duration(elapsed),
                }

            elif job_name == "investor":
                gaps = await _detect_gaps_investor(conn, trading_days, symbols)
                logger.info("[investor] gap: %d일 누락 (전체 거래일 %d)",
                            len(gaps), len(trading_days))
                if gaps:
                    logger.info("[investor] 첫 gap: %s, 마지막 gap: %s", gaps[0], gaps[-1])

                if dry_run or not gaps:
                    summary["investor"] = {"gaps": len(gaps), "filled": 0, "status": "dry-run" if dry_run else "no-gaps"}
                    continue

                result = await _run_investor(gaps, symbols)
                elapsed = time.monotonic() - t0_job
                summary["investor"] = {
                    "gaps": len(gaps), **result,
                    "elapsed": _fmt_duration(elapsed),
                }

            elif job_name == "margin_short":
                gaps = await _detect_gaps_margin_short(conn, trading_days, symbols)
                logger.info("[margin_short] gap: %d일 누락 (전체 거래일 %d)",
                            len(gaps), len(trading_days))
                if gaps:
                    logger.info("[margin_short] 첫 gap: %s, 마지막 gap: %s", gaps[0], gaps[-1])

                if dry_run or not gaps:
                    summary["margin_short"] = {"gaps": len(gaps), "filled": 0, "status": "dry-run" if dry_run else "no-gaps"}
                    continue

                result = await _run_margin_short(gaps, symbols)
                elapsed = time.monotonic() - t0_job
                summary["margin_short"] = {
                    "gaps": len(gaps), **result,
                    "elapsed": _fmt_duration(elapsed),
                }

            elif job_name == "news":
                gaps = await _detect_gaps_news(conn, trading_days, symbols)
                logger.info("[news] gap: %d일 누락 (전체 거래일 %d)",
                            len(gaps), len(trading_days))
                if gaps:
                    logger.info("[news] 첫 gap: %s, 마지막 gap: %s", gaps[0], gaps[-1])

                if dry_run or not gaps:
                    summary["news"] = {"gaps": len(gaps), "filled": 0, "status": "dry-run" if dry_run else "no-gaps"}
                    continue

                result = await _run_news(gaps, symbols)
                elapsed = time.monotonic() - t0_job
                summary["news"] = {
                    "gaps": len(gaps), **result,
                    "elapsed": _fmt_duration(elapsed),
                }

            elif job_name == "dart":
                gaps = await _detect_gaps_dart(conn, trading_days, symbols)
                logger.info("[dart] gap: %d 연도x분기 조합 누락", len(gaps))
                if gaps:
                    for y, q in gaps[:10]:
                        logger.info("[dart]   %d %s", y, q)
                    if len(gaps) > 10:
                        logger.info("[dart]   ... 외 %d개", len(gaps) - 10)

                if dry_run or not gaps:
                    summary["dart"] = {"gaps": len(gaps), "filled": 0, "status": "dry-run" if dry_run else "no-gaps"}
                    continue

                result = await _run_dart(gaps, symbols)
                elapsed = time.monotonic() - t0_job
                summary["dart"] = {
                    "gaps": len(gaps), **result,
                    "elapsed": _fmt_duration(elapsed),
                }

            else:
                logger.warning("알 수 없는 잡: %s — 스킵", job_name)

        except Exception as e:
            elapsed = time.monotonic() - t0_job
            logger.error("[%s] 예외: %s", job_name, str(e)[:500])
            summary[job_name] = {
                "gaps": "?", "filled": 0, "failed": 1,
                "error": str(e)[:200],
                "elapsed": _fmt_duration(elapsed),
            }

    await conn.close()

    # 4) 최종 요약
    total_elapsed = time.monotonic() - t0_total
    logger.info("")
    logger.info("=" * 60)
    logger.info("=== Backfill complete (총 소요: %s) ===", _fmt_duration(total_elapsed))
    for job_name, info in summary.items():
        gaps_count = info.get("gaps", 0)
        filled = info.get("filled", 0)
        failed = info.get("failed", 0)
        status = info.get("status", "")
        elapsed = info.get("elapsed", "")
        candles = info.get("candles", "")

        parts = [f"gaps={gaps_count}"]
        if status:
            parts.append(status)
        else:
            parts.append(f"filled={filled}")
            if failed:
                parts.append(f"failed={failed}")
            if candles:
                parts.append(f"candles={candles}")
            if elapsed:
                parts.append(elapsed)

        logger.info("  %-14s %s", job_name, ", ".join(parts))
    logger.info("=" * 60)


# ── CLI ───────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="통합 데이터 백필 — 수집 공백 기간 메꾸기",
    )
    parser.add_argument(
        "--start", required=True,
        help="시작일 (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end", default=None,
        help="종료일 (YYYY-MM-DD). 기본: 어제",
    )
    parser.add_argument(
        "--jobs", default=",".join(ALL_JOBS),
        help=f"수집 잡 (쉼표 구분). 기본: {','.join(ALL_JOBS)}",
    )
    parser.add_argument(
        "--symbols", default=None,
        help="특정 종목만 (쉼표 구분). 기본: 전체",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="갭 분석만 (실제 수집 없이)",
    )
    args = parser.parse_args()

    start = _parse_date(args.start)

    if args.end:
        end = _parse_date(args.end)
    else:
        end = date.today() - timedelta(days=1)

    if start > end:
        logger.error("시작일(%s)이 종료일(%s)보다 큽니다.", start, end)
        return

    jobs = tuple(j.strip() for j in args.jobs.split(",") if j.strip())
    invalid_jobs = [j for j in jobs if j not in ALL_JOBS]
    if invalid_jobs:
        logger.error("유효하지 않은 잡: %s (가능: %s)", invalid_jobs, ALL_JOBS)
        return

    symbols = None
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    asyncio.run(run(start, end, jobs, symbols, args.dry_run))


if __name__ == "__main__":
    main()
