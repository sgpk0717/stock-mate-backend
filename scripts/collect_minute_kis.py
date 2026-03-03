"""KIS API로 전 종목 1분봉 수집 (역순 — 뒤에서 앞으로).

KIS REST API (inquire_time_dailychartprice, FHKST03010230)를 활용하여
전 종목의 과거 1분봉 데이터를 수집한다.

Usage:
    # 일반 모드: stock_masters 전 종목 1년치 수집
    docker-compose run --rm app python -m scripts.collect_minute_kis

    # 보충 모드: 키움이 수집한 종목의 부족분(~4.5개월)을 KIS로 채움
    docker-compose run --rm app python -m scripts.collect_minute_kis --fill-kiwoom
"""

import argparse
import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta

from sqlalchemy import text

from app.core.database import async_session
from app.services.candle_writer import write_candles_bulk
from app.trading.kis_client import get_kis_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROGRESS_FILE = "kis_minute_progress.json"
FILL_PROGRESS_FILE = "kis_fill_progress.json"
RETRY_PROGRESS_FILE = "kis_retry_progress.json"
CLAIMS_FILE = "minute_claims.json"


def _load_claims() -> dict:
    try:
        with open(CLAIMS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


# ── 유틸리티 ─────────────────────────────────────────────

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


def load_progress() -> dict:
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed": [], "total": 0, "total_candles": 0}


def save_progress(progress: dict):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


# ── DB 조회 ──────────────────────────────────────────────

async def get_all_symbols() -> list[dict]:
    """stock_masters에서 전 종목 조회 (역순 — 뒤에서부터)."""
    async with async_session() as db:
        result = await db.execute(
            text("SELECT symbol, name FROM stock_masters ORDER BY symbol DESC")
        )
        return [{"symbol": row[0], "name": row[1]} for row in result.fetchall()]


async def get_kiwoom_fill_targets() -> list[dict]:
    """키움이 수집한 종목 중 stock_masters에 있는 것들의 가장 오래된 dt 조회.

    Returns:
        [{"symbol": "000660", "name": "SK하이닉스", "oldest_dt": "20250801"}]
    """
    claims = _load_claims()
    kiwoom_syms = [sym for sym, col in claims.items() if col == "kiwoom"]

    if not kiwoom_syms:
        return []

    # stock_masters와 교차하여 이름 가져오기 + DB에서 가장 오래된 dt 조회
    async with async_session() as db:
        # stock_masters에 있는 키움 종목만 필터
        placeholders = ", ".join(f":s{i}" for i in range(len(kiwoom_syms)))
        params = {f"s{i}": s for i, s in enumerate(kiwoom_syms)}

        result = await db.execute(
            text(f"""
                SELECT m.symbol, m.name,
                       MIN(c.dt) AS oldest_dt
                FROM stock_masters m
                LEFT JOIN stock_candles c
                    ON c.symbol = m.symbol AND c.interval = '1m'
                WHERE m.symbol IN ({placeholders})
                GROUP BY m.symbol, m.name
                ORDER BY m.symbol
            """),
            params,
        )
        rows = result.fetchall()

    targets = []
    for row in rows:
        sym, name, oldest_dt = row[0], row[1], row[2]
        if oldest_dt is None:
            # DB에 데이터가 없는 종목 → 오늘부터 1년치 전체 수집
            targets.append({"symbol": sym, "name": name, "oldest_dt": None})
        else:
            # oldest_dt를 KST 문자열로 변환
            from datetime import timezone as tz
            KST = tz(timedelta(hours=9))
            if hasattr(oldest_dt, "astimezone"):
                dt_kst = oldest_dt.astimezone(KST)
            else:
                dt_kst = oldest_dt
            targets.append({
                "symbol": sym,
                "name": name,
                "oldest_dt": dt_kst.strftime("%Y%m%d"),
            })

    return targets


# ── 수집 로직 ────────────────────────────────────────────

async def collect_one(
    client, symbol: str, start_date: str, max_days: int,
) -> int:
    """한 종목의 1분봉을 수집 (역순, 최대 1년)."""
    date = start_date
    hour = "160000"
    total = 0
    cutoff = (datetime.strptime(start_date, "%Y%m%d") - timedelta(days=max_days)).strftime("%Y%m%d")

    consec_errors = 0

    while True:
        try:
            candles, next_date, next_hour = await client.get_minute_candles(
                symbol, date, hour,
            )
            consec_errors = 0
        except Exception as e:
            consec_errors += 1
            if consec_errors <= 3:
                logger.warning(
                    "%s: API error (attempt %d/3, date=%s): %s",
                    symbol, consec_errors, date, e,
                )
                await asyncio.sleep(2 ** consec_errors)
                continue
            logger.warning("%s: 3회 연속 실패, 종료: %s", symbol, e)
            break

        if not candles:
            break

        rows = []
        for c in candles:
            dt_str = c.get("stck_bsop_date", "") + c.get("stck_cntg_hour", "")
            if len(dt_str) != 14:
                continue
            cl = float(c.get("stck_prpr", 0) or 0)
            if cl <= 0:
                continue
            rows.append({
                "dt": dt_str,
                "open": float(c.get("stck_oprc", 0) or 0),
                "high": float(c.get("stck_hgpr", 0) or 0),
                "low": float(c.get("stck_lwpr", 0) or 0),
                "close": cl,
                "volume": int(c.get("cntg_vol", 0) or 0),
            })

        if rows:
            await write_candles_bulk(symbol, rows, "1m")
            total += len(rows)

        if next_date and next_date < cutoff:
            break
        if not next_date or not next_hour:
            break

        # pagination이 진행되지 않으면 종료 (무한 루프 방지)
        if next_date == date and next_hour == hour:
            break

        date, hour = next_date, next_hour

    return total


async def _get_oldest_minute_dt(symbol: str) -> str | None:
    """종목의 1분봉 MIN(dt)을 KST YYYYMMDD로 반환. 없으면 None."""
    from datetime import timezone as tz

    KST = tz(timedelta(hours=9))
    async with async_session() as db:
        result = await db.execute(
            text(
                "SELECT MIN(dt) FROM stock_candles"
                " WHERE symbol = :sym AND interval = '1m'"
            ),
            {"sym": symbol},
        )
        row = result.scalar()
    if row is None:
        return None
    if hasattr(row, "astimezone"):
        row = row.astimezone(KST)
    return row.strftime("%Y%m%d")


async def collect_one_verified(
    client,
    symbol: str,
    start_date: str,
    max_days: int,
    max_retries: int = 2,
) -> int:
    """수집 후 DB 검증 — 1년치 미달이면 부족분 재시도.

    Args:
        max_retries: 검증 실패 시 추가 재시도 횟수 (기본 2 → 총 3회 시도).
    """
    cutoff = (
        datetime.strptime(start_date, "%Y%m%d") - timedelta(days=max_days)
    ).strftime("%Y%m%d")
    total = 0

    cur_start = start_date
    cur_days = max_days

    for attempt in range(max_retries + 1):  # 0=initial, 1..max_retries=재시도
        count = await collect_one(client, symbol, cur_start, cur_days)
        total += count

        if count == 0:
            break  # API에 데이터 자체가 없음

        actual_oldest = await _get_oldest_minute_dt(symbol)
        if actual_oldest is None:
            break
        if actual_oldest <= cutoff:
            break  # 1년치 충족

        if attempt >= max_retries:
            logger.warning(
                "%s: %d회 재시도 후에도 부족 (oldest=%s, cutoff=%s)",
                symbol, max_retries, actual_oldest, cutoff,
            )
            break

        # 부족분 재수집: actual_oldest + 2일 오버랩 → cutoff까지
        overlap_dt = datetime.strptime(actual_oldest, "%Y%m%d") + timedelta(days=2)
        cur_start = overlap_dt.strftime("%Y%m%d")
        cur_days = (overlap_dt - datetime.strptime(cutoff, "%Y%m%d")).days
        logger.info(
            "%s: 검증 — oldest=%s > cutoff=%s, 재시도 %d/%d (%d일 부족분)",
            symbol, actual_oldest, cutoff, attempt + 1, max_retries, cur_days,
        )

    return total


# ── retry 모드 ────────────────────────────────────────────

async def get_retry_targets(max_days: int = 365) -> list[dict]:
    """DB에서 데이터가 부족한 종목 목록 조회.

    Returns:
        [{"symbol": "...", "name": "...", "oldest_dt": "YYYYMMDD" or None}]
    """
    from datetime import timezone as tz
    KST = tz(timedelta(hours=9))
    cutoff = (datetime.now() - timedelta(days=max_days)).strftime("%Y%m%d")

    async with async_session() as db:
        result = await db.execute(
            text("""
                SELECT m.symbol, m.name, MIN(c.dt) AS oldest_dt
                FROM stock_masters m
                LEFT JOIN stock_candles c
                    ON c.symbol = m.symbol AND c.interval = '1m'
                GROUP BY m.symbol, m.name
                ORDER BY m.symbol
            """)
        )
        rows = result.fetchall()

    targets = []
    for row in rows:
        sym, name, oldest_dt = row[0], row[1], row[2]
        if oldest_dt is None:
            targets.append({"symbol": sym, "name": name, "oldest_dt": None})
        else:
            if hasattr(oldest_dt, "astimezone"):
                dt_kst = oldest_dt.astimezone(KST)
            else:
                dt_kst = oldest_dt
            oldest_str = dt_kst.strftime("%Y%m%d")
            if oldest_str > cutoff:
                targets.append({"symbol": sym, "name": name, "oldest_dt": oldest_str})
            # oldest_str <= cutoff → 이미 1년치 있음 → 스킵

    return targets


async def run_retry(max_days: int = 365):
    """데이터 부족 종목만 재수집."""
    targets = await get_retry_targets(max_days)
    if not targets:
        logger.info("재수집 대상 없음 — 모든 종목 1년치 충족")
        return

    # 진행률 로딩
    progress: dict = {}
    if os.path.exists(RETRY_PROGRESS_FILE):
        with open(RETRY_PROGRESS_FILE, "r", encoding="utf-8") as f:
            progress = json.load(f)
    completed = set(progress.get("completed", []))
    total_candles = progress.get("total_candles", 0)

    remaining = [t for t in targets if t["symbol"] not in completed]
    total = len(targets)
    done = len(completed)

    today = datetime.now().strftime("%Y%m%d")
    one_year_ago = (datetime.now() - timedelta(days=max_days)).strftime("%Y%m%d")
    start_time = time.time()

    logger.info("=== KIS 재수집 (--retry) ===")
    logger.info("  대상: %d종목 (DB 부족분)", total)
    logger.info("  완료: %d, 남은: %d", done, len(remaining))

    client = get_kis_client(is_mock=False)

    for i, target in enumerate(remaining):
        sym = target["symbol"]
        name = target["name"]
        oldest_dt = target["oldest_dt"]

        elapsed = time.time() - start_time
        if i > 0:
            avg_per = elapsed / i
            eta = _format_duration(avg_per * (len(remaining) - i))
        else:
            eta = "..."

        current = done + i + 1

        if oldest_dt is None:
            fill_start = today
            fill_days = max_days
            logger.info(
                "[%d/%d] %s %s — 데이터 없음, 전체 수집 (elapsed %s, ETA %s)",
                current, total, sym, name, _format_duration(elapsed), eta,
            )
        else:
            overlap_dt = datetime.strptime(oldest_dt, "%Y%m%d") + timedelta(days=2)
            fill_start = overlap_dt.strftime("%Y%m%d")
            fill_days = (overlap_dt - datetime.strptime(one_year_ago, "%Y%m%d")).days
            logger.info(
                "[%d/%d] %s %s — oldest=%s, %s~%s 보충 (%d일) (elapsed %s, ETA %s)",
                current, total, sym, name, oldest_dt,
                one_year_ago, fill_start, fill_days,
                _format_duration(elapsed), eta,
            )

        try:
            count = await collect_one_verified(client, sym, fill_start, fill_days)
            total_candles += count
            logger.info("  %s -> %s candles", sym, f"{count:,}")
        except Exception as e:
            logger.error("  %s -> failed: %s", sym, e)

        completed.add(sym)
        progress["completed"] = sorted(completed)
        progress["total"] = total
        progress["total_candles"] = total_candles
        progress["last_symbol"] = sym
        with open(RETRY_PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start_time
    logger.info("=== 재수집 완료 ===")
    logger.info("  총 %s건, 소요 %s", f"{total_candles:,}", _format_duration(elapsed))

    await client.close()


# ── fill-kiwoom 메인 ──────────────────────────────────────

async def run_fill_kiwoom(max_days: int = 365):
    """키움이 수집한 종목들의 부족 구간을 KIS로 보충."""
    targets = await get_kiwoom_fill_targets()
    if not targets:
        logger.error("키움 수집 종목 없음 (minute_claims.json 확인)")
        return

    # 진행률 로딩
    progress: dict = {}
    if os.path.exists(FILL_PROGRESS_FILE):
        with open(FILL_PROGRESS_FILE, "r", encoding="utf-8") as f:
            progress = json.load(f)
    completed = set(progress.get("completed", []))
    total_candles = progress.get("total_candles", 0)

    remaining = [t for t in targets if t["symbol"] not in completed]
    total = len(targets)
    done = len(completed)

    today = datetime.now().strftime("%Y%m%d")
    one_year_ago = (datetime.now() - timedelta(days=max_days)).strftime("%Y%m%d")
    start_time = time.time()

    logger.info("=== KIS 보충 수집 (--fill-kiwoom) ===")
    logger.info("  대상: %d종목 (키움 수집 중 stock_masters 존재)", total)
    logger.info("  완료: %d, 남은: %d", done, len(remaining))
    logger.info("  max_days=%d (1년 전: %s)", max_days, one_year_ago)

    client = get_kis_client(is_mock=False)

    for i, target in enumerate(remaining):
        sym = target["symbol"]
        name = target["name"]
        oldest_dt = target["oldest_dt"]

        elapsed = time.time() - start_time
        if i > 0:
            avg_per = elapsed / i
            eta = _format_duration(avg_per * (len(remaining) - i))
        else:
            eta = "..."

        current = done + i + 1

        if oldest_dt is None:
            # DB에 데이터 없음 → 오늘부터 1년치 전체 수집
            fill_start = today
            fill_days = max_days
            logger.info(
                "[%d/%d] %s %s — DB 데이터 없음, 전체 수집 (elapsed %s, ETA %s)",
                current, total, sym, name, _format_duration(elapsed), eta,
            )
        elif oldest_dt <= one_year_ago:
            # 이미 1년치 이상 있음 → 스킵
            logger.info(
                "[%d/%d] %s %s — oldest=%s, 이미 충분 (스킵)",
                current, total, sym, name, oldest_dt,
            )
            completed.add(sym)
            progress["completed"] = sorted(completed)
            progress["total"] = total
            progress["total_candles"] = total_candles
            with open(FILL_PROGRESS_FILE, "w", encoding="utf-8") as f:
                json.dump(progress, f, indent=2, ensure_ascii=False)
            continue
        else:
            # 키움 최초 수집일 + 2일 오버랩 → 그 시점부터 역순 수집
            overlap_dt = datetime.strptime(oldest_dt, "%Y%m%d") + timedelta(days=2)
            fill_start = overlap_dt.strftime("%Y%m%d")
            fill_days = (overlap_dt - datetime.strptime(one_year_ago, "%Y%m%d")).days
            logger.info(
                "[%d/%d] %s %s — oldest=%s, %s~%s 보충 (%d일) (elapsed %s, ETA %s)",
                current, total, sym, name, oldest_dt,
                one_year_ago, fill_start, fill_days,
                _format_duration(elapsed), eta,
            )

        try:
            count = await collect_one_verified(client, sym, fill_start, fill_days)
            total_candles += count
            logger.info("  %s -> %s candles", sym, f"{count:,}")
        except Exception as e:
            logger.error("  %s -> failed: %s", sym, e)

        completed.add(sym)
        progress["completed"] = sorted(completed)
        progress["total"] = total
        progress["total_candles"] = total_candles
        progress["last_symbol"] = sym
        with open(FILL_PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start_time
    logger.info("=== 보충 수집 완료 ===")
    logger.info("  총 %s건, 소요 %s", f"{total_candles:,}", _format_duration(elapsed))

    await client.close()


# ── 메인 ─────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="KIS API 전 종목 1분봉 수집")
    parser.add_argument("--max-days", type=int, default=365, help="최대 수집 일수 (기본 365)")
    parser.add_argument(
        "--fill-kiwoom", action="store_true",
        help="키움이 수집한 종목의 부족분을 KIS로 보충 (7.5개월→1년)",
    )
    parser.add_argument(
        "--retry", action="store_true",
        help="DB에 데이터가 부족한 종목만 재수집 (에러로 누락된 종목 복구)",
    )
    args = parser.parse_args()

    if args.retry:
        await run_retry(max_days=args.max_days)
        return

    if args.fill_kiwoom:
        await run_fill_kiwoom(max_days=args.max_days)
        return

    all_stocks = await get_all_symbols()  # 역순 (뒤에서 앞으로)
    if not all_stocks:
        logger.error("stock_masters empty. Run seed_stock_masters first.")
        return

    progress = load_progress()
    completed = set(progress.get("completed", []))
    total_candles = progress.get("total_candles", 0)

    remaining = [s for s in all_stocks if s["symbol"] not in completed]
    total = len(all_stocks)
    done = len(completed)
    skipped = 0

    today = datetime.now().strftime("%Y%m%d")
    start_time = time.time()

    logger.info("=== KIS 전 종목 1분봉 수집 시작 (역순) ===")
    logger.info("  전체: %d종목, 완료: %d, 남은: %d", total, done, len(remaining))
    logger.info("  max_days=%d", args.max_days)

    client = get_kis_client(is_mock=False)

    for i, stock in enumerate(remaining):
        sym = stock["symbol"]
        name = stock["name"]

        elapsed = time.time() - start_time
        processed = i - skipped
        if processed > 0:
            avg_per = elapsed / processed
            eta = _format_duration(avg_per * (len(remaining) - i))
        else:
            eta = "..."

        current = done + i + 1
        logger.info(
            "[%d/%d] %s %s (elapsed %s, ETA %s, skipped %d)",
            current, total, sym, name,
            _format_duration(elapsed), eta, skipped,
        )

        try:
            count = await collect_one_verified(client, sym, today, args.max_days)
            total_candles += count
            logger.info("  %s -> %s candles", sym, f"{count:,}")
        except Exception as e:
            logger.error("  %s -> failed: %s", sym, e)

        completed.add(sym)
        progress["completed"] = sorted(completed)
        progress["total"] = total
        progress["total_candles"] = total_candles
        progress["last_symbol"] = sym
        save_progress(progress)

    elapsed = time.time() - start_time
    logger.info("=== 수집 완료 ===")
    logger.info("  총 %s건, 소요 %s, 스킵 %d", f"{total_candles:,}", _format_duration(elapsed), skipped)

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
