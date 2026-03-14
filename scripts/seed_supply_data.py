"""수급 데이터 통합 벌크 시딩 — KIS API 기반.

공매도 + 신용잔고 + 투자자별 수급을 하나의 스크립트로 수집한다.
종목별 수집 후 DB 검증 → 실패 시 최대 3회 시도 → 그래도 실패 시 스킵.

Usage:
    docker compose exec app python -m scripts.seed_supply_data
    docker compose exec app python -m scripts.seed_supply_data --start 2024-01-01 --end 2025-12-31
    docker compose exec app python -m scripts.seed_supply_data --symbols 005930,000660
    docker compose exec app python -m scripts.seed_supply_data --only investor
    docker compose exec app python -m scripts.seed_supply_data --only short
    docker compose exec app python -m scripts.seed_supply_data --only credit

주의: 앱이 실행 중이면 KIS 토큰 경쟁 발생 가능.
      `docker compose stop app` 후 실행 권장.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import date, timedelta

import asyncpg
import httpx

from app.core.config import settings
from app.trading.kis_client import KISClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("seed_supply_data")

MAX_ATTEMPTS = 3  # 총 3회 (초회 + 2회 재시도)


def _dsn() -> str:
    s = settings
    return (
        f"postgresql://{s.POSTGRES_USER}:{s.POSTGRES_PASSWORD}"
        f"@{s.POSTGRES_HOST}:{s.POSTGRES_PORT}/{s.POSTGRES_DB}"
    )


async def _ensure_token(kis: KISClient, max_retries: int = 3) -> None:
    """KIS 토큰 발급 (1분당 1회 제한 대응)."""
    for attempt in range(max_retries):
        try:
            await kis._get_token()
            logger.info("KIS 토큰 발급 성공")
            return
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 403 and attempt < max_retries - 1:
                logger.info(
                    "토큰 1분 제한 (EGW00133) — 65초 대기 (attempt %d/%d)",
                    attempt + 1, max_retries,
                )
                await asyncio.sleep(65)
            else:
                raise
        except Exception:
            if attempt < max_retries - 1:
                logger.info("토큰 발급 실패 — 10초 후 재시도 (attempt %d/%d)", attempt + 1, max_retries)
                await asyncio.sleep(10)
            else:
                raise


async def _get_universe(conn: asyncpg.Connection, symbols: list[str] | None) -> list[str]:
    if symbols:
        return symbols
    rows = await conn.fetch(
        "SELECT symbol FROM stock_masters WHERE market IN ('KOSPI', 'KOSDAQ') ORDER BY symbol"
    )
    return [r["symbol"] for r in rows]


async def _ensure_tables(conn: asyncpg.Connection) -> None:
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS margin_short_daily (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            dt DATE NOT NULL,
            margin_balance BIGINT DEFAULT 0,
            margin_rate FLOAT DEFAULT 0,
            short_volume BIGINT DEFAULT 0,
            short_balance BIGINT DEFAULT 0,
            short_balance_rate FLOAT DEFAULT 0,
            CONSTRAINT uq_margin_short UNIQUE (symbol, dt)
        )
    """)
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_margin_short_lookup ON margin_short_daily (symbol, dt)"
    )
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS investor_trading (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            dt DATE NOT NULL,
            foreign_net BIGINT DEFAULT 0,
            inst_net BIGINT DEFAULT 0,
            retail_net BIGINT DEFAULT 0,
            foreign_buy_vol BIGINT DEFAULT 0,
            foreign_sell_vol BIGINT DEFAULT 0,
            inst_buy_vol BIGINT DEFAULT 0,
            inst_sell_vol BIGINT DEFAULT 0,
            retail_buy_vol BIGINT DEFAULT 0,
            retail_sell_vol BIGINT DEFAULT 0,
            CONSTRAINT uq_investor_trading UNIQUE (symbol, dt)
        )
    """)
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_investor_trading_lookup ON investor_trading (symbol, dt)"
    )


# ── DB 검증 ──────────────────────────────────────────


async def _verify_short(conn: asyncpg.Connection, symbol: str, start_date: date, end_date: date) -> int:
    """공매도 데이터 건수 확인 (행 존재 여부)."""
    row = await conn.fetchrow(
        "SELECT COUNT(*) AS cnt FROM margin_short_daily WHERE symbol=$1 AND dt BETWEEN $2 AND $3",
        symbol, start_date, end_date,
    )
    return row["cnt"]


async def _verify_credit(conn: asyncpg.Connection, symbol: str, start_date: date, end_date: date) -> int:
    """신용잔고 데이터 건수 확인 (margin_balance > 0인 행)."""
    row = await conn.fetchrow(
        "SELECT COUNT(*) AS cnt FROM margin_short_daily WHERE symbol=$1 AND dt BETWEEN $2 AND $3 AND margin_balance > 0",
        symbol, start_date, end_date,
    )
    return row["cnt"]


async def _verify_investor(conn: asyncpg.Connection, symbol: str, start_date: date, end_date: date) -> int:
    """투자자 수급 데이터 건수 확인 (전 기간)."""
    row = await conn.fetchrow(
        "SELECT COUNT(*) AS cnt FROM investor_trading WHERE symbol=$1 AND dt BETWEEN $2 AND $3",
        symbol, start_date, end_date,
    )
    return row["cnt"]


# ── 수집 함수 ────────────────────────────────────────


async def _collect_short(
    conn: asyncpg.Connection,
    kis: KISClient,
    symbol: str,
    start_date: date,
    end_date: date,
    batch_days: int = 90,
) -> int:
    """한 종목 공매도 수집 (전 기간)."""
    total = 0
    current = start_date
    while current <= end_date:
        chunk_end = min(current + timedelta(days=batch_days - 1), end_date)
        start_str = current.strftime("%Y%m%d")
        end_str = chunk_end.strftime("%Y%m%d")

        try:
            rows = await kis.inquire_daily_short_sale(symbol, start_str, end_str)
        except Exception as e:
            logger.warning("  %s short API err: %s", symbol, e)
            current = chunk_end + timedelta(days=1)
            await asyncio.sleep(0.12)
            continue

        if rows:
            batch = []
            for row in rows:
                dt_str = row["dt"]
                try:
                    dt_val = date(int(dt_str[:4]), int(dt_str[4:6]), int(dt_str[6:8]))
                except (ValueError, IndexError):
                    continue
                batch.append((
                    symbol, dt_val, 0, 0.0,
                    row["short_volume"], 0, row["short_volume_ratio"],
                ))
            if batch:
                await conn.executemany(
                    """
                    INSERT INTO margin_short_daily (symbol, dt, margin_balance, margin_rate,
                        short_volume, short_balance, short_balance_rate)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (symbol, dt) DO UPDATE
                    SET short_volume = EXCLUDED.short_volume,
                        short_balance_rate = EXCLUDED.short_balance_rate
                    """,
                    batch,
                )
                total += len(batch)

        current = chunk_end + timedelta(days=1)
        await asyncio.sleep(0.12)

    return total


async def _collect_credit(
    conn: asyncpg.Connection,
    kis: KISClient,
    symbol: str,
    start_date: date,
    end_date: date,
) -> int:
    """한 종목 신용잔고 수집 (전 기간).

    API가 기준일로부터 과거 ~30거래일 데이터를 반환하므로,
    end_date부터 30일씩 뒤로 이동하며 반복 호출한다.
    """
    total = 0
    cursor = end_date
    seen_dates: set[str] = set()

    while cursor >= start_date:
        cursor_str = cursor.strftime("%Y%m%d")
        try:
            rows = await kis.inquire_daily_credit_balance(symbol, cursor_str)
        except Exception as e:
            logger.warning("  %s credit API err: %s", symbol, e)
            cursor -= timedelta(days=30)
            await asyncio.sleep(0.12)
            continue

        if not rows:
            break

        batch = []
        earliest_dt: date | None = None
        for row in rows:
            dt_str = row["dt"]
            if dt_str in seen_dates:
                continue
            seen_dates.add(dt_str)
            try:
                dt_val = date(int(dt_str[:4]), int(dt_str[4:6]), int(dt_str[6:8]))
            except (ValueError, IndexError):
                continue
            if dt_val < start_date:
                continue
            if earliest_dt is None or dt_val < earliest_dt:
                earliest_dt = dt_val
            batch.append((symbol, dt_val, row["margin_balance"], row["margin_rate"]))

        if batch:
            await conn.executemany(
                """
                INSERT INTO margin_short_daily (symbol, dt, margin_balance, margin_rate,
                    short_volume, short_balance, short_balance_rate)
                VALUES ($1, $2, $3, $4, 0, 0, 0)
                ON CONFLICT (symbol, dt) DO UPDATE
                SET margin_balance = EXCLUDED.margin_balance,
                    margin_rate = EXCLUDED.margin_rate
                """,
                batch,
            )
            total += len(batch)

        if not batch:
            break

        if earliest_dt:
            cursor = earliest_dt - timedelta(days=1)
        else:
            break

        await asyncio.sleep(0.12)

    return total


async def _collect_investor(
    conn: asyncpg.Connection,
    kis: KISClient,
    symbol: str,
    start_date: date,
    end_date: date,
) -> int:
    """한 종목 투자자별 매매동향 수집 (전 기간).

    API가 기준일로부터 과거 ~30거래일 데이터를 반환하므로,
    end_date부터 30일씩 뒤로 이동하며 반복 호출한다.
    """
    total = 0
    cursor = end_date
    seen_dates: set[str] = set()  # 중복 방지

    while cursor >= start_date:
        cursor_str = cursor.strftime("%Y%m%d")
        try:
            rows = await kis.inquire_daily_investor(symbol, cursor_str)
        except Exception as e:
            logger.warning("  %s investor API err: %s", symbol, e)
            cursor -= timedelta(days=30)
            await asyncio.sleep(0.12)
            continue

        if not rows:
            break  # 더 이상 데이터 없음

        batch = []
        earliest_dt: date | None = None
        for row in rows:
            dt_str = row["dt"]
            if dt_str in seen_dates:
                continue
            seen_dates.add(dt_str)
            try:
                dt_val = date(int(dt_str[:4]), int(dt_str[4:6]), int(dt_str[6:8]))
            except (ValueError, IndexError):
                continue
            if dt_val < start_date:
                continue
            if earliest_dt is None or dt_val < earliest_dt:
                earliest_dt = dt_val
            batch.append((
                symbol, dt_val,
                row["frgn_net"], row["orgn_net"], row["prsn_net"],
                row["frgn_buy_vol"], row["frgn_sell_vol"],
                row["orgn_buy_vol"], row["orgn_sell_vol"],
                row["prsn_buy_vol"], row["prsn_sell_vol"],
            ))

        if batch:
            await conn.executemany(
                """
                INSERT INTO investor_trading
                    (symbol, dt, foreign_net, inst_net, retail_net,
                     foreign_buy_vol, foreign_sell_vol,
                     inst_buy_vol, inst_sell_vol,
                     retail_buy_vol, retail_sell_vol)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                ON CONFLICT (symbol, dt) DO UPDATE
                SET foreign_net = EXCLUDED.foreign_net,
                    inst_net = EXCLUDED.inst_net,
                    retail_net = EXCLUDED.retail_net,
                    foreign_buy_vol = EXCLUDED.foreign_buy_vol,
                    foreign_sell_vol = EXCLUDED.foreign_sell_vol,
                    inst_buy_vol = EXCLUDED.inst_buy_vol,
                    inst_sell_vol = EXCLUDED.inst_sell_vol,
                    retail_buy_vol = EXCLUDED.retail_buy_vol,
                    retail_sell_vol = EXCLUDED.retail_sell_vol
                """,
                batch,
            )
            total += len(batch)

        # 새 데이터가 없으면 (이미 본 데이터만 반환) 종료
        if not batch:
            break

        # 가장 오래된 날짜 전날로 커서 이동
        if earliest_dt:
            cursor = earliest_dt - timedelta(days=1)
        else:
            break

        await asyncio.sleep(0.12)

    return total


# ── 메인 ────────────────────────────────────────────


async def seed(
    start_date: date,
    end_date: date,
    symbols: list[str] | None = None,
    only: set[str] | None = None,
    batch_days: int = 90,
) -> None:
    """KIS API 기반 수급 데이터 통합 시딩 — 종목별 검증 + 3회 시도."""
    do_short = only is None or "short" in only
    do_credit = only is None or "credit" in only
    do_investor = only is None or "investor" in only

    conn = await asyncpg.connect(_dsn())
    await _ensure_tables(conn)

    universe = await _get_universe(conn, symbols)
    logger.info("Universe: %d symbols, period: %s ~ %s, only=%s",
                len(universe), start_date, end_date, ",".join(only) if only else "all")

    kis = KISClient(is_mock=False)
    await _ensure_token(kis)

    # 통계
    stats = {
        "short_ok": 0, "short_skip": 0, "short_rows": 0,
        "credit_ok": 0, "credit_skip": 0, "credit_rows": 0,
        "investor_ok": 0, "investor_skip": 0, "investor_rows": 0,
    }
    skipped_symbols: list[tuple[str, str]] = []  # (symbol, reason)

    for i, symbol in enumerate(universe):
        logger.info("[%d/%d] %s", i + 1, len(universe), symbol)

        # ── 1) 공매도 ──
        if do_short:
            ok = False
            for attempt in range(1, MAX_ATTEMPTS + 1):
                cnt = await _collect_short(conn, kis, symbol, start_date, end_date, batch_days)
                db_cnt = await _verify_short(conn, symbol, start_date, end_date)
                if db_cnt > 0:
                    stats["short_ok"] += 1
                    stats["short_rows"] += db_cnt
                    ok = True
                    break
                if attempt < MAX_ATTEMPTS:
                    logger.info("  %s 공매도 검증 실패 (DB %d건) — 재시도 %d/%d",
                                symbol, db_cnt, attempt + 1, MAX_ATTEMPTS)
                    await asyncio.sleep(1)
            if not ok:
                stats["short_skip"] += 1
                skipped_symbols.append((symbol, "short"))
                logger.warning("  %s 공매도 3회 실패 — 데이터 없음 판단, 스킵", symbol)

        # ── 2) 신용잔고 ──
        if do_credit:
            ok = False
            for attempt in range(1, MAX_ATTEMPTS + 1):
                cnt = await _collect_credit(conn, kis, symbol, start_date, end_date)
                db_cnt = await _verify_credit(conn, symbol, start_date, end_date)
                if db_cnt > 0:
                    stats["credit_ok"] += 1
                    stats["credit_rows"] += db_cnt
                    ok = True
                    break
                if attempt < MAX_ATTEMPTS:
                    logger.info("  %s 신용잔고 검증 실패 (DB %d건) — 재시도 %d/%d",
                                symbol, db_cnt, attempt + 1, MAX_ATTEMPTS)
                    await asyncio.sleep(1)
            if not ok:
                stats["credit_skip"] += 1
                skipped_symbols.append((symbol, "credit"))
                logger.warning("  %s 신용잔고 3회 실패 — 데이터 없음 판단, 스킵", symbol)

        # ── 3) 투자자별 수급 ──
        if do_investor:
            ok = False
            for attempt in range(1, MAX_ATTEMPTS + 1):
                cnt = await _collect_investor(conn, kis, symbol, start_date, end_date)
                db_cnt = await _verify_investor(conn, symbol, start_date, end_date)
                if db_cnt > 0:
                    stats["investor_ok"] += 1
                    stats["investor_rows"] += db_cnt
                    ok = True
                    break
                if attempt < MAX_ATTEMPTS:
                    logger.info("  %s 투자자 검증 실패 (DB %d건) — 재시도 %d/%d",
                                symbol, db_cnt, attempt + 1, MAX_ATTEMPTS)
                    await asyncio.sleep(1)
            if not ok:
                stats["investor_skip"] += 1
                skipped_symbols.append((symbol, "investor"))
                logger.warning("  %s 투자자 3회 실패 — 데이터 없음 판단, 스킵", symbol)

        # 진행률 로그
        if (i + 1) % 100 == 0:
            logger.info(
                "  === 진행 %d/%d — 공매도: %d건(%d스킵) 신용: %d건(%d스킵) 투자자: %d건(%d스킵)",
                i + 1, len(universe),
                stats["short_rows"], stats["short_skip"],
                stats["credit_rows"], stats["credit_skip"],
                stats["investor_rows"], stats["investor_skip"],
            )

    await kis.close()
    await conn.close()

    # ── 최종 리포트 ──
    logger.info("=" * 60)
    logger.info("수집 완료 리포트")
    logger.info("=" * 60)
    if do_short:
        logger.info("  공매도:   성공 %d종목 (%d건), 스킵 %d종목",
                     stats["short_ok"], stats["short_rows"], stats["short_skip"])
    if do_credit:
        logger.info("  신용잔고: 성공 %d종목 (%d건), 스킵 %d종목",
                     stats["credit_ok"], stats["credit_rows"], stats["credit_skip"])
    if do_investor:
        logger.info("  투자자:   성공 %d종목 (%d건), 스킵 %d종목",
                     stats["investor_ok"], stats["investor_rows"], stats["investor_skip"])

    if skipped_symbols:
        logger.info("스킵된 종목 (%d건):", len(skipped_symbols))
        for sym, reason in skipped_symbols[:50]:
            logger.info("  %s (%s)", sym, reason)
        if len(skipped_symbols) > 50:
            logger.info("  ... 외 %d건", len(skipped_symbols) - 50)

    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="수급 데이터 통합 시딩 (KIS API)")
    parser.add_argument(
        "--start", type=str, default="2020-01-01",
        help="시작일 (YYYY-MM-DD, 기본 2020-01-01)",
    )
    parser.add_argument(
        "--end", type=str, default="",
        help="종료일 (YYYY-MM-DD, 기본 오늘)",
    )
    parser.add_argument(
        "--symbols", type=str, default="",
        help="종목코드 (쉼표 구분, 기본 전체 유니버스)",
    )
    parser.add_argument(
        "--only", type=str, default="",
        help="수집 대상 (콤마 구분: short,credit,investor). 미지정 시 전체",
    )
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else date.today()
    sym_list = [s.strip() for s in args.symbols.split(",") if s.strip()] or None

    only_set: set[str] | None = None
    if args.only:
        only_set = {s.strip() for s in args.only.split(",") if s.strip()}
        valid = {"short", "credit", "investor"}
        invalid = only_set - valid
        if invalid:
            parser.error(f"--only 유효값: {valid}, 잘못된 값: {invalid}")

    asyncio.run(seed(start, end, symbols=sym_list, only=only_set))
