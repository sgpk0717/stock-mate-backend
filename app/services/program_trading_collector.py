"""KIS API 기반 프로그램 매매 데이터 수집기.

장 중(09:00~15:30) N분 간격으로 상위 종목의 프로그램 매매 현황을 폴링하여 DB에 저장.

Robustness 설계:
- 종목 목록: stock_masters에서 직접 로드 (WebSocket 의존 제거)
- DB: asyncpg 커넥션 풀 (매번 connect/close 반복 방지)
- 장 경계: 09:00 첫 수집 + 15:25 마지막 수집 보장
- 텔레그램: 연속 실패 경고 + 일일 요약 알림
- 토큰: 갱신 실패 시 지수 백오프 재시도
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timedelta, timezone

import asyncpg

from app.core.config import settings

logger = logging.getLogger(__name__)

_KST = timezone(timedelta(hours=9))
_running = False
_pool: asyncpg.Pool | None = None

# 일별 통계
_daily_stats = {
    "date": "",
    "success": 0,
    "fail": 0,
    "total_snapshots": 0,
}


def _now_kst() -> datetime:
    return datetime.now(_KST)


def _is_market_hours() -> bool:
    """KST 기준 장 시간(09:00~15:30), 평일."""
    now = _now_kst()
    hm = now.hour * 100 + now.minute
    return now.weekday() < 5 and 900 <= hm <= 1530


def _is_near_close() -> bool:
    """장 마감 직전(15:25~15:30) — 마지막 수집 보장."""
    now = _now_kst()
    hm = now.hour * 100 + now.minute
    return now.weekday() < 5 and 1525 <= hm <= 1530


def _seconds_until_market_open() -> float:
    """다음 장 시작(09:00)까지 남은 초."""
    now = _now_kst()
    # 오늘 09:00
    today_open = now.replace(hour=9, minute=0, second=0, microsecond=0)

    if now < today_open and now.weekday() < 5:
        return (today_open - now).total_seconds()

    # 다음 영업일 09:00
    days_ahead = 1
    if now.weekday() == 4:  # 금요일
        days_ahead = 3
    elif now.weekday() == 5:  # 토요일
        days_ahead = 2
    next_open = (now + timedelta(days=days_ahead)).replace(
        hour=9, minute=0, second=0, microsecond=0,
    )
    return (next_open - now).total_seconds()


async def _get_pool() -> asyncpg.Pool:
    """asyncpg 커넥션 풀 싱글턴."""
    global _pool
    if _pool is None or _pool._closed:
        dsn = (
            f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
            f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
        )
        _pool = await asyncpg.create_pool(dsn, min_size=2, max_size=5)
    return _pool


async def _load_symbols() -> list[str]:
    """stock_masters에서 상위 N개 종목 코드 로드."""
    pool = await _get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT symbol FROM stock_masters ORDER BY symbol LIMIT $1",
            settings.PGM_TRADING_SYMBOLS_LIMIT,
        )
    return [r[0] for r in rows]


async def _save_batch(rows: list[tuple]) -> int:
    """프로그램 매매 배치 저장. 반환: 저장 건수."""
    if not rows:
        return 0
    pool = await _get_pool()
    async with pool.acquire() as conn:
        await conn.executemany(
            """
            INSERT INTO program_trading (symbol, dt, pgm_buy_qty, pgm_sell_qty,
                pgm_net_qty, pgm_buy_amount, pgm_sell_amount, pgm_net_amount, collected_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (symbol, dt) DO UPDATE
            SET pgm_buy_qty = EXCLUDED.pgm_buy_qty,
                pgm_sell_qty = EXCLUDED.pgm_sell_qty,
                pgm_net_qty = EXCLUDED.pgm_net_qty,
                pgm_buy_amount = EXCLUDED.pgm_buy_amount,
                pgm_sell_amount = EXCLUDED.pgm_sell_amount,
                pgm_net_amount = EXCLUDED.pgm_net_amount,
                collected_at = EXCLUDED.collected_at
            """,
            rows,
        )
    return len(rows)


async def _send_telegram(text: str) -> None:
    """텔레그램 알림 (실패해도 무시)."""
    try:
        from app.telegram.bot import send_message
        await send_message(text, category="system")
    except Exception:
        pass


async def _collect_round(
    client: object,
    symbols: list[str],
    dt: datetime,
) -> tuple[int, int]:
    """한 라운드 수집. 반환: (성공, 실패)."""
    collected_at = _now_kst()
    batch: list[tuple] = []
    fail_count = 0

    for symbol in symbols:
        try:
            data = await client.inquire_program_trading(symbol)  # type: ignore[attr-defined]
            batch.append((
                symbol, dt,
                data.get("pgm_buy_qty", 0),
                data.get("pgm_sell_qty", 0),
                data.get("pgm_net_qty", 0),
                data.get("pgm_buy_amount", 0),
                data.get("pgm_sell_amount", 0),
                data.get("pgm_net_amount", 0),
                collected_at,
            ))
        except Exception as e:
            fail_count += 1
            if fail_count <= 3:  # 처음 3건만 로깅
                logger.warning("프로그램 매매 조회 실패 (%s): %s", symbol, e)

        # KIS rate limit (15 req/s)
        await asyncio.sleep(0.08)

    saved = await _save_batch(batch)
    return saved, fail_count


async def _daily_summary() -> None:
    """장 마감 후 일일 수집 요약 알림."""
    today = _now_kst().strftime("%Y-%m-%d")
    stats = _daily_stats

    if stats["date"] != today:
        return

    pool = await _get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT COUNT(*) as cnt FROM program_trading WHERE dt::date = $1",
            date.fromisoformat(today),
        )
    db_count = row["cnt"] if row else 0

    msg = (
        f"[프로그램매매 일일요약] {today}\n"
        f"수집 라운드: {stats['success']}회 성공 / {stats['fail']}회 실패\n"
        f"DB 저장: {db_count:,}건"
    )

    if db_count == 0:
        msg += "\n\n수집 데이터 0건 — 확인 필요!"

    logger.info(msg)
    await _send_telegram(msg)


async def start_collector() -> None:
    """프로그램 매매 수집기 메인 루프."""
    global _running, _daily_stats
    _running = True
    consecutive_failures = 0

    interval_sec = settings.PGM_TRADING_COLLECT_INTERVAL_MINUTES * 60
    logger.info(
        "프로그램 매매 수집기 시작 (간격: %d분, 종목: 상위 %d개)",
        settings.PGM_TRADING_COLLECT_INTERVAL_MINUTES,
        settings.PGM_TRADING_SYMBOLS_LIMIT,
    )

    # KIS 클라이언트 초기화
    from app.trading.kis_client import get_kis_client
    client = get_kis_client(is_mock=False)

    # 종목 목록 로드 (1시간마다 갱신)
    symbols: list[str] = []
    symbols_loaded_at = 0.0
    sent_close_summary = False

    while _running:
        now = _now_kst()
        today_str = now.strftime("%Y-%m-%d")

        # 일별 통계 리셋
        if _daily_stats["date"] != today_str:
            _daily_stats = {"date": today_str, "success": 0, "fail": 0, "total_snapshots": 0}
            sent_close_summary = False

        # 장외 시간 대기
        if not _is_market_hours():
            # 장 마감 직후: 일일 요약 알림 (1회)
            if not sent_close_summary and now.hour >= 15 and now.minute >= 31 and _daily_stats["success"] > 0:
                await _daily_summary()
                sent_close_summary = True

            wait = min(_seconds_until_market_open(), 300)  # 최대 5분 단위 sleep
            logger.debug("프로그램 매매 수집기: 장외 대기 (%.0f초)", wait)
            await asyncio.sleep(wait)
            continue

        # 종목 목록 갱신 (1시간마다)
        import time
        if not symbols or (time.time() - symbols_loaded_at) > 3600:
            try:
                symbols = await _load_symbols()
                symbols_loaded_at = time.time()
                logger.info("프로그램 매매 종목 로드: %d개", len(symbols))
            except Exception as e:
                logger.error("프로그램 매매 종목 로드 실패: %s", e)
                await asyncio.sleep(60)
                continue

        if not symbols:
            logger.warning("프로그램 매매 수집 대상 종목 없음")
            await asyncio.sleep(60)
            continue

        # 스냅샷 시점 (분 truncate)
        dt = now.replace(second=0, microsecond=0)

        # 수집 실행
        try:
            success, fail = await _collect_round(client, symbols, dt)
            _daily_stats["total_snapshots"] += success

            if success > 0:
                consecutive_failures = 0
                _daily_stats["success"] += 1
                logger.info(
                    "프로그램 매매 수집: %d/%d 성공 (실패 %d)",
                    success, len(symbols), fail,
                )
            else:
                consecutive_failures += 1
                _daily_stats["fail"] += 1

        except Exception as e:
            consecutive_failures += 1
            _daily_stats["fail"] += 1
            logger.error("프로그램 매매 수집 라운드 실패: %s", e)

        # 연속 실패 시 지수 백오프 + 텔레그램 경고
        if consecutive_failures >= 3:
            backoff = min(60 * (2 ** (consecutive_failures - 3)), 600)
            logger.warning(
                "프로그램 매매 %d연속 실패 — %ds 대기", consecutive_failures, backoff,
            )
            if consecutive_failures == 3:
                await _send_telegram(
                    f"[프로그램매매 경고] {consecutive_failures}회 연속 수집 실패\n"
                    f"KIS API 또는 네트워크 확인 필요",
                )
            await asyncio.sleep(backoff)
            continue

        # 장 마감 직전이면 짧은 간격으로 재수집
        if _is_near_close():
            await asyncio.sleep(60)  # 1분 간격
        else:
            await asyncio.sleep(interval_sec)


def stop_collector() -> None:
    """수집기 중지."""
    global _running
    _running = False
    logger.info("프로그램 매매 수집기 중지")
