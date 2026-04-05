"""프로그램 매매 일별 수집 (장후 배치) — KIS API.

장 마감 후 전 종목의 당일 프로그램 매매 데이터를 수집한다.
장중 폴링(program_trading_collector.py)에서 빠진 종목을 보충하는 역할.

- 종목별 프로그램매매추이(일별): FHPPG04650201
- /uapi/domestic-stock/v1/quotations/program-trade-by-stock-daily
"""

from __future__ import annotations

from app.core.timezone import KST, now_kst

import asyncio
import logging
from datetime import date as date_type, datetime, timedelta, timezone

from sqlalchemy import text

from app.core.database import async_session
from app.scheduler.circuit_breaker import CircuitBreaker, CircuitBreakerOpen
from app.scheduler.collectors import LogCb, ProgressCb
from app.scheduler.schemas import CollectionResult
from app.trading.kis_client import get_kis_client

logger = logging.getLogger(__name__)

_cb = CircuitBreaker(name="program_trading_daily", failure_threshold=5, reset_timeout=300)


async def _get_symbols() -> list[str]:
    """stock_masters에서 KOSPI/KOSDAQ 전 종목."""
    async with async_session() as db:
        result = await db.execute(
            text("SELECT symbol FROM stock_masters WHERE market IN ('KOSPI', 'KOSDAQ') ORDER BY symbol"),
        )
        return [r[0] for r in result.fetchall()]


async def _get_existing_symbols(target_date: date_type) -> set[str]:
    """이미 수집된 종목 (빠진 데이터만 보충하기 위해)."""
    async with async_session() as db:
        result = await db.execute(
            text("""
                SELECT DISTINCT symbol FROM program_trading
                WHERE dt::date = :dt AND (pgm_buy_amount != 0 OR pgm_sell_amount != 0)
            """),
            {"dt": target_date},
        )
        return {r[0] for r in result.fetchall()}


async def collect_program_trading_daily(
    target_date: date_type | str | None = None,
    *,
    progress_cb: ProgressCb = None,
    log_cb: LogCb = None,
    cb: CircuitBreaker | None = None,
) -> CollectionResult:
    """장후 프로그램 매매 일일 수집 (빠진 종목 보충).

    Args:
        target_date: 수집 대상일 (None이면 오늘, str이면 YYYYMMDD)
        cb: 외부 CircuitBreaker (None이면 모듈 내부 _cb 사용)
    """
    breaker = cb or _cb
    if target_date is None:
        target_date = now_kst().date()
    # manual_runner가 문자열("20260324")로 전달
    if isinstance(target_date, str):
        date_str = target_date.replace("-", "")
        target_date = datetime.strptime(date_str[:8], "%Y%m%d").date()
    else:
        date_str = target_date.strftime("%Y%m%d")

    if log_cb:
        await log_cb(f"프로그램 매매 수집 시작: {target_date}")

    # 전 종목 로드
    all_symbols = await _get_symbols()
    if not all_symbols:
        if log_cb:
            await log_cb("종목 없음 — 수집 중단")
        return CollectionResult(job="program_trading")

    # 이미 수집된 종목 제외
    existing = await _get_existing_symbols(target_date)
    missing = [s for s in all_symbols if s not in existing]

    if log_cb:
        await log_cb(
            f"전체 {len(all_symbols)}종목, 기존 {len(existing)}종목, "
            f"보충 대상 {len(missing)}종목"
        )

    if not missing:
        if log_cb:
            await log_cb("보충 필요 없음 — 전체 수집 완료 상태")
        return CollectionResult(
            job="program_trading",
            total=len(all_symbols),
            completed=len(existing),
        )

    # KIS 클라이언트 + 토큰 warmup
    client = get_kis_client()

    for attempt in range(3):
        try:
            await client._get_token()
            break
        except Exception as e:
            if "403" in str(e) and attempt < 2:
                logger.info("[프로그램 매매] 토큰 1분 제한 — 65초 대기 (attempt %d)", attempt + 1)
                if log_cb:
                    await log_cb(f"KIS 토큰 1분 제한 — 65초 대기 (시도 {attempt + 1}/3)")
                for remaining in range(65, 0, -10):
                    await asyncio.sleep(min(10, remaining))
                    if log_cb and remaining > 10:
                        await log_cb(f"  토큰 대기 중... {remaining - 10}초 남음")
            else:
                logger.error("[프로그램 매매] 토큰 발급 실패: %s", e)
                if log_cb:
                    await log_cb(f"토큰 발급 실패: {e}")
                return CollectionResult(
                    job="program_trading",
                    total=len(missing),
                    error=f"토큰 발급 실패: {e}",
                )

    collected = 0
    failed = 0
    skipped = 0
    total = len(missing)

    dt = datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=KST)

    for i, symbol in enumerate(missing):
        try:
            data = await breaker.call(
                client.inquire_program_trading, symbol, date=date_str,
            )

            # 값이 전부 0이면 스킵 (데이터 없는 종목)
            if data["pgm_buy_amount"] == 0 and data["pgm_sell_amount"] == 0:
                skipped += 1
                continue

            # DB UPSERT
            async with async_session() as db:
                await db.execute(
                    text("""
                        INSERT INTO program_trading (symbol, dt, pgm_buy_amount, pgm_sell_amount,
                            pgm_net_amount, arbt_buy_amount, arbt_sell_amount,
                            nabt_buy_amount, nabt_sell_amount, collected_at)
                        VALUES (:symbol, :dt, :buy, :sell, :net, :arbt_buy, :arbt_sell,
                            :nabt_buy, :nabt_sell, NOW())
                        ON CONFLICT (symbol, dt) DO UPDATE
                        SET pgm_buy_amount = EXCLUDED.pgm_buy_amount,
                            pgm_sell_amount = EXCLUDED.pgm_sell_amount,
                            pgm_net_amount = EXCLUDED.pgm_net_amount,
                            collected_at = EXCLUDED.collected_at
                    """),
                    {
                        "symbol": symbol, "dt": dt,
                        "buy": data["pgm_buy_amount"],
                        "sell": data["pgm_sell_amount"],
                        "net": data["pgm_net_amount"],
                        "arbt_buy": data.get("arbt_buy_amount", 0),
                        "arbt_sell": data.get("arbt_sell_amount", 0),
                        "nabt_buy": data.get("nabt_buy_amount", 0),
                        "nabt_sell": data.get("nabt_sell_amount", 0),
                    },
                )
                await db.commit()
            collected += 1

        except CircuitBreakerOpen:
            remaining_count = total - i
            logger.warning(
                "[프로그램 매매] 서킷 OPEN — 나머지 %d종목 스킵", remaining_count
            )
            if log_cb:
                await log_cb(f"서킷 브레이커 OPEN — 나머지 {remaining_count}종목 스킵")
            return CollectionResult(
                job="program_trading",
                total=total,
                completed=collected,
                failed=failed,
                skipped=remaining_count,
                error="KIS 서킷 브레이커 OPEN",
            )

        except Exception as e:
            failed += 1
            if log_cb and failed % 5 == 0:
                await log_cb(f"  누적 실패 {failed}건 (최근: {symbol} — {str(e)[:60]})")

        # 진행률 — 10건마다 보고 (무음 구간 최대 ~1초)
        if progress_cb and (i + 1) % 10 == 0:
            await progress_cb(total, i + 1, symbol)

        # Rate limit (15req/s)
        await asyncio.sleep(0.08)

    # 최종 진행률 보고
    if progress_cb:
        await progress_cb(total, total, "done")
    if log_cb:
        await log_cb(
            f"프로그램 매매 수집 완료: {collected}종목 저장, "
            f"{failed}실패, {skipped}스킵 (데이터 없음)"
        )

    return CollectionResult(
        job="program_trading",
        total=total,
        completed=collected,
        failed=failed,
        skipped=skipped,
    )
