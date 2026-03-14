"""투자자별 매매동향 일별 수집 — KIS API.

당일 전 종목의 외국인/기관/개인 매수·매도·순매수 데이터를 수집한다.
- 투자자매매동향(일별): FHPTJ04160001

KIS Open API를 통해 데이터를 수집한다.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date as date_type, datetime, timedelta, timezone
from typing import Callable

from sqlalchemy import text

from app.core.database import async_session
from app.scheduler.circuit_breaker import CircuitBreaker, CircuitBreakerOpen
from app.scheduler.schemas import CollectionResult
from app.trading.kis_client import get_kis_client

logger = logging.getLogger(__name__)

ProgressCb = Callable[[int, int, str], object] | None


async def _get_all_symbols() -> list[str]:
    """stock_masters에서 전 종목 코드 조회."""
    async with async_session() as db:
        result = await db.execute(
            text("SELECT symbol FROM stock_masters ORDER BY symbol"),
        )
        return [r[0] for r in result.fetchall()]


async def _upsert_investor(
    symbol: str,
    date_str: str,
    rows: list[dict],
) -> int:
    """investor_trading에 UPSERT."""
    upserted = 0

    async with async_session() as db:
        for row in rows:
            dt_str = row["dt"]
            if dt_str != date_str:
                continue
            await db.execute(
                text("""
                    INSERT INTO investor_trading
                        (symbol, dt, foreign_net, inst_net, retail_net,
                         foreign_buy_vol, foreign_sell_vol,
                         inst_buy_vol, inst_sell_vol,
                         retail_buy_vol, retail_sell_vol, collected_at)
                    VALUES (:symbol, :dt, :foreign_net, :inst_net, :retail_net,
                            :foreign_buy_vol, :foreign_sell_vol,
                            :inst_buy_vol, :inst_sell_vol,
                            :retail_buy_vol, :retail_sell_vol, :collected_at)
                    ON CONFLICT (symbol, dt) DO UPDATE
                    SET foreign_net = EXCLUDED.foreign_net,
                        inst_net = EXCLUDED.inst_net,
                        retail_net = EXCLUDED.retail_net,
                        foreign_buy_vol = EXCLUDED.foreign_buy_vol,
                        foreign_sell_vol = EXCLUDED.foreign_sell_vol,
                        inst_buy_vol = EXCLUDED.inst_buy_vol,
                        inst_sell_vol = EXCLUDED.inst_sell_vol,
                        retail_buy_vol = EXCLUDED.retail_buy_vol,
                        retail_sell_vol = EXCLUDED.retail_sell_vol,
                        collected_at = EXCLUDED.collected_at
                """),
                {
                    "symbol": symbol,
                    "dt": date_type(int(dt_str[:4]), int(dt_str[4:6]), int(dt_str[6:8])),
                    "foreign_net": row["frgn_net"],
                    "inst_net": row["orgn_net"],
                    "retail_net": row["prsn_net"],
                    "foreign_buy_vol": row["frgn_buy_vol"],
                    "foreign_sell_vol": row["frgn_sell_vol"],
                    "inst_buy_vol": row["orgn_buy_vol"],
                    "inst_sell_vol": row["orgn_sell_vol"],
                    "retail_buy_vol": row["prsn_buy_vol"],
                    "retail_sell_vol": row["prsn_sell_vol"],
                    "collected_at": datetime.now(timezone(timedelta(hours=9))),
                },
            )
            upserted += 1

        await db.commit()

    return upserted


async def collect_investor(
    date: str,
    *,
    progress_cb: ProgressCb = None,
    cb: CircuitBreaker,
) -> CollectionResult:
    """전 종목 당일 투자자별 매매동향 수집.

    Args:
        date: YYYYMMDD (당일).
        progress_cb: 진행률 콜백.
        cb: KIS API 서킷 브레이커.
    """
    logger.info("[투자자별수급] 수집 시작 (date=%s)", date)

    symbols = await _get_all_symbols()
    client = get_kis_client(is_mock=False)

    # 토큰 warmup: 다른 모듈이 먼저 토큰을 발급받아
    # 1분 쿼터를 소진했을 수 있으므로, 403 시 65초 대기 후 재시도
    for attempt in range(3):
        try:
            await client._get_token()
            break
        except Exception as e:
            if "403" in str(e) and attempt < 2:
                logger.info("[투자자별수급] 토큰 1분 제한 — 65초 대기 (attempt %d)", attempt + 1)
                await asyncio.sleep(65)
            else:
                logger.error("[투자자별수급] 토큰 발급 실패: %s", e)
                return CollectionResult(
                    job="investor",
                    total=len(symbols),
                    error=f"토큰 발급 실패: {e}",
                )

    completed = 0
    failed = 0

    for i, sym in enumerate(symbols):
        try:
            rows = await cb.call(
                client.inquire_daily_investor, sym, date,
            )
            await asyncio.sleep(0.12)

            await _upsert_investor(sym, date, rows)
            completed += 1

        except CircuitBreakerOpen:
            logger.warning(
                "[투자자별수급] 서킷 OPEN — 나머지 %d종목 스킵",
                len(symbols) - i,
            )
            return CollectionResult(
                job="investor",
                total=len(symbols),
                completed=completed,
                failed=failed,
                skipped=len(symbols) - i,
                error="KIS 서킷 브레이커 OPEN",
            )
        except Exception as e:
            failed += 1
            logger.warning("[투자자별수급] %s 실패: %s", sym, e)

        if progress_cb and (i + 1) % 50 == 0:
            await progress_cb(len(symbols), i + 1, sym)

    logger.info("[투자자별수급] 완료: %d종목 성공, %d 실패", completed, failed)
    return CollectionResult(
        job="investor",
        total=len(symbols),
        completed=completed,
        failed=failed,
    )
