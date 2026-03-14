"""뉴스 수집 — 기존 파이프라인 래핑.

당일 거래량 상위 종목에 대해 뉴스 수집 + Claude 감성 분석.
"""

from __future__ import annotations

import logging
from typing import Callable

from sqlalchemy import text

from app.core.config import settings
from app.core.database import async_session
from app.scheduler.circuit_breaker import CircuitBreaker
from app.scheduler.schemas import CollectionResult

logger = logging.getLogger(__name__)

ProgressCb = Callable[[int, int, str], object] | None


async def _get_top_volume_symbols(date: str, limit: int) -> list[str]:
    """당일 거래량 상위 종목 조회.

    일봉이 아직 없으면 stock_masters 전체에서 상위 N개 반환.
    """
    async with async_session() as db:
        # 당일 일봉 기준 거래량 정렬
        result = await db.execute(
            text("""
                SELECT symbol FROM stock_candles
                WHERE interval = '1d'
                  AND dt::date = :dt::date
                  AND volume > 0
                ORDER BY volume DESC
                LIMIT :lim
            """),
            {"dt": date, "lim": limit},
        )
        symbols = [r[0] for r in result.fetchall()]

        if symbols:
            return symbols

        # fallback: stock_masters에서 심볼 목록
        result = await db.execute(
            text("SELECT symbol FROM stock_masters ORDER BY symbol LIMIT :lim"),
            {"lim": limit},
        )
        return [r[0] for r in result.fetchall()]


async def collect_news(
    date: str,
    *,
    progress_cb: ProgressCb = None,
    cb: CircuitBreaker,
) -> CollectionResult:
    """당일 뉴스 수집 + 감성 분석.

    기존 app/news/scheduler.py의 collect_and_analyze()를 래핑.
    Claude API 서킷 브레이커로 보호.
    """
    from app.news.scheduler import collect_and_analyze

    logger.info("[뉴스] 수집 시작 (date=%s)", date)

    top_n = settings.DAILY_NEWS_TOP_N
    symbols = await _get_top_volume_symbols(date, limit=top_n)

    if not symbols:
        logger.info("[뉴스] 대상 종목 없음")
        return CollectionResult(job="news")

    async with async_session() as session:
        try:
            stats = await cb.call(
                collect_and_analyze, session, symbols, days=1,
            )
        except Exception as e:
            logger.error("[뉴스] 수집 실패: %s", e)
            return CollectionResult(
                job="news",
                total=len(symbols),
                failed=len(symbols),
                error=str(e)[:500],
            )

    collected = stats.get("collected", 0)
    analyzed = stats.get("analyzed", 0)
    scored = stats.get("scored", 0)

    if progress_cb:
        await progress_cb(len(symbols), len(symbols), symbols[-1])

    logger.info(
        "[뉴스] 완료: %d종목, collected=%d, analyzed=%d, scored=%d",
        len(symbols), collected, analyzed, scored,
    )
    return CollectionResult(
        job="news",
        total=len(symbols),
        completed=scored,
    )
