"""뉴스 수집 — 기존 파이프라인 래핑.

당일 거래량 상위 종목에 대해 뉴스 수집 + LLM 감성 분석.
Gemini 우선, Anthropic 폴백.
"""

from __future__ import annotations

import logging

from sqlalchemy import text

from app.core.config import settings
from app.core.database import async_session
from app.scheduler.circuit_breaker import CircuitBreaker
from app.scheduler.collectors import LogCb, ProgressCb
from app.scheduler.schemas import CollectionResult

logger = logging.getLogger(__name__)


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
                  AND CAST(dt AS date) = to_date(:dt, 'YYYYMMDD')
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


def _detect_llm_provider() -> str:
    """현재 설정된 LLM 프로바이더를 확인."""
    if settings.GEMINI_API_KEY:
        return f"Gemini ({getattr(settings, 'GEMINI_MODEL', 'default')})"
    if settings.ANTHROPIC_API_KEY:
        return f"Anthropic ({getattr(settings, 'AGENT_MODEL', 'default')})"
    return "없음 (분석 스킵)"


async def collect_news(
    date: str,
    *,
    progress_cb: ProgressCb = None,
    log_cb: LogCb = None,
    cb: CircuitBreaker,
) -> CollectionResult:
    """당일 뉴스 수집 + 감성 분석.

    기존 app/news/scheduler.py의 collect_and_analyze()를 래핑.
    """
    from app.news.scheduler import collect_and_analyze

    logger.info("[뉴스] 수집 시작 (date=%s)", date)

    top_n = settings.DAILY_NEWS_TOP_N
    symbols = await _get_top_volume_symbols(date, limit=top_n)

    if not symbols:
        logger.info("[뉴스] 대상 종목 없음")
        if log_cb:
            await log_cb("대상 종목 없음 — 스킵")
        return CollectionResult(job="news")

    provider = _detect_llm_provider()
    if log_cb:
        await log_cb(
            f"거래량 상위 {len(symbols)}종목 뉴스 크롤링 + 감성 분석 (LLM: {provider})"
        )

    async with async_session() as session:
        try:
            stats = await cb.call(
                collect_and_analyze, session, symbols, days=1,
                log_cb=log_cb, progress_cb=progress_cb,
            )
        except Exception as e:
            logger.error("[뉴스] 수집 실패: %s", e)
            if log_cb:
                await log_cb(f"수집 실패: {str(e)[:100]}")
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
        await progress_cb(len(symbols), len(symbols), "done")

    if log_cb:
        await log_cb(
            f"완료: 크롤링 {collected}건, 감성 분석 {analyzed}건, "
            f"스코어 산출 {scored}건 (LLM: {provider})"
        )

    logger.info(
        "[뉴스] 완료: %d종목, collected=%d, analyzed=%d, scored=%d",
        len(symbols), collected, analyzed, scored,
    )
    return CollectionResult(
        job="news",
        total=len(symbols),
        completed=scored,
    )
