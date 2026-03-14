"""뉴스 수집 + 감성 분석 스케줄러.

장마감 후 (18:00 KST) 일괄 수집 → 분석 → 스코어 산출 파이프라인.
수동 트리거도 지원.
"""

from __future__ import annotations

import logging
import uuid
from datetime import date, datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.stock_master import get_stock_name
from app.news.analyzer import analyze_batch
from app.news.collectors.bigkinds import collect_stock_news as bigkinds_collect
from app.news.collectors.dart import collect_disclosures
from app.news.collectors.naver import RawArticle
from app.news.collectors.naver import collect_stock_news as naver_collect
from app.news.models import NewsArticle
from app.news.scorer import compute_daily_scores

logger = logging.getLogger(__name__)


async def collect_and_analyze(
    session: AsyncSession,
    symbols: list[str],
    *,
    days: int = 1,
) -> dict:
    """종목 리스트에 대해 뉴스 수집 → 감성 분석 → 스코어 산출을 수행한다.

    Args:
        session: DB 세션
        symbols: 종목 코드 리스트
        days: 수집 기간 (일)

    Returns:
        {"collected": N, "analyzed": N, "scored": N}
    """
    stats = {"collected": 0, "analyzed": 0, "scored": 0}

    for symbol in symbols:
        stock_name = get_stock_name(symbol) or symbol

        # ── 1. 수집 ──
        all_articles: list[RawArticle] = []

        # 네이버 금융
        try:
            naver_articles = await naver_collect(symbol, page=1)
            logger.info("네이버 수집 결과 (%s): %d건", symbol, len(naver_articles))
            all_articles.extend(naver_articles)
        except Exception as e:
            logger.warning("네이버 수집 실패 (%s): %s: %s", symbol, type(e).__name__, e)

        # BigKinds (API 키 있을 때만)
        if settings.BIGKINDS_API_KEY:
            try:
                bk_articles = await bigkinds_collect(stock_name, symbol, days=days)
                all_articles.extend(bk_articles)
            except Exception as e:
                logger.warning("BigKinds 수집 실패 (%s): %s", symbol, e)

        # DART 공시 (API 키 있을 때만)
        if settings.DART_API_KEY:
            try:
                dart_articles = await collect_disclosures(days=days)
                all_articles.extend(dart_articles)
            except Exception as e:
                logger.warning("DART 공시 수집 실패: %s", e)

        if not all_articles:
            continue

        # ── 2. DB 저장 (중복 URL 건너뜀) ──
        new_articles: list[NewsArticle] = []
        for art in all_articles:
            existing = await session.execute(
                select(NewsArticle).where(NewsArticle.url == art.url)
            )
            if existing.scalar_one_or_none():
                continue

            db_article = NewsArticle(
                id=uuid.uuid4(),
                source=art.source,
                title=art.title,
                content=art.content,
                url=art.url,
                published_at=art.published_at.replace(tzinfo=timezone.utc)
                if art.published_at.tzinfo is None
                else art.published_at,
                symbols=art.symbols,
            )
            session.add(db_article)
            new_articles.append(db_article)

        await session.flush()
        stats["collected"] += len(new_articles)

        # ── 3. 감성 분석 (미분석 기사만) ──
        unanalyzed = [a for a in new_articles if a.sentiment_score is None]
        if unanalyzed:
            batch_size = settings.NEWS_BATCH_SIZE
            for i in range(0, len(unanalyzed), batch_size):
                batch = unanalyzed[i : i + batch_size]
                batch_dicts = [
                    {
                        "title": a.title,
                        "content": a.content,
                        "source": a.source,
                    }
                    for a in batch
                ]

                results = await analyze_batch(batch_dicts)

                for result in results:
                    idx = result.article_index
                    if 0 <= idx < len(batch):
                        article = batch[idx]
                        article.sentiment_score = result.sentiment_score
                        article.sentiment_magnitude = result.sentiment_magnitude
                        article.market_impact = result.market_impact
                        article.analyzed_at = datetime.now(timezone.utc)

                        # NER로 발견된 추가 종목 매핑
                        if result.entities:
                            existing_symbols = set(article.symbols or [])
                            for entity in result.entities:
                                sym = entity.get("symbol")
                                if sym and sym not in existing_symbols:
                                    existing_symbols.add(sym)
                            article.symbols = list(existing_symbols)

                        stats["analyzed"] += 1

            await session.flush()

        # ── 4. 이벤트 스코어 산출 ──
        today = date.today()
        record = await compute_daily_scores(session, symbol, today)
        if record:
            stats["scored"] += 1

    await session.commit()

    logger.info(
        "수집+분석 파이프라인 완료: collected=%d, analyzed=%d, scored=%d",
        stats["collected"],
        stats["analyzed"],
        stats["scored"],
    )
    return stats


async def collect_dart_disclosures(
    session: AsyncSession,
    *,
    days: int = 7,
) -> int:
    """DART 전체 공시를 수집하고 DB에 저장한다."""
    if not settings.DART_API_KEY:
        return 0

    articles = await collect_disclosures(days=days)
    count = 0

    for art in articles:
        existing = await session.execute(
            select(NewsArticle).where(NewsArticle.url == art.url)
        )
        if existing.scalar_one_or_none():
            continue

        db_article = NewsArticle(
            id=uuid.uuid4(),
            source=art.source,
            title=art.title,
            content=art.content,
            url=art.url,
            published_at=art.published_at.replace(tzinfo=timezone.utc)
            if art.published_at.tzinfo is None
            else art.published_at,
            symbols=art.symbols,
        )
        session.add(db_article)
        count += 1

    await session.commit()
    logger.info("DART 공시 저장: %d건", count)
    return count
