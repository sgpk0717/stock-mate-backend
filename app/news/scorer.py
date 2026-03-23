"""이벤트 스코어 산출.

수집된 뉴스 감성 데이터를 기반으로 종목별 일별 이벤트 스코어를 계산한다.

공식:
  event_score = weighted_avg(sentiment_score × market_impact)
                × log(article_count + 1) / normalization_factor

가중치:
  - DART 공시: 1.5 (팩트 데이터)
  - 주요 경제지: 1.2 (향후 매체별 분류 시 사용)
  - 일반 뉴스: 1.0
"""

from __future__ import annotations

import logging
import math
from datetime import date, datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.news.models import NewsArticle, NewsSentimentDaily

logger = logging.getLogger(__name__)

# 소스별 신뢰도 가중치
SOURCE_WEIGHTS: dict[str, float] = {
    "dart": 1.5,
    "naver": 1.0,
    "bigkinds": 1.2,
}

# 정규화 상수 (log(100+1) ≈ 4.6, 일반적으로 100건 이상은 드뭄)
NORMALIZATION_FACTOR = math.log(101)


async def compute_daily_scores(
    session: AsyncSession,
    symbol: str,
    target_date: date,
) -> NewsSentimentDaily | None:
    """특정 종목의 특정 날짜 이벤트 스코어를 산출하고 DB에 저장한다.

    Returns:
        NewsSentimentDaily 레코드 또는 기사 없으면 None
    """
    # 해당 날짜+종목의 분석 완료된 기사 조회 (KST aware → DB timestamptz 자동 변환)
    _KST = timezone(timedelta(hours=9))
    start_dt = datetime.combine(target_date, datetime.min.time()).replace(tzinfo=_KST)
    end_dt = datetime.combine(target_date, datetime.max.time()).replace(tzinfo=_KST)

    stmt = (
        select(NewsArticle)
        .where(
            NewsArticle.published_at >= start_dt,
            NewsArticle.published_at <= end_dt,
            NewsArticle.sentiment_score.isnot(None),
        )
    )
    result = await session.execute(stmt)
    articles = result.scalars().all()

    # 종목 관련 기사만 필터
    relevant = [
        a for a in articles
        if a.symbols and symbol in a.symbols
    ]

    if not relevant:
        return None

    # 가중 평균 계산
    total_weight = 0.0
    weighted_sum = 0.0
    headlines: list[str] = []

    for art in relevant:
        source_w = SOURCE_WEIGHTS.get(art.source, 1.0)
        impact = art.market_impact or 0.5
        score = art.sentiment_score or 0.0

        w = source_w * impact
        weighted_sum += score * w
        total_weight += w

        if len(headlines) < 3:
            headlines.append(art.title)

    avg_sentiment = weighted_sum / total_weight if total_weight > 0 else 0.0
    article_count = len(relevant)

    # 이벤트 스코어
    event_score = avg_sentiment * math.log(article_count + 1) / NORMALIZATION_FACTOR

    # 기존 레코드 업데이트 또는 새로 생성
    existing_stmt = (
        select(NewsSentimentDaily)
        .where(
            NewsSentimentDaily.symbol == symbol,
            NewsSentimentDaily.date == target_date,
        )
    )
    existing = (await session.execute(existing_stmt)).scalar_one_or_none()

    if existing:
        existing.avg_sentiment = avg_sentiment
        existing.article_count = article_count
        existing.event_score = event_score
        existing.top_headlines = headlines
        record = existing
    else:
        record = NewsSentimentDaily(
            symbol=symbol,
            date=target_date,
            avg_sentiment=round(avg_sentiment, 4),
            article_count=article_count,
            event_score=round(event_score, 4),
            top_headlines=headlines,
        )
        session.add(record)

    await session.flush()
    logger.info(
        "이벤트 스코어 산출: %s/%s — score=%.4f, count=%d",
        symbol, target_date, event_score, article_count,
    )
    return record


async def compute_scores_for_symbols(
    session: AsyncSession,
    symbols: list[str],
    target_date: date,
) -> list[NewsSentimentDaily]:
    """여러 종목의 이벤트 스코어를 일괄 산출한다."""
    results: list[NewsSentimentDaily] = []
    for symbol in symbols:
        record = await compute_daily_scores(session, symbol, target_date)
        if record:
            results.append(record)
    await session.commit()
    return results
