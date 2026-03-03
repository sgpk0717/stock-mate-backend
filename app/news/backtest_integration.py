"""백테스트 엔진용 감성 데이터 로더.

news_sentiment_daily 테이블에서 종목별 일별 감성 데이터를 로딩하여
Polars DataFrame으로 변환한다. 엔진의 generate_signals()에서 join된다.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import polars as pl
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import async_session
from app.news.models import NewsSentimentDaily

logger = logging.getLogger(__name__)


async def load_sentiment_data(
    symbols: list[str] | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pl.DataFrame:
    """뉴스 감성 데이터를 Polars DataFrame으로 로딩한다.

    룩어헤드 편향 방지: 반환 시 dt를 T+1로 shift한다.
    즉, T일 뉴스 감성 → T+1일 시가 매매에 사용.

    Returns:
        DataFrame with columns: [symbol, dt, sentiment_score, article_count, event_score]
    """
    async with async_session() as session:
        stmt = select(NewsSentimentDaily)

        if symbols:
            stmt = stmt.where(NewsSentimentDaily.symbol.in_(symbols))
        if start_date:
            stmt = stmt.where(NewsSentimentDaily.date >= start_date)
        if end_date:
            stmt = stmt.where(NewsSentimentDaily.date <= end_date)

        stmt = stmt.order_by(NewsSentimentDaily.symbol, NewsSentimentDaily.date)

        result = await session.execute(stmt)
        rows = result.scalars().all()

    if not rows:
        return pl.DataFrame(
            schema={
                "symbol": pl.Utf8,
                "dt": pl.Date,
                "sentiment_score": pl.Float64,
                "article_count": pl.Int64,
                "event_score": pl.Float64,
            }
        )

    # T+1 shift: 뉴스 날짜를 1일 뒤로 이동 (룩어헤드 방지)
    # candle data_loader와 동일하게 pl.Date 타입 사용
    data = {
        "symbol": [r.symbol for r in rows],
        "dt": [r.date + timedelta(days=1) for r in rows],
        "sentiment_score": [r.avg_sentiment for r in rows],
        "article_count": [r.article_count for r in rows],
        "event_score": [r.event_score for r in rows],
    }

    df = pl.DataFrame(data)

    logger.info(
        "감성 데이터 로딩: %d행 (%s종목)",
        df.height,
        df["symbol"].n_unique(),
    )
    return df


def has_sentiment_conditions(strategy: dict) -> bool:
    """전략에 뉴스 감성 조건이 포함되어 있는지 확인한다."""
    sentiment_indicators = {"sentiment_score", "article_count", "event_score"}

    for cond in strategy.get("buy_conditions", []):
        if cond.get("indicator") in sentiment_indicators:
            return True
    for cond in strategy.get("sell_conditions", []):
        if cond.get("indicator") in sentiment_indicators:
            return True
    return False
