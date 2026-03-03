"""뉴스 감성 분석 REST API."""

from __future__ import annotations

import asyncio
from datetime import date, datetime

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.core.database import async_session
from app.news.models import NewsArticle, NewsSentimentDaily
from app.news.scheduler import collect_and_analyze, collect_dart_disclosures

from sqlalchemy import select, func

router = APIRouter(prefix="/news", tags=["news"])


# ── 응답 스키마 ──


class SentimentDailyResponse(BaseModel):
    symbol: str
    date: str
    avg_sentiment: float
    article_count: int
    event_score: float
    top_headlines: list[str] | None = None


class NewsArticleResponse(BaseModel):
    id: str
    source: str
    title: str
    url: str
    published_at: str
    sentiment_score: float | None = None
    sentiment_magnitude: float | None = None
    market_impact: float | None = None


class CollectResponse(BaseModel):
    collected: int
    analyzed: int
    scored: int


# ── 엔드포인트 ──


@router.get("/sentiment/{symbol}", response_model=list[SentimentDailyResponse])
async def get_sentiment(
    symbol: str,
    start_date: date | None = Query(None),
    end_date: date | None = Query(None),
):
    """종목의 일별 감성 스코어를 조회한다."""
    async with async_session() as session:
        stmt = (
            select(NewsSentimentDaily)
            .where(NewsSentimentDaily.symbol == symbol)
            .order_by(NewsSentimentDaily.date.desc())
        )

        if start_date:
            stmt = stmt.where(NewsSentimentDaily.date >= start_date)
        if end_date:
            stmt = stmt.where(NewsSentimentDaily.date <= end_date)

        result = await session.execute(stmt)
        rows = result.scalars().all()

    return [
        SentimentDailyResponse(
            symbol=r.symbol,
            date=r.date.isoformat(),
            avg_sentiment=r.avg_sentiment,
            article_count=r.article_count,
            event_score=r.event_score,
            top_headlines=r.top_headlines,
        )
        for r in rows
    ]


@router.get("/articles/{symbol}", response_model=list[NewsArticleResponse])
async def get_articles(
    symbol: str,
    limit: int = Query(20, ge=1, le=100),
):
    """종목 관련 최근 뉴스 기사를 조회한다."""
    async with async_session() as session:
        # symbols JSON 배열에 symbol 포함된 기사 조회
        # PostgreSQL JSON 포함 여부 확인
        stmt = (
            select(NewsArticle)
            .where(
                func.cast(NewsArticle.symbols, type_=func.text()).contains(symbol)
            )
            .order_by(NewsArticle.published_at.desc())
            .limit(limit)
        )

        result = await session.execute(stmt)
        rows = result.scalars().all()

    return [
        NewsArticleResponse(
            id=str(r.id),
            source=r.source,
            title=r.title,
            url=r.url,
            published_at=r.published_at.isoformat(),
            sentiment_score=r.sentiment_score,
            sentiment_magnitude=r.sentiment_magnitude,
            market_impact=r.market_impact,
        )
        for r in rows
    ]


@router.post("/collect", response_model=CollectResponse)
async def trigger_collection(
    symbols: list[str] = Query(..., description="종목 코드 리스트"),
    days: int = Query(1, ge=1, le=30),
):
    """수동 뉴스 수집 + 분석 트리거."""
    async with async_session() as session:
        stats = await collect_and_analyze(session, symbols, days=days)

    return CollectResponse(**stats)


@router.get("/sentiment/batch", response_model=list[SentimentDailyResponse])
async def get_sentiment_batch(
    symbols: str = Query(..., description="종목 코드 콤마 구분 (예: 005930,000660)"),
    target_date: date | None = Query(None),
):
    """복수 종목의 감성 스코어를 일괄 조회한다."""
    symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
    if not symbol_list:
        raise HTTPException(400, "종목 코드를 입력하세요.")

    query_date = target_date or date.today()

    async with async_session() as session:
        stmt = (
            select(NewsSentimentDaily)
            .where(
                NewsSentimentDaily.symbol.in_(symbol_list),
                NewsSentimentDaily.date == query_date,
            )
        )
        result = await session.execute(stmt)
        rows = result.scalars().all()

    return [
        SentimentDailyResponse(
            symbol=r.symbol,
            date=r.date.isoformat(),
            avg_sentiment=r.avg_sentiment,
            article_count=r.article_count,
            event_score=r.event_score,
            top_headlines=r.top_headlines,
        )
        for r in rows
    ]
