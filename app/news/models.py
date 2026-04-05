"""뉴스 기사 및 감성 분석 DB 모델."""

import uuid
from datetime import date, datetime

from sqlalchemy import (
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class NewsArticle(Base):
    """뉴스 기사 원문 + 감성 분석 결과."""

    __tablename__ = "news_articles"
    __table_args__ = (
        UniqueConstraint("url", name="uq_news_url"),
        Index("ix_news_published", "published_at"),
        Index("ix_news_source", "source"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    source: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # "naver" | "dart" | "bigkinds"
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    published_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    symbols: Mapped[list | None] = mapped_column(
        JSON, nullable=True
    )  # ["005930", "000660"]

    # 감성 분석 결과
    sentiment_score: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )  # -1.0 ~ +1.0
    sentiment_magnitude: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )  # 0.0 ~ 1.0 (확신도)
    market_impact: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )  # 0.0 ~ 1.0
    analyzed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class DiscussionPost(Base):
    """네이버 종토방 개별 게시글 (초 단위 저장)."""

    __tablename__ = "discussion_posts"
    __table_args__ = (
        UniqueConstraint("nid", name="uq_discussion_nid"),
        Index("ix_disc_symbol_published", "symbol", "published_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    nid: Mapped[str] = mapped_column(String(50), nullable=False)  # 네이버 게시글 ID
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    author: Mapped[str | None] = mapped_column(String(100), nullable=True)
    published_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )  # 초 단위 (writtenAt 그대로)

    likes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    dislikes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    comment_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # 스팸/홍보글 판별
    is_spam: Mapped[bool] = mapped_column(default=False)

    # 감성 분석 (is_spam=True면 분석 제외)
    sentiment_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    sentiment_magnitude: Mapped[float | None] = mapped_column(Float, nullable=True)  # 확신도 0~1
    analyzed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class DiscussionSentimentHourly(Base):
    """종토방 시간별 종목 집계 (알파 피처 JOIN용)."""

    __tablename__ = "discussion_sentiment_hourly"
    __table_args__ = (
        UniqueConstraint("symbol", "dt", name="uq_disc_hourly_symbol_dt"),
        Index("ix_disc_hourly_lookup", "symbol", "dt"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    dt: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )  # 시간 버킷 (정각)
    post_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    avg_sentiment: Mapped[float | None] = mapped_column(Float, nullable=True)
    positive_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    negative_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_likes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_dislikes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class NewsSentimentDaily(Base):
    """일별/종목별 집계 감성 스코어."""

    __tablename__ = "news_sentiment_daily"
    __table_args__ = (
        Index("ix_sentiment_daily_lookup", "symbol", "date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)

    avg_sentiment: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    article_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    event_score: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0
    )  # 최종 이벤트 스코어
    top_headlines: Mapped[list | None] = mapped_column(
        JSON, nullable=True
    )  # 주요 헤드라인 3개

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
