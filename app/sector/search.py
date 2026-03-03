"""의미론적 섹터 검색.

쿼리 텍스트를 임베딩하여 stock_masters의 임베딩과 코사인 유사도로 비교,
관련 종목을 Top-K로 반환한다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.base import StockMaster

from .embedder import cosine_similarity_batch, encode_text

logger = logging.getLogger(__name__)


@dataclass
class SectorSearchResult:
    """섹터 검색 결과."""

    symbol: str
    name: str
    market: str
    sector: str | None
    sub_sector: str | None
    similarity: float


async def search_stocks(
    session: AsyncSession,
    query: str,
    *,
    top_k: int = 20,
    min_similarity: float = 0.3,
) -> list[SectorSearchResult]:
    """자연어 쿼리로 관련 종목을 검색한다.

    Args:
        session: DB 세션
        query: 검색 쿼리 (예: "반도체 관련주", "2차전지 배터리")
        top_k: 반환할 최대 종목 수
        min_similarity: 최소 유사도 임계값

    Returns:
        유사도 내림차순 정렬된 SectorSearchResult 리스트
    """
    # 1. 쿼리 임베딩
    query_vec = encode_text(query)

    # 2. 임베딩이 있는 종목 조회
    stmt = select(StockMaster).where(StockMaster.embedding.isnot(None))
    result = await session.execute(stmt)
    stocks = result.scalars().all()

    if not stocks:
        logger.warning("임베딩된 종목이 없습니다. seed_sectors.py를 실행하세요.")
        return []

    # 3. 배치 코사인 유사도 계산
    embeddings = [s.embedding for s in stocks]
    similarities = cosine_similarity_batch(query_vec, embeddings)

    # 4. 유사도 기준 정렬 + 필터링
    results: list[SectorSearchResult] = []
    for stock, sim in zip(stocks, similarities):
        if sim >= min_similarity:
            results.append(
                SectorSearchResult(
                    symbol=stock.symbol,
                    name=stock.name,
                    market=stock.market,
                    sector=stock.sector,
                    sub_sector=stock.sub_sector,
                    similarity=round(sim, 4),
                )
            )

    results.sort(key=lambda r: r.similarity, reverse=True)
    results = results[:top_k]

    logger.info("섹터 검색 '%s': %d건 (top sim=%.4f)",
                query, len(results), results[0].similarity if results else 0)
    return results


async def search_by_sector(
    session: AsyncSession,
    sector: str,
) -> list[SectorSearchResult]:
    """특정 섹터의 종목을 DB 쿼리로 조회한다 (정확 매칭)."""
    stmt = (
        select(StockMaster)
        .where(StockMaster.sector == sector)
        .order_by(StockMaster.name)
    )
    result = await session.execute(stmt)
    stocks = result.scalars().all()

    return [
        SectorSearchResult(
            symbol=s.symbol,
            name=s.name,
            market=s.market,
            sector=s.sector,
            sub_sector=s.sub_sector,
            similarity=1.0,
        )
        for s in stocks
    ]


async def get_all_sectors(session: AsyncSession) -> list[dict]:
    """모든 섹터 목록과 종목 수를 반환한다."""
    from sqlalchemy import func as sqla_func

    stmt = (
        select(
            StockMaster.sector,
            sqla_func.count(StockMaster.symbol).label("count"),
        )
        .where(StockMaster.sector.isnot(None))
        .group_by(StockMaster.sector)
        .order_by(sqla_func.count(StockMaster.symbol).desc())
    )
    result = await session.execute(stmt)
    rows = result.all()

    return [{"sector": row[0], "count": row[1]} for row in rows]
