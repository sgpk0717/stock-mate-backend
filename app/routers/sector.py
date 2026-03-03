"""섹터 검색 REST API."""

from __future__ import annotations

from fastapi import APIRouter, Query
from pydantic import BaseModel

from app.core.database import async_session
from app.sector.search import get_all_sectors, search_by_sector, search_stocks

router = APIRouter(prefix="/sector", tags=["sector"])


class SectorSearchResponse(BaseModel):
    symbol: str
    name: str
    market: str
    sector: str | None = None
    sub_sector: str | None = None
    similarity: float


class SectorInfo(BaseModel):
    sector: str
    count: int


@router.get("/search", response_model=list[SectorSearchResponse])
async def search(
    q: str = Query(..., description="검색 쿼리 (예: '반도체 관련주')"),
    top_k: int = Query(20, ge=1, le=100),
    min_similarity: float = Query(0.3, ge=0.0, le=1.0),
):
    """의미론적 종목 검색. 쿼리와 유사한 종목을 반환한다."""
    async with async_session() as session:
        results = await search_stocks(
            session, q, top_k=top_k, min_similarity=min_similarity
        )

    return [
        SectorSearchResponse(
            symbol=r.symbol,
            name=r.name,
            market=r.market,
            sector=r.sector,
            sub_sector=r.sub_sector,
            similarity=r.similarity,
        )
        for r in results
    ]


@router.get("/list", response_model=list[SectorInfo])
async def list_sectors():
    """모든 섹터 목록과 종목 수를 반환한다."""
    async with async_session() as session:
        sectors = await get_all_sectors(session)

    return [SectorInfo(**s) for s in sectors]


@router.get("/stocks/{sector_name}", response_model=list[SectorSearchResponse])
async def get_sector_stocks(sector_name: str):
    """특정 섹터의 종목 목록을 반환한다."""
    async with async_session() as session:
        results = await search_by_sector(session, sector_name)

    return [
        SectorSearchResponse(
            symbol=r.symbol,
            name=r.name,
            market=r.market,
            sector=r.sector,
            sub_sector=r.sub_sector,
            similarity=r.similarity,
        )
        for r in results
    ]
