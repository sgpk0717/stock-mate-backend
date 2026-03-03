from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.stock_master import get_all_stocks, search_stocks
from app.schemas.stock import CandleResponse, StockInfoResponse, TickResponse
from app.services.candle_service import get_candles
from app.services.candle_writer import write_candles_bulk

router = APIRouter(prefix="/stocks", tags=["stocks"])

KST_OFFSET = 9 * 3600  # 한국시간 UTC+9


@router.get("", response_model=list[StockInfoResponse])
async def list_stocks(
    q: str | None = Query(None, description="Search by name or symbol"),
    market: str | None = Query(None, description="Filter by market (KOSPI/KOSDAQ)"),
):
    if q:
        return search_stocks(q, market=market)
    return get_all_stocks()


@router.get("/{symbol}/candles")
async def get_candles_endpoint(
    symbol: str,
    interval: str = "1d",
    count: int = 200,
    indicators: str | None = Query(None, description="Comma-separated: rsi,macd,bb"),
    db: AsyncSession = Depends(get_db),
):
    """멀티타임프레임 캔들 조회. 1m, 3m, 5m, 15m, 30m, 1h, 1d, 1w, 1M 지원."""
    candles = await get_candles(db, symbol, interval, count)

    if not indicators:
        return candles

    from app.services.indicator_service import compute_indicators

    ind_list = [i.strip() for i in indicators.split(",") if i.strip()]
    return {
        "candles": candles,
        "indicators": compute_indicators(candles, ind_list),
    }


@router.get("/{symbol}/ticks", response_model=list[TickResponse])
async def get_ticks(
    symbol: str,
    limit: int = Query(1000, ge=1, le=10000, description="Number of ticks to return"),
    db: AsyncSession = Depends(get_db),
):
    """최근 틱 데이터 조회 (초 단위 그룹핑)."""
    result = await db.execute(
        text(
            "SELECT "
            "  date_trunc('second', ts) AS sec, "
            "  (array_agg(price ORDER BY ts DESC))[1] AS last_price, "
            "  sum(volume)::bigint AS total_volume "
            "FROM stock_ticks "
            "WHERE symbol = :symbol "
            "GROUP BY sec "
            "ORDER BY sec DESC LIMIT :limit"
        ),
        {"symbol": symbol, "limit": limit},
    )
    rows = result.fetchall()
    return [
        {
            "time": int(row[0].timestamp()) + KST_OFFSET,
            "price": float(row[1]),
            "volume": int(row[2]),
        }
        for row in reversed(rows)  # 오래된 순으로 반환
    ]


class CandleImportItem(BaseModel):
    dt: str  # YYYYMMDD or YYYYMMDDHHMMSS
    open: float
    high: float
    low: float
    close: float
    volume: int = 0


@router.post("/{symbol}/candles/import")
async def import_candles(
    symbol: str,
    interval: str = Query("1d", description="Candle interval: 1m, 5m, 1h, 1d"),
    candles: list[CandleImportItem] = [],
):
    """과거 캔들 데이터 수동 import (개발/테스트용)."""
    await write_candles_bulk(
        symbol,
        [c.model_dump() for c in candles],
        interval,
    )
    return {"imported": len(candles)}
