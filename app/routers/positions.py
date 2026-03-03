from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.stock_master import get_current_price, get_stock_name
from app.models.base import Position
from app.schemas.position import PositionResponse

router = APIRouter(prefix="/positions", tags=["positions"])


@router.get("", response_model=list[PositionResponse])
async def list_positions(mode: str = "PAPER", db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Position).where(Position.mode == mode))
    rows = result.scalars().all()

    response = []
    for pos in rows:
        current_price = float(get_current_price(pos.symbol))
        avg = float(pos.avg_price)
        pnl = (current_price - avg) * pos.qty
        pnl_percent = ((current_price - avg) / avg * 100) if avg else 0.0

        response.append(
            PositionResponse(
                id=pos.id,
                symbol=pos.symbol,
                name=get_stock_name(pos.symbol),
                mode=pos.mode,
                qty=pos.qty,
                avg_price=avg,
                current_price=current_price,
                pnl=round(pnl),
                pnl_percent=round(pnl_percent, 2),
            )
        )
    return response
