from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.stock_master import get_current_price, get_stock_name
from app.models.base import Order
from app.schemas.order import OrderCreate, OrderResponse

router = APIRouter(prefix="/orders", tags=["orders"])


def _to_response(order: Order) -> OrderResponse:
    return OrderResponse(
        order_id=str(order.order_id),
        symbol=order.symbol,
        name=get_stock_name(order.symbol),
        side=order.side,
        type=order.type,
        price=float(order.price) if order.price is not None else None,
        qty=order.qty,
        status=order.status,
        mode=order.mode,
        created_at=order.created_at.isoformat() if order.created_at else "",
    )


@router.get("", response_model=list[OrderResponse])
async def list_orders(
    mode: str | None = None,
    side: str | None = None,
    status: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    stmt = select(Order)
    if mode:
        stmt = stmt.where(Order.mode == mode)
    if side:
        stmt = stmt.where(Order.side == side)
    if status:
        stmt = stmt.where(Order.status == status)
    stmt = stmt.order_by(Order.created_at.desc())

    result = await db.execute(stmt)
    rows = result.scalars().all()
    return [_to_response(o) for o in rows]


@router.post("", response_model=OrderResponse, status_code=201)
async def create_order(data: OrderCreate, db: AsyncSession = Depends(get_db)):
    price = data.price
    if data.type == "MARKET":
        price = float(get_current_price(data.symbol))

    order = Order(
        symbol=data.symbol,
        side=data.side,
        type=data.type,
        price=price,
        qty=data.qty,
        mode=data.mode,
    )
    db.add(order)
    await db.flush()
    await db.refresh(order)
    return _to_response(order)


@router.patch("/{order_id}/cancel", response_model=OrderResponse)
async def cancel_order(order_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Order).where(Order.order_id == order_id))
    order = result.scalar_one_or_none()
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    if order.status != "PENDING":
        raise HTTPException(status_code=400, detail="대기 중인 주문만 취소할 수 있습니다")

    order.status = "CANCELLED"
    await db.flush()
    await db.refresh(order)
    return _to_response(order)
