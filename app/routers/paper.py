"""모의투자 API 라우터.

기존 Order/Position/Account 모델을 mode="PAPER"로 활용한다.
체결은 paper_engine이 실시간 틱으로 자동 매칭한다.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from decimal import Decimal

from app.core.database import get_db
from app.core.stock_master import get_current_price, get_stock_name
from app.models.base import Account, Order, Position
from app.schemas.account import AccountResponse
from app.schemas.order import OrderCreate, OrderResponse
from app.schemas.position import PositionResponse
from app.services import paper_engine

router = APIRouter(prefix="/paper", tags=["paper"])


def _order_response(order: Order) -> OrderResponse:
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


@router.post("/orders", response_model=OrderResponse, status_code=201)
async def create_paper_order(data: OrderCreate, db: AsyncSession = Depends(get_db)):
    """모의주문 접수. 시장가는 즉시 체결 시도."""
    price = data.price
    if data.type == "MARKET":
        current = float(get_current_price(data.symbol))
        if current <= 0:
            raise HTTPException(status_code=400, detail="현재가 정보 없음 (시장 데이터 대기 중)")
        price = current

    order = Order(
        symbol=data.symbol,
        side=data.side,
        type=data.type,
        price=price,
        qty=data.qty,
        mode="PAPER",
    )
    db.add(order)
    await db.flush()
    await db.refresh(order)

    # 시장가 주문: 같은 세션에서 즉시 체결
    if data.type == "MARKET" and price and price > 0:
        fill_price = Decimal(str(price))
        account = await paper_engine._ensure_paper_account(db)
        total_cost = fill_price * order.qty

        if order.side == "BUY":
            if account.current_balance >= total_cost:
                account.current_balance = Decimal(str(account.current_balance)) - total_cost
                # 포지션 업데이트
                result = await db.execute(
                    select(Position).where(
                        Position.symbol == order.symbol, Position.mode == "PAPER"
                    )
                )
                position = result.scalar_one_or_none()
                if position:
                    old_total = Decimal(str(position.avg_price)) * position.qty
                    new_total = old_total + total_cost
                    position.qty += order.qty
                    position.avg_price = new_total / position.qty
                else:
                    db.add(Position(
                        symbol=order.symbol, mode="PAPER",
                        qty=order.qty, avg_price=fill_price,
                    ))
                order.status = "FILLED"
            else:
                order.status = "REJECTED"
        elif order.side == "SELL":
            result = await db.execute(
                select(Position).where(
                    Position.symbol == order.symbol, Position.mode == "PAPER"
                )
            )
            position = result.scalar_one_or_none()
            if position and position.qty >= order.qty:
                position.qty -= order.qty
                if position.qty == 0:
                    await db.delete(position)
                account.current_balance = Decimal(str(account.current_balance)) + total_cost
                order.status = "FILLED"
            else:
                order.status = "REJECTED"

        await db.flush()
        await db.refresh(order)

    return _order_response(order)


@router.get("/orders", response_model=list[OrderResponse])
async def list_paper_orders(
    status: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    """모의주문 내역 조회."""
    stmt = select(Order).where(Order.mode == "PAPER")
    if status:
        stmt = stmt.where(Order.status == status)
    stmt = stmt.order_by(Order.created_at.desc())

    result = await db.execute(stmt)
    return [_order_response(o) for o in result.scalars().all()]


@router.patch("/orders/{order_id}/cancel", response_model=OrderResponse)
async def cancel_paper_order(order_id: str, db: AsyncSession = Depends(get_db)):
    """미체결 모의주문 취소."""
    result = await db.execute(
        select(Order).where(Order.order_id == order_id, Order.mode == "PAPER")
    )
    order = result.scalar_one_or_none()
    if not order:
        raise HTTPException(status_code=404, detail="주문을 찾을 수 없습니다")
    if order.status != "PENDING":
        raise HTTPException(status_code=400, detail="대기 중인 주문만 취소할 수 있습니다")

    order.status = "CANCELLED"
    await db.flush()
    await db.refresh(order)
    return _order_response(order)


@router.get("/positions", response_model=list[PositionResponse])
async def list_paper_positions(db: AsyncSession = Depends(get_db)):
    """모의 포지션 조회."""
    result = await db.execute(
        select(Position).where(Position.mode == "PAPER")
    )
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


@router.get("/account", response_model=AccountResponse)
async def get_paper_account(db: AsyncSession = Depends(get_db)):
    """모의 계좌 잔고 조회."""
    result = await db.execute(
        select(Account).where(Account.mode == "PAPER")
    )
    account = result.scalar_one_or_none()
    if not account:
        # 자동 생성
        account = await paper_engine._ensure_paper_account(db)
        await db.commit()
    return account


@router.post("/reset")
async def reset_paper():
    """모의투자 초기화 (전체 리셋)."""
    await paper_engine.reset_paper_account()
    return {"message": "모의투자가 초기화되었습니다."}
