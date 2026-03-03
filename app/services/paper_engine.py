"""모의투자 체결 엔진.

실시간 틱을 감시하며 미체결 PAPER 주문을 매칭한다.
- 지정가 매수: 시장가 ≤ 주문가 → 체결
- 지정가 매도: 시장가 ≥ 주문가 → 체결
- 시장가: 즉시 체결 (현재 틱 가격)
"""

import logging
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import async_session
from app.models.base import Account, Order, Position
from app.services.ws_manager import manager

logger = logging.getLogger(__name__)

# 모의투자 계좌 ID (DB에 없으면 자동 생성)
PAPER_ACCOUNT_ID = 1
PAPER_INITIAL_CAPITAL = Decimal("100000000")  # 1억원


async def _ensure_paper_account(db: AsyncSession) -> Account:
    """PAPER 계좌가 없으면 생성한다."""
    result = await db.execute(
        select(Account).where(Account.mode == "PAPER")
    )
    account = result.scalar_one_or_none()
    if not account:
        account = Account(
            mode="PAPER",
            total_capital=PAPER_INITIAL_CAPITAL,
            current_balance=PAPER_INITIAL_CAPITAL,
        )
        db.add(account)
        await db.flush()
        await db.refresh(account)
        logger.info(f"모의투자 계좌 생성: ID={account.id}, 자본금={PAPER_INITIAL_CAPITAL:,}")
    return account


async def on_tick(symbol: str, price: int):
    """틱 수신 시 호출 — 해당 종목의 미체결 PAPER 주문을 체결 시도."""
    if price <= 0:
        return

    async with async_session() as db:
        # 해당 종목의 PENDING 모의주문 조회
        result = await db.execute(
            select(Order).where(
                Order.symbol == symbol,
                Order.mode == "PAPER",
                Order.status == "PENDING",
            )
        )
        pending_orders = result.scalars().all()

        if not pending_orders:
            return

        for order in pending_orders:
            should_fill = False

            if order.type == "MARKET":
                # 시장가: 무조건 체결
                should_fill = True
                fill_price = Decimal(str(price))
            elif order.type == "LIMIT":
                order_price = Decimal(str(order.price))
                if order.side == "BUY" and price <= order_price:
                    # 지정가 매수: 시장가 ≤ 주문가
                    should_fill = True
                    fill_price = order_price
                elif order.side == "SELL" and price >= order_price:
                    # 지정가 매도: 시장가 ≥ 주문가
                    should_fill = True
                    fill_price = order_price

            if should_fill:
                await _fill_order(db, order, fill_price)

        await db.commit()


async def _fill_order(db: AsyncSession, order: Order, fill_price: Decimal):
    """주문 체결 처리: Order 상태 변경 + Position 업데이트 + Account 잔고 변경."""
    account = await _ensure_paper_account(db)
    total_cost = fill_price * order.qty

    if order.side == "BUY":
        # 잔고 확인
        if account.current_balance < total_cost:
            order.status = "REJECTED"
            logger.info(f"주문 거부 (잔고 부족): {order.symbol} {order.qty}주 × {fill_price}")
            return

        # 잔고 차감
        account.current_balance = Decimal(str(account.current_balance)) - total_cost

        # 포지션 업데이트 (기존 포지션 있으면 평균단가 재계산)
        result = await db.execute(
            select(Position).where(
                Position.symbol == order.symbol,
                Position.mode == "PAPER",
            )
        )
        position = result.scalar_one_or_none()

        if position:
            old_total = Decimal(str(position.avg_price)) * position.qty
            new_total = old_total + total_cost
            position.qty += order.qty
            position.avg_price = new_total / position.qty
        else:
            position = Position(
                symbol=order.symbol,
                mode="PAPER",
                qty=order.qty,
                avg_price=fill_price,
            )
            db.add(position)

    elif order.side == "SELL":
        # 포지션 확인
        result = await db.execute(
            select(Position).where(
                Position.symbol == order.symbol,
                Position.mode == "PAPER",
            )
        )
        position = result.scalar_one_or_none()

        if not position or position.qty < order.qty:
            order.status = "REJECTED"
            logger.info(f"주문 거부 (보유수량 부족): {order.symbol} {order.qty}주")
            return

        # 포지션 수량 감소
        position.qty -= order.qty
        if position.qty == 0:
            await db.delete(position)

        # 잔고 증가 (매도대금)
        account.current_balance = Decimal(str(account.current_balance)) + total_cost

    # 체결 완료
    order.status = "FILLED"
    order.price = fill_price
    logger.info(
        f"모의 체결: {order.side} {order.symbol} {order.qty}주 × {fill_price:,} "
        f"(잔고: {account.current_balance:,})"
    )

    # WebSocket으로 체결 알림 broadcast
    await manager.broadcast("paper:fills", {
        "type": "fill",
        "order_id": str(order.order_id),
        "symbol": order.symbol,
        "side": order.side,
        "price": float(fill_price),
        "qty": order.qty,
    })


async def reset_paper_account():
    """모의투자 초기화 — 모든 PAPER 주문/포지션 삭제, 잔고 리셋."""
    async with async_session() as db:
        # PAPER 주문 전체 삭제
        result = await db.execute(
            select(Order).where(Order.mode == "PAPER")
        )
        for order in result.scalars().all():
            await db.delete(order)

        # PAPER 포지션 전체 삭제
        result = await db.execute(
            select(Position).where(Position.mode == "PAPER")
        )
        for pos in result.scalars().all():
            await db.delete(pos)

        # 계좌 잔고 리셋
        account = await _ensure_paper_account(db)
        account.total_capital = PAPER_INITIAL_CAPITAL
        account.current_balance = PAPER_INITIAL_CAPITAL

        await db.commit()
        logger.info("모의투자 초기화 완료")
