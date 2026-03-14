"""초기 테스트 데이터 삽입 스크립트.

Usage:
    docker-compose run --rm app python seed.py
"""

import asyncio

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import async_session, engine
from app.models.base import Account, Base, Order, Position


async def seed():
    # 테이블이 없으면 생성 (Alembic 이후라면 무시됨)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with async_session() as session:
        session: AsyncSession

        # 이미 시드 데이터가 있으면 스킵
        result = await session.execute(select(Account))
        if result.scalar_one_or_none():
            print("Seed data already exists — skipping.")
            return

        # ── Account ──
        account = Account(
            mode="PAPER",
            total_capital=50_000_000,
            current_balance=12_340_000,
        )
        session.add(account)

        # ── Positions ──
        positions = [
            Position(symbol="005930", mode="PAPER", qty=100, avg_price=71500),
            Position(symbol="000660", mode="PAPER", qty=30, avg_price=180000),
            Position(symbol="035720", mode="PAPER", qty=200, avg_price=41800),
            Position(symbol="005380", mode="PAPER", qty=20, avg_price=228000),
            Position(symbol="035420", mode="PAPER", qty=50, avg_price=201000),
        ]
        session.add_all(positions)

        # ── Orders ──
        orders = [
            Order(symbol="005930", side="BUY", type="MARKET", price=72800, qty=100, status="FILLED", mode="PAPER"),
            Order(symbol="000660", side="BUY", type="LIMIT", price=180000, qty=30, status="FILLED", mode="PAPER"),
            Order(symbol="035720", side="BUY", type="MARKET", price=41800, qty=200, status="FILLED", mode="PAPER"),
            Order(symbol="005380", side="BUY", type="LIMIT", price=228000, qty=20, status="FILLED", mode="PAPER"),
            Order(symbol="035420", side="BUY", type="MARKET", price=201000, qty=50, status="FILLED", mode="PAPER"),
            Order(symbol="005930", side="SELL", type="LIMIT", price=73500, qty=50, status="PENDING", mode="PAPER"),
            Order(symbol="051910", side="BUY", type="LIMIT", price=310000, qty=10, status="PENDING", mode="PAPER"),
            Order(symbol="068270", side="BUY", type="MARKET", price=184000, qty=30, status="CANCELLED", mode="PAPER"),
            Order(symbol="000660", side="SELL", type="MARKET", price=177500, qty=10, status="FILLED", mode="PAPER"),
            Order(symbol="035720", side="SELL", type="LIMIT", price=43000, qty=100, status="REJECTED", mode="PAPER"),
            Order(symbol="005930", side="BUY", type="LIMIT", price=72000, qty=50, status="PENDING", mode="PAPER"),
            Order(symbol="006400", side="BUY", type="MARKET", price=372000, qty=5, status="PARTIAL", mode="PAPER"),
        ]
        session.add_all(orders)

        await session.commit()
        print("Seed data inserted successfully!")
        print(f"  - 1 account (PAPER, 50M KRW)")
        print(f"  - {len(positions)} positions")
        print(f"  - {len(orders)} orders")


if __name__ == "__main__":
    asyncio.run(seed())
