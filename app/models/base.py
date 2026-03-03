import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    Enum as SAEnum,
    Index,
    Integer,
    Numeric,
    PrimaryKeyConstraint,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Account(Base):
    __tablename__ = "accounts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    mode: Mapped[str] = mapped_column(
        SAEnum("REAL", "PAPER", name="account_mode"), nullable=False
    )
    total_capital: Mapped[float] = mapped_column(
        Numeric(18, 2), nullable=False, server_default="0"
    )
    current_balance: Mapped[float] = mapped_column(
        Numeric(18, 2), nullable=False, server_default="0"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class Position(Base):
    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    mode: Mapped[str] = mapped_column(
        SAEnum("REAL", "PAPER", name="account_mode", create_type=False),
        nullable=False,
    )
    qty: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    avg_price: Mapped[float] = mapped_column(
        Numeric(18, 2), nullable=False, server_default="0"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class Order(Base):
    __tablename__ = "orders"

    order_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    side: Mapped[str] = mapped_column(
        SAEnum("BUY", "SELL", name="order_side"), nullable=False
    )
    type: Mapped[str] = mapped_column(
        SAEnum("MARKET", "LIMIT", name="order_type"), nullable=False
    )
    price: Mapped[float] = mapped_column(Numeric(18, 2), nullable=True)
    qty: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(
        SAEnum(
            "PENDING", "FILLED", "PARTIAL", "CANCELLED", "REJECTED",
            name="order_status",
        ),
        nullable=False,
        server_default="PENDING",
    )
    mode: Mapped[str] = mapped_column(
        SAEnum("REAL", "PAPER", name="account_mode", create_type=False),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class StockTick(Base):
    __tablename__ = "stock_ticks"
    __table_args__ = (
        PrimaryKeyConstraint("ts", "id"),
        Index("ix_stock_ticks_symbol", "symbol"),
    )

    id: Mapped[int] = mapped_column(BigInteger, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
    )
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    price: Mapped[float] = mapped_column(Numeric(18, 2), nullable=False)
    volume: Mapped[int] = mapped_column(BigInteger, nullable=False, server_default="0")


class StockMaster(Base):
    __tablename__ = "stock_masters"

    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    market: Mapped[str] = mapped_column(String(10), nullable=False)
    sector: Mapped[str | None] = mapped_column(String(50), nullable=True)
    sub_sector: Mapped[str | None] = mapped_column(String(100), nullable=True)
    description: Mapped[str | None] = mapped_column(String(500), nullable=True)
    embedding: Mapped[list | None] = mapped_column(JSON, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class StockCandle(Base):
    __tablename__ = "stock_candles"
    __table_args__ = (
        UniqueConstraint("symbol", "dt", "interval", name="uq_candle"),
        Index("ix_candle_lookup", "symbol", "interval", "dt"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    dt: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    interval: Mapped[str] = mapped_column(String(5), nullable=False, server_default="'1d'")
    open: Mapped[float] = mapped_column(Numeric(18, 2), nullable=False)
    high: Mapped[float] = mapped_column(Numeric(18, 2), nullable=False)
    low: Mapped[float] = mapped_column(Numeric(18, 2), nullable=False)
    close: Mapped[float] = mapped_column(Numeric(18, 2), nullable=False)
    volume: Mapped[int] = mapped_column(BigInteger, nullable=False, server_default="0")
