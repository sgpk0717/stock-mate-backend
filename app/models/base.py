import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Date,
    DateTime,
    Enum as SAEnum,
    Float,
    Index,
    Integer,
    Numeric,
    PrimaryKeyConstraint,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSON, JSONB, UUID
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
    collected_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class InvestorTrading(Base):
    """투자자 주체별 매매 데이터 (KIS API 기반)."""

    __tablename__ = "investor_trading"
    __table_args__ = (
        UniqueConstraint("symbol", "dt", name="uq_investor_trading"),
        Index("ix_investor_trading_lookup", "symbol", "dt"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    dt: Mapped[datetime] = mapped_column(Date, nullable=False)
    foreign_net: Mapped[int] = mapped_column(BigInteger, server_default="0")
    inst_net: Mapped[int] = mapped_column(BigInteger, server_default="0")
    retail_net: Mapped[int] = mapped_column(BigInteger, server_default="0")
    foreign_buy_vol: Mapped[int] = mapped_column(BigInteger, server_default="0")
    foreign_sell_vol: Mapped[int] = mapped_column(BigInteger, server_default="0")
    inst_buy_vol: Mapped[int] = mapped_column(BigInteger, server_default="0")
    inst_sell_vol: Mapped[int] = mapped_column(BigInteger, server_default="0")
    retail_buy_vol: Mapped[int] = mapped_column(BigInteger, server_default="0")
    retail_sell_vol: Mapped[int] = mapped_column(BigInteger, server_default="0")
    collected_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class DartFinancial(Base):
    """DART 공시 기반 재무 데이터."""

    __tablename__ = "dart_financials"
    __table_args__ = (
        UniqueConstraint("symbol", "fiscal_year", "fiscal_quarter", name="uq_dart_financial"),
        Index("ix_dart_financials_lookup", "symbol", "disclosure_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    disclosure_date: Mapped[datetime] = mapped_column(Date, nullable=False)
    fiscal_year: Mapped[str] = mapped_column(String(4), nullable=False)
    fiscal_quarter: Mapped[str] = mapped_column(String(2), nullable=False)
    fiscal_period_end: Mapped[datetime | None] = mapped_column(Date, nullable=True)
    eps: Mapped[float | None] = mapped_column(Float, nullable=True)
    bps: Mapped[float | None] = mapped_column(Float, nullable=True)
    operating_margin: Mapped[float | None] = mapped_column(Float, nullable=True)
    debt_to_equity: Mapped[float | None] = mapped_column(Float, nullable=True)
    collected_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class ProgramTrading(Base):
    """KIS API 기반 프로그램 매매 스냅샷."""

    __tablename__ = "program_trading"
    __table_args__ = (
        UniqueConstraint("symbol", "dt", name="uq_program_trading"),
        Index("ix_program_trading_lookup", "symbol", "dt"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    dt: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    pgm_buy_qty: Mapped[int] = mapped_column(BigInteger, server_default="0")
    pgm_sell_qty: Mapped[int] = mapped_column(BigInteger, server_default="0")
    pgm_net_qty: Mapped[int] = mapped_column(BigInteger, server_default="0")
    pgm_buy_amount: Mapped[int] = mapped_column(BigInteger, server_default="0")
    pgm_sell_amount: Mapped[int] = mapped_column(BigInteger, server_default="0")
    pgm_net_amount: Mapped[int] = mapped_column(BigInteger, server_default="0")
    collected_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class MarginShortDaily(Base):
    """pykrx 기반 신용잔고 / 공매도 일별 데이터."""

    __tablename__ = "margin_short_daily"
    __table_args__ = (
        UniqueConstraint("symbol", "dt", name="uq_margin_short"),
        Index("ix_margin_short_lookup", "symbol", "dt"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    dt: Mapped[datetime] = mapped_column(Date, nullable=False)
    margin_balance: Mapped[int] = mapped_column(BigInteger, server_default="0")
    margin_rate: Mapped[float] = mapped_column(Float, server_default="0")
    short_volume: Mapped[int] = mapped_column(BigInteger, server_default="0")
    short_balance: Mapped[int] = mapped_column(BigInteger, server_default="0")
    short_balance_rate: Mapped[float] = mapped_column(Float, server_default="0")
    collected_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class WorkerState(Base):
    """워커 상태 (단일 행 singleton). 워커→API 상태 전달용."""

    __tablename__ = "worker_state"
    __table_args__ = (CheckConstraint("id = 1", name="ck_worker_state_singleton"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    factory_status: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")
    causal_jobs: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class WorkerCommand(Base):
    """API→워커 명령 큐."""

    __tablename__ = "worker_commands"
    __table_args__ = (
        Index("idx_worker_commands_pending", "status", postgresql_where=Text("status = 'pending'")),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    command: Mapped[str] = mapped_column(String(50), nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")
    status: Mapped[str] = mapped_column(String(20), nullable=False, server_default="'pending'")
    result: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    picked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
