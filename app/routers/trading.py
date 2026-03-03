"""실거래 REST API — TradingContext 관리 + KIS 주문 + 세션 제어."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.trading.context import (
    TradingContext,
    delete_context,
    get_context,
    list_contexts,
    save_context,
)
from app.trading.kis_client import get_kis_client
from app.trading.kis_order import KISOrderExecutor
from app.trading.live_runner import (
    get_session,
    list_sessions,
    start_session,
    stop_session,
)

router = APIRouter(prefix="/trading", tags=["trading"])


# ── 스키마 ──────────────────────────────────────────────


class ContextCreateRequest(BaseModel):
    """백테스트 결과에서 Context 생성."""
    mode: str = "paper"  # "paper" | "real"
    strategy: dict
    strategy_name: str = ""
    position_sizing: dict | None = None
    scaling: dict | None = None
    risk_management: dict | None = None
    initial_capital: float = 100_000_000
    position_size_pct: float = 0.1
    max_positions: int = 10
    symbols: list[str] = []
    source_backtest_id: str | None = None
    cost_config: dict | None = None


class ContextFromBacktestRequest(BaseModel):
    """백테스트 run에서 Context 직접 생성."""
    run: dict
    mode: str = "paper"


class OrderRequest(BaseModel):
    symbol: str
    qty: int
    price: int = 0
    order_type: str = "LIMIT"  # "LIMIT" | "MARKET"
    is_mock: bool = True


class CancelRequest(BaseModel):
    order_id: str
    symbol: str
    qty: int
    price: int = 0
    is_mock: bool = True


# ── Context CRUD ────────────────────────────────────────


@router.post("/context")
async def create_context(req: ContextCreateRequest):
    """TradingContext 생성."""
    from app.trading.context import CostConfig

    cost_raw = req.cost_config or {}
    cost = CostConfig(
        buy_commission=cost_raw.get("buy_commission", 0.00015),
        sell_commission=cost_raw.get("sell_commission", 0.00215),
        slippage_pct=cost_raw.get("slippage_pct", 0.001),
    )

    ctx = TradingContext(
        mode=req.mode,
        strategy=req.strategy,
        strategy_name=req.strategy_name,
        position_sizing=req.position_sizing or {},
        scaling=req.scaling,
        risk_management=req.risk_management,
        cost_config=cost,
        initial_capital=req.initial_capital,
        position_size_pct=req.position_size_pct,
        max_positions=req.max_positions,
        symbols=req.symbols,
        source_backtest_id=req.source_backtest_id,
    )
    save_context(ctx)
    return ctx.to_dict()


@router.post("/context/from-backtest")
async def create_context_from_backtest(req: ContextFromBacktestRequest):
    """백테스트 결과에서 Context 생성."""
    ctx = TradingContext.from_backtest_run(req.run, req.mode)
    save_context(ctx)
    return ctx.to_dict()


@router.get("/contexts")
async def get_contexts():
    """모든 Context 목록."""
    return [c.to_dict() for c in list_contexts()]


@router.get("/context/{context_id}")
async def get_context_detail(context_id: str):
    ctx = get_context(context_id)
    if not ctx:
        raise HTTPException(404, "Context not found")
    return ctx.to_dict()


@router.delete("/context/{context_id}")
async def remove_context(context_id: str):
    if not delete_context(context_id):
        raise HTTPException(404, "Context not found")
    return {"status": "deleted"}


# ── 세션 (실거래 실행) ──────────────────────────────────


@router.post("/start")
async def start_trading(context_id: str):
    """실거래 세션 시작."""
    ctx = get_context(context_id)
    if not ctx:
        raise HTTPException(404, "Context not found")

    # 이미 실행 중인 세션 확인
    existing = get_session(ctx.id)
    if existing and existing.status == "running":
        raise HTTPException(409, "Session already running")

    session = await start_session(ctx)
    return session.to_dict()


@router.post("/stop")
async def stop_trading(session_id: str):
    """실거래 세션 중지."""
    session = await stop_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return session.to_dict()


@router.get("/status")
async def trading_status():
    """모든 세션 상태."""
    return [s.to_dict() for s in list_sessions()]


@router.get("/session/{session_id}")
async def get_session_detail(session_id: str):
    session = get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return session.to_dict()


@router.get("/session/{session_id}/trades")
async def get_session_trades(session_id: str):
    session = get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return session.trade_log


# ── KIS 직접 주문 ───────────────────────────────────────


@router.post("/order/buy")
async def place_buy_order(req: OrderRequest):
    """KIS 매수 주문."""
    client = get_kis_client(is_mock=req.is_mock)
    executor = KISOrderExecutor(client)
    result = await executor.buy(req.symbol, req.qty, req.price, req.order_type)
    return result


@router.post("/order/sell")
async def place_sell_order(req: OrderRequest):
    """KIS 매도 주문."""
    client = get_kis_client(is_mock=req.is_mock)
    executor = KISOrderExecutor(client)
    result = await executor.sell(req.symbol, req.qty, req.price, req.order_type)
    return result


@router.post("/order/cancel")
async def cancel_order(req: CancelRequest):
    """KIS 주문 취소."""
    client = get_kis_client(is_mock=req.is_mock)
    executor = KISOrderExecutor(client)
    result = await executor.cancel(req.order_id, req.symbol, req.qty, req.price)
    return result


# ── KIS 계좌 조회 ───────────────────────────────────────


@router.get("/accounts")
async def get_kis_accounts(is_mock: bool = True):
    """KIS 잔고/계좌 조회."""
    client = get_kis_client(is_mock=is_mock)
    return await client.inquire_balance()


@router.get("/orders")
async def get_kis_orders(is_mock: bool = True):
    """KIS 당일 체결 내역."""
    client = get_kis_client(is_mock=is_mock)
    return await client.inquire_daily_ccld()


@router.get("/buyable")
async def get_buyable(symbol: str, price: int = 0, is_mock: bool = True):
    """매수 가능 수량 조회."""
    client = get_kis_client(is_mock=is_mock)
    return await client.inquire_psbl_order(symbol, price)
