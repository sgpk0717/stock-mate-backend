"""실거래 REST API — TradingContext 관리 + KIS 주문 + 세션 제어."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.trading.context import (
    TradingContext,
    delete_context_from_db,
    get_context,
    list_contexts,
    save_context_to_db,
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
    await save_context_to_db(ctx)
    return ctx.to_dict()


@router.post("/context/from-backtest")
async def create_context_from_backtest(req: ContextFromBacktestRequest):
    """백테스트 결과에서 Context 생성."""
    ctx = TradingContext.from_backtest_run(req.run, req.mode)
    await save_context_to_db(ctx)
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
    if not await delete_context_from_db(context_id):
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
    """모든 세션 상태.

    Phase 2: Redis 캐시 우선, 실패 시 메모리+DB 폴백.
    """
    # Redis에서 읽기 시도
    try:
        import json
        from app.core.redis import get_client
        r = get_client()
        session_ids = await r.smembers("sessions:index")
        if session_ids:
            result = []
            for sid in session_ids:
                cached = await r.hgetall(f"sessions:{sid}")
                if cached and cached.get("id"):
                    d = dict(cached)
                    # JSON 필드 역직렬화
                    if d.get("positions"):
                        try:
                            d["positions"] = json.loads(d["positions"])
                        except (json.JSONDecodeError, TypeError):
                            d["positions"] = {}
                    d["trade_count"] = int(d.get("trade_count", 0))
                    result.append(d)
            if result:
                # Redis에 있으면 DB trade_count 보강
                for d in result:
                    if d["trade_count"] == 0:
                        try:
                            from app.core.database import async_session
                            from sqlalchemy import text
                            import uuid as _uuid
                            async with async_session() as db:
                                row = await db.execute(
                                    text("SELECT COUNT(*) FROM live_trades WHERE context_id = :cid"),
                                    {"cid": _uuid.UUID(d["id"])},
                                )
                                db_count = row.scalar() or 0
                                if db_count > 0:
                                    d["trade_count"] = db_count
                        except Exception:
                            pass
                return result
    except Exception:
        pass

    # 폴백: 기존 방식 (메모리 + DB)
    sessions = list_sessions()
    result = []
    for s in sessions:
        d = s.to_dict()
        if d["trade_count"] == 0:
            try:
                from app.core.database import async_session
                from sqlalchemy import text
                import uuid as _uuid
                async with async_session() as db:
                    row = await db.execute(
                        text("SELECT COUNT(*) FROM live_trades WHERE context_id = :cid"),
                        {"cid": _uuid.UUID(s.id)},
                    )
                    db_count = row.scalar() or 0
                    if db_count > 0:
                        d["trade_count"] = db_count
            except Exception:
                pass
        result.append(d)
    return result


@router.get("/session/{session_id}")
async def get_session_detail(session_id: str):
    session = get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return session.to_dict()


@router.get("/session/{session_id}/trades")
async def get_session_trades(session_id: str):
    session = get_session(session_id)
    if session and session.trade_log:
        return session.trade_log
    # 메모리에 없거나 비었으면 DB에서 조회
    try:
        from app.core.database import async_session
        from app.workflow.models import LiveTrade
        from sqlalchemy import select
        import uuid as _uuid

        async with async_session() as db:
            stmt = (
                select(LiveTrade)
                .where(LiveTrade.context_id == _uuid.UUID(session_id))
                .order_by(LiveTrade.executed_at)
            )
            result = await db.execute(stmt)
            trades = result.scalars().all()
            if trades:
                return [
                    {
                        "symbol": t.symbol, "name": t.name, "side": t.side,
                        "step": t.step, "qty": t.qty, "price": float(t.price),
                        "pnl_pct": t.pnl_pct,
                        "pnl_amount": float(t.pnl_amount) if t.pnl_amount is not None else None,
                        "holding_minutes": t.holding_minutes,
                        "success": t.success, "order_id": t.order_id,
                        "reason": t.reason,
                        "timestamp": t.executed_at.isoformat() if t.executed_at else "",
                        "snapshot": t.snapshot,
                    }
                    for t in trades
                ]
    except Exception:
        pass
    raise HTTPException(404, "Session not found")


@router.get("/session/{session_id}/decisions")
async def get_session_decisions(
    session_id: str,
    action: str | None = None,
    symbol: str | None = None,
    limit: int = 200,
):
    """판단 로그 조회 (매매 실행 + 스킵 포함).

    query params:
      - action: BUY, SELL, SKIP_BUY, SKIP_DATA, SKIP_ERROR, RISK_STOP, RISK_TRAIL 등
      - symbol: 특정 종목 필터
      - limit: 최대 반환 건수 (기본 200)

    Phase 2: Redis List 우선 → 메모리 폴백 (서비스 분리 대응).
    """
    # 1. Redis에서 읽기 (Worker가 RPUSH한 데이터)
    try:
        import json
        from app.core.redis import lrange
        raw = await lrange(f"decisions:{session_id}")
        if raw:
            logs = [json.loads(r) for r in raw]
            if action:
                logs = [d for d in logs if d.get("action") == action]
            if symbol:
                logs = [d for d in logs if d.get("symbol") == symbol]
            return logs[-limit:]
    except Exception:
        pass

    # 2. 폴백: 메모리 (Worker 내부 또는 inline 모드)
    session = get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    logs = session.decision_log
    if action:
        logs = [d for d in logs if d.get("action") == action]
    if symbol:
        logs = [d for d in logs if d.get("symbol") == symbol]
    return logs[-limit:]


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


@router.get("/history/{date}")
async def get_trading_history_by_date(date: str):
    """특정 날짜의 매매 기록 조회 (live_trades)."""
    from datetime import datetime

    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import AsyncSession

    from app.core.database import async_session
    from app.workflow.models import LiveTrade, WorkflowRun

    try:
        target_date = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(400, "날짜 형식: YYYY-MM-DD")

    async with async_session() as db:
        # 해당 날짜의 workflow_run에서 context_ids 조회
        run_stmt = select(WorkflowRun).where(WorkflowRun.date == target_date)
        run_result = await db.execute(run_stmt)
        run = run_result.scalar_one_or_none()

        if not run or not run.config:
            return {"date": date, "trades": [], "trade_count": 0}

        context_ids = run.config.get("trading_context_ids", [])
        if not context_ids:
            return {"date": date, "trades": [], "trade_count": 0}

        # live_trades 조회
        import uuid

        trade_stmt = (
            select(LiveTrade)
            .where(LiveTrade.context_id.in_([uuid.UUID(c) for c in context_ids]))
            .order_by(LiveTrade.executed_at.desc())
        )
        trade_result = await db.execute(trade_stmt)
        trades = trade_result.scalars().all()

        return {
            "date": date,
            "trade_count": len(trades),
            "trades": [
                {
                    "id": str(t.id),
                    "context_id": str(t.context_id),
                    "symbol": t.symbol,
                    "side": t.side,
                    "step": t.step,
                    "qty": t.qty,
                    "price": t.price,
                    "pnl_pct": t.pnl_pct,
                    "pnl_amount": t.pnl_amount,
                    "holding_minutes": getattr(t, "holding_minutes", None),
                    "executed_at": t.executed_at.isoformat() if t.executed_at else None,
                }
                for t in trades
            ],
        }


# ── 알파 스코어 랭킹 ─────────────────────────────────────


@router.get("/alpha-ranking")
async def alpha_ranking(top_n: int = 10):
    """실시간 알파 스코어 랭킹 — 매수/매도 임박 종목 TOP N.

    Redis Sorted Set에서 즉시 읽기 (<50ms).
    """
    try:
        from app.core.redis import get_client

        r = get_client()

        # 매수 후보 (score 높은 순)
        buy_raw = await r.zrevrange("alpha:buy_ranking", 0, top_n - 1, withscores=True)
        buy_candidates = []
        for sym, score in buy_raw:
            detail = await r.hgetall(f"alpha:detail:{sym}")
            buy_candidates.append({
                "symbol": sym,
                "name": detail.get("name", sym),
                "score": round(score, 4),
                "close": detail.get("close", "0"),
                "rsi": detail.get("rsi", "0"),
                "volume_ratio": detail.get("volume_ratio", "0"),
                "delta_to_buy": round(max(0, 0.7 - score), 4),
            })

        # 매도 후보 (score 낮은 순 = 1-score 높은 순)
        sell_raw = await r.zrevrange("alpha:sell_ranking", 0, top_n - 1, withscores=True)
        sell_candidates = []
        for sym, inv_score in sell_raw:
            actual_score = round(1.0 - inv_score, 4)
            detail = await r.hgetall(f"alpha:detail:{sym}")
            sell_candidates.append({
                "symbol": sym,
                "name": detail.get("name", sym),
                "score": actual_score,
                "close": detail.get("close", "0"),
                "rsi": detail.get("rsi", "0"),
                "volume_ratio": detail.get("volume_ratio", "0"),
                "delta_to_sell": round(max(0, actual_score - 0.3), 4),
            })

        updated_at = await r.get("alpha:updated_at")
        version = await r.get("alpha:version")

        return {
            "buy_candidates": buy_candidates,
            "sell_candidates": sell_candidates,
            "updated_at": updated_at or "",
            "version": int(version) if version else 0,
            "scored_count": await r.zcard("alpha:buy_ranking"),
        }
    except Exception as e:
        # Redis 미연결 또는 스코어 엔진 미시작
        return {
            "buy_candidates": [],
            "sell_candidates": [],
            "updated_at": "",
            "version": 0,
            "scored_count": 0,
            "error": str(e)[:100],
        }


@router.get("/alpha-ranking/status")
async def alpha_ranking_status():
    """알파 스코어 엔진 상태."""
    try:
        from app.trading.alpha_score_engine import get_score_engine
        return get_score_engine().get_status()
    except Exception:
        return {"running": False, "error": "엔진 미초기화"}
