"""실시간 전략 실행기 — 틱 감시 → 시그널 → KIS 주문.

TradingContext 기반으로 전략을 실시간 실행한다.
- 캔들 단위(1d 등): 장 시작 시 1회 시그널 체크
- 분봉/틱 단위: 주기적 시그널 체크
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import polars as pl

from app.backtest.cost_model import CostConfig, effective_buy_price, effective_sell_price
from app.backtest.engine import generate_signals
from app.core.stock_master import get_stock_name
from app.services.ws_manager import manager
from .context import TradingContext
from .kis_client import get_kis_client
from .kis_order import KISOrderExecutor

logger = logging.getLogger(__name__)


@dataclass
class LivePosition:
    """실시간 포지션 추적."""
    symbol: str
    qty: int
    avg_price: float
    highest_price: float = 0.0
    entry_date: str = ""


@dataclass
class LiveSession:
    """실거래 세션 상태."""
    id: str
    context: TradingContext
    status: str = "stopped"  # "running" | "stopped" | "error"
    positions: dict[str, LivePosition] = field(default_factory=dict)
    pending_orders: list[dict] = field(default_factory=list)
    trade_log: list[dict] = field(default_factory=list)
    error_message: str = ""
    started_at: str = ""
    stopped_at: str = ""

    # 내부 태스크
    _task: asyncio.Task | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "context_id": self.context.id,
            "mode": self.context.mode,
            "strategy_name": self.context.strategy_name,
            "status": self.status,
            "positions": {
                sym: {"symbol": p.symbol, "qty": p.qty, "avg_price": p.avg_price}
                for sym, p in self.positions.items()
            },
            "trade_count": len(self.trade_log),
            "error_message": self.error_message,
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
        }


# 세션 저장소
_sessions: dict[str, LiveSession] = {}


async def start_session(ctx: TradingContext) -> LiveSession:
    """전략 실거래 세션 시작."""
    session = LiveSession(
        id=ctx.id,
        context=ctx,
        status="running",
        started_at=datetime.now().isoformat(),
    )
    _sessions[session.id] = session

    # 백그라운드 러너 시작
    session._task = asyncio.create_task(_run_loop(session))
    logger.info("실거래 세션 시작: %s (mode=%s)", session.id, ctx.mode)

    await manager.broadcast("trading:status", {
        "session_id": session.id,
        "status": "running",
    })

    return session


async def stop_session(session_id: str) -> LiveSession | None:
    """세션 중지."""
    session = _sessions.get(session_id)
    if not session:
        return None

    session.status = "stopped"
    session.stopped_at = datetime.now().isoformat()

    if session._task and not session._task.done():
        session._task.cancel()
        try:
            await session._task
        except asyncio.CancelledError:
            pass

    logger.info("실거래 세션 중지: %s", session_id)

    await manager.broadcast("trading:status", {
        "session_id": session_id,
        "status": "stopped",
    })

    return session


def get_session(session_id: str) -> LiveSession | None:
    return _sessions.get(session_id)


def list_sessions() -> list[LiveSession]:
    return list(_sessions.values())


async def _run_loop(session: LiveSession) -> None:
    """전략 실행 메인 루프.

    일봉 전략: 30초마다 시그널 체크.
    """
    ctx = session.context
    is_mock = ctx.mode != "real"
    client = get_kis_client(is_mock=is_mock)
    executor = KISOrderExecutor(client)

    cost = CostConfig(
        buy_commission=ctx.cost_config.buy_commission,
        sell_commission=ctx.cost_config.sell_commission,
        slippage_pct=ctx.cost_config.slippage_pct,
    )

    strategy = ctx.strategy
    risk = ctx.risk_management or {}
    stop_loss_pct = risk.get("stop_loss_pct")
    trailing_stop_pct = risk.get("trailing_stop_pct")

    check_interval = 30  # 초

    try:
        while session.status == "running":
            try:
                # 1. KIS 잔고 동기화
                balance_data = await client.inquire_balance()
                kis_positions = {p["symbol"]: p for p in balance_data["positions"]}
                cash = balance_data["account"]["cash"]

                # 세션 포지션 동기화
                for sym, kp in kis_positions.items():
                    if sym in session.positions:
                        session.positions[sym].qty = kp["qty"]
                    else:
                        session.positions[sym] = LivePosition(
                            symbol=sym,
                            qty=kp["qty"],
                            avg_price=kp["avg_price"],
                            entry_date=datetime.now().strftime("%Y-%m-%d"),
                        )

                # 종료된 포지션 제거
                for sym in list(session.positions.keys()):
                    if sym not in kis_positions:
                        session.positions.pop(sym)

                # 2. 리스크 관리 체크 (기존 포지션)
                for sym, pos in list(session.positions.items()):
                    current_price = kis_positions.get(sym, {}).get("current_price", 0)
                    if current_price <= 0 or pos.avg_price <= 0:
                        continue

                    # 고점 갱신
                    if current_price > pos.highest_price:
                        pos.highest_price = current_price

                    # 고정 손절
                    if stop_loss_pct is not None:
                        loss_pct = (current_price - pos.avg_price) / pos.avg_price * 100
                        if loss_pct <= -stop_loss_pct:
                            result = await executor.sell(
                                sym, pos.qty, order_type="MARKET"
                            )
                            _log_trade(
                                session, sym, "SELL", "S-STOP", pos.qty, current_price, result,
                                reason=f"손절: 평단 대비 {round(loss_pct, 2):+.2f}%",
                                snapshot={"close": current_price, "avg_price": round(pos.avg_price)},
                            )
                            continue

                    # 트레일링 스탑
                    if trailing_stop_pct is not None and pos.highest_price > 0:
                        drop_pct = (pos.highest_price - current_price) / pos.highest_price * 100
                        if drop_pct >= trailing_stop_pct:
                            result = await executor.sell(
                                sym, pos.qty, order_type="MARKET"
                            )
                            _log_trade(
                                session, sym, "SELL", "S-TRAIL", pos.qty, current_price, result,
                                reason=f"트레일링: 고점({round(pos.highest_price):,}) 대비 -{round(drop_pct, 2):.2f}%",
                                snapshot={"close": current_price, "highest_price": round(pos.highest_price)},
                            )
                            continue

                # 3. 시그널 기반 매매 (일봉 기준 간단 구현)
                # 실제 운영에서는 캔들 데이터를 DB에서 로딩
                symbols = ctx.symbols or [sym for sym in kis_positions]
                if symbols:
                    await _check_signals_and_trade(
                        session, executor, strategy, symbols, cash, cost, ctx
                    )

                # WebSocket 상태 업데이트
                await manager.broadcast("trading:update", {
                    "session_id": session.id,
                    "positions": len(session.positions),
                    "trades": len(session.trade_log),
                    "cash": cash,
                })

            except Exception as e:
                logger.error("실거래 루프 에러: %s", e)
                session.error_message = str(e)

            await asyncio.sleep(check_interval)

    except asyncio.CancelledError:
        logger.info("실거래 루프 취소: %s", session.id)
    except Exception as e:
        session.status = "error"
        session.error_message = str(e)
        logger.error("실거래 루프 치명적 에러: %s", e)


async def _check_signals_and_trade(
    session: LiveSession,
    executor: KISOrderExecutor,
    strategy: dict,
    symbols: list[str],
    cash: float,
    cost: CostConfig,
    ctx: TradingContext,
) -> None:
    """캔들 데이터 기반 시그널 체크 → 주문."""
    from app.backtest.data_loader import load_candles
    from datetime import date, timedelta

    end = date.today()
    start = end - timedelta(days=120)
    interval = strategy.get("timeframe", "1d")

    try:
        df = await load_candles(symbols, start, end, interval)
    except Exception as e:
        logger.warning("캔들 로딩 실패: %s", e)
        return

    if df.is_empty():
        return

    for sym in symbols:
        sym_df = df.filter(pl.col("symbol") == sym).sort("dt")
        if sym_df.height < 30:
            continue

        try:
            sym_df = generate_signals(sym_df, strategy)
        except Exception:
            continue

        # 마지막 봉의 시그널 확인
        last_row = sym_df.tail(1).to_dicts()[0]
        signal = last_row.get("signal", 0)

        # 지표 스냅샷 수집
        snap_keys = ("rsi", "macd_hist", "volume_ratio", "close")
        snapshot = {k: round(v, 4) if isinstance(v, float) else v
                    for k in snap_keys if (v := last_row.get(k)) is not None}

        if signal == 1 and sym not in session.positions:
            # 매수 시그널
            if len(session.positions) >= ctx.max_positions:
                continue
            alloc = ctx.initial_capital * ctx.position_size_pct
            alloc = min(alloc, cash * 0.95)
            price = int(last_row.get("close", 0))
            if price <= 0:
                continue
            qty = int(alloc / price)
            if qty <= 0:
                continue

            result = await executor.buy(sym, qty, price, order_type="LIMIT")
            _log_trade(session, sym, "BUY", "B1", qty, price, result,
                       reason="매수 시그널", snapshot=snapshot)

        elif signal == -1 and sym in session.positions:
            # 매도 시그널
            pos = session.positions[sym]
            if pos.qty <= 0:
                continue
            result = await executor.sell(sym, pos.qty, order_type="MARKET")
            _log_trade(session, sym, "SELL", "", pos.qty, last_row.get("close", 0), result,
                       reason="매도 시그널", snapshot=snapshot)


def _log_trade(
    session: LiveSession,
    symbol: str,
    side: str,
    step: str,
    qty: int,
    price: float,
    result: dict,
    reason: str = "",
    snapshot: dict | None = None,
) -> None:
    """매매 기록."""
    session.trade_log.append({
        "symbol": symbol,
        "name": get_stock_name(symbol),
        "side": side,
        "step": step,
        "qty": qty,
        "price": price,
        "success": result.get("success", False),
        "order_id": result.get("order_id", ""),
        "message": result.get("message", ""),
        "timestamp": datetime.now().isoformat(),
        "reason": reason,
        "snapshot": snapshot,
    })
    logger.info(
        "매매 기록: %s %s %s %d주 @ %s — %s (%s)",
        side, symbol, step, qty, price,
        "성공" if result.get("success") else "실패",
        reason or "-",
    )
