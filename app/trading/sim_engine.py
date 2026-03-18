"""시뮬레이션 엔진 — LOB 기반 주문 실행 시뮬레이터.

MockExecutor + MarketSimulator + SimulationRunner를 제공한다.
SSE 리플레이 엔드포인트와 test_order_sim.py 양쪽에서 사용.

관측 가능성: 모든 판단에 SimEvent(decision trail) 기록.
제어 가능성: SimHooks 콜백으로 사용자 개입 가능.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import polars as pl

from app.simulation.orderbook import LimitOrderBook
from app.trading.order_manager import ManagedOrder, OrderCheckResult, OrderManager
from app.trading.tick_size import get_tick_size, round_to_tick


# ═══════════════════════════════════════════════════════════
# 이벤트 로그 (관측 가능성)
# ═══════════════════════════════════════════════════════════


@dataclass
class SimEvent:
    """시뮬레이션 이벤트."""

    tick: int
    event_type: str
    symbol: str
    detail: str

    def __str__(self) -> str:
        return f"[bar {self.tick:>3}] {self.event_type:<9} {self.symbol}  {self.detail}"

    def to_dict(self) -> dict:
        return {
            "bar": self.tick,
            "type": self.event_type,
            "symbol": self.symbol,
            "detail": self.detail,
        }


class EventLog:
    """이벤트 로그 수집기."""

    def __init__(self) -> None:
        self.events: list[SimEvent] = []

    def log(self, tick: int, event_type: str, symbol: str, detail: str) -> None:
        self.events.append(SimEvent(tick, event_type, symbol, detail))

    def count(self, event_type: str) -> int:
        return sum(1 for e in self.events if e.event_type == event_type)

    def clear(self) -> None:
        self.events.clear()


# ═══════════════════════════════════════════════════════════
# 사용자 개입 훅 (제어 가능성)
# ═══════════════════════════════════════════════════════════


@dataclass
class SimHooks:
    """시뮬레이션 개입 포인트."""

    on_before_order: Callable[[dict], bool] | None = None
    on_order_expire: Callable[[ManagedOrder], str] | None = None
    on_tick_end: Callable[[int, dict], None] | None = None
    blocked_symbols: set[str] = field(default_factory=set)


# ═══════════════════════════════════════════════════════════
# MockExecutor + MockClient (LOB 기반)
# ═══════════════════════════════════════════════════════════


class MockClient:
    """KISClient 대체."""

    def __init__(self, executor: MockExecutor) -> None:
        self._executor = executor

    async def inquire_daily_ccld(self) -> list[dict]:
        result = []
        for oid, info in self._executor._order_state.items():
            ord_qty = info["qty"]
            filled = info["filled_qty"]
            remaining = ord_qty - filled
            if filled >= ord_qty and ord_qty > 0:
                status = "FILLED"
            elif filled > 0:
                status = "PARTIAL"
            else:
                status = "PENDING"
            result.append({
                "order_id": oid,
                "symbol": info["symbol"],
                "side": info["side"],
                "price": info.get("fill_avg_price", info["price"]),
                "qty": filled,
                "order_qty": ord_qty,
                "remaining_qty": remaining,
                "tot_ccld_qty": filled,
                "status": status,
                "order_time": "",
            })
        return result

    async def inquire_balance(self) -> dict:
        return {"positions": [], "account": {"cash": 100_000_000, "total_deposit": 100_000_000}}


class MockExecutor:
    """KISOrderExecutor 대체."""

    def __init__(self, lob: LimitOrderBook) -> None:
        self.lob = lob
        self.client = MockClient(self)
        self._next_id = 1
        self._order_state: dict[str, dict[str, Any]] = {}

    def _gen_id(self) -> str:
        oid = f"SIM_{self._next_id:06d}"
        self._next_id += 1
        return oid

    async def buy(
        self, symbol: str, qty: int, price: int = 0, order_type: str = "LIMIT",
    ) -> dict[str, Any]:
        oid = self._gen_id()
        filled_qty = 0
        fill_avg = 0.0
        lob_oid = ""

        if order_type == "MARKET":
            fills = self.lob.market_order("BUY", qty)
            filled_qty = sum(f.qty for f in fills)
            if filled_qty > 0:
                fill_avg = sum(f.qty * f.price for f in fills) / filled_qty
        else:
            order, fills = self.lob.limit_order("BUY", price, qty)
            lob_oid = order.id
            filled_qty = sum(f.qty for f in fills)
            if filled_qty > 0:
                fill_avg = sum(f.qty * f.price for f in fills) / filled_qty

        self._order_state[oid] = {
            "symbol": symbol, "side": "BUY", "qty": qty,
            "filled_qty": filled_qty, "price": price,
            "lob_order_id": lob_oid, "fill_avg_price": fill_avg,
        }
        return {"success": True, "order_id": oid, "message": "OK"}

    async def sell(
        self, symbol: str, qty: int, price: int = 0, order_type: str = "LIMIT",
    ) -> dict[str, Any]:
        oid = self._gen_id()
        filled_qty = 0
        fill_avg = 0.0
        lob_oid = ""

        if order_type == "MARKET":
            fills = self.lob.market_order("SELL", qty)
            filled_qty = sum(f.qty for f in fills)
            if filled_qty > 0:
                fill_avg = sum(f.qty * f.price for f in fills) / filled_qty
        else:
            order, fills = self.lob.limit_order("SELL", price, qty)
            lob_oid = order.id
            filled_qty = sum(f.qty for f in fills)
            if filled_qty > 0:
                fill_avg = sum(f.qty * f.price for f in fills) / filled_qty

        self._order_state[oid] = {
            "symbol": symbol, "side": "SELL", "qty": qty,
            "filled_qty": filled_qty, "price": price,
            "lob_order_id": lob_oid, "fill_avg_price": fill_avg,
        }
        return {"success": True, "order_id": oid, "message": "OK"}

    async def cancel(
        self, order_id: str, symbol: str, qty: int, price: int = 0,
    ) -> dict[str, Any]:
        info = self._order_state.get(order_id)
        if not info:
            return {"success": False, "message": "unknown order"}
        lob_oid = info.get("lob_order_id", "")
        if lob_oid:
            self.lob.cancel(lob_oid)
        info["qty"] = info["filled_qty"]
        return {"success": True, "message": "cancelled"}


# ═══════════════════════════════════════════════════════════
# MarketSimulator (가격 시나리오 + 유동성)
# ═══════════════════════════════════════════════════════════


class MarketSimulator:
    """시장 환경 시뮬레이터."""

    def __init__(
        self, base_price: int = 50000, tick_size: int = 100, seed: int = 42,
    ) -> None:
        self.base_price = base_price
        self.tick_size = tick_size
        self.rng = np.random.default_rng(seed)
        self.lob = LimitOrderBook(tick_size=float(tick_size))
        self.current_price = base_price

    def inject_liquidity(
        self, mid_price: int, depth: int = 10, qty_per_level: int = 500,
    ) -> None:
        """호가창에 매수/매도 유동성 주입."""
        tick = self.tick_size
        for i in range(1, depth + 1):
            ask_price = mid_price + i * tick
            bid_price = mid_price - i * tick
            qty = max(50, qty_per_level + int(self.rng.integers(-100, 100)))
            self.lob.limit_order("SELL", ask_price, qty)
            self.lob.limit_order("BUY", bid_price, qty)

    def step(self, new_price: int) -> None:
        """가격 변동 → LOB 유동성 갱신."""
        self.current_price = new_price
        self.inject_liquidity(new_price, depth=5, qty_per_level=300)

    def generate_candles(self, n_bars: int, prices: list[int]) -> pl.DataFrame:
        """가격 시퀀스 → OHLCV 캔들 DataFrame."""
        opens, highs, lows, closes, volumes = [], [], [], [], []
        for p in prices[:n_bars]:
            noise = self.rng.normal(0, 0.003)
            o = int(p * (1 + noise))
            h = max(o, p) + int(self.rng.integers(0, self.tick_size * 3))
            l = min(o, p) - int(self.rng.integers(0, self.tick_size * 3))
            l = max(l, 1)
            opens.append(o)
            highs.append(h)
            lows.append(l)
            closes.append(p)
            volumes.append(int(self.rng.integers(50000, 500000)))

        dates = pl.date_range(
            pl.date(2026, 3, 18),
            pl.date(2026, 3, 18) + pl.duration(days=n_bars - 1),
            eager=True,
        )
        return pl.DataFrame({
            "dt": dates[:n_bars],
            "symbol": ["005930"] * n_bars,
            "open": opens, "high": highs, "low": lows,
            "close": closes, "volume": volumes,
        })

    def scenario_normal(self, n: int) -> list[int]:
        prices = [self.base_price]
        for _ in range(n - 1):
            ret = self.rng.normal(0.0003, 0.005)
            prices.append(round_to_tick(int(prices[-1] * (1 + ret))))
        return prices

    def scenario_flash_crash(
        self, n: int, crash_at: int = 30, drop_pct: float = 5.0,
    ) -> list[int]:
        prices = self.scenario_normal(n)
        if crash_at < n:
            prices[crash_at] = round_to_tick(int(prices[crash_at] * (1 - drop_pct / 100)))
            for i in range(crash_at + 1, min(crash_at + 10, n)):
                prices[i] = round_to_tick(int(prices[i - 1] * (1 + 0.002)))
        return prices

    def scenario_gap_up(
        self, n: int, gap_at: int = 20, gap_pct: float = 3.0,
    ) -> list[int]:
        prices = self.scenario_normal(n)
        if gap_at < n:
            for i in range(gap_at, n):
                prices[i] = round_to_tick(int(prices[i] * (1 + gap_pct / 100)))
        return prices


# ═══════════════════════════════════════════════════════════
# SimulationRunner
# ═══════════════════════════════════════════════════════════


class SimulationRunner:
    """OrderManager를 LOB 기반으로 시뮬레이션."""

    def __init__(
        self,
        market: MarketSimulator,
        hooks: SimHooks | None = None,
    ) -> None:
        self.market = market
        self.hooks = hooks or SimHooks()
        self.executor = MockExecutor(market.lob)
        self.om = OrderManager(self.executor)
        self.event_log = EventLog()
        self.positions: dict[str, dict] = {}
        self.cash: float = 100_000_000
        self.initial_capital: float = 100_000_000
        self._last_event_idx = 0

    def _log(self, tick: int, event_type: str, symbol: str, detail: str) -> None:
        self.event_log.log(tick, event_type, symbol, detail)

    def get_new_events(self) -> list[SimEvent]:
        """마지막 조회 이후 새 이벤트만 반환."""
        events = self.event_log.events[self._last_event_idx:]
        self._last_event_idx = len(self.event_log.events)
        return events

    async def run_tick(
        self,
        tick: int,
        symbol: str,
        price: int,
        signal: int,
        signal_detail: str = "",
        stop_loss_pct: float = 5.0,
        trailing_stop_pct: float = 3.0,
    ) -> None:
        """1틱 실행."""
        self.market.step(price)

        # 체결 확인
        result = await self.om.check_orders()
        for filled in result.newly_filled:
            self._log(tick, "FILL", filled.symbol,
                      f"{filled.side} 체결 완료: {filled.filled_qty}주 @ {filled.filled_avg_price:,.0f}")
            self._apply_fill(filled)
        for partial in result.partially_filled:
            self._log(tick, "PARTIAL", partial.symbol,
                      f"{partial.filled_qty}/{partial.qty}주 체결. 잔량 {partial.remaining_qty}주")
        for expired in result.expired:
            policy = "잔량→MARKET" if expired.market_reorder_on_expire else "취소"
            self._log(tick, "EXPIRE", expired.symbol,
                      f"TTL 만료 → 잔량 {expired.remaining_qty}주 ({policy})")
        for reordered in result.reordered:
            self._log(tick, "REORDER", reordered.symbol,
                      f"잔량 {reordered.qty}주 → MARKET 재주문")
        for cancelled in result.cancel_confirmed:
            self._log(tick, "CANCEL", cancelled.symbol,
                      f"취소 확인: 체결 {cancelled.filled_qty}주")

        # 리스크 체크
        for sym, pos in list(self.positions.items()):
            if pos["qty"] <= 0:
                continue
            if price > pos.get("highest_price", 0):
                pos["highest_price"] = price

            if pos["avg_price"] > 0:
                loss_pct = (price - pos["avg_price"]) / pos["avg_price"] * 100
                if loss_pct <= -stop_loss_pct:
                    self._log(tick, "RISK", sym,
                              f"손절: {loss_pct:+.2f}% (기준 -{stop_loss_pct}%) → MARKET SELL")
                    await self.om.submit_sell(sym, pos["qty"], reason="stop_loss", urgent=True)
                    continue

            hp = pos.get("highest_price", 0)
            if hp > 0:
                drop = (hp - price) / hp * 100
                if drop >= trailing_stop_pct:
                    self._log(tick, "RISK", sym,
                              f"트레일링: 고점 {hp:,} → {price:,} = -{drop:.1f}% → MARKET SELL")
                    await self.om.submit_sell(sym, pos["qty"], reason="trailing", urgent=True)

        # 시그널
        if signal != 0 and signal_detail:
            self._log(tick, "SIGNAL", symbol, signal_detail)

        if symbol in self.hooks.blocked_symbols:
            if signal != 0:
                self._log(tick, "SKIP", symbol, "blocked_symbols → 스킵")
            return

        if self.om.has_pending(symbol):
            if signal != 0:
                self._log(tick, "SKIP", symbol, "미체결 주문 존재 → 스킵")
            return

        if signal == 1 and symbol not in self.positions:
            alloc = min(self.initial_capital * 0.1, self.cash * 0.95)
            qty = int(alloc / price)
            if qty <= 0:
                return

            if self.hooks.on_before_order:
                if not self.hooks.on_before_order({"symbol": symbol, "side": "BUY", "qty": qty, "price": price}):
                    self._log(tick, "SKIP", symbol, "on_before_order 훅 거부")
                    return

            managed = await self.om.submit_buy(symbol, qty, price)
            if managed:
                self._log(tick, "SUBMIT", symbol,
                          f"LIMIT BUY {qty}주 @ {managed.price:,} (TTL={managed.ttl_seconds:.0f}s)")

        elif signal == -1 and symbol in self.positions:
            pos = self.positions[symbol]
            if pos["qty"] <= 0:
                return

            if self.hooks.on_before_order:
                if not self.hooks.on_before_order({"symbol": symbol, "side": "SELL", "qty": pos["qty"], "price": price}):
                    self._log(tick, "SKIP", symbol, "on_before_order 훅 거부")
                    return

            managed = await self.om.submit_sell(symbol, pos["qty"], price, reason="signal")
            if managed:
                self._log(tick, "SUBMIT", symbol,
                          f"LIMIT SELL {pos['qty']}주 @ {managed.price:,} (TTL={managed.ttl_seconds:.0f}s)")

        if self.hooks.on_tick_end:
            self.hooks.on_tick_end(tick, {
                "price": price, "positions": dict(self.positions),
                "cash": self.cash, "pending": len(self.om._orders),
            })

    def _apply_fill(self, order: ManagedOrder) -> None:
        sym = order.symbol
        if order.side == "BUY":
            if sym in self.positions:
                pos = self.positions[sym]
                old_total = pos["avg_price"] * pos["qty"]
                new_total = old_total + order.filled_avg_price * order.filled_qty
                pos["qty"] += order.filled_qty
                pos["avg_price"] = new_total / pos["qty"] if pos["qty"] > 0 else 0
            else:
                self.positions[sym] = {
                    "qty": order.filled_qty,
                    "avg_price": order.filled_avg_price,
                    "highest_price": order.filled_avg_price,
                }
            self.cash -= order.filled_avg_price * order.filled_qty
        elif order.side == "SELL":
            if sym in self.positions:
                pos = self.positions[sym]
                pnl = (order.filled_avg_price - pos["avg_price"]) * order.filled_qty
                pos["qty"] -= order.filled_qty
                self.cash += order.filled_avg_price * order.filled_qty
                if pos["qty"] <= 0:
                    del self.positions[sym]
                    self._log(0, "SETTLED", sym, f"포지션 청산. PnL: {pnl:+,.0f}원")

    def get_state(self) -> dict:
        """현재 상태 스냅샷."""
        total_eval = self.cash
        for sym, pos in self.positions.items():
            total_eval += pos.get("avg_price", 0) * pos["qty"]
        pnl = total_eval - self.initial_capital
        pnl_pct = pnl / self.initial_capital * 100 if self.initial_capital > 0 else 0

        return {
            "cash": round(self.cash),
            "positions": {
                sym: {"qty": p["qty"], "avg_price": round(p["avg_price"])}
                for sym, p in self.positions.items()
            },
            "pending_orders": len(self.om._orders),
            "total_eval": round(total_eval),
            "pnl": round(pnl),
            "pnl_pct": round(pnl_pct, 4),
        }
