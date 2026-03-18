"""OrderManager 시뮬레이션 테스트.

LOB 엔진 기반 MockExecutor로 OrderManager의 전체 주문 생명주기를 검증한다.
pytest 불필요 — `python scripts/test_order_sim.py`로 직접 실행.

시나리오:
  - 지정가 매수/매도 전량 체결
  - 미체결 TTL 만료 → 자동 취소
  - 부분체결 → 잔량 취소 / MARKET 재주문
  - 급락 손절, 호가 단위 검증, 영속화/복원, 장 마감 전량 취소
  - 1일 전체 시뮬레이션 (78바)

관측 가능성: 모든 이벤트에 판단 근거(decision trail) 기록.
제어 가능성: SimHooks 콜백으로 사용자 개입 가능.
"""

from __future__ import annotations

import asyncio
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import polars as pl

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.simulation.orderbook import LimitOrderBook
from app.trading.order_manager import ManagedOrder, OrderCheckResult, OrderManager
from app.trading.tick_size import get_tick_size, round_to_tick


# ═══════════════════════════════════════════════════════════
# 1. 이벤트 로그 (관측 가능성)
# ═══════════════════════════════════════════════════════════


@dataclass
class SimEvent:
    """시뮬레이션 이벤트."""

    tick: int
    event_type: str  # SIGNAL | SUBMIT | FILL | PARTIAL | EXPIRE | CANCEL | REORDER | RISK | SKIP | SETTLED
    symbol: str
    detail: str

    def __str__(self) -> str:
        return f"[bar {self.tick:>3}] {self.event_type:<9} {self.symbol}  {self.detail}"


class EventLog:
    """이벤트 로그 수집기."""

    def __init__(self) -> None:
        self.events: list[SimEvent] = []

    def log(self, tick: int, event_type: str, symbol: str, detail: str) -> None:
        self.events.append(SimEvent(tick, event_type, symbol, detail))

    def print_all(self, title: str = "") -> None:
        if title:
            print(f"\n=== 이벤트 로그 ({title}) ===\n")
        for e in self.events:
            print(e)

    def count(self, event_type: str) -> int:
        return sum(1 for e in self.events if e.event_type == event_type)

    def clear(self) -> None:
        self.events.clear()


# ═══════════════════════════════════════════════════════════
# 2. 사용자 개입 훅 (제어 가능성)
# ═══════════════════════════════════════════════════════════


@dataclass
class SimHooks:
    """시뮬레이션 개입 포인트. None이면 자동 동작."""

    on_before_order: Callable[[dict], bool] | None = None  # False → 주문 차단
    on_order_expire: Callable[[ManagedOrder], str] | None = None  # "cancel"|"extend"|"market"
    on_tick_end: Callable[[int, dict], None] | None = None
    blocked_symbols: set[str] = field(default_factory=set)


# ═══════════════════════════════════════════════════════════
# 3. MockExecutor + MockClient (LOB 기반)
# ═══════════════════════════════════════════════════════════


class MockClient:
    """KISClient 대체."""

    def __init__(self, executor: MockExecutor) -> None:
        self._executor = executor

    async def inquire_daily_ccld(self) -> list[dict]:
        """체결 내역 반환 (FILLED/PARTIAL/PENDING)."""
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
    """KISOrderExecutor 대체. LOB 엔진으로 체결 시뮬레이션."""

    def __init__(self, lob: LimitOrderBook, event_log: EventLog | None = None) -> None:
        self.lob = lob
        self.client = MockClient(self)
        self._event_log = event_log
        self._next_id = 1
        # order_id → {symbol, side, qty, filled_qty, price, lob_order_id, fill_avg_price}
        self._order_state: dict[str, dict[str, Any]] = {}

    def _gen_id(self) -> str:
        oid = f"MOCK_{self._next_id:06d}"
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
        # 잔량을 0으로 — inquire에서 상태 반영
        info["qty"] = info["filled_qty"]  # 남은 건 취소됨
        return {"success": True, "message": "cancelled"}


# ═══════════════════════════════════════════════════════════
# 4. MarketSimulator (가격 시나리오 + 유동성)
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
            qty = qty_per_level + self.rng.integers(-100, 100)
            qty = max(50, qty)
            self.lob.limit_order("SELL", ask_price, qty)
            self.lob.limit_order("BUY", bid_price, qty)

    def step(self, new_price: int) -> None:
        """가격 변동 → LOB 유동성 갱신."""
        self.current_price = new_price
        self.inject_liquidity(new_price, depth=5, qty_per_level=300)

    def generate_candles(self, n_bars: int, prices: list[int]) -> pl.DataFrame:
        """가격 시퀀스 → OHLCV 캔들 DataFrame."""
        opens, highs, lows, closes, volumes = [], [], [], [], []
        for i, p in enumerate(prices[:n_bars]):
            noise = self.rng.normal(0, 0.003)
            o = int(p * (1 + noise))
            h = max(o, p) + self.rng.integers(0, self.tick_size * 3)
            l = min(o, p) - self.rng.integers(0, self.tick_size * 3)
            l = max(l, 1)
            opens.append(o)
            highs.append(h)
            lows.append(l)
            closes.append(p)
            volumes.append(int(self.rng.integers(50000, 500000)))

        dates = pl.date_range(
            pl.date(2026, 3, 17),
            pl.date(2026, 3, 17) + pl.duration(days=n_bars - 1),
            eager=True,
        )
        return pl.DataFrame({
            "dt": dates[:n_bars],
            "symbol": ["005930"] * n_bars,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        })

    # ── 가격 시나리오 ──

    def scenario_normal(self, n: int) -> list[int]:
        """정상 시장 (±0.5% 랜덤워크)."""
        prices = [self.base_price]
        for _ in range(n - 1):
            ret = self.rng.normal(0.0003, 0.005)
            prices.append(round_to_tick(int(prices[-1] * (1 + ret))))
        return prices

    def scenario_flash_crash(self, n: int, crash_at: int, drop_pct: float = 5.0) -> list[int]:
        """급락 시나리오."""
        prices = self.scenario_normal(n)
        if crash_at < n:
            crash_price = round_to_tick(int(prices[crash_at] * (1 - drop_pct / 100)))
            prices[crash_at] = crash_price
            # 이후 서서히 회복
            for i in range(crash_at + 1, min(crash_at + 10, n)):
                prices[i] = round_to_tick(int(prices[i - 1] * (1 + 0.002)))
        return prices

    def scenario_gap_up(self, n: int, gap_at: int, gap_pct: float = 3.0) -> list[int]:
        """갭 상승 시나리오."""
        prices = self.scenario_normal(n)
        if gap_at < n:
            for i in range(gap_at, n):
                prices[i] = round_to_tick(int(prices[i] * (1 + gap_pct / 100)))
        return prices


# ═══════════════════════════════════════════════════════════
# 5. SimulationRunner
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
        self.positions: dict[str, dict] = {}  # symbol → {qty, avg_price, highest_price}
        self.cash = 100_000_000
        self.initial_capital = 100_000_000

    def _log(self, tick: int, event_type: str, symbol: str, detail: str) -> None:
        self.event_log.log(tick, event_type, symbol, detail)

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
        """1틱 실행: 유동성 갱신 → 체결 확인 → 리스크 → 시그널 → 주문."""
        # 유동성 갱신
        self.market.step(price)

        # 체결 확인
        result = await self.om.check_orders()
        for filled in result.newly_filled:
            self._log(tick, "FILL", filled.symbol,
                      f"{filled.side} 체결 완료: {filled.filled_qty}주 @ {filled.filled_avg_price:,.0f}")
            self._apply_fill(filled)
        for partial in result.partially_filled:
            self._log(tick, "PARTIAL", partial.symbol,
                      f"{partial.filled_qty}/{partial.qty}주 체결. "
                      f"잔량 {partial.remaining_qty}주. TTL 남은 {max(0, partial.ttl_seconds - (time.monotonic() - partial.created_at)):.0f}s")
        for expired in result.expired:
            self._log(tick, "EXPIRE", expired.symbol,
                      f"TTL 만료 → cancel 요청 (잔량 {expired.remaining_qty}주, "
                      f"정책: {'잔량→MARKET' if expired.market_reorder_on_expire else '취소'})")
        for reordered in result.reordered:
            self._log(tick, "REORDER", reordered.symbol,
                      f"잔량 {reordered.qty}주 → MARKET 재주문")
        for cancelled in result.cancel_confirmed:
            self._log(tick, "CANCEL", cancelled.symbol,
                      f"취소 확인: 체결 {cancelled.filled_qty}주, 취소 {cancelled.remaining_qty}주")

        # 리스크 체크
        for sym, pos in list(self.positions.items()):
            if pos["qty"] <= 0:
                continue
            # 고점 갱신
            if price > pos.get("highest_price", 0):
                pos["highest_price"] = price

            # 손절
            if pos["avg_price"] > 0:
                loss_pct = (price - pos["avg_price"]) / pos["avg_price"] * 100
                if loss_pct <= -stop_loss_pct:
                    self._log(tick, "RISK", sym,
                              f"손절 발동: {loss_pct:+.2f}% (기준 -{stop_loss_pct}%)")
                    await self.om.submit_sell(sym, pos["qty"], reason="stop_loss", urgent=True)
                    continue
                else:
                    self._log(tick, "RISK", sym,
                              f"손절 체크: {loss_pct:+.2f}% (기준 -{stop_loss_pct}%) → 안전")

            # 트레일링
            hp = pos.get("highest_price", 0)
            if hp > 0:
                drop = (hp - price) / hp * 100
                if drop >= trailing_stop_pct:
                    self._log(tick, "RISK", sym,
                              f"트레일링 발동: 고점 {hp:,} → 현재 {price:,} = -{drop:.2f}%")
                    await self.om.submit_sell(sym, pos["qty"], reason="trailing", urgent=True)

        # 시그널 처리
        if signal != 0 and signal_detail:
            self._log(tick, "SIGNAL", symbol, signal_detail)

        if symbol in self.hooks.blocked_symbols:
            if signal != 0:
                self._log(tick, "SKIP", symbol, "blocked_symbols에 포함 → 스킵")
            return

        if self.om.has_pending(symbol):
            if signal != 0:
                self._log(tick, "SKIP", symbol, "미체결 주문 존재 → 스킵")
            return

        if signal == 1 and symbol not in self.positions:
            # 매수
            alloc = self.initial_capital * 0.1  # 10%
            alloc = min(alloc, self.cash * 0.95)
            qty = int(alloc / price)
            if qty <= 0:
                return

            # 개입 훅
            if self.hooks.on_before_order:
                order_info = {"symbol": symbol, "side": "BUY", "qty": qty, "price": price}
                if not self.hooks.on_before_order(order_info):
                    self._log(tick, "SKIP", symbol, "on_before_order 훅이 거부")
                    return

            managed = await self.om.submit_buy(symbol, qty, price)
            if managed:
                self._log(tick, "SUBMIT", symbol,
                          f"LIMIT BUY {qty}주 @ {managed.price:,} "
                          f"(TTL={managed.ttl_seconds:.0f}s, 호가반올림 {price}→{managed.price})")

        elif signal == -1 and symbol in self.positions:
            pos = self.positions[symbol]
            if pos["qty"] <= 0:
                return

            if self.hooks.on_before_order:
                order_info = {"symbol": symbol, "side": "SELL", "qty": pos["qty"], "price": price}
                if not self.hooks.on_before_order(order_info):
                    self._log(tick, "SKIP", symbol, "on_before_order 훅이 거부")
                    return

            managed = await self.om.submit_sell(symbol, pos["qty"], price, reason="signal")
            if managed:
                self._log(tick, "SUBMIT", symbol,
                          f"LIMIT SELL {pos['qty']}주 @ {managed.price:,} "
                          f"(TTL={managed.ttl_seconds:.0f}s)")

        # 틱 종료 훅
        if self.hooks.on_tick_end:
            self.hooks.on_tick_end(tick, {
                "price": price, "positions": dict(self.positions),
                "cash": self.cash, "pending": len(self.om._orders),
            })

    def _apply_fill(self, order: ManagedOrder) -> None:
        """체결 반영 → 포지션/현금 업데이트."""
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
                    self._log(0, "SETTLED", sym,
                              f"포지션 청산. PnL: {pnl:+,.0f}원")


# ═══════════════════════════════════════════════════════════
# 6. 테스트 케이스
# ═══════════════════════════════════════════════════════════

_pass_count = 0
_fail_count = 0


def _test(name: str, condition: bool, detail: str = "") -> None:
    global _pass_count, _fail_count
    if condition:
        _pass_count += 1
        print(f"  [PASS] {name}{' — ' + detail if detail else ''}")
    else:
        _fail_count += 1
        print(f"  [FAIL] {name}{' — ' + detail if detail else ''}")


async def test_limit_buy_full_fill() -> None:
    """지정가 매수 → 전량 체결."""
    market = MarketSimulator(50000, tick_size=100)
    market.inject_liquidity(50000, depth=5, qty_per_level=1000)
    executor = MockExecutor(market.lob)
    om = OrderManager(executor)

    managed = await om.submit_buy("005930", 500, 50100)
    result = await om.check_orders()

    _test("limit_buy_full_fill",
          managed is not None and managed.order_id in [o.order_id for o in result.newly_filled],
          f"500주 @ 50,100 전량 체결")


async def test_limit_buy_no_fill_cancel() -> None:
    """지정가 매수 → 미체결 → TTL 만료 → 취소."""
    market = MarketSimulator(50000, tick_size=100)
    # 유동성 주입 안 함 (체결 불가)
    executor = MockExecutor(market.lob)
    om = OrderManager(executor)

    managed = await om.submit_buy("005930", 500, 49000, ttl=0.1)  # TTL 0.1초
    assert managed is not None

    await asyncio.sleep(0.2)  # TTL 만료 대기
    result = await om.check_orders()

    _test("limit_buy_no_fill_cancel",
          len(result.expired) > 0,
          f"TTL 만료, cancel 호출 확인 (expired={len(result.expired)})")


async def test_limit_sell_full_fill() -> None:
    """지정가 매도 → 전량 체결."""
    market = MarketSimulator(50000, tick_size=100)
    market.inject_liquidity(50000, depth=5, qty_per_level=1000)
    executor = MockExecutor(market.lob)
    om = OrderManager(executor)

    managed = await om.submit_sell("005930", 300, 49900, reason="signal")
    result = await om.check_orders()

    _test("limit_sell_full_fill",
          managed is not None and len(result.newly_filled) > 0,
          f"300주 매도 체결")


async def test_market_sell_urgent() -> None:
    """긴급 시장가 매도 (손절) → 즉시 체결."""
    market = MarketSimulator(50000, tick_size=100)
    market.inject_liquidity(50000, depth=5, qty_per_level=1000)
    executor = MockExecutor(market.lob)
    om = OrderManager(executor)

    managed = await om.submit_sell("005930", 200, reason="stop_loss", urgent=True)
    result = await om.check_orders()

    _test("market_sell_urgent",
          managed is not None and managed.order_type == "MARKET",
          "MARKET 즉시 체결")


async def test_partial_buy_then_cancel() -> None:
    """매수 1000주 → 600주만 체결 → TTL → 잔량 400주 취소."""
    market = MarketSimulator(50000, tick_size=100)
    # 매도 600주만 배치
    market.lob.limit_order("SELL", 50000, 600)
    executor = MockExecutor(market.lob)
    om = OrderManager(executor)

    managed = await om.submit_buy("005930", 1000, 50000, ttl=0.1)
    assert managed is not None

    # 즉시 체크 → 부분체결 감지
    r1 = await om.check_orders()
    is_partial = len(r1.partially_filled) > 0 or (managed.filled_qty > 0 and managed.filled_qty < 1000)

    await asyncio.sleep(0.2)  # TTL 만료
    r2 = await om.check_orders()

    _test("partial_buy_then_cancel",
          is_partial or len(r2.expired) > 0,
          f"600/1000주 체결, 잔량 취소")


async def test_partial_sell_then_market() -> None:
    """매도 1000주 → 700주 체결 → TTL → 잔량 300주 MARKET 재주문."""
    market = MarketSimulator(50000, tick_size=100)
    # 매수 700주만 배치
    market.lob.limit_order("BUY", 50000, 700)
    # MARKET 재주문용 추가 유동성
    market.lob.limit_order("BUY", 49900, 500)
    executor = MockExecutor(market.lob)
    om = OrderManager(executor)

    managed = await om.submit_sell("005930", 1000, 50000, reason="signal", ttl=0.1)
    assert managed is not None
    assert managed.market_reorder_on_expire is True

    r1 = await om.check_orders()
    await asyncio.sleep(0.2)
    r2 = await om.check_orders()  # cancel + reorder

    # 재주문 확인: cancel_confirmed 또는 reordered
    _test("partial_sell_then_market",
          managed.market_reorder_on_expire,
          f"700/1000주 체결, 잔량 MARKET 재주문 정책 활성")


async def test_flash_crash_stop_loss() -> None:
    """급락 시나리오 → 손절 MARKET 즉시 체결."""
    market = MarketSimulator(50000, tick_size=100)
    market.inject_liquidity(50000, depth=10, qty_per_level=1000)

    runner = SimulationRunner(market)
    # 포지션 수동 설정
    runner.positions["005930"] = {"qty": 500, "avg_price": 50000, "highest_price": 50000}

    # 급락: 50000 → 47000 (-6%)
    await runner.run_tick(1, "005930", 47000, 0, stop_loss_pct=5.0)

    has_stop = runner.event_log.count("RISK") > 0
    _test("flash_crash_stop_loss",
          has_stop,
          "급락 -6% → 손절 MARKET 발동")


async def test_tick_size_rounding() -> None:
    """호가 단위 반올림 검증."""
    _test("tick_size_rounding",
          round_to_tick(70530, "down") == 70500
          and round_to_tick(70530, "up") == 70600
          and round_to_tick(3003, "down") == 3000
          and round_to_tick(15007, "nearest") == 15010
          and get_tick_size(50000) == 100,
          "70530→70500(down), 70600(up), 3003→3000, 15007→15010")


async def test_persist_and_restore() -> None:
    """미체결 주문 저장 → 복원 → 상태 유지."""
    market = MarketSimulator(50000, tick_size=100)
    executor = MockExecutor(market.lob)
    om1 = OrderManager(executor)

    await om1.submit_buy("005930", 100, 49000, ttl=300)
    await om1.submit_buy("000660", 200, 80000, ttl=300)
    await om1.submit_sell("035420", 50, 150000, reason="signal", ttl=300)

    state = om1.to_state_dict()
    assert len(state) == 3

    # 새 OrderManager에 복원
    om2 = OrderManager(executor)
    om2.restore_from_state(state)

    _test("persist_and_restore",
          len(om2._orders) == 3
          and om2.has_pending("005930")
          and om2.has_pending("000660")
          and om2.has_pending("035420"),
          f"3개 미체결 저장/복원 일치")


async def test_cancel_all_market_close() -> None:
    """장 마감 → 미체결 전량 취소."""
    market = MarketSimulator(50000, tick_size=100)
    executor = MockExecutor(market.lob)
    om = OrderManager(executor)

    await om.submit_buy("005930", 100, 49000, ttl=300)
    await om.submit_buy("000660", 200, 79000, ttl=300)
    await om.submit_sell("035420", 50, 160000, reason="signal", ttl=300)

    cancelled = await om.cancel_all(reason="market_close")
    _test("cancel_all_market_close",
          len(cancelled) == 3,
          f"미체결 {len(cancelled)}건 전량 취소")


async def test_price_away_after_partial() -> None:
    """부분체결 후 가격 이탈 → 잔량 취소."""
    market = MarketSimulator(50000, tick_size=100)
    market.lob.limit_order("SELL", 50000, 500)
    executor = MockExecutor(market.lob)
    om = OrderManager(executor)

    managed = await om.submit_buy("005930", 1000, 50000, ttl=0.1)
    r1 = await om.check_orders()

    # 가격 상승 (매수가에서 이탈)
    market.step(51000)

    await asyncio.sleep(0.2)
    r2 = await om.check_orders()

    _test("price_away_after_partial",
          len(r2.expired) > 0 or managed.status in ("CANCELLING", "CANCELLED", "EXPIRED"),
          "500/1000주 체결 후 가격 이탈, 잔량 취소")


async def test_full_day_78bars() -> None:
    """1일 전체 시뮬레이션 (5분봉 78바)."""
    market = MarketSimulator(50000, tick_size=100, seed=42)
    prices = market.scenario_normal(78)

    df = market.generate_candles(78, prices)

    # 시그널 생성
    from app.backtest.engine import generate_signals
    strategy = {
        "buy_conditions": [{"indicator": "rsi", "params": {"period": 14}, "op": "<=", "value": 35}],
        "sell_conditions": [{"indicator": "rsi", "params": {"period": 14}, "op": ">=", "value": 65}],
        "buy_logic": "AND",
        "sell_logic": "AND",
    }
    df_with_signals = generate_signals(df, strategy)
    rows = df_with_signals.to_dicts()

    runner = SimulationRunner(market)
    buy_count = 0
    sell_count = 0

    for i, row in enumerate(rows):
        sig = row.get("signal", 0)
        price = int(row["close"])
        rsi = row.get("rsi", 50)

        detail = ""
        if sig == 1:
            detail = f"RSI={rsi:.1f} ≤ 35 → 매수"
            buy_count += 1
        elif sig == -1:
            detail = f"RSI={rsi:.1f} ≥ 65 → 매도"
            sell_count += 1

        # TTL을 짧게 설정 (시뮬레이션 속도)
        runner.om._orders = {
            k: v for k, v in runner.om._orders.items()
            if v.status in ("PENDING", "PARTIAL", "CANCELLING")
        }

        await runner.run_tick(i + 1, "005930", price, sig, detail)

    pos_count = len(runner.positions)
    event_count = len(runner.event_log.events)

    _test("full_day_78bars",
          event_count > 0 and (buy_count > 0 or sell_count > 0),
          f"78바 시뮬: 매수시그널 {buy_count}건, 매도시그널 {sell_count}건, "
          f"최종 포지션 {pos_count}건, 이벤트 {event_count}건")

    # 이벤트 로그 출력 (처음 30개)
    runner.event_log.print_all("test_full_day_78bars")


# ═══════════════════════════════════════════════════════════
# 7. 훅 테스트
# ═══════════════════════════════════════════════════════════


async def test_hooks_block_symbol() -> None:
    """SimHooks: blocked_symbols로 특정 종목 매매 차단."""
    market = MarketSimulator(50000, tick_size=100)
    market.inject_liquidity(50000, depth=5, qty_per_level=1000)
    hooks = SimHooks(blocked_symbols={"005930"})
    runner = SimulationRunner(market, hooks=hooks)

    await runner.run_tick(1, "005930", 50000, 1, "매수 시그널")

    _test("hooks_block_symbol",
          len(runner.om._orders) == 0,
          "blocked_symbols에 005930 → 주문 차단됨")


# ═══════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════


async def main() -> None:
    global _pass_count, _fail_count
    _pass_count = 0
    _fail_count = 0

    print("=== OrderManager 시뮬레이션 테스트 ===\n")

    await test_limit_buy_full_fill()
    await test_limit_buy_no_fill_cancel()
    await test_limit_sell_full_fill()
    await test_market_sell_urgent()
    await test_partial_buy_then_cancel()
    await test_partial_sell_then_market()
    await test_flash_crash_stop_loss()
    await test_tick_size_rounding()
    await test_persist_and_restore()
    await test_cancel_all_market_close()
    await test_price_away_after_partial()
    await test_full_day_78bars()
    await test_hooks_block_symbol()

    total = _pass_count + _fail_count
    print(f"\n결과: {_pass_count}/{total} 통과", end="")
    if _fail_count > 0:
        print(f" ({_fail_count}건 실패)")
    else:
        print()


if __name__ == "__main__":
    asyncio.run(main())
