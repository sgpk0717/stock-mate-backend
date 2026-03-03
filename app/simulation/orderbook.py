"""경량 Limit Order Book (LOB) 엔진.

외부 의존성 없이 heapq + dataclasses로 구현.
Price-time priority 매칭, lazy cancellation.
"""

from __future__ import annotations

import heapq
import uuid
from dataclasses import dataclass, field


@dataclass
class Fill:
    """체결 결과."""

    order_id: str
    price: float
    qty: int
    side: str  # "BUY" | "SELL"
    timestamp: int


@dataclass
class Order:
    """주문."""

    id: str
    side: str  # "BUY" | "SELL"
    price: float
    qty: int
    remaining: int
    timestamp: int


class LimitOrderBook:
    """호가창 시뮬레이터.

    bids: max-heap (negated price for heapq)
    asks: min-heap
    Price-time priority matching.
    """

    def __init__(self, tick_size: float = 10.0) -> None:
        # (-price, timestamp, order_id) for max-heap behavior
        self._bids: list[tuple[float, int, str]] = []
        # (price, timestamp, order_id) for min-heap
        self._asks: list[tuple[float, int, str]] = []
        self._orders: dict[str, Order] = {}
        self._cancelled: set[str] = set()
        self._tick_size = tick_size
        self._ts_counter = 0
        self._last_trade_price: float = 0.0

    def _next_ts(self) -> int:
        self._ts_counter += 1
        return self._ts_counter

    def _snap_price(self, price: float) -> float:
        """Round price to nearest tick."""
        return round(price / self._tick_size) * self._tick_size

    # ── Public API ────────────────────────────────────────

    def limit_order(
        self, side: str, price: float, qty: int
    ) -> tuple[Order, list[Fill]]:
        """Submit limit order. If crossing, fills immediately."""
        price = self._snap_price(price)
        ts = self._next_ts()
        order_id = uuid.uuid4().hex[:12]
        order = Order(
            id=order_id,
            side=side,
            price=price,
            qty=qty,
            remaining=qty,
            timestamp=ts,
        )
        self._orders[order_id] = order

        # Try to match against opposite side
        fills = self._match(order)

        # If remaining, rest on the book
        if order.remaining > 0:
            if side == "BUY":
                heapq.heappush(self._bids, (-price, ts, order_id))
            else:
                heapq.heappush(self._asks, (price, ts, order_id))

        return order, fills

    def market_order(self, side: str, qty: int) -> list[Fill]:
        """Execute market order, walking the book."""
        ts = self._next_ts()
        order_id = uuid.uuid4().hex[:12]
        # Use extreme price to guarantee crossing
        price = 1e12 if side == "BUY" else 0.0
        order = Order(
            id=order_id,
            side=side,
            price=price,
            qty=qty,
            remaining=qty,
            timestamp=ts,
        )
        self._orders[order_id] = order
        return self._match(order)

    def cancel(self, order_id: str) -> bool:
        """Cancel order by ID."""
        if order_id in self._orders and order_id not in self._cancelled:
            self._cancelled.add(order_id)
            return True
        return False

    def get_mid_price(self) -> float:
        """Best bid/ask midpoint. Falls back to last trade price."""
        bid = self.get_best_bid()
        ask = self.get_best_ask()
        if bid is not None and ask is not None:
            return (bid + ask) / 2.0
        if bid is not None:
            return bid
        if ask is not None:
            return ask
        return self._last_trade_price

    def get_spread(self) -> float:
        """Best ask - best bid. Returns 0 if one side empty."""
        bid = self.get_best_bid()
        ask = self.get_best_ask()
        if bid is not None and ask is not None:
            return ask - bid
        return 0.0

    def get_best_bid(self) -> float | None:
        """Best (highest) bid price."""
        self._clean_stale(self._bids, "BUY")
        if self._bids:
            return -self._bids[0][0]
        return None

    def get_best_ask(self) -> float | None:
        """Best (lowest) ask price."""
        self._clean_stale(self._asks, "SELL")
        if self._asks:
            return self._asks[0][0]
        return None

    def get_depth(self, levels: int = 5) -> dict:
        """Returns aggregated depth: {bids: [(price, qty)], asks: [(price, qty)]}."""
        bid_depth: dict[float, int] = {}
        for neg_p, _ts, oid in self._bids:
            if oid in self._cancelled:
                continue
            order = self._orders.get(oid)
            if order is None or order.remaining <= 0:
                continue
            p = -neg_p
            bid_depth[p] = bid_depth.get(p, 0) + order.remaining

        ask_depth: dict[float, int] = {}
        for p, _ts, oid in self._asks:
            if oid in self._cancelled:
                continue
            order = self._orders.get(oid)
            if order is None or order.remaining <= 0:
                continue
            ask_depth[p] = ask_depth.get(p, 0) + order.remaining

        bids = sorted(bid_depth.items(), key=lambda x: -x[0])[:levels]
        asks = sorted(ask_depth.items(), key=lambda x: x[0])[:levels]

        return {"bids": bids, "asks": asks}

    # ── Matching Engine ───────────────────────────────────

    def _match(self, incoming: Order) -> list[Fill]:
        """Match incoming order against the opposite side."""
        fills: list[Fill] = []

        if incoming.side == "BUY":
            book = self._asks
            while incoming.remaining > 0 and book:
                self._clean_stale(book, "SELL")
                if not book:
                    break
                best_price, _ts, resting_id = book[0]
                if best_price > incoming.price:
                    break  # No more crossable prices
                resting = self._orders.get(resting_id)
                if resting is None or resting.remaining <= 0:
                    heapq.heappop(book)
                    continue

                fill_qty = min(incoming.remaining, resting.remaining)
                fill_price = resting.price  # Price improvement for aggressor
                incoming.remaining -= fill_qty
                resting.remaining -= fill_qty

                fill = Fill(
                    order_id=incoming.id,
                    price=fill_price,
                    qty=fill_qty,
                    side="BUY",
                    timestamp=incoming.timestamp,
                )
                fills.append(fill)
                self._last_trade_price = fill_price

                if resting.remaining <= 0:
                    heapq.heappop(book)

        else:  # SELL
            book = self._bids
            while incoming.remaining > 0 and book:
                self._clean_stale(book, "BUY")
                if not book:
                    break
                neg_best, _ts, resting_id = book[0]
                best_price = -neg_best
                if best_price < incoming.price:
                    break
                resting = self._orders.get(resting_id)
                if resting is None or resting.remaining <= 0:
                    heapq.heappop(book)
                    continue

                fill_qty = min(incoming.remaining, resting.remaining)
                fill_price = resting.price
                incoming.remaining -= fill_qty
                resting.remaining -= fill_qty

                fill = Fill(
                    order_id=incoming.id,
                    price=fill_price,
                    qty=fill_qty,
                    side="SELL",
                    timestamp=incoming.timestamp,
                )
                fills.append(fill)
                self._last_trade_price = fill_price

                if resting.remaining <= 0:
                    heapq.heappop(book)

        return fills

    def _clean_stale(self, heap: list, side: str) -> None:
        """Remove cancelled/filled orders from top of heap."""
        while heap:
            if side == "BUY":
                _neg_p, _ts, oid = heap[0]
            else:
                _p, _ts, oid = heap[0]

            if oid in self._cancelled:
                heapq.heappop(heap)
                self._cancelled.discard(oid)
                continue

            order = self._orders.get(oid)
            if order is None or order.remaining <= 0:
                heapq.heappop(heap)
                continue

            break  # Top is valid
