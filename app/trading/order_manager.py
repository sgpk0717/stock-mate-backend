"""주문 생명주기 관리 (OrderManager).

KISOrderExecutor를 감싸고, 미체결/부분체결/TTL 만료/취소를 관리한다.
live_runner.py가 직접 KISOrderExecutor 대신 이 클래스를 사용.

주문 상태 전이:
  PENDING → PARTIAL → FILLED
  PENDING → CANCELLING → CANCELLED
  PARTIAL → CANCELLING → CANCELLED (+ 잔량 MARKET 재주문)
  PENDING/PARTIAL → EXPIRED (TTL 만료 → 자동 cancel)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from app.core.config import settings

from .tick_size import round_to_tick

logger = logging.getLogger(__name__)


@dataclass
class ManagedOrder:
    """추적 중인 주문."""

    order_id: str
    symbol: str
    side: str  # "BUY" | "SELL"
    order_type: str  # "LIMIT" | "MARKET"
    qty: int  # 원래 주문 수량
    price: int  # 주문 가격 (시장가면 0)
    filled_qty: int = 0
    filled_avg_price: float = 0.0
    status: str = "PENDING"  # PENDING | PARTIAL | CANCELLING | FILLED | CANCELLED | EXPIRED
    created_at: float = 0.0  # time.monotonic()
    ttl_seconds: float = 0.0
    cancel_requested_at: float | None = None
    reason: str = ""  # signal | stop_loss | trailing | circuit_breaker | close
    market_reorder_on_expire: bool = False
    child_order_id: str | None = None  # cancel 후 MARKET 재주문 ID
    meta: dict = field(default_factory=dict)  # 분할매매 메타: {"step": "B1", "conviction": ...}

    @property
    def remaining_qty(self) -> int:
        return max(0, self.qty - self.filled_qty)

    @property
    def is_expired(self) -> bool:
        if self.ttl_seconds <= 0:
            return False
        return (time.monotonic() - self.created_at) > self.ttl_seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "order_type": self.order_type,
            "qty": self.qty,
            "price": self.price,
            "filled_qty": self.filled_qty,
            "filled_avg_price": self.filled_avg_price,
            "status": self.status,
            "created_at": self.created_at,
            "ttl_seconds": self.ttl_seconds,
            "reason": self.reason,
            "market_reorder_on_expire": self.market_reorder_on_expire,
            "child_order_id": self.child_order_id,
            "meta": self.meta,
        }


@dataclass
class OrderCheckResult:
    """check_orders() 결과."""

    newly_filled: list[ManagedOrder] = field(default_factory=list)
    partially_filled: list[ManagedOrder] = field(default_factory=list)
    expired: list[ManagedOrder] = field(default_factory=list)
    cancel_confirmed: list[ManagedOrder] = field(default_factory=list)
    reordered: list[ManagedOrder] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class OrderManager:
    """주문 생명주기 관리.

    KISOrderExecutor를 감싸고, 주문 추적/부분체결/만료/취소를 관리한다.
    """

    def __init__(self, executor: Any):
        self._executor = executor
        self._orders: dict[str, ManagedOrder] = {}

    # ── 주문 제출 ────────────────────────────────────────────

    async def submit_buy(
        self,
        symbol: str,
        qty: int,
        price: int,
        ttl: float | None = None,
    ) -> ManagedOrder | None:
        """매수 주문 (LIMIT, 호가 단위 반올림)."""
        rounded_price = round_to_tick(price)
        if rounded_price <= 0 or qty <= 0:
            return None

        result = await self._executor.buy(
            symbol, qty, rounded_price, order_type="LIMIT"
        )
        if not result.get("success"):
            logger.warning(
                "매수 주문 실패: %s %d주 @ %d — %s",
                symbol, qty, rounded_price, result.get("message", ""),
            )
            return None

        order = ManagedOrder(
            order_id=result["order_id"],
            symbol=symbol,
            side="BUY",
            order_type="LIMIT",
            qty=qty,
            price=rounded_price,
            created_at=time.monotonic(),
            ttl_seconds=ttl or settings.ORDER_BUY_TTL_SECONDS,
            reason="signal",
            market_reorder_on_expire=False,  # 매수: 잔량 취소만
        )
        self._orders[order.order_id] = order
        logger.info(
            "매수 주문 등록: %s %s %d주 @ %d (TTL=%ds)",
            order.order_id, symbol, qty, rounded_price, order.ttl_seconds,
        )
        return order

    async def submit_sell(
        self,
        symbol: str,
        qty: int,
        price: int = 0,
        *,
        reason: str = "signal",
        urgent: bool = False,
        ttl: float | None = None,
    ) -> ManagedOrder | None:
        """매도 주문.

        urgent=True: MARKET 즉시 체결 (손절/트레일링/서킷브레이커).
        urgent=False: LIMIT (호가 내림) + TTL 만료 시 잔량 MARKET 재주문.
        """
        if qty <= 0:
            return None

        if urgent or not settings.ORDER_SELL_USE_LIMIT:
            # MARKET 즉시 — 추적하지만 TTL 없음
            result = await self._executor.sell(symbol, qty, order_type="MARKET")
            if not result.get("success"):
                logger.warning(
                    "시장가 매도 실패: %s %d주 — %s",
                    symbol, qty, result.get("message", ""),
                )
                return None

            order = ManagedOrder(
                order_id=result["order_id"],
                symbol=symbol,
                side="SELL",
                order_type="MARKET",
                qty=qty,
                price=0,
                created_at=time.monotonic(),
                ttl_seconds=60,  # MARKET도 60초 추적 (확인용)
                reason=reason,
                market_reorder_on_expire=False,
            )
            self._orders[order.order_id] = order
            return order

        # LIMIT 매도
        rounded_price = round_to_tick(price, "down") if price > 0 else 0
        if rounded_price <= 0:
            # 가격 정보 없으면 MARKET 폴백
            return await self.submit_sell(
                symbol, qty, reason=reason, urgent=True,
            )

        result = await self._executor.sell(
            symbol, qty, rounded_price, order_type="LIMIT"
        )
        if not result.get("success"):
            logger.warning(
                "지정가 매도 실패: %s %d주 @ %d — %s",
                symbol, qty, rounded_price, result.get("message", ""),
            )
            return None

        order = ManagedOrder(
            order_id=result["order_id"],
            symbol=symbol,
            side="SELL",
            order_type="LIMIT",
            qty=qty,
            price=rounded_price,
            created_at=time.monotonic(),
            ttl_seconds=ttl or settings.ORDER_SELL_TTL_SECONDS,
            reason=reason,
            market_reorder_on_expire=True,  # 매도: TTL 후 잔량 MARKET
        )
        self._orders[order.order_id] = order
        logger.info(
            "매도 주문 등록: %s %s %d주 @ %d (TTL=%ds, 만료→MARKET)",
            order.order_id, symbol, qty, rounded_price, order.ttl_seconds,
        )
        return order

    # ── 주문 확인 + 만료 처리 ─────────────────────────────────

    async def check_orders(self) -> OrderCheckResult:
        """미체결 주문 체결 상태 확인 + TTL 만료 처리.

        30초마다 호출. KIS inquire_daily_ccld() 1회 + 만료 주문 cancel.
        """
        result = OrderCheckResult()

        if not self._orders:
            return result

        # KIS 체결 내역 조회
        try:
            ccld_list = await self._executor.client.inquire_daily_ccld()
        except Exception as e:
            result.errors.append(f"체결 조회 실패: {e}")
            return result

        ccld_map: dict[str, dict] = {}
        for item in ccld_list:
            oid = item.get("order_id", "")
            if oid:
                ccld_map[oid] = item

        orders_to_remove: list[str] = []

        for order_id, order in list(self._orders.items()):
            ccld = ccld_map.get(order_id)

            if ccld:
                prev_filled = order.filled_qty
                order.filled_qty = ccld.get("tot_ccld_qty", 0)
                order.filled_avg_price = ccld.get("price", 0)

                if ccld["status"] == "FILLED":
                    order.status = "FILLED"
                    result.newly_filled.append(order)
                    orders_to_remove.append(order_id)
                    logger.info(
                        "체결 완료: %s %s %s %d주 @ %.0f",
                        order_id, order.side, order.symbol,
                        order.filled_qty, order.filled_avg_price,
                    )
                    continue

                if ccld["status"] == "PARTIAL" and order.filled_qty > prev_filled:
                    order.status = "PARTIAL"
                    result.partially_filled.append(order)
                    logger.info(
                        "부분체결: %s %s %s %d/%d주",
                        order_id, order.side, order.symbol,
                        order.filled_qty, order.qty,
                    )

            # CANCELLING 상태: cancel 확인
            if order.status == "CANCELLING":
                cancel_elapsed = time.monotonic() - (order.cancel_requested_at or 0)
                if cancel_elapsed > settings.ORDER_CANCEL_TIMEOUT_SECONDS:
                    # 취소 타임아웃 — 강제 제거
                    order.status = "CANCELLED"
                    result.cancel_confirmed.append(order)
                    orders_to_remove.append(order_id)
                    logger.warning("취소 타임아웃: %s %s", order_id, order.symbol)
                elif ccld and ccld["status"] == "FILLED":
                    # cancel 보냈는데 그새 전부 체결됨
                    order.status = "FILLED"
                    result.newly_filled.append(order)
                    orders_to_remove.append(order_id)
                elif not ccld or ccld.get("remaining_qty", 0) == 0:
                    # 미체결 사라짐 → 취소 성공
                    order.status = "CANCELLED"
                    result.cancel_confirmed.append(order)
                    orders_to_remove.append(order_id)
                    # SELL + market_reorder → 잔량 MARKET 재주문
                    if order.market_reorder_on_expire and order.remaining_qty > 0:
                        await self._reorder_market(order, result)
                continue

            # TTL 만료 체크
            if order.is_expired and order.status in ("PENDING", "PARTIAL"):
                await self._handle_expiry(order, result)

        for oid in orders_to_remove:
            self._orders.pop(oid, None)

        return result

    async def _handle_expiry(
        self, order: ManagedOrder, result: OrderCheckResult
    ) -> None:
        """TTL 만료 주문 처리: KIS cancel() 호출."""
        logger.info(
            "TTL 만료: %s %s %s %d주 (체결 %d주, 잔량 %d주)",
            order.order_id, order.side, order.symbol,
            order.qty, order.filled_qty, order.remaining_qty,
        )

        try:
            cancel_result = await self._executor.cancel(
                order.order_id, order.symbol, order.remaining_qty, order.price,
            )
            if cancel_result.get("success"):
                order.status = "CANCELLING"
                order.cancel_requested_at = time.monotonic()
                result.expired.append(order)
                logger.info("취소 요청: %s %s", order.order_id, order.symbol)
            else:
                logger.warning(
                    "취소 실패: %s — %s",
                    order.order_id, cancel_result.get("message", ""),
                )
        except Exception as e:
            result.errors.append(f"취소 에러 {order.order_id}: {e}")
            logger.error("취소 에러: %s — %s", order.order_id, e)

    async def _reorder_market(
        self, order: ManagedOrder, result: OrderCheckResult
    ) -> None:
        """취소 확인 후 잔량을 MARKET으로 재주문 (매도 전용)."""
        remaining = order.remaining_qty
        if remaining <= 0:
            return

        logger.info(
            "잔량 MARKET 재주문: %s %s %d주",
            order.symbol, order.side, remaining,
        )

        try:
            sell_result = await self._executor.sell(
                order.symbol, remaining, order_type="MARKET"
            )
            if sell_result.get("success"):
                child_id = sell_result["order_id"]
                order.child_order_id = child_id
                # 자식 주문도 추적
                child = ManagedOrder(
                    order_id=child_id,
                    symbol=order.symbol,
                    side="SELL",
                    order_type="MARKET",
                    qty=remaining,
                    price=0,
                    created_at=time.monotonic(),
                    ttl_seconds=60,
                    reason=f"reorder_from_{order.order_id}",
                )
                self._orders[child_id] = child
                result.reordered.append(child)
            else:
                logger.warning(
                    "잔량 MARKET 재주문 실패: %s — %s",
                    order.symbol, sell_result.get("message", ""),
                )
        except Exception as e:
            result.errors.append(f"재주문 에러 {order.symbol}: {e}")

    # ── 전량 취소 ─────────────────────────────────────────────

    async def cancel_all(self, *, reason: str = "") -> list[ManagedOrder]:
        """미체결 전량 취소 (장 마감 등).

        SELL 주문은 잔량을 MARKET으로 재주문 (포지션 정리 우선).
        """
        cancelled: list[ManagedOrder] = []

        for order_id, order in list(self._orders.items()):
            if order.status not in ("PENDING", "PARTIAL"):
                continue

            try:
                cancel_result = await self._executor.cancel(
                    order.order_id, order.symbol,
                    order.remaining_qty, order.price,
                )
                if cancel_result.get("success"):
                    order.status = "CANCELLING"
                    order.cancel_requested_at = time.monotonic()
                    cancelled.append(order)
                    logger.info(
                        "전량 취소 (%s): %s %s %s 잔량 %d주",
                        reason, order.order_id, order.side,
                        order.symbol, order.remaining_qty,
                    )
            except Exception as e:
                logger.error("전량 취소 에러: %s — %s", order.order_id, e)

        return cancelled

    # ── 조회 ──────────────────────────────────────────────────

    def has_pending(self, symbol: str) -> bool:
        """해당 종목에 활성 주문이 있는지 확인."""
        return any(
            o.symbol == symbol and o.status in ("PENDING", "PARTIAL", "CANCELLING")
            for o in self._orders.values()
        )

    def get_orders(self) -> dict[str, ManagedOrder]:
        return dict(self._orders)

    # ── 영속화 ────────────────────────────────────────────────

    def to_state_dict(self) -> list[dict]:
        """session_state 저장용 직렬화."""
        return [o.to_dict() for o in self._orders.values()]

    def restore_from_state(self, data: list[dict]) -> None:
        """session_state에서 복원. 서버 재시작 시 사용."""
        now = time.monotonic()
        for d in data or []:
            oid = d.get("order_id", "")
            if not oid:
                continue
            order = ManagedOrder(
                order_id=oid,
                symbol=d.get("symbol", ""),
                side=d.get("side", "BUY"),
                order_type=d.get("order_type", "LIMIT"),
                qty=d.get("qty", 0),
                price=d.get("price", 0),
                filled_qty=d.get("filled_qty", 0),
                filled_avg_price=d.get("filled_avg_price", 0.0),
                status=d.get("status", "PENDING"),
                created_at=now,  # 재시작 시점 기준 재시작
                ttl_seconds=d.get("ttl_seconds", 120),
                reason=d.get("reason", ""),
                market_reorder_on_expire=d.get("market_reorder_on_expire", False),
                child_order_id=d.get("child_order_id"),
            )
            self._orders[oid] = order

        if self._orders:
            logger.info("OrderManager: %d개 주문 복원", len(self._orders))


# ── 싱글턴 ────────────────────────────────────────────────

_manager: OrderManager | None = None


def get_order_manager(executor: Any | None = None) -> OrderManager:
    """OrderManager 싱글턴. executor가 없으면 기존 인스턴스 반환."""
    global _manager
    if _manager is None:
        if executor is None:
            raise RuntimeError("OrderManager 초기화 시 executor 필요")
        _manager = OrderManager(executor)
    return _manager


def reset_order_manager() -> None:
    """테스트/재시작용."""
    global _manager
    _manager = None
