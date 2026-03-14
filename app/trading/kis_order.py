"""KIS 주문 실행기 — 매수/매도/정정/취소 + 체결 추적.

한국투자증권 REST API를 통한 국내 주식 주문.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .kis_client import KISClient

logger = logging.getLogger(__name__)


# ── 미체결 주문 추적 ────────────────────────────────────────
# key: order_id, value: {"symbol", "side", "qty", "price", "order_time", "retries"}
_pending_orders: dict[str, dict[str, Any]] = {}


def get_pending_orders() -> dict[str, dict[str, Any]]:
    """현재 미체결 주문 목록."""
    return dict(_pending_orders)


def has_pending_order(symbol: str) -> bool:
    """해당 종목에 미체결 주문이 있는지 확인."""
    return any(o["symbol"] == symbol for o in _pending_orders.values())


class KISOrderExecutor:
    """KIS API 주문 실행."""

    def __init__(self, client: KISClient):
        self.client = client

    async def buy(
        self,
        symbol: str,
        qty: int,
        price: int = 0,
        order_type: str = "LIMIT",
    ) -> dict[str, Any]:
        """매수 주문.

        Args:
            symbol: 종목코드 (6자리)
            qty: 수량
            price: 주문가 (시장가일 때 0)
            order_type: "LIMIT" | "MARKET"

        Returns:
            {"order_id": str, "success": bool, ...}
        """
        # 모의: VTTC0802U, 실전: TTTC0802U
        tr_id = "VTTC0802U" if self.client.is_mock else "TTTC0802U"

        # 주문구분: 00=지정가, 01=시장가
        ord_dvsn = "01" if order_type == "MARKET" else "00"

        body = {
            "CANO": self.client.cano,
            "ACNT_PRDT_CD": self.client.acnt_prdt_cd,
            "PDNO": symbol,
            "ORD_DVSN": ord_dvsn,
            "ORD_QTY": str(qty),
            "ORD_UNPR": str(price) if order_type == "LIMIT" else "0",
        }

        try:
            data = await self.client._post(
                "/uapi/domestic-stock/v1/trading/order-cash",
                tr_id,
                body,
            )
            output = data.get("output", {})
            rt_cd = data.get("rt_cd", "1")
            msg = data.get("msg1", "")

            result = {
                "success": rt_cd == "0",
                "order_id": output.get("ODNO", ""),
                "order_time": output.get("ORD_TMD", ""),
                "message": msg,
            }
            if rt_cd == "0":
                logger.info("매수 주문 성공: %s %d주 @ %s", symbol, qty, price or "시장가")
                # 미체결 추적 등록
                order_id = output.get("ODNO", "")
                if order_id:
                    _pending_orders[order_id] = {
                        "symbol": symbol, "side": "BUY", "qty": qty,
                        "price": price, "order_time": output.get("ORD_TMD", ""),
                        "retries": 0,
                    }
            else:
                logger.warning("매수 주문 실패: %s — %s", symbol, msg)
            return result

        except Exception as e:
            logger.error("매수 주문 에러: %s — %s", symbol, e)
            return {"success": False, "order_id": "", "message": str(e)}

    async def sell(
        self,
        symbol: str,
        qty: int,
        price: int = 0,
        order_type: str = "LIMIT",
    ) -> dict[str, Any]:
        """매도 주문.

        모의: VTTC0801U, 실전: TTTC0801U
        """
        tr_id = "VTTC0801U" if self.client.is_mock else "TTTC0801U"
        ord_dvsn = "01" if order_type == "MARKET" else "00"

        body = {
            "CANO": self.client.cano,
            "ACNT_PRDT_CD": self.client.acnt_prdt_cd,
            "PDNO": symbol,
            "ORD_DVSN": ord_dvsn,
            "ORD_QTY": str(qty),
            "ORD_UNPR": str(price) if order_type == "LIMIT" else "0",
        }

        try:
            data = await self.client._post(
                "/uapi/domestic-stock/v1/trading/order-cash",
                tr_id,
                body,
            )
            output = data.get("output", {})
            rt_cd = data.get("rt_cd", "1")
            msg = data.get("msg1", "")

            result = {
                "success": rt_cd == "0",
                "order_id": output.get("ODNO", ""),
                "order_time": output.get("ORD_TMD", ""),
                "message": msg,
            }
            if rt_cd == "0":
                logger.info("매도 주문 성공: %s %d주 @ %s", symbol, qty, price or "시장가")
                order_id = output.get("ODNO", "")
                if order_id:
                    _pending_orders[order_id] = {
                        "symbol": symbol, "side": "SELL", "qty": qty,
                        "price": price, "order_time": output.get("ORD_TMD", ""),
                        "retries": 0,
                    }
            else:
                logger.warning("매도 주문 실패: %s — %s", symbol, msg)
            return result

        except Exception as e:
            logger.error("매도 주문 에러: %s — %s", symbol, e)
            return {"success": False, "order_id": "", "message": str(e)}

    async def cancel(
        self,
        order_id: str,
        symbol: str,
        qty: int,
        price: int = 0,
    ) -> dict[str, Any]:
        """주문 취소.

        모의: VTTC0803U, 실전: TTTC0803U
        """
        tr_id = "VTTC0803U" if self.client.is_mock else "TTTC0803U"

        body = {
            "CANO": self.client.cano,
            "ACNT_PRDT_CD": self.client.acnt_prdt_cd,
            "KRX_FWDG_ORD_ORGNO": "",
            "ORGN_ODNO": order_id,
            "ORD_DVSN": "00",
            "RVSE_CNCL_DVSN_CD": "02",  # 02=취소
            "ORD_QTY": str(qty),
            "ORD_UNPR": str(price),
            "QTY_ALL_ORD_YN": "Y",
        }

        try:
            data = await self.client._post(
                "/uapi/domestic-stock/v1/trading/order-rvsecncl",
                tr_id,
                body,
            )
            rt_cd = data.get("rt_cd", "1")
            msg = data.get("msg1", "")
            return {
                "success": rt_cd == "0",
                "message": msg,
            }
        except Exception as e:
            logger.error("주문 취소 에러: %s — %s", order_id, e)
            return {"success": False, "message": str(e)}

    async def modify(
        self,
        order_id: str,
        symbol: str,
        qty: int,
        new_price: int,
    ) -> dict[str, Any]:
        """주문 정정 (가격 변경).

        모의: VTTC0803U, 실전: TTTC0803U
        """
        tr_id = "VTTC0803U" if self.client.is_mock else "TTTC0803U"

        body = {
            "CANO": self.client.cano,
            "ACNT_PRDT_CD": self.client.acnt_prdt_cd,
            "KRX_FWDG_ORD_ORGNO": "",
            "ORGN_ODNO": order_id,
            "ORD_DVSN": "00",  # 지정가
            "RVSE_CNCL_DVSN_CD": "01",  # 01=정정
            "ORD_QTY": str(qty),
            "ORD_UNPR": str(new_price),
            "QTY_ALL_ORD_YN": "N",
        }

        try:
            data = await self.client._post(
                "/uapi/domestic-stock/v1/trading/order-rvsecncl",
                tr_id,
                body,
            )
            rt_cd = data.get("rt_cd", "1")
            msg = data.get("msg1", "")
            return {
                "success": rt_cd == "0",
                "message": msg,
            }
        except Exception as e:
            logger.error("주문 정정 에러: %s — %s", order_id, e)
            return {"success": False, "message": str(e)}

    async def check_pending_orders(self) -> list[dict[str, Any]]:
        """미체결 주문 체결 상태 확인 + 완료된 주문 제거.

        KIS 당일 체결 내역(inquire_daily_ccld)을 조회하여
        _pending_orders에서 체결 완료된 주문을 제거한다.

        Returns:
            체결 확인된 주문 목록.
        """
        if not _pending_orders:
            return []

        try:
            ccld_list = await self.client.inquire_daily_ccld()
        except Exception as e:
            logger.warning("체결 조회 실패: %s", e)
            return []

        # order_id → 체결 정보 매핑
        ccld_map: dict[str, dict] = {}
        for item in ccld_list:
            oid = item.get("order_id", "")
            if oid:
                ccld_map[oid] = item

        filled: list[dict[str, Any]] = []
        expired_ids: list[str] = []

        for order_id, info in list(_pending_orders.items()):
            ccld = ccld_map.get(order_id)
            if ccld and ccld.get("status") == "FILLED":
                filled.append({
                    "order_id": order_id,
                    "symbol": info["symbol"],
                    "side": info["side"],
                    "qty": ccld.get("qty", info["qty"]),
                    "price": ccld.get("price", info["price"]),
                })
                expired_ids.append(order_id)
                logger.info(
                    "체결 확인: %s %s %s %d주",
                    order_id, info["side"], info["symbol"], info["qty"],
                )
            else:
                # 재시도 카운터 증가 (최대 3회 = 90초 후 포기)
                info["retries"] = info.get("retries", 0) + 1
                if info["retries"] >= 3:
                    expired_ids.append(order_id)
                    logger.warning(
                        "미체결 만료: %s %s %s (3회 조회 후 미체결)",
                        order_id, info["side"], info["symbol"],
                    )

        for oid in expired_ids:
            _pending_orders.pop(oid, None)

        return filled
