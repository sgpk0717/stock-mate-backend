"""KIS 주문 실행기 — 매수/매도/정정/취소.

한국투자증권 REST API를 통한 국내 주식 주문.
"""

from __future__ import annotations

import logging
from typing import Any

from .kis_client import KISClient

logger = logging.getLogger(__name__)


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
