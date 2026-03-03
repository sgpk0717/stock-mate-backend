"""KIS (한국투자증권) REST API 클라이언트.

httpx 비동기 기반, 자동 토큰 갱신 + Rate Limiting.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from app.core.config import settings
from .token_bucket import kis_rate_limiter

logger = logging.getLogger(__name__)


class KISClient:
    """한국투자증권 Open API 클라이언트."""

    def __init__(self, is_mock: bool = True):
        """
        Args:
            is_mock: True = 모의투자 서버(VTS), False = 실전 서버
        """
        self.is_mock = is_mock
        self.base_url = settings.KIS_MOCK_URL if is_mock else settings.KIS_BASE_URL
        self.app_key = settings.KIS_APP_KEY
        self.app_secret = settings.KIS_APP_SECRET
        self.account_no = settings.KIS_ACCOUNT_NO

        # 계좌 파싱: XXXXXXXX-XX
        parts = self.account_no.split("-") if "-" in self.account_no else [self.account_no[:8], self.account_no[8:]]
        self.cano = parts[0]  # 종합계좌번호 (8자리)
        self.acnt_prdt_cd = parts[1] if len(parts) > 1 else "01"  # 계좌상품코드 (2자리)

        # 토큰 관리
        self._access_token: str = ""
        self._token_expires_at: float = 0.0

        self._client: httpx.AsyncClient | None = None

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(30.0),
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ── 토큰 관리 ──────────────────────────────────────────

    async def _get_token(self) -> str:
        """Access Token을 발급/갱신하여 반환."""
        now = time.time()
        # 유효기간 5분 전에 갱신
        if self._access_token and now < self._token_expires_at - 300:
            return self._access_token

        client = await self._ensure_client()
        resp = await client.post(
            "/oauth2/tokenP",
            json={
                "grant_type": "client_credentials",
                "appkey": self.app_key,
                "appsecret": self.app_secret,
            },
        )
        resp.raise_for_status()
        data = resp.json()

        self._access_token = data["access_token"]
        # KIS 토큰 유효기간: 약 24시간 (86400초)
        expires_in = data.get("expires_in", 86400)
        self._token_expires_at = now + expires_in

        logger.info("KIS 토큰 발급 완료 (mock=%s, expires_in=%ds)", self.is_mock, expires_in)
        return self._access_token

    async def _headers(self, tr_id: str) -> dict[str, str]:
        """공통 헤더 구성."""
        token = await self._get_token()
        return {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": tr_id,
        }

    # ── 공통 요청 ──────────────────────────────────────────

    async def _get(self, path: str, tr_id: str, params: dict | None = None) -> dict[str, Any]:
        """GET 요청 + Rate Limiting."""
        await kis_rate_limiter.acquire()
        client = await self._ensure_client()
        headers = await self._headers(tr_id)
        resp = await client.get(path, params=params, headers=headers)
        resp.raise_for_status()
        return resp.json()

    async def _post(self, path: str, tr_id: str, body: dict | None = None) -> dict[str, Any]:
        """POST 요청 + Rate Limiting."""
        await kis_rate_limiter.acquire()
        client = await self._ensure_client()
        headers = await self._headers(tr_id)
        resp = await client.post(path, json=body or {}, headers=headers)
        resp.raise_for_status()
        return resp.json()

    # ── 계좌 조회 ──────────────────────────────────────────

    async def inquire_balance(self) -> dict[str, Any]:
        """주식 잔고 조회.

        모의: VTTC8434R, 실전: TTTC8434R
        """
        tr_id = "VTTC8434R" if self.is_mock else "TTTC8434R"
        params = {
            "CANO": self.cano,
            "ACNT_PRDT_CD": self.acnt_prdt_cd,
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "02",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "01",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        }
        data = await self._get("/uapi/domestic-stock/v1/trading/inquire-balance", tr_id, params)

        positions = []
        for item in data.get("output1", []):
            if int(item.get("hldg_qty", "0")) > 0:
                positions.append({
                    "symbol": item["pdno"],
                    "name": item.get("prdt_name", ""),
                    "qty": int(item["hldg_qty"]),
                    "avg_price": float(item.get("pchs_avg_pric", "0")),
                    "current_price": float(item.get("prpr", "0")),
                    "pnl": float(item.get("evlu_pfls_amt", "0")),
                    "pnl_pct": float(item.get("evlu_pfls_rt", "0")),
                })

        summary = data.get("output2", [{}])
        account_info = summary[0] if summary else {}
        account = {
            "total_eval": float(account_info.get("sma_evlu_amt", "0")),
            "total_pnl": float(account_info.get("evlu_pfls_smtl_amt", "0")),
            "cash": float(account_info.get("dnca_tot_amt", "0")),
            "total_deposit": float(account_info.get("tot_evlu_amt", "0")),
        }

        return {"positions": positions, "account": account}

    async def inquire_daily_ccld(self) -> list[dict]:
        """당일 체결 내역 조회.

        모의: VTTC8001R, 실전: TTTC8001R
        """
        tr_id = "VTTC8001R" if self.is_mock else "TTTC8001R"
        import datetime as dt
        today = dt.date.today().strftime("%Y%m%d")

        params = {
            "CANO": self.cano,
            "ACNT_PRDT_CD": self.acnt_prdt_cd,
            "INQR_STRT_DT": today,
            "INQR_END_DT": today,
            "SLL_BUY_DVSN_CD": "00",
            "INQR_DVSN": "00",
            "PDNO": "",
            "CCLD_DVSN": "00",
            "ORD_GNO_BRNO": "",
            "ODNO": "",
            "INQR_DVSN_3": "00",
            "INQR_DVSN_1": "",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        }
        data = await self._get("/uapi/domestic-stock/v1/trading/inquire-daily-ccld", tr_id, params)

        orders = []
        for item in data.get("output1", []):
            orders.append({
                "order_id": item.get("odno", ""),
                "symbol": item.get("pdno", ""),
                "name": item.get("prdt_name", ""),
                "side": "BUY" if item.get("sll_buy_dvsn_cd") == "02" else "SELL",
                "price": float(item.get("avg_prvs", "0")),
                "qty": int(item.get("tot_ccld_qty", "0")),
                "order_qty": int(item.get("ord_qty", "0")),
                "status": "FILLED" if int(item.get("tot_ccld_qty", "0")) > 0 else "PENDING",
                "order_time": item.get("ord_tmd", ""),
            })

        return orders

    # ── 시세 조회 ──────────────────────────────────────────

    async def get_minute_candles(
        self, symbol: str, date: str, hour: str = "160000",
    ) -> tuple[list[dict], str, str]:
        """주식일별분봉조회 (inquire_time_dailychartprice).

        과거 분봉 데이터를 역순으로 조회한다 (최대 120건/요청, 1년).

        Args:
            symbol: 종목코드 (예: "005930")
            date: 기준일자 YYYYMMDD
            hour: 기준시각 HHMMSS (기본 "160000" = 장마감 후부터)

        Returns:
            (candles, next_date, next_hour) — 다음 페이지 호출용.
            candles가 빈 리스트면 더 이상 데이터 없음.
        """
        resp = await self._get(
            "/uapi/domestic-stock/v1/quotations/inquire-time-dailychartprice",
            tr_id="FHKST03010230",
            params={
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": symbol,
                "FID_INPUT_DATE_1": date,
                "FID_INPUT_HOUR_1": hour,
                "FID_PW_DATA_INCU_YN": "Y",
                "FID_FAKE_TICK_INCU_YN": "N",
            },
        )
        candles = resp.get("output2", [])
        if not candles:
            return [], "", ""
        last = candles[-1]
        return candles, last.get("stck_bsop_date", ""), last.get("stck_cntg_hour", "")

    async def inquire_psbl_order(self, symbol: str, price: int = 0) -> dict:
        """매수 가능 수량 조회.

        모의: VTTC8908R, 실전: TTTC8908R
        """
        tr_id = "VTTC8908R" if self.is_mock else "TTTC8908R"
        params = {
            "CANO": self.cano,
            "ACNT_PRDT_CD": self.acnt_prdt_cd,
            "PDNO": symbol,
            "ORD_UNPR": str(price),
            "ORD_DVSN": "01",  # 지정가
            "CMA_EVLU_AMT_ICLD_YN": "N",
            "OVRS_ICLD_YN": "N",
        }
        data = await self._get("/uapi/domestic-stock/v1/trading/inquire-psbl-order", tr_id, params)
        output = data.get("output", {})
        return {
            "max_buy_qty": int(output.get("nrcvb_buy_qty", "0")),
            "max_buy_amount": float(output.get("nrcvb_buy_amt", "0")),
            "cash": float(output.get("dnca_ord_psbl_amt", "0")),
        }


# 싱글톤 클라이언트 (모의투자)
_mock_client: KISClient | None = None
_real_client: KISClient | None = None


def get_kis_client(is_mock: bool = True) -> KISClient:
    """KIS 클라이언트 싱글톤."""
    global _mock_client, _real_client
    if is_mock:
        if _mock_client is None:
            _mock_client = KISClient(is_mock=True)
        return _mock_client
    else:
        if _real_client is None:
            _real_client = KISClient(is_mock=False)
        return _real_client
