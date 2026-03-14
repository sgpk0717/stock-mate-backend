"""KIS (한국투자증권) REST API 클라이언트.

httpx 비동기 기반, 자동 토큰 갱신 + Rate Limiting.
"""

from __future__ import annotations

import asyncio
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
        self._token_lock = asyncio.Lock()

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
        """Access Token을 발급/갱신하여 반환.

        asyncio.Lock으로 동시 토큰 요청을 직렬화하여
        KIS 1분당 1회 발급 제한(EGW00133) race condition 방지.
        """
        # 빠른 경로: 유효 토큰이 있으면 락 없이 반환
        now = time.time()
        if self._access_token and now < self._token_expires_at - 300:
            return self._access_token

        async with self._token_lock:
            # Double-check: 락 획득 사이에 다른 코루틴이 갱신했을 수 있음
            now = time.time()
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

    async def inquire_program_trading(self, symbol: str) -> dict[str, Any]:
        """주식현재가 프로그램매매 조회 (FHKST01010600).

        실전 전용 (모의투자 미지원).

        Returns:
            프로그램 매수/매도 수량 및 금액
        """
        data = await self._get(
            "/uapi/domestic-stock/v1/quotations/inquire-price-pgm",
            tr_id="FHKST01010600",
            params={
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": symbol,
            },
        )
        output = data.get("output", {})
        return {
            "pgm_buy_qty": int(output.get("pgtr_ntby_qty", "0")),   # 프로그램 순매수 수량
            "pgm_sell_qty": int(output.get("pgtr_ntsl_qty", "0")),   # 프로그램 순매도 수량
            "pgm_net_qty": int(output.get("pgtr_ntby_qty", "0")) - int(output.get("pgtr_ntsl_qty", "0")),
            "pgm_buy_amount": int(output.get("pgtr_ntby_tr_pbmn", "0")),
            "pgm_sell_amount": int(output.get("pgtr_ntsl_tr_pbmn", "0")),
            "pgm_net_amount": int(output.get("pgtr_ntby_tr_pbmn", "0")) - int(output.get("pgtr_ntsl_tr_pbmn", "0")),
        }

    async def inquire_daily_short_sale(
        self, symbol: str, start_date: str, end_date: str,
    ) -> list[dict[str, Any]]:
        """국내주식 공매도 일별추이 (FHPST04830000).

        Args:
            symbol: 종목코드 (예: "005930")
            start_date: 시작일자 YYYYMMDD
            end_date: 종료일자 YYYYMMDD

        Returns:
            일별 공매도 데이터 리스트
        """
        data = await self._get(
            "/uapi/domestic-stock/v1/quotations/daily-short-sale",
            tr_id="FHPST04830000",
            params={
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": symbol,
                "FID_INPUT_DATE_1": start_date,
                "FID_INPUT_DATE_2": end_date,
            },
        )
        rows = []
        for item in data.get("output2", []):
            dt_str = item.get("stck_bsop_date", "")
            if not dt_str:
                continue
            rows.append({
                "dt": dt_str,
                "short_volume": int(item.get("ssts_cntg_qty", "0")),
                "short_volume_ratio": float(item.get("ssts_vol_rlim", "0")),
                "short_amount": int(item.get("ssts_tr_pbmn", "0")),
                "total_volume": int(item.get("acml_vol", "0")),
            })
        return rows

    async def inquire_daily_credit_balance(
        self, symbol: str, settle_date: str,
    ) -> list[dict[str, Any]]:
        """국내주식 신용잔고 일별추이 (FHPST04760000).

        최대 30건/회. 거래일자 기준 과거 방향 조회.

        Args:
            symbol: 종목코드 (예: "005930")
            settle_date: 기준일자 YYYYMMDD

        Returns:
            일별 신용잔고 데이터 리스트
        """
        data = await self._get(
            "/uapi/domestic-stock/v1/quotations/daily-credit-balance",
            tr_id="FHPST04760000",
            params={
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_COND_SCR_DIV_CODE": "20476",
                "FID_INPUT_ISCD": symbol,
                "FID_INPUT_DATE_1": settle_date,
            },
        )
        rows = []
        for item in data.get("output", []):
            dt_str = item.get("deal_date", "")
            if not dt_str:
                continue
            rows.append({
                "dt": dt_str,
                "margin_balance": int(item.get("whol_loan_rmnd_stcn", "0") or "0"),
                "margin_rate": float(item.get("whol_loan_rmnd_rate", "0") or "0"),
            })
        return rows

    async def inquire_daily_investor(
        self, symbol: str, date: str,
    ) -> list[dict[str, Any]]:
        """종목별 투자자매매동향(일별) (FHPTJ04160001).

        Args:
            symbol: 종목코드 (예: "005930")
            date: 기준일자 YYYYMMDD

        Returns:
            투자자별 매수/매도/순매수 데이터 리스트
        """
        data = await self._get(
            "/uapi/domestic-stock/v1/quotations/investor-trade-by-stock-daily",
            tr_id="FHPTJ04160001",
            params={
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": symbol,
                "FID_INPUT_DATE_1": date,
                "FID_ORG_ADJ_PRC": "",
                "FID_ETC_CLS_CODE": "",
            },
        )

        # output1 = 종목 기본 시세 (dict), output2 = 투자자별 매매 (list)
        output2 = data.get("output2")
        if not output2 or not isinstance(output2, list):
            return []

        # 첫 호출 시 필드명 확인용 로깅 (DEBUG 레벨)
        if output2 and isinstance(output2[0], dict):
            logger.debug("inquire_daily_investor raw keys: %s", list(output2[0].keys()))

        rows = []
        for item in output2:
            if not isinstance(item, dict):
                continue
            dt_str = item.get("stck_bsop_date", "")
            if not dt_str:
                continue
            rows.append({
                "dt": dt_str,
                # 외국인
                "frgn_buy_vol": int(item.get("frgn_shnu_vol", "0") or "0"),
                "frgn_sell_vol": int(item.get("frgn_seln_vol", "0") or "0"),
                "frgn_net": int(item.get("frgn_ntby_qty", "0") or "0"),
                # 기관
                "orgn_buy_vol": int(item.get("orgn_shnu_vol", "0") or "0"),
                "orgn_sell_vol": int(item.get("orgn_seln_vol", "0") or "0"),
                "orgn_net": int(item.get("orgn_ntby_qty", "0") or "0"),
                # 개인
                "prsn_buy_vol": int(item.get("prsn_shnu_vol", "0") or "0"),
                "prsn_sell_vol": int(item.get("prsn_seln_vol", "0") or "0"),
                "prsn_net": int(item.get("prsn_ntby_qty", "0") or "0"),
            })
        return rows

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
