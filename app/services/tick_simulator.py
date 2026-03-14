"""KIS API 기반 실시간 시세 폴링.

키움 Data Pump 미연결 상태에서 구독된 종목에 대해
KIS REST API로 현재가/호가를 폴링하여 broadcast.
"""

import asyncio
import logging
import random
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from app.core.config import settings
from app.core.stock_master import get_current_price, update_price
from app.services import paper_engine
from app.services.tick_writer import enqueue_tick
from app.services.ws_manager import manager

logger = logging.getLogger(__name__)

# 시뮬레이터 상태
_running = False
_prev_close: dict[str, int] = {}  # 종목별 전일종가
_KST = timezone(timedelta(hours=9))

# KIS API 직접 호출용 (kis_client 싱글톤과 별도)
_kis_http: "httpx.AsyncClient | None" = None
_kis_token: str = ""
_kis_token_expires: float = 0.0


async def _ensure_kis() -> "httpx.AsyncClient":
    """KIS httpx 클라이언트 + 토큰 보장."""
    import httpx
    import time

    global _kis_http, _kis_token, _kis_token_expires

    if _kis_http is None or _kis_http.is_closed:
        _kis_http = httpx.AsyncClient(
            base_url=settings.KIS_BASE_URL,
            timeout=httpx.Timeout(10.0),
        )

    now = time.time()
    if not _kis_token or now >= _kis_token_expires - 300:
        resp = await _kis_http.post(
            "/oauth2/tokenP",
            json={
                "grant_type": "client_credentials",
                "appkey": settings.KIS_APP_KEY,
                "appsecret": settings.KIS_APP_SECRET,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        _kis_token = data["access_token"]
        _kis_token_expires = now + data.get("expires_in", 86400)
        logger.info("KIS 시세 폴링 토큰 발급 완료")

    return _kis_http


def _kis_headers(tr_id: str) -> dict[str, str]:
    return {
        "content-type": "application/json; charset=utf-8",
        "authorization": f"Bearer {_kis_token}",
        "appkey": settings.KIS_APP_KEY,
        "appsecret": settings.KIS_APP_SECRET,
        "tr_id": tr_id,
    }


async def _fetch_price(symbol: str) -> dict | None:
    """KIS 주식현재가 시세 조회 (FHKST01010100)."""
    try:
        client = await _ensure_kis()
        resp = await client.get(
            "/uapi/domestic-stock/v1/quotations/inquire-price",
            params={
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": symbol,
            },
            headers=_kis_headers("FHKST01010100"),
        )
        resp.raise_for_status()
        return resp.json().get("output", {})
    except Exception as e:
        logger.warning("KIS 시세 조회 실패 (%s): %s", symbol, e)
        return None


async def _fetch_orderbook(symbol: str) -> dict | None:
    """KIS 주식현재가 호가/예상체결 조회 (FHKST01010200)."""
    try:
        client = await _ensure_kis()
        resp = await client.get(
            "/uapi/domestic-stock/v1/quotations/inquire-asking-price-exp-ccn",
            params={
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": symbol,
            },
            headers=_kis_headers("FHKST01010200"),
        )
        resp.raise_for_status()
        return resp.json().get("output1", {})
    except Exception as e:
        logger.warning("KIS 호가 조회 실패 (%s): %s", symbol, e)
        return None


async def simulate_ticks():
    """구독 중인 종목에 대해 KIS API로 실시간 시세 폴링 및 broadcast."""
    global _running
    _running = True
    _prev_close.clear()
    logger.info("KIS 실시간 시세 폴링 시작")

    while _running:
        symbols = manager.get_subscribed_symbols("ticks")
        for symbol in symbols:
            data = await _fetch_price(symbol)
            if not data:
                continue

            try:
                new_price = int(data.get("stck_prpr", "0"))
                if new_price <= 0:
                    continue

                # 전일종가 저장
                if symbol not in _prev_close:
                    prev = int(data.get("stck_sdpr", "0"))
                    _prev_close[symbol] = prev if prev > 0 else new_price

                volume = int(data.get("cntg_vol", "0"))
                if volume <= 0:
                    volume = int(data.get("acml_vol", "0"))

                change = new_price - _prev_close[symbol]

                # 캐시 가격 업데이트
                update_price(symbol, Decimal(str(new_price)))

                await manager.broadcast(f"ticks:{symbol}", {
                    "type": "tick",
                    "symbol": symbol,
                    "price": new_price,
                    "volume": volume,
                    "change": change,
                })

                # DB에 틱 저장 (장중만: 09:00~15:30 KST)
                now_kst = datetime.now(_KST)
                if 9 <= now_kst.hour < 16:
                    await enqueue_tick(symbol, new_price, volume)

                # 모의투자 엔진에 틱 전달
                await paper_engine.on_tick(symbol, new_price)

            except (ValueError, KeyError) as e:
                logger.warning("시세 파싱 실패 (%s): %s", symbol, e)

        # KIS API rate limit 고려 (초당 15건)
        await asyncio.sleep(2.0)


async def simulate_orderbook():
    """구독 중인 종목에 대해 KIS API로 호가 폴링 및 broadcast."""
    global _running

    while _running:
        symbols = manager.get_subscribed_symbols("orderbook")
        for symbol in symbols:
            data = await _fetch_orderbook(symbol)
            if not data:
                continue

            try:
                current_price = int(get_current_price(symbol))
                if current_price <= 0:
                    continue

                asks = []
                bids = []
                for i in range(1, 11):
                    ask_p = int(data.get(f"askp{i}", "0"))
                    ask_v = int(data.get(f"askp_rsqn{i}", "0"))
                    bid_p = int(data.get(f"bidp{i}", "0"))
                    bid_v = int(data.get(f"bidp_rsqn{i}", "0"))

                    if ask_p > 0:
                        asks.append({"price": ask_p, "volume": ask_v})
                    if bid_p > 0:
                        bids.append({"price": bid_p, "volume": bid_v})

                await manager.broadcast(f"orderbook:{symbol}", {
                    "type": "orderbook",
                    "symbol": symbol,
                    "currentPrice": current_price,
                    "asks": asks,
                    "bids": bids,
                })

            except (ValueError, KeyError) as e:
                logger.warning("호가 파싱 실패 (%s): %s", symbol, e)

        await asyncio.sleep(3.0)


def _get_tick_size(price: float) -> int:
    """가격대별 호가 단위 (KRX 규정)."""
    if price < 2000:
        return 1
    elif price < 5000:
        return 5
    elif price < 20000:
        return 10
    elif price < 50000:
        return 50
    elif price < 200000:
        return 100
    elif price < 500000:
        return 500
    else:
        return 1000


def stop_simulator():
    global _running
    _running = False
