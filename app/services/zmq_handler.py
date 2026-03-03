"""ZMQ 메시지 핸들러 — Data Pump 수신 데이터를 WebSocket으로 브로드캐스트.

Data Pump(키움)에서 ZMQ로 수신한 tick/quote 메시지를
프론트엔드가 기대하는 RealtimeTick / RealtimeOrderBook 형태로 변환하여
WebSocket 채널에 broadcast한다.
"""

import logging
from decimal import Decimal

from app.core.stock_master import get_current_price, update_price
from app.services import paper_engine
from app.services.tick_writer import enqueue_tick
from app.services.ws_manager import manager

logger = logging.getLogger(__name__)


async def handle_zmq_message(topic: str, payload: dict):
    """ZMQ 토픽 메시지를 파싱하여 적절한 WebSocket 채널에 broadcast."""
    try:
        msg_type, symbol = topic.split(".", 1)
    except ValueError:
        logger.warning(f"[ZMQ] 알 수 없는 토픽: {topic}")
        return

    if msg_type == "tick":
        await _handle_tick(symbol, payload)
    elif msg_type == "quote":
        await _handle_quote(symbol, payload)
    elif msg_type == "candle":
        await _handle_candle(symbol, payload)
    else:
        logger.debug(f"[ZMQ] 무시된 토픽: {topic}")


async def _handle_tick(symbol: str, payload: dict):
    """틱 데이터 → ticks:{symbol} 채널 broadcast + 가격 캐시 업데이트."""
    price = payload.get("price", 0)
    if price <= 0:
        return

    # 가격 캐시 업데이트
    update_price(symbol, Decimal(str(price)))

    # 프론트엔드 RealtimeTick 형태로 broadcast
    await manager.broadcast(f"ticks:{symbol}", {
        "type": "tick",
        "symbol": symbol,
        "price": price,
        "volume": payload.get("volume", 0),
        "change": payload.get("change", 0),
    })

    # DB에 틱 저장
    await enqueue_tick(symbol, price, payload.get("volume", 0))

    # 모의투자 엔진에 틱 전달 (미체결 주문 매칭)
    await paper_engine.on_tick(symbol, price)


async def _handle_quote(symbol: str, payload: dict):
    """호가 데이터 → orderbook:{symbol} 채널 broadcast."""
    asks = payload.get("asks", [])
    bids = payload.get("bids", [])

    if not asks and not bids:
        return

    # 현재가: 캐시에서 조회 (틱이 먼저 오므로 보통 존재)
    current_price = float(get_current_price(symbol) or 0)

    # 프론트엔드 RealtimeOrderBook 형태로 broadcast
    await manager.broadcast(f"orderbook:{symbol}", {
        "type": "orderbook",
        "symbol": symbol,
        "currentPrice": current_price,
        "asks": asks,
        "bids": bids,
    })


async def _handle_candle(symbol: str, payload: dict):
    """과거 캔들 데이터 → stock_candles 테이블 저장."""
    from app.services.candle_writer import write_candle

    await write_candle(symbol, payload)
