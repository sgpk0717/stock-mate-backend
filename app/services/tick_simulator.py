"""개발용 틱/호가 시뮬레이터.

키움 Data Pump 미연결 상태에서 구독된 종목에 대해
랜덤 워크 가격과 호가 데이터를 broadcast.
나중에 ZMQ 연결 시 이 모듈만 교체하면 됨.
"""

import asyncio
import logging
import random
from decimal import Decimal

from app.core.stock_master import get_current_price, update_price
from app.services import paper_engine
from app.services.tick_writer import enqueue_tick
from app.services.ws_manager import manager

logger = logging.getLogger(__name__)

# 시뮬레이터 상태
_running = False


async def simulate_ticks():
    """구독 중인 종목에 대해 틱 데이터 생성 및 broadcast."""
    global _running
    _running = True
    logger.info("Tick simulator started")

    while _running:
        symbols = manager.get_subscribed_symbols("ticks")
        for symbol in symbols:
            price = get_current_price(symbol)
            if price <= 0:
                continue

            # 랜덤 워크: -0.3% ~ +0.3%
            change_pct = (random.random() - 0.48) * 0.006
            new_price = round(float(price) * (1 + change_pct))
            if new_price <= 0:
                new_price = 1

            volume = random.randint(1, 500)

            # 캐시 가격 업데이트
            update_price(symbol, Decimal(str(new_price)))

            await manager.broadcast(f"ticks:{symbol}", {
                "type": "tick",
                "symbol": symbol,
                "price": new_price,
                "volume": volume,
                "change": new_price - float(price),
            })

            # DB에 틱 저장
            await enqueue_tick(symbol, new_price, volume)

            # 모의투자 엔진에 틱 전달
            await paper_engine.on_tick(symbol, new_price)

        await asyncio.sleep(random.uniform(0.5, 1.5))


async def simulate_orderbook():
    """구독 중인 종목에 대해 호가 데이터 생성 및 broadcast."""
    global _running

    while _running:
        symbols = manager.get_subscribed_symbols("orderbook")
        for symbol in symbols:
            price = float(get_current_price(symbol))
            if price <= 0:
                continue

            # 호가 단위 결정
            tick_size = _get_tick_size(price)

            # 현재가 기준으로 10단계 매도/매수 호가 생성
            asks = []
            bids = []
            for i in range(1, 11):
                ask_price = price + tick_size * i
                bid_price = price - tick_size * i
                asks.append({
                    "price": int(ask_price),
                    "volume": random.randint(100, 10000),
                })
                bids.append({
                    "price": max(int(bid_price), tick_size),
                    "volume": random.randint(100, 10000),
                })

            await manager.broadcast(f"orderbook:{symbol}", {
                "type": "orderbook",
                "symbol": symbol,
                "currentPrice": int(price),
                "asks": asks,   # 매도 (가격 오름차순)
                "bids": bids,   # 매수 (가격 내림차순)
            })

        await asyncio.sleep(random.uniform(1.0, 2.0))


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
