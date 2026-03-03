"""WebSocket 종목별 구독 관리.

종목 코드로 구독/해제, 특정 종목 구독자에게 broadcast.
"""

import asyncio
import logging
from collections import defaultdict

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """종목별 WebSocket 구독 관리."""

    def __init__(self):
        # channel → set of websockets  (channel = "ticks:005930", "orderbook:005930")
        self._subscriptions: dict[str, set[WebSocket]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def subscribe(self, ws: WebSocket, channel: str):
        async with self._lock:
            self._subscriptions[channel].add(ws)
        logger.debug("subscribe %s  (total=%d)", channel, len(self._subscriptions[channel]))

    async def unsubscribe(self, ws: WebSocket, channel: str):
        async with self._lock:
            self._subscriptions[channel].discard(ws)
            if not self._subscriptions[channel]:
                del self._subscriptions[channel]

    async def unsubscribe_all(self, ws: WebSocket):
        """연결 끊긴 ws를 모든 채널에서 제거."""
        async with self._lock:
            empty_channels = []
            for channel, sockets in self._subscriptions.items():
                sockets.discard(ws)
                if not sockets:
                    empty_channels.append(channel)
            for ch in empty_channels:
                del self._subscriptions[ch]

    async def broadcast(self, channel: str, data: dict):
        """특정 채널의 모든 구독자에게 JSON 전송."""
        async with self._lock:
            sockets = list(self._subscriptions.get(channel, []))

        dead: list[WebSocket] = []
        for ws in sockets:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)

        # dead connection 정리
        if dead:
            async with self._lock:
                for ws in dead:
                    self._subscriptions[channel].discard(ws)
                if not self._subscriptions.get(channel):
                    self._subscriptions.pop(channel, None)

    def get_subscribed_symbols(self, prefix: str = "ticks") -> set[str]:
        """현재 구독 중인 종목 심볼 목록 반환."""
        symbols = set()
        for channel in self._subscriptions:
            if channel.startswith(f"{prefix}:"):
                symbols.add(channel.split(":", 1)[1])
        return symbols


# 싱글턴 인스턴스
manager = ConnectionManager()
