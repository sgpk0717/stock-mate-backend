"""ZeroMQ SUB 소켓 — Data Pump로부터 틱/호가 데이터를 수신.

FastAPI lifespan에서 백그라운드 태스크로 실행한다.
수신된 데이터는 WebSocket 브로드캐스트 및 DB 저장에 활용된다.
"""

import asyncio
import json
import logging

import zmq
import zmq.asyncio

from app.core.config import settings

logger = logging.getLogger(__name__)


class ZmqSubscriber:
    """비동기 ZeroMQ SUB 소켓."""

    def __init__(self):
        self._ctx = zmq.asyncio.Context()
        self._socket = self._ctx.socket(zmq.SUB)
        self._address = f"tcp://{settings.ZMQ_HOST}:{settings.ZMQ_PORT}"
        self._running = False
        self._handlers: list = []

    def on_message(self, handler):
        """메시지 수신 핸들러를 등록한다."""
        self._handlers.append(handler)

    async def start(self):
        """SUB 소켓을 연결하고 수신 루프를 시작한다."""
        self._socket.connect(self._address)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")  # 모든 토픽 구독
        self._running = True
        logger.info(f"[ZMQ] SUB 연결: {self._address}")

        while self._running:
            try:
                raw = await asyncio.wait_for(
                    self._socket.recv_string(),
                    timeout=1.0,
                )
                # 메시지 형식: "topic payload_json"
                space_idx = raw.index(" ")
                topic = raw[:space_idx]
                payload = json.loads(raw[space_idx + 1:])

                for handler in self._handlers:
                    try:
                        await handler(topic, payload)
                    except Exception:
                        logger.exception(f"[ZMQ] 핸들러 오류: {topic}")

            except asyncio.TimeoutError:
                continue
            except zmq.ZMQError as e:
                if self._running:
                    logger.error(f"[ZMQ] 소켓 오류: {e}")
                    await asyncio.sleep(1)

    async def stop(self):
        self._running = False
        self._socket.close()
        self._ctx.term()
        logger.info("[ZMQ] SUB 소켓 종료")
