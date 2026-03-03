"""버퍼링 틱 저장 — 주기적으로 bulk insert하여 DB 부하 최소화.

1초마다 또는 버퍼가 500건에 도달하면 flush.
실패 시 버퍼에 재삽입하여 데이터 손실 방지.
"""

import asyncio
import logging
from collections import deque
from datetime import datetime, timezone

from sqlalchemy import text

from app.core.database import async_session

logger = logging.getLogger(__name__)

_buffer: deque[dict] = deque()
_running = False
_FLUSH_INTERVAL = 1.0  # seconds
_FLUSH_THRESHOLD = 500
_MAX_BATCH = 2000


async def enqueue_tick(
    symbol: str, price: int | float, volume: int, ts: datetime | None = None
):
    """버퍼에 틱 추가."""
    _buffer.append({
        "ts": ts or datetime.now(timezone.utc),
        "symbol": symbol,
        "price": price,
        "volume": volume,
    })


async def start_writer():
    """백그라운드 태스크: 주기적으로 버퍼를 DB에 flush."""
    global _running
    _running = True
    logger.info("Tick writer started")

    while _running:
        await asyncio.sleep(_FLUSH_INTERVAL)
        if len(_buffer) >= _FLUSH_THRESHOLD or _buffer:
            await _flush()


async def stop_writer():
    """종료 시 남은 버퍼 flush."""
    global _running
    _running = False
    await _flush()
    logger.info("Tick writer stopped")


async def _flush():
    """버퍼 drain → DB bulk insert."""
    if not _buffer:
        return

    batch: list[dict] = []
    while _buffer and len(batch) < _MAX_BATCH:
        batch.append(_buffer.popleft())

    if not batch:
        return

    try:
        async with async_session() as db:
            await db.execute(
                text(
                    "INSERT INTO stock_ticks (ts, symbol, price, volume) "
                    "VALUES (:ts, :symbol, :price, :volume)"
                ),
                batch,
            )
            await db.commit()
        logger.debug("Flushed %d ticks to DB", len(batch))
    except Exception:
        logger.exception("Failed to flush %d ticks — re-enqueuing", len(batch))
        for item in reversed(batch):
            _buffer.appendleft(item)
