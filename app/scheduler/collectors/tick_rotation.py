"""틱 순환 스케줄 JSON 생성.

장 시작 전(08:50 KST)에 전 종목을 배치로 분할한 스케줄 파일을 생성한다.
Data Pump(32-bit)이 이 파일을 읽어 QTimer 기반으로 실시간 수신 종목을 순환한다.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import text

from app.core.config import settings
from app.core.database import async_session

logger = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9))


async def _get_all_symbols() -> list[str]:
    """stock_masters에서 전 종목 코드 조회."""
    async with async_session() as db:
        result = await db.execute(
            text("SELECT symbol FROM stock_masters ORDER BY symbol"),
        )
        return [r[0] for r in result.fetchall()]


def _split_batches(symbols: list[str], batch_size: int) -> list[list[str]]:
    """종목 리스트를 batch_size 단위로 분할."""
    return [
        symbols[i : i + batch_size]
        for i in range(0, len(symbols), batch_size)
    ]


async def generate_tick_schedule() -> Path:
    """틱 순환 스케줄 JSON 생성.

    Returns:
        생성된 JSON 파일 경로.
    """
    symbols = await _get_all_symbols()
    batch_size = settings.TICK_ROTATION_BATCH_SIZE
    interval_min = settings.TICK_ROTATION_INTERVAL_MIN

    batches = _split_batches(symbols, batch_size)

    schedule = {
        "batches": batches,
        "batch_count": len(batches),
        "total_symbols": len(symbols),
        "batch_size": batch_size,
        "interval_minutes": interval_min,
        "generated_at": datetime.now(KST).isoformat(),
    }

    out_path = Path(settings.TICK_ROTATION_SCHEDULE_FILE)
    out_path.write_text(json.dumps(schedule, ensure_ascii=False, indent=2))

    logger.info(
        "틱 순환 스케줄 생성: %d종목 → %d배치 (배치당 %d, %d분 간격)",
        len(symbols), len(batches), batch_size, interval_min,
    )
    return out_path
