"""과거 캔들/틱 데이터 저장.

키움 TR 응답이나 수동 import로 받은 과거 데이터를 저장한다.
- daily/minute → stock_candles (upsert)
- tick → stock_ticks (insert)
"""

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import text

from app.core.database import async_session

logger = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9))


async def write_candle(symbol: str, payload: dict):
    """단일 캔들/틱을 적절한 테이블에 저장."""
    subtype = payload.get("subtype", "")

    if subtype == "tick":
        await _write_tick(symbol, payload)
        return

    if subtype == "daily":
        dt_str = payload.get("date", "")
        if not dt_str:
            return
        dt = datetime.strptime(dt_str, "%Y%m%d").replace(tzinfo=KST)
        interval = "1d"
    elif subtype == "minute":
        dt_str = payload.get("datetime", "")
        if not dt_str:
            return
        dt = datetime.strptime(dt_str, "%Y%m%d%H%M%S").replace(tzinfo=KST)
        interval = payload.get("interval", "1m")
    else:
        logger.warning("Unknown candle subtype: %s", subtype)
        return

    async with async_session() as db:
        await db.execute(
            text("""
                INSERT INTO stock_candles (symbol, dt, interval, open, high, low, close, volume, collected_at)
                VALUES (:symbol, :dt, :interval, :open, :high, :low, :close, :volume, :collected_at)
                ON CONFLICT ON CONSTRAINT uq_candle
                DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume,
                    collected_at = EXCLUDED.collected_at
            """),
            {
                "symbol": symbol,
                "dt": dt,
                "interval": interval,
                "open": payload.get("open", 0),
                "high": payload.get("high", 0),
                "low": payload.get("low", 0),
                "close": payload.get("close", 0),
                "volume": payload.get("volume", 0),
                "collected_at": datetime.now(KST),
            },
        )
        await db.commit()


async def write_candles_bulk(
    symbol: str, candles: list[dict], interval: str
):
    """캔들 배치를 stock_candles에 upsert."""
    if not candles:
        return

    params = []
    for c in candles:
        dt = c.get("dt")
        if isinstance(dt, str):
            if len(dt) == 8:  # YYYYMMDD
                dt = datetime.strptime(dt, "%Y%m%d").replace(tzinfo=KST)
            elif len(dt) == 14:  # YYYYMMDDHHMMSS
                dt = datetime.strptime(dt, "%Y%m%d%H%M%S").replace(tzinfo=KST)
        if not dt:
            continue
        params.append({
            "symbol": symbol,
            "dt": dt,
            "interval": interval,
            "open": c.get("open", 0),
            "high": c.get("high", 0),
            "low": c.get("low", 0),
            "close": c.get("close", 0),
            "volume": c.get("volume", 0),
            "collected_at": datetime.now(KST),
        })

    if not params:
        return

    async with async_session() as db:
        await db.execute(
            text("""
                INSERT INTO stock_candles (symbol, dt, interval, open, high, low, close, volume, collected_at)
                VALUES (:symbol, :dt, :interval, :open, :high, :low, :close, :volume, :collected_at)
                ON CONFLICT ON CONSTRAINT uq_candle
                DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume,
                    collected_at = EXCLUDED.collected_at
            """),
            params,
        )
        await db.commit()
    logger.info("Bulk wrote %d candles for %s (%s)", len(params), symbol, interval)


async def _write_tick(symbol: str, payload: dict):
    """틱 데이터를 stock_ticks에 저장."""
    dt_str = payload.get("datetime", "")
    if not dt_str:
        return
    ts = datetime.strptime(dt_str, "%Y%m%d%H%M%S").replace(tzinfo=KST)
    price = payload.get("close", 0)
    volume = payload.get("volume", 0)

    async with async_session() as db:
        await db.execute(
            text(
                "INSERT INTO stock_ticks (ts, symbol, price, volume) "
                "VALUES (:ts, :symbol, :price, :volume)"
            ),
            {"ts": ts, "symbol": symbol, "price": price, "volume": volume},
        )
        await db.commit()
