"""멀티타임프레임 캔들 쿼리 서비스.

continuous aggregate(실시간 틱 → 자동 캔들)과
stock_candles(키움 TR 과거 데이터)를 병합(stitching)하여
1m ~ 1M 전 인터벌의 캔들 데이터를 통합 제공한다.
"""

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# Lightweight Charts는 UTC 기준 표시 → KST 오프셋을 더해 한국 시간으로 보이게 함
_KST_OFFSET = 9 * 3600
_KST_TZ = timezone(timedelta(hours=9))

# 일봉 이상 인터벌 (KST 거래일 기준 정규화 대상)
_DAILY_OR_ABOVE = {"1d", "1w", "1M"}


def _to_chart_ts(dt_val: datetime, interval: str) -> int:
    """DB timestamp → Lightweight Charts timestamp.

    일봉 이상: KST 거래일로 정규화하여 pykrx/키움 중복 방지.
    분봉/시봉: KST 오프셋으로 한국 시간 표시.
    """
    if interval in _DAILY_OR_ABOVE:
        kst = dt_val.astimezone(_KST_TZ)
        return int(datetime(kst.year, kst.month, kst.day,
                            tzinfo=timezone.utc).timestamp())
    return int(dt_val.timestamp()) + _KST_OFFSET

# Continuous aggregate 뷰 매핑
_CA_MAP = {
    "1m": "candles_1m",
    "1h": "candles_1h",
    "1d": "candles_1d",
}

# 1분봉에서 집계하는 인터벌
_DERIVED_FROM_1M = {"3m", "5m", "15m", "30m"}


async def get_candles(
    db: AsyncSession,
    symbol: str,
    interval: str = "1d",
    count: int = 200,
) -> list[dict]:
    """전 타임프레임 통합 캔들 조회.

    전략:
    1. 1w/1M: 일봉 데이터를 집계
    2. 3m/5m/15m/30m: 1분봉 데이터를 집계
    3. 1m/1h/1d: CA + stock_candles 병합
    """
    if interval in ("1w", "1M"):
        daily = await _get_stitched_candles(db, symbol, "1d", count * (5 if interval == "1w" else 22))
        return _aggregate_to_period(daily, interval, count)

    if interval in _DERIVED_FROM_1M:
        minutes = int(interval.replace("m", ""))
        raw_1m = await _get_stitched_candles(db, symbol, "1m", count * minutes)
        return _aggregate_minutes(raw_1m, minutes, count)

    return await _get_stitched_candles(db, symbol, interval, count)


async def _get_stitched_candles(
    db: AsyncSession, symbol: str, interval: str, count: int
) -> list[dict]:
    """CA + stock_candles 데이터를 병합. 겹치는 시간대는 CA 우선."""
    candle_map: dict[int, dict] = {}

    # 1. stock_candles (과거 데이터)
    hist_result = await db.execute(
        text("""
            SELECT dt, open, high, low, close, volume
            FROM stock_candles
            WHERE symbol = :symbol AND interval = :interval
            ORDER BY dt DESC
            LIMIT :count
        """),
        {"symbol": symbol, "interval": interval, "count": count},
    )
    for row in hist_result.fetchall():
        ts = _to_chart_ts(row[0], interval)
        candle_map[ts] = {
            "time": ts,
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": int(row[5]),
        }

    # 2. Continuous aggregate (실시간 데이터)
    ca_view = _CA_MAP.get(interval)
    if ca_view:
        try:
            agg_result = await db.execute(
                text(f"""
                    SELECT bucket, open, high, low, close, volume
                    FROM {ca_view}
                    WHERE symbol = :symbol
                    ORDER BY bucket DESC
                    LIMIT :count
                """),
                {"symbol": symbol, "count": count},
            )
            for row in agg_result.fetchall():
                ts = _to_chart_ts(row[0], interval)
                # CA 데이터가 우선 (실제 틱 기반이므로 더 정확)
                candle_map[ts] = {
                    "time": ts,
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": int(row[5]),
                }
        except Exception:
            logger.debug("CA view %s not available, using stock_candles only", ca_view)

    # 시간순 정렬, 최근 count개
    sorted_candles = sorted(candle_map.values(), key=lambda c: c["time"])
    return sorted_candles[-count:]


def _aggregate_to_period(daily: list[dict], interval: str, count: int) -> list[dict]:
    """일봉 → 주봉/월봉 집계."""
    if not daily:
        return []

    groups: dict[str, list[dict]] = defaultdict(list)
    for c in daily:
        dt = datetime.fromtimestamp(c["time"], tz=timezone.utc)
        if interval == "1w":
            key = dt.strftime("%G-W%V")
        else:  # 1M
            key = dt.strftime("%Y-%m")
        groups[key].append(c)

    result = []
    for _key, candles in groups.items():
        result.append({
            "time": candles[0]["time"],
            "open": candles[0]["open"],
            "high": max(c["high"] for c in candles),
            "low": min(c["low"] for c in candles),
            "close": candles[-1]["close"],
            "volume": sum(c["volume"] for c in candles),
        })

    result.sort(key=lambda c: c["time"])
    return result[-count:]


def _aggregate_minutes(candles_1m: list[dict], minutes: int, count: int) -> list[dict]:
    """1분봉 → N분봉 집계."""
    if not candles_1m:
        return []

    bucket_seconds = minutes * 60
    groups: dict[int, list[dict]] = defaultdict(list)

    for c in candles_1m:
        bucket = (c["time"] // bucket_seconds) * bucket_seconds
        groups[bucket].append(c)

    result = []
    for bucket_ts in sorted(groups.keys()):
        group = groups[bucket_ts]
        result.append({
            "time": bucket_ts,
            "open": group[0]["open"],
            "high": max(c["high"] for c in group),
            "low": min(c["low"] for c in group),
            "close": group[-1]["close"],
            "volume": sum(c["volume"] for c in group),
        })

    return result[-count:]
