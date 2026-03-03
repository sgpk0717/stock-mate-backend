"""유니버스 리졸버.

pykrx 런타임 동적 조회 + 24시간 인메모리 캐시.
KOSPI200/KOSDAQ150/KRX300/ALL 유니버스를 symbol list로 변환.
"""

from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum

from pykrx import stock as krx

from app.core.stock_master import get_all_stocks

logger = logging.getLogger(__name__)

_CACHE_TTL_SECONDS = 86_400  # 24시간 (인덱스 구성은 분기별 변경)


class Universe(str, Enum):
    KOSPI200 = "KOSPI200"
    KOSDAQ150 = "KOSDAQ150"
    KRX300 = "KRX300"
    ALL = "ALL"


UNIVERSE_LABELS: dict[Universe, str] = {
    Universe.KOSPI200: "KOSPI 200",
    Universe.KOSDAQ150: "KOSDAQ 150",
    Universe.KRX300: "KRX 300",
    Universe.ALL: "전종목 (KOSPI+KOSDAQ)",
}

# pykrx 인덱스 코드 매핑
_INDEX_CODES: dict[Universe, str] = {
    Universe.KOSPI200: "1028",
    Universe.KOSDAQ150: "2203",
    Universe.KRX300: "1035",
}

# 인메모리 캐시: universe -> (symbols, timestamp)
_cache: dict[str, tuple[list[str], float]] = {}


def _fetch_index_members(index_code: str) -> list[str]:
    """pykrx 동기 호출로 인덱스 구성종목 조회."""
    try:
        members = krx.get_index_portfolio_deposit_file(index_code)
        return list(members)
    except Exception as e:
        logger.warning("pykrx index %s fetch failed: %s", index_code, e)
        return []


async def resolve_universe(universe: Universe) -> list[str]:
    """Universe enum → symbol list 변환.

    pykrx 조회 결과를 24시간 캐시.
    실패 시 이전 캐시 반환 (서킷 브레이커 원칙).
    """
    cache_key = universe.value
    now = time.monotonic()

    # 캐시 히트
    if cache_key in _cache:
        symbols, cached_at = _cache[cache_key]
        if now - cached_at < _CACHE_TTL_SECONDS and symbols:
            return symbols

    if universe == Universe.ALL:
        all_stocks = get_all_stocks()
        symbols = [s["symbol"] for s in all_stocks]
        _cache[cache_key] = (symbols, now)
        return symbols

    index_code = _INDEX_CODES[universe]
    symbols = await asyncio.to_thread(_fetch_index_members, index_code)

    if symbols:
        _cache[cache_key] = (symbols, now)
        logger.info(
            "Universe %s resolved: %d symbols (index %s)",
            universe.value, len(symbols), index_code,
        )
        return symbols

    # 실패 시 이전 캐시 반환
    if cache_key in _cache:
        prev_symbols, _ = _cache[cache_key]
        if prev_symbols:
            logger.warning(
                "Universe %s fetch failed, returning stale cache (%d symbols)",
                universe.value, len(prev_symbols),
            )
            return prev_symbols

    # pykrx 실패 + 캐시 없음 → stock_masters 마켓 기반 폴백
    fallback = _fallback_from_stock_masters(universe)
    if fallback:
        _cache[cache_key] = (fallback, now)
        logger.warning(
            "Universe %s: pykrx failed, DB fallback (%d symbols from stock_masters)",
            universe.value, len(fallback),
        )
        return fallback

    raise RuntimeError(
        f"Universe {universe.value} 구성종목을 가져올 수 없습니다. "
        "네트워크 연결을 확인하세요."
    )


_MARKET_FALLBACK: dict[Universe, str | None] = {
    Universe.KOSPI200: "KOSPI",
    Universe.KOSDAQ150: "KOSDAQ",
    Universe.KRX300: None,  # KOSPI + KOSDAQ 전체
}


def _fallback_from_stock_masters(universe: Universe) -> list[str]:
    """pykrx 실패 시 stock_masters 마켓 기반 폴백.

    정확한 인덱스 구성종목은 아니지만, 해당 시장 전체 종목을 반환.
    """
    all_stocks = get_all_stocks()
    if not all_stocks:
        return []

    market_filter = _MARKET_FALLBACK.get(universe)
    if market_filter:
        return [s["symbol"] for s in all_stocks if s["market"] == market_filter]
    # KRX300 → 전체
    return [s["symbol"] for s in all_stocks]


async def get_universe_info() -> list[dict]:
    """사용 가능한 유니버스 목록과 종목 수 반환."""
    result = []
    for univ in Universe:
        try:
            symbols = await resolve_universe(univ)
            count = len(symbols)
        except Exception:
            count = 0
        result.append({
            "code": univ.value,
            "label": UNIVERSE_LABELS[univ],
            "count": count,
        })
    return result
