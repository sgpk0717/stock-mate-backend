"""유니버스 리졸버.

pykrx 런타임 동적 조회 + 24시간 인메모리 캐시.
KOSPI200/KOSDAQ150/KRX300/ALL 유니버스를 symbol list로 변환.

Point-in-Time (PIT) 유니버스:
  as_of_date가 주어지면 해당 시점에 실제 거래되던 종목 기반으로
  유동성(거래대금) 상위 N개를 반환한다.
  stock_candles 테이블의 존재 여부를 거래 가능성 증거로 사용.
  (정확한 과거 인덱스 구성종목이 아닌 유동성 기반 PIT 근사치)
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import date as date_type
from datetime import timedelta
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

# PIT 유니버스 크기 매핑 (None = 제한 없음)
_UNIVERSE_SIZE: dict[Universe, int | None] = {
    Universe.KOSPI200: 200,
    Universe.KOSDAQ150: 150,
    Universe.KRX300: 300,
    Universe.ALL: None,
}

# PIT 유니버스 마켓 필터 (None = 전체)
_PIT_MARKET_FILTER: dict[Universe, str | None] = {
    Universe.KOSPI200: "KOSPI",
    Universe.KOSDAQ150: "KOSDAQ",
    Universe.KRX300: None,   # KOSPI + KOSDAQ 전체
    Universe.ALL: None,
}

# 인메모리 캐시: universe -> (symbols, timestamp)
_cache: dict[str, tuple[list[str], float]] = {}

# PIT 유니버스 캐시: "universe:YYYY-MM-DD" -> (symbols, timestamp)
# 동일 날짜 재조회 방지 (TTL 24시간 — 과거 데이터는 변하지 않으므로 사실상 영구)
_pit_cache: dict[str, tuple[list[str], float]] = {}


def _fetch_index_members(index_code: str) -> list[str]:
    """pykrx 동기 호출로 인덱스 구성종목 조회."""
    try:
        members = krx.get_index_portfolio_deposit_file(index_code)
        return list(members)
    except Exception as e:
        logger.warning("pykrx index %s fetch failed: %s", index_code, e)
        return []


# [DAILY_MINING] 생존편향 Point-in-Time 유니버스
# 상태: 구현됨 (유동성 기반 PIT 근사치)
# 접근법: stock_candles 테이블 존재 여부를 거래 가능성 증거로 사용.
#   해당 날짜 ±7일 내 캔들 데이터가 있는 종목 중
#   거래대금(close*volume) 상위 N개를 유니버스로 반환.
# 한계: 정확한 과거 KOSPI200 구성종목이 아닌 유동성 기반 근사치.
#   KOSPI200 정기 변경 이력이 DB에 없고, pykrx도 과거 구성종목 미지원.
# 효과: 상장폐지 종목 제거 + 당시 존재하던 종목 포함으로
#   생존편향 대폭 완화 (1~4% 연간 수익률 부풀림 방지)


async def resolve_universe(
    universe: Universe,
    as_of_date: date_type | None = None,
) -> list[str]:
    """Universe enum → symbol list 변환 (SSOT 패턴).

    Args:
        universe: 유니버스 종류 (KOSPI200/KOSDAQ150/KRX300/ALL).
        as_of_date: Point-in-Time 기준일. None이면 현재 유니버스 반환 (기존 동작).
                    날짜가 주어지면 해당 시점에 거래되던 종목 기반 PIT 유니버스 반환.

    현재 유니버스 (as_of_date=None) 조회 순서:
    1. 메모리 캐시 (24시간 TTL)
    2. Redis 캐시 (24시간 TTL, 컨테이너 재시작 시에도 유지)
    3. DB stock_masters 마켓 기반 폴백 (정확한 인덱스는 아니지만 즉시 사용 가능)
    4. pykrx KRX API (최후 수단, 하루 1회만)

    PIT 유니버스 (as_of_date 지정) 조회 순서:
    1. PIT 메모리 캐시 (24시간 TTL)
    2. DB stock_candles 거래대금 기반 상위 N종목

    pykrx는 IP 밴 위험이 있으므로 가능한 한 호출하지 않는다.
    """
    # PIT 유니버스 분기
    if as_of_date is not None:
        return await _resolve_pit_universe(universe, as_of_date)
    cache_key = universe.value
    now = time.monotonic()

    # 1. 메모리 캐시 히트
    if cache_key in _cache:
        symbols, cached_at = _cache[cache_key]
        if now - cached_at < _CACHE_TTL_SECONDS and symbols:
            return symbols

    if universe == Universe.ALL:
        all_stocks = get_all_stocks()
        symbols = [s["symbol"] for s in all_stocks]
        if symbols:
            _cache[cache_key] = (symbols, now)
            return symbols

    # 2. Redis 캐시 (컨테이너 재시작 후에도 유지)
    redis_symbols = await _load_from_redis(cache_key)
    if redis_symbols:
        _cache[cache_key] = (redis_symbols, now)
        logger.info(
            "Universe %s: Redis 캐시 히트 (%d symbols)",
            universe.value, len(redis_symbols),
        )
        return redis_symbols

    # 3. DB stock_masters 폴백 (pykrx 호출 없이 즉시)
    fallback = _fallback_from_stock_masters(universe)
    if fallback:
        _cache[cache_key] = (fallback, now)
        await _save_to_redis(cache_key, fallback)
        logger.info(
            "Universe %s: DB fallback (%d symbols from stock_masters)",
            universe.value, len(fallback),
        )
        return fallback

    # 4. pykrx KRX API (최후 수단)
    index_code = _INDEX_CODES.get(universe)
    if index_code:
        symbols = await asyncio.to_thread(_fetch_index_members, index_code)
        if symbols:
            _cache[cache_key] = (symbols, now)
            await _save_to_redis(cache_key, symbols)
            logger.info(
                "Universe %s: pykrx 조회 성공 (%d symbols, index %s)",
                universe.value, len(symbols), index_code,
            )
            return symbols

    # 이전 메모리 캐시 (stale이라도)
    if cache_key in _cache:
        prev_symbols, _ = _cache[cache_key]
        if prev_symbols:
            logger.warning(
                "Universe %s: 모든 소스 실패, stale 캐시 반환 (%d symbols)",
                universe.value, len(prev_symbols),
            )
            return prev_symbols

    raise RuntimeError(
        f"Universe {universe.value} 구성종목을 가져올 수 없습니다. "
        "Redis/DB/pykrx 모두 실패."
    )


async def _resolve_pit_universe(
    universe: Universe,
    as_of_date: date_type,
) -> list[str]:
    """Point-in-Time 유니버스: 특정 날짜에 실제 거래되던 종목 반환.

    stock_candles 테이블에서 as_of_date ±7일 내 일봉 데이터가 있는 종목을 찾고,
    거래대금(close*volume) 합산 기준으로 상위 N개를 반환한다.

    이 접근법은:
    1. 상장폐지 종목 제거 (폐지 후 캔들 데이터 없음)
    2. 당시 존재하던 종목 포함 (시딩된 과거 데이터가 있으면)
    3. 유동성 순위로 인덱스 구성종목 근사 (시가총액 대용)
    4. 추가 데이터 수집 불필요 (기존 stock_candles 활용)

    Args:
        universe: 유니버스 종류.
        as_of_date: 기준일.

    Returns:
        해당 시점 거래 가능 종목의 symbol 리스트 (유동성 내림차순 정렬).
    """
    now = time.monotonic()
    pit_cache_key = f"{universe.value}:{as_of_date.isoformat()}"

    # 1. PIT 메모리 캐시 (과거 데이터는 변하지 않으므로 사실상 영구)
    if pit_cache_key in _pit_cache:
        symbols, cached_at = _pit_cache[pit_cache_key]
        if now - cached_at < _CACHE_TTL_SECONDS and symbols:
            return symbols

    # 2. DB 조회
    try:
        symbols = await _query_pit_symbols(universe, as_of_date)
    except Exception as e:
        logger.error(
            "PIT universe 조회 실패 (universe=%s, date=%s): %s",
            universe.value, as_of_date, e,
        )
        symbols = []

    # 3. 결과가 없으면 현재 유니버스로 폴백
    if not symbols:
        logger.warning(
            "PIT universe 결과 없음 (universe=%s, date=%s) → 현재 유니버스 폴백",
            universe.value, as_of_date,
        )
        return await resolve_universe(universe, as_of_date=None)

    # 4. 캐시 저장
    _pit_cache[pit_cache_key] = (symbols, now)
    logger.info(
        "PIT universe %s (as_of=%s): %d symbols (거래대금 기준)",
        universe.value, as_of_date, len(symbols),
    )
    return symbols


async def _query_pit_symbols(
    universe: Universe,
    as_of_date: date_type,
) -> list[str]:
    """stock_candles + stock_masters 조인으로 PIT 유니버스 조회.

    SQL 전략:
    - interval='1d'인 일봉 데이터 중 as_of_date ±7일 범위 필터
    - stock_masters.market으로 KOSPI/KOSDAQ 필터 (해당 시)
    - close*volume 합산으로 거래대금 순위 산출
    - 상위 N개 반환 (KOSPI200=200, KOSDAQ150=150, KRX300=300, ALL=무제한)
    """
    from sqlalchemy import text as sa_text

    from app.core.database import async_session

    market_filter = _PIT_MARKET_FILTER.get(universe)
    limit = _UNIVERSE_SIZE.get(universe)

    # ±7일 범위 (주말/공휴일 고려)
    date_from = as_of_date - timedelta(days=7)
    date_to = as_of_date + timedelta(days=7)

    # 마켓 필터 WHERE 절
    market_where = ""
    if market_filter:
        market_where = "AND sm.market = :market"

    # LIMIT 절
    limit_clause = ""
    if limit is not None:
        limit_clause = "LIMIT :limit"

    query = sa_text(f"""
        SELECT sc.symbol
        FROM stock_candles sc
        JOIN stock_masters sm ON sc.symbol = sm.symbol
        WHERE sc.interval = '1d'
          AND sc.dt::date BETWEEN :date_from AND :date_to
          AND sc.close > 0
          AND sc.volume > 0
          {market_where}
        GROUP BY sc.symbol
        HAVING COUNT(*) >= 1
        ORDER BY SUM(sc.close * sc.volume) DESC
        {limit_clause}
    """)

    params: dict = {
        "date_from": date_from,
        "date_to": date_to,
    }
    if market_filter:
        params["market"] = market_filter
    if limit is not None:
        params["limit"] = limit

    async with async_session() as session:
        result = await session.execute(query, params)
        rows = result.fetchall()

    return [row[0] for row in rows]


async def prefetch_universe(universe: Universe) -> list[str]:
    """하루 1회 프리페치 — handle_pre_market()에서 호출.

    pykrx로 정확한 인덱스 구성종목을 조회하여 Redis에 캐싱.
    """
    cache_key = universe.value
    index_code = _INDEX_CODES.get(universe)

    if not index_code:
        return await resolve_universe(universe)

    symbols = await asyncio.to_thread(_fetch_index_members, index_code)
    if symbols:
        _cache[cache_key] = (symbols, time.monotonic())
        await _save_to_redis(cache_key, symbols)
        logger.info(
            "Universe %s prefetched: %d symbols (pykrx)",
            universe.value, len(symbols),
        )
        return symbols

    # pykrx 실패해도 기존 캐시 유지
    logger.warning("Universe %s prefetch 실패 — 기존 캐시 유지", universe.value)
    return await resolve_universe(universe)


async def _save_to_redis(cache_key: str, symbols: list[str]) -> None:
    """유니버스를 Redis에 저장 (TTL 24시간)."""
    try:
        import json
        from app.core.redis import get_client
        r = get_client()
        await r.set(
            f"universe:{cache_key}",
            json.dumps(symbols),
            ex=86_400,  # 24시간
        )
    except Exception as e:
        logger.debug("Redis universe 저장 실패: %s", e)


async def _load_from_redis(cache_key: str) -> list[str]:
    """Redis에서 유니버스 로드."""
    try:
        import json
        from app.core.redis import get_client
        r = get_client()
        data = await r.get(f"universe:{cache_key}")
        if data:
            return json.loads(data)
    except Exception as e:
        logger.debug("Redis universe 로드 실패: %s", e)
    return []


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
