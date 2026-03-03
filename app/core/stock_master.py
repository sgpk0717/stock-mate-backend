"""종목 마스터 — DB에서 로딩한 메모리 캐시.

앱 시작 시 load_stock_cache()로 DB → 메모리 캐시.
나중에 실시간 데이터 들어오면 가격만 업데이트.
"""

from decimal import Decimal

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

# 메모리 캐시: symbol → {name, market, price}
_cache: dict[str, dict] = {}

# 초기 가격 (캔들 데이터 로드 전 fallback)
_DEFAULT_PRICES: dict[str, Decimal] = {
    "005930": Decimal("72800"),
    "000660": Decimal("178000"),
    "035720": Decimal("42350"),
    "005380": Decimal("231500"),
    "035420": Decimal("198500"),
    "051910": Decimal("315000"),
    "006400": Decimal("372000"),
    "068270": Decimal("185500"),
}


async def load_stock_cache(session: AsyncSession):
    """앱 시작 시 호출 — stock_masters 테이블 전체를 메모리에 로딩."""
    result = await session.execute(
        text("SELECT symbol, name, market FROM stock_masters")
    )
    rows = result.fetchall()

    _cache.clear()
    for symbol, name, market in rows:
        _cache[symbol] = {
            "name": name,
            "market": market,
            "price": _DEFAULT_PRICES.get(symbol, Decimal("0")),
        }

    # stock_masters가 비어있으면 기본 8종목으로 fallback
    if not _cache:
        for sym, price in _DEFAULT_PRICES.items():
            _cache[sym] = {
                "name": _FALLBACK_NAMES.get(sym, sym),
                "market": "KOSPI",
                "price": price,
            }

    # 캔들 테이블에서 최신 종가 로딩
    candle_result = await session.execute(
        text(
            "SELECT DISTINCT ON (symbol) symbol, close "
            "FROM stock_candles WHERE interval = '1d' "
            "ORDER BY symbol, dt DESC"
        )
    )
    for symbol, close_price in candle_result.fetchall():
        if symbol in _cache:
            _cache[symbol]["price"] = Decimal(str(close_price))

    print(f"Stock cache loaded: {len(_cache)} stocks")


def get_stock_name(symbol: str) -> str:
    return _cache.get(symbol, {}).get("name", symbol)


def get_current_price(symbol: str) -> Decimal:
    return _cache.get(symbol, {}).get("price", Decimal("0"))


def get_stock_market(symbol: str) -> str:
    return _cache.get(symbol, {}).get("market", "")


def update_price(symbol: str, price: Decimal):
    """실시간 틱이 들어올 때 캐시 가격 업데이트."""
    if symbol in _cache:
        _cache[symbol]["price"] = price


def get_all_stocks() -> list[dict]:
    """전체 종목 리스트 반환."""
    return [
        {
            "symbol": sym,
            "name": info["name"],
            "market": info["market"],
        }
        for sym, info in _cache.items()
    ]


def search_stocks(query: str, market: str | None = None, limit: int = 50) -> list[dict]:
    """종목 이름/코드로 검색."""
    q = query.lower()
    results = []
    for sym, info in _cache.items():
        if market and info["market"] != market:
            continue
        if q in sym.lower() or q in info["name"].lower():
            results.append({
                "symbol": sym,
                "name": info["name"],
                "market": info["market"],
            })
            if len(results) >= limit:
                break
    return results


# 캐시 미로딩 시 fallback 이름
_FALLBACK_NAMES = {
    "005930": "삼성전자",
    "000660": "SK하이닉스",
    "035720": "카카오",
    "005380": "현대차",
    "035420": "NAVER",
    "051910": "LG화학",
    "006400": "삼성SDI",
    "068270": "셀트리온",
}
