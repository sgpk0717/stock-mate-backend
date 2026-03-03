"""에이전트 오케스트레이션.

Manager Agent가 전문 에이전트(Technical, Risk)와 섹터 검색을
호출할 수 있는 비동기 도구 핸들러들을 제공한다.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.core.database import async_session
from app.core.stock_master import get_stock_name

logger = logging.getLogger(__name__)


async def handle_ask_technical(input_data: dict[str, Any]) -> str:
    """Technical Analyst Agent를 호출한다."""
    from app.agents.technical import analyze_stock

    symbol = input_data.get("symbol", "")
    stock_name = input_data.get("stock_name") or get_stock_name(symbol) or symbol
    strategy_draft = input_data.get("strategy_draft")

    result = await analyze_stock(symbol, stock_name, strategy_draft)
    return json.dumps(result, ensure_ascii=False)


async def handle_ask_risk(input_data: dict[str, Any]) -> str:
    """Risk Manager Agent를 호출한다."""
    from app.agents.risk import assess_strategy

    strategy_draft = input_data.get("strategy_draft", {})
    result = await assess_strategy(strategy_draft)
    return json.dumps(result, ensure_ascii=False)


async def handle_search_sector(input_data: dict[str, Any]) -> str:
    """섹터 의미론적 검색을 수행한다."""
    from app.sector.search import search_stocks

    query = input_data.get("query", "")
    top_k = input_data.get("top_k", 10)

    async with async_session() as session:
        results = await search_stocks(session, query, top_k=top_k)

    return json.dumps(
        {
            "query": query,
            "results": [
                {
                    "symbol": r.symbol,
                    "name": r.name,
                    "sector": r.sector,
                    "similarity": r.similarity,
                }
                for r in results
            ],
            "total": len(results),
        },
        ensure_ascii=False,
    )


# 비동기 도구 핸들러 레지스트리
ASYNC_TOOL_HANDLERS: dict[str, Any] = {
    "ask_technical_analyst": handle_ask_technical,
    "ask_risk_manager": handle_ask_risk,
    "search_sector_stocks": handle_search_sector,
}
