"""FastMCP 서버 — Stock Mate 도구 5개 + 리소스 1개.

KIS/캔들/뉴스/알파/포트폴리오를 MCP 표준 도구로 캡슐화.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date

from fastmcp import FastMCP

from app.mcp.governance import GovernanceCheck, audit_log

logger = logging.getLogger(__name__)

mcp = FastMCP("stock-mate")


# ── Tool 1: execute_order ─────────────────────────────────


@mcp.tool()
async def execute_order(
    ticker: str,
    action: str,
    qty: int,
    price: int = 0,
    mode: str = "paper",
) -> str:
    """증권사 주문 실행 (KIS API). Governance 검증 후 실행.

    Args:
        ticker: 6자리 종목코드 (예: 005930)
        action: BUY 또는 SELL
        qty: 주문 수량
        price: 지정가 (0이면 시장가)
        mode: paper (모의) 또는 real (실전)
    """
    start_ms = time.monotonic()
    params = {
        "ticker": ticker,
        "action": action,
        "qty": qty,
        "price": price,
        "mode": mode,
    }

    # Governance pre-check
    allowed, reason = GovernanceCheck.pre_check("execute_order", params)
    if not allowed:
        await audit_log("execute_order", params, None, "blocked", reason)
        return json.dumps(
            {"success": False, "blocked": True, "reason": reason}, ensure_ascii=False
        )

    # Execute via KIS
    from app.trading.kis_client import KISClient
    from app.trading.kis_order import KISOrderExecutor

    is_mock = mode != "real"
    client = KISClient(is_mock=is_mock)
    executor = KISOrderExecutor(client)

    order_type = "LIMIT" if price > 0 else "MARKET"
    if action.upper() == "BUY":
        result = await executor.buy(ticker, qty, price, order_type)
    elif action.upper() == "SELL":
        result = await executor.sell(ticker, qty, price, order_type)
    else:
        result = {"success": False, "message": f"Unknown action: {action}"}

    elapsed = int((time.monotonic() - start_ms) * 1000)
    status = "success" if result.get("success") else "error"
    await audit_log("execute_order", params, result, status, execution_ms=elapsed)

    return json.dumps(result, ensure_ascii=False)


# ── Tool 2: query_stock_data ──────────────────────────────


@mcp.tool()
async def query_stock_data(
    symbol: str,
    interval: str = "1d",
    count: int = 200,
) -> str:
    """시계열 OHLCV 데이터 조회 (TimescaleDB).

    Args:
        symbol: 6자리 종목코드
        interval: 캔들 간격 (1m, 5m, 1h, 1d, 1w, 1M)
        count: 캔들 수 (기본 200)
    """
    start_ms = time.monotonic()
    params = {"symbol": symbol, "interval": interval, "count": count}

    from app.core.database import async_session
    from app.services.candle_service import get_candles

    async with async_session() as db:
        candles = await get_candles(db, symbol, interval, count)

    elapsed = int((time.monotonic() - start_ms) * 1000)
    result = {"symbol": symbol, "interval": interval, "count": len(candles), "candles": candles[-10:]}
    await audit_log("query_stock_data", params, {"count": len(candles)}, "success", execution_ms=elapsed)

    return json.dumps(result, ensure_ascii=False)


# ── Tool 3: fetch_sentiment ───────────────────────────────


@mcp.tool()
async def fetch_sentiment(
    symbol: str,
    target_date: str = "",
) -> str:
    """종목 뉴스 감성 분석 데이터 조회.

    Args:
        symbol: 6자리 종목코드
        target_date: 날짜 YYYY-MM-DD (기본: 오늘)
    """
    start_ms = time.monotonic()
    params = {"symbol": symbol, "date": target_date}

    from sqlalchemy import select

    from app.core.database import async_session
    from app.news.models import NewsSentimentDaily

    dt = date.fromisoformat(target_date) if target_date else date.today()

    async with async_session() as db:
        result = await db.execute(
            select(NewsSentimentDaily).where(
                NewsSentimentDaily.symbol == symbol,
                NewsSentimentDaily.date == dt,
            )
        )
        row = result.scalar_one_or_none()

    if row:
        data = {
            "symbol": symbol,
            "date": str(row.date),
            "avg_sentiment": row.avg_sentiment,
            "article_count": row.article_count,
            "event_score": row.event_score,
        }
    else:
        data = {
            "symbol": symbol,
            "date": str(dt),
            "avg_sentiment": None,
            "article_count": 0,
            "event_score": None,
            "message": "해당 날짜 데이터 없음",
        }

    elapsed = int((time.monotonic() - start_ms) * 1000)
    await audit_log("fetch_sentiment", params, data, "success", execution_ms=elapsed)

    return json.dumps(data, ensure_ascii=False)


# ── Tool 4: run_alpha_scan ────────────────────────────────


@mcp.tool()
async def run_alpha_scan(
    context: str = "한국 주식시장 알파 팩터 탐색",
) -> str:
    """AI 알파 팩터 탐색 — 최근 발견된 팩터 요약 반환.

    Args:
        context: 시장 맥락 설명
    """
    start_ms = time.monotonic()
    params = {"context": context}

    from sqlalchemy import select

    from app.alpha.models import AlphaFactor
    from app.core.database import async_session

    async with async_session() as db:
        result = await db.execute(
            select(AlphaFactor)
            .where(AlphaFactor.ic_mean.isnot(None))
            .order_by(AlphaFactor.ic_mean.desc())
            .limit(10)
        )
        factors = result.scalars().all()

    data = {
        "context": context,
        "top_factors": [
            {
                "name": f.name,
                "expression": f.expression_str[:100],
                "ic_mean": f.ic_mean,
                "status": f.status,
                "factor_type": f.factor_type,
            }
            for f in factors
        ],
    }

    elapsed = int((time.monotonic() - start_ms) * 1000)
    await audit_log("run_alpha_scan", params, {"count": len(factors)}, "success", execution_ms=elapsed)

    return json.dumps(data, ensure_ascii=False)


# ── Tool 5: get_portfolio_status ──────────────────────────


@mcp.tool()
async def get_portfolio_status(mode: str = "paper") -> str:
    """현재 포트폴리오 (보유종목 + 평가손익) 조회.

    Args:
        mode: paper (모의투자) 또는 real (실전)
    """
    start_ms = time.monotonic()
    params = {"mode": mode}

    from sqlalchemy import select

    from app.core.database import async_session
    from app.models.base import Position

    async with async_session() as db:
        result = await db.execute(
            select(Position).where(Position.mode == mode.upper())
        )
        positions = result.scalars().all()

    data = {
        "mode": mode,
        "positions": [
            {
                "symbol": p.symbol,
                "qty": p.qty,
                "avg_price": float(p.avg_price),
            }
            for p in positions
        ],
        "total_positions": len(positions),
    }

    elapsed = int((time.monotonic() - start_ms) * 1000)
    await audit_log("get_portfolio_status", params, {"count": len(positions)}, "success", execution_ms=elapsed)

    return json.dumps(data, ensure_ascii=False)


# ── Resource: orderbook ───────────────────────────────────


@mcp.resource("realtime_orderbook/{symbol}")
async def get_realtime_orderbook(symbol: str) -> str:
    """실시간 호가창 스냅샷.

    Args:
        symbol: 6자리 종목코드
    """
    from app.services.ws_manager import manager

    # Return latest order book from tick store if available
    data = {
        "symbol": symbol,
        "message": "실시간 호가 데이터는 WebSocket 스트림 참조",
        "ws_channel": f"orderbook:{symbol}",
    }
    return json.dumps(data, ensure_ascii=False)
