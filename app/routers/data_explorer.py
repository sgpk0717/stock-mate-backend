"""수집 데이터 탐색 REST API.

서버사이드 페이지네이션 지원 (page + limit → {items, total}).
"""

from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any, Generic, TypeVar

from fastapi import APIRouter, Query
from pydantic import BaseModel
from sqlalchemy import text

from app.core.database import async_session

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/data", tags=["data-explorer"])

T = TypeVar("T")


# ── 공통 스키마 ──


class PagedResponse(BaseModel, Generic[T]):
    items: list[T]
    total: int
    page: int
    limit: int


class CollectionStatusItem(BaseModel):
    table_name: str
    display_name: str
    total_rows: int
    earliest_date: str | None
    latest_date: str | None
    last_collected_at: str | None


class InvestorTradingRow(BaseModel):
    id: int
    symbol: str
    name: str | None
    dt: str
    foreign_net: int
    inst_net: int
    retail_net: int
    foreign_buy_vol: int
    foreign_sell_vol: int
    inst_buy_vol: int
    inst_sell_vol: int
    retail_buy_vol: int
    retail_sell_vol: int
    collected_at: str | None


class MarginShortRow(BaseModel):
    id: int
    symbol: str
    name: str | None
    dt: str
    margin_balance: int
    margin_rate: float
    short_volume: int
    short_balance: int
    short_balance_rate: float
    collected_at: str | None


class DartFinancialRow(BaseModel):
    id: int
    symbol: str
    name: str | None
    disclosure_date: str
    fiscal_year: str
    fiscal_quarter: str
    eps: float | None
    bps: float | None
    operating_margin: float | None
    debt_to_equity: float | None
    collected_at: str | None


class ProgramTradingRow(BaseModel):
    id: int
    symbol: str
    name: str | None
    dt: str
    pgm_buy_qty: int
    pgm_sell_qty: int
    pgm_net_qty: int
    pgm_buy_amount: int
    pgm_sell_amount: int
    pgm_net_amount: int
    collected_at: str | None


class CandleCoverageItem(BaseModel):
    symbol: str
    name: str | None
    interval: str
    total_candles: int
    earliest_date: str
    latest_date: str


class NewsExplorerRow(BaseModel):
    id: str
    symbol: str | None
    name: str | None
    source: str
    title: str
    url: str | None
    published_at: str
    sentiment_score: float | None
    market_impact: float | None


class DataGapItem(BaseModel):
    data_type: str
    missing_dates: list[str]
    gap_count: int


# ── 헬퍼 ──


def _fmt_date(val: Any) -> str | None:
    if val is None:
        return None
    return val.isoformat()


async def _count_query(session: Any, table: str, where: str, params: dict) -> int:
    """WHERE 절을 공유하는 COUNT 쿼리."""
    sql = f"SELECT COUNT(*) FROM {table} {where}"
    result = await session.execute(text(sql), params)
    row = result.fetchone()
    return row[0] if row else 0


# ── 엔드포인트 ──


@router.get("/collection-status", response_model=list[CollectionStatusItem])
async def collection_status():
    """6개 테이블의 데이터 수집 현황."""
    queries: list[tuple[str, str, str]] = [
        ("stock_candles_1d", "일봉 캔들",
         "SELECT COUNT(*), MIN(dt), MAX(dt), MAX(collected_at) FROM stock_candles WHERE interval = '1d'"),
        ("stock_candles_1m", "분봉 캔들",
         "SELECT COUNT(*), MIN(dt), MAX(dt), MAX(collected_at) FROM stock_candles WHERE interval = '1m'"),
        ("investor_trading", "투자자별 매매동향",
         "SELECT COUNT(*), MIN(dt), MAX(dt), MAX(collected_at) FROM investor_trading"),
        ("margin_short_daily", "신용잔고/공매도",
         "SELECT COUNT(*), MIN(dt), MAX(dt), MAX(collected_at) FROM margin_short_daily"),
        ("dart_financials", "DART 재무",
         "SELECT COUNT(*), MIN(disclosure_date), MAX(disclosure_date), MAX(collected_at) FROM dart_financials"),
        ("program_trading", "프로그램 매매",
         "SELECT COUNT(*), MIN(dt), MAX(dt), MAX(collected_at) FROM program_trading"),
        ("news_articles", "뉴스 기사",
         "SELECT COUNT(*), MIN(published_at), MAX(published_at), MAX(created_at) FROM news_articles"),
    ]
    items: list[CollectionStatusItem] = []
    async with async_session() as session:
        for table_name, display_name, sql in queries:
            result = await session.execute(text(sql))
            row = result.fetchone()
            total_rows = row[0] if row and row[0] else 0
            items.append(CollectionStatusItem(
                table_name=table_name, display_name=display_name,
                total_rows=total_rows,
                earliest_date=_fmt_date(row[1]) if row else None,
                latest_date=_fmt_date(row[2]) if row else None,
                last_collected_at=_fmt_date(row[3]) if row else None,
            ))
    return items


@router.get("/investor-trading")
async def get_investor_trading(
    symbol: str | None = Query(None),
    start: date | None = Query(None),
    end: date | None = Query(None),
    page: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
) -> PagedResponse[InvestorTradingRow]:
    clauses: list[str] = []
    params: dict = {}
    if symbol:
        clauses.append("t.symbol = :symbol")
        params["symbol"] = symbol
    if start:
        clauses.append("t.dt >= :start")
        params["start"] = start.isoformat()
    if end:
        clauses.append("t.dt <= :end")
        params["end"] = end.isoformat()

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    offset = page * limit
    params["limit"] = limit
    params["offset"] = offset

    async with async_session() as session:
        total = await _count_query(session, "investor_trading t", where, {k: v for k, v in params.items() if k not in ("limit", "offset")})
        sql = (
            f"SELECT t.id, t.symbol, m.name, t.dt, t.foreign_net, t.inst_net, t.retail_net, "
            f"t.foreign_buy_vol, t.foreign_sell_vol, t.inst_buy_vol, t.inst_sell_vol, "
            f"t.retail_buy_vol, t.retail_sell_vol, t.collected_at "
            f"FROM investor_trading t LEFT JOIN stock_masters m ON t.symbol = m.symbol "
            f"{where} ORDER BY t.dt DESC LIMIT :limit OFFSET :offset"
        )
        result = await session.execute(text(sql), params)
        rows = result.fetchall()

    items = [
        InvestorTradingRow(
            id=r[0], symbol=r[1], name=r[2], dt=_fmt_date(r[3]),
            foreign_net=r[4], inst_net=r[5], retail_net=r[6],
            foreign_buy_vol=r[7], foreign_sell_vol=r[8],
            inst_buy_vol=r[9], inst_sell_vol=r[10],
            retail_buy_vol=r[11], retail_sell_vol=r[12],
            collected_at=_fmt_date(r[13]),
        ) for r in rows
    ]
    return PagedResponse(items=items, total=total, page=page, limit=limit)


@router.get("/margin-short")
async def get_margin_short(
    symbol: str | None = Query(None),
    start: date | None = Query(None),
    end: date | None = Query(None),
    page: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
) -> PagedResponse[MarginShortRow]:
    clauses: list[str] = []
    params: dict = {}
    if symbol:
        clauses.append("t.symbol = :symbol")
        params["symbol"] = symbol
    if start:
        clauses.append("t.dt >= :start")
        params["start"] = start.isoformat()
    if end:
        clauses.append("t.dt <= :end")
        params["end"] = end.isoformat()

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    offset = page * limit
    params["limit"] = limit
    params["offset"] = offset

    async with async_session() as session:
        total = await _count_query(session, "margin_short_daily t", where, {k: v for k, v in params.items() if k not in ("limit", "offset")})
        sql = (
            f"SELECT t.id, t.symbol, m.name, t.dt, t.margin_balance, t.margin_rate, "
            f"t.short_volume, t.short_balance, t.short_balance_rate, t.collected_at "
            f"FROM margin_short_daily t LEFT JOIN stock_masters m ON t.symbol = m.symbol "
            f"{where} ORDER BY t.dt DESC LIMIT :limit OFFSET :offset"
        )
        result = await session.execute(text(sql), params)
        rows = result.fetchall()

    items = [
        MarginShortRow(
            id=r[0], symbol=r[1], name=r[2], dt=_fmt_date(r[3]),
            margin_balance=r[4], margin_rate=r[5],
            short_volume=r[6], short_balance=r[7], short_balance_rate=r[8],
            collected_at=_fmt_date(r[9]),
        ) for r in rows
    ]
    return PagedResponse(items=items, total=total, page=page, limit=limit)


@router.get("/dart-financials")
async def get_dart_financials(
    symbol: str | None = Query(None),
    year: str | None = Query(None),
    page: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
) -> PagedResponse[DartFinancialRow]:
    clauses: list[str] = []
    params: dict = {}
    if symbol:
        clauses.append("t.symbol = :symbol")
        params["symbol"] = symbol
    if year:
        clauses.append("t.fiscal_year = :year")
        params["year"] = year

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    offset = page * limit
    params["limit"] = limit
    params["offset"] = offset

    async with async_session() as session:
        total = await _count_query(session, "dart_financials t", where, {k: v for k, v in params.items() if k not in ("limit", "offset")})
        sql = (
            f"SELECT t.id, t.symbol, m.name, t.disclosure_date, t.fiscal_year, t.fiscal_quarter, "
            f"t.eps, t.bps, t.operating_margin, t.debt_to_equity, t.collected_at "
            f"FROM dart_financials t LEFT JOIN stock_masters m ON t.symbol = m.symbol "
            f"{where} ORDER BY t.fiscal_year DESC, t.fiscal_quarter DESC LIMIT :limit OFFSET :offset"
        )
        result = await session.execute(text(sql), params)
        rows = result.fetchall()

    items = [
        DartFinancialRow(
            id=r[0], symbol=r[1], name=r[2], disclosure_date=_fmt_date(r[3]),
            fiscal_year=r[4], fiscal_quarter=r[5],
            eps=r[6], bps=r[7], operating_margin=r[8], debt_to_equity=r[9],
            collected_at=_fmt_date(r[10]),
        ) for r in rows
    ]
    return PagedResponse(items=items, total=total, page=page, limit=limit)


@router.get("/program-trading")
async def get_program_trading(
    symbol: str | None = Query(None),
    start: date | None = Query(None),
    end: date | None = Query(None),
    page: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
) -> PagedResponse[ProgramTradingRow]:
    clauses: list[str] = []
    params: dict = {}
    if symbol:
        clauses.append("t.symbol = :symbol")
        params["symbol"] = symbol
    if start:
        clauses.append("t.dt >= :start")
        params["start"] = start.isoformat()
    if end:
        clauses.append("t.dt <= :end")
        params["end"] = end.isoformat()

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    offset = page * limit
    params["limit"] = limit
    params["offset"] = offset

    async with async_session() as session:
        total = await _count_query(session, "program_trading t", where, {k: v for k, v in params.items() if k not in ("limit", "offset")})
        sql = (
            f"SELECT t.id, t.symbol, m.name, t.dt, t.pgm_buy_qty, t.pgm_sell_qty, t.pgm_net_qty, "
            f"t.pgm_buy_amount, t.pgm_sell_amount, t.pgm_net_amount, t.collected_at "
            f"FROM program_trading t LEFT JOIN stock_masters m ON t.symbol = m.symbol "
            f"{where} ORDER BY t.dt DESC LIMIT :limit OFFSET :offset"
        )
        result = await session.execute(text(sql), params)
        rows = result.fetchall()

    items = [
        ProgramTradingRow(
            id=r[0], symbol=r[1], name=r[2], dt=_fmt_date(r[3]),
            pgm_buy_qty=r[4], pgm_sell_qty=r[5], pgm_net_qty=r[6],
            pgm_buy_amount=r[7], pgm_sell_amount=r[8], pgm_net_amount=r[9],
            collected_at=_fmt_date(r[10]),
        ) for r in rows
    ]
    return PagedResponse(items=items, total=total, page=page, limit=limit)


@router.get("/news")
async def get_news(
    symbol: str | None = Query(None),
    page: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
) -> PagedResponse[NewsExplorerRow]:
    clauses: list[str] = []
    params: dict = {}
    if symbol:
        clauses.append("n.symbols::text LIKE :sym_like")
        params["sym_like"] = f"%{symbol}%"

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    offset = page * limit
    params["limit"] = limit
    params["offset"] = offset

    async with async_session() as session:
        total = await _count_query(session, "news_articles n", where, {k: v for k, v in params.items() if k not in ("limit", "offset")})
        name_result = await session.execute(text("SELECT symbol, name FROM stock_masters"))
        name_map: dict[str, str] = {r[0]: r[1] for r in name_result.fetchall()}

        sql = (
            f"SELECT n.id, n.symbols, n.source, n.title, n.url, n.published_at, "
            f"n.sentiment_score, n.market_impact "
            f"FROM news_articles n {where} "
            f"ORDER BY n.published_at DESC LIMIT :limit OFFSET :offset"
        )
        result = await session.execute(text(sql), params)
        rows = result.fetchall()

    items: list[NewsExplorerRow] = []
    for r in rows:
        raw_symbols = r[1]
        first_symbol = None
        first_name = None
        if raw_symbols:
            try:
                syms = json.loads(raw_symbols) if isinstance(raw_symbols, str) else raw_symbols
                if isinstance(syms, list) and syms:
                    first_symbol = syms[0]
                    first_name = name_map.get(first_symbol)
            except (json.JSONDecodeError, TypeError):
                pass
        items.append(NewsExplorerRow(
            id=str(r[0]), symbol=first_symbol, name=first_name,
            source=r[2], title=r[3], url=r[4],
            published_at=_fmt_date(r[5]), sentiment_score=r[6], market_impact=r[7],
        ))

    return PagedResponse(items=items, total=total, page=page, limit=limit)


@router.get("/candle-coverage", response_model=list[CandleCoverageItem])
async def candle_coverage(
    symbol: str | None = Query(None),
):
    """종목x인터벌별 캔들 수집 현황."""
    if symbol:
        sql = (
            "SELECT c.symbol, m.name, c.interval, COUNT(*) AS cnt, "
            "MIN(c.dt) AS earliest, MAX(c.dt) AS latest "
            "FROM stock_candles c LEFT JOIN stock_masters m ON c.symbol = m.symbol "
            "WHERE c.symbol = :symbol GROUP BY c.symbol, m.name, c.interval "
            "ORDER BY c.symbol, c.interval"
        )
        params: dict = {"symbol": symbol}
    else:
        sql = (
            "SELECT c.symbol, m.name, c.interval, COUNT(*) AS cnt, "
            "MIN(c.dt) AS earliest, MAX(c.dt) AS latest "
            "FROM stock_candles c LEFT JOIN stock_masters m ON c.symbol = m.symbol "
            "GROUP BY c.symbol, m.name, c.interval ORDER BY cnt DESC LIMIT 50"
        )
        params = {}

    async with async_session() as session:
        result = await session.execute(text(sql), params)
        rows = result.fetchall()

    return [
        CandleCoverageItem(
            symbol=r[0], name=r[1], interval=r[2], total_candles=r[3],
            earliest_date=_fmt_date(r[4]), latest_date=_fmt_date(r[5]),
        ) for r in rows
    ]


@router.get("/gaps", response_model=list[DataGapItem])
async def data_gaps(
    data_type: str | None = Query(None),
    limit: int = Query(30, ge=1, le=90),
):
    """최근 N 거래일 중 누락 날짜."""
    trading_days_sql = (
        "SELECT DISTINCT dt::date AS d FROM stock_candles "
        "WHERE interval = '1d' ORDER BY d DESC LIMIT :limit"
    )
    type_queries: dict[str, str] = {
        "daily": "SELECT DISTINCT dt::date AS d FROM stock_candles WHERE interval = '1d'",
        "minute": "SELECT DISTINCT dt::date AS d FROM stock_candles WHERE interval = '1m'",
        "investor": "SELECT DISTINCT dt AS d FROM investor_trading",
        "margin_short": "SELECT DISTINCT dt AS d FROM margin_short_daily",
    }

    check_types = {data_type: type_queries[data_type]} if data_type and data_type in type_queries else type_queries
    items: list[DataGapItem] = []

    async with async_session() as session:
        result = await session.execute(text(trading_days_sql), {"limit": limit})
        trading_days: set[date] = {row[0] for row in result.fetchall()}
        if not trading_days:
            return items

        for dtype, sql in check_types.items():
            result = await session.execute(text(sql))
            existing: set[date] = {row[0] for row in result.fetchall()}
            missing = sorted(trading_days - existing, reverse=True)
            items.append(DataGapItem(data_type=dtype, missing_dates=[d.isoformat() for d in missing], gap_count=len(missing)))

    return items
