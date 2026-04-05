"""수집 데이터 탐색 REST API.

서버사이드 페이지네이션 지원 (page + limit → {items, total}).
"""

from __future__ import annotations

from app.core.timezone import KST

import asyncio
import json
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Generic, TypeVar

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
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
    latest_date: str | None          # 실제 데이터 최신 날짜 (MAX(dt))
    last_collected_at: str | None    # 수집 시도 시각 (참고용)


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
    pgm_buy_amount: int
    pgm_sell_amount: int
    pgm_net_amount: int
    arbt_buy_amount: int = 0
    arbt_sell_amount: int = 0
    nabt_buy_amount: int = 0
    nabt_sell_amount: int = 0
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


class DiscussionExplorerRow(BaseModel):
    id: str
    symbol: str
    name: str | None = None
    title: str
    content: str | None = None
    author: str | None = None
    published_at: str
    likes: int
    dislikes: int
    comment_count: int
    sentiment_score: float | None = None


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
    """7개 테이블의 데이터 수집 현황 (UNION ALL 단일 쿼리)."""
    sql = """
    SELECT 'stock_candles_1d' AS tbl, '일봉 캔들' AS disp,
           COUNT(*), MIN(dt), MAX(dt), MAX(collected_at)
    FROM stock_candles WHERE interval = '1d'
    UNION ALL
    SELECT 'stock_candles_1m', '분봉 캔들',
           COUNT(*), MIN(dt), MAX(dt), MAX(collected_at)
    FROM stock_candles WHERE interval = '1m'
    UNION ALL
    SELECT 'investor_trading', '투자자별 매매동향',
           COUNT(*), MIN(dt), MAX(dt), MAX(collected_at)
    FROM investor_trading
    UNION ALL
    SELECT 'margin_short_daily', '신용잔고/공매도',
           COUNT(*), MIN(dt), MAX(dt), MAX(collected_at)
    FROM margin_short_daily
    UNION ALL
    SELECT 'dart_financials', 'DART 재무',
           COUNT(*), MIN(disclosure_date), MAX(disclosure_date), MAX(collected_at)
    FROM dart_financials
    UNION ALL
    SELECT 'program_trading', '프로그램 매매',
           COUNT(*), MIN(dt), MAX(dt), MAX(collected_at)
    FROM program_trading
    UNION ALL
    SELECT 'news_articles', '뉴스 기사',
           COUNT(*), MIN(published_at), MAX(published_at), MAX(created_at)
    FROM news_articles
    UNION ALL
    SELECT 'discussion_posts', '종토방',
           COUNT(*), MIN(published_at), MAX(published_at), MAX(created_at)
    FROM discussion_posts
    """
    async with async_session() as session:
        result = await session.execute(text(sql))
        rows = result.fetchall()

    return [
        CollectionStatusItem(
            table_name=r[0], display_name=r[1],
            total_rows=r[2] or 0,
            earliest_date=_fmt_date(r[3]),
            latest_date=_fmt_date(r[4]),
            last_collected_at=_fmt_date(r[5]),
        ) for r in rows
    ]


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
        params["start"] = start
    if end:
        clauses.append("t.dt <= :end")
        params["end"] = end

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
        params["start"] = start
    if end:
        clauses.append("t.dt <= :end")
        params["end"] = end

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
        # program_trading.dt는 DateTime → 날짜 범위를 timestamp으로 변환 (인덱스 활용)
        clauses.append("t.dt >= :start")
        params["start"] = start
    if end:
        clauses.append("t.dt < :end_next")
        params["end_next"] = end + timedelta(days=1)

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    offset = page * limit
    params["limit"] = limit
    params["offset"] = offset

    async with async_session() as session:
        total = await _count_query(session, "program_trading t", where, {k: v for k, v in params.items() if k not in ("limit", "offset")})
        sql = (
            f"SELECT t.id, t.symbol, m.name, t.dt, "
            f"t.pgm_buy_amount, t.pgm_sell_amount, t.pgm_net_amount, "
            f"COALESCE(t.arbt_buy_amount, 0), COALESCE(t.arbt_sell_amount, 0), "
            f"COALESCE(t.nabt_buy_amount, 0), COALESCE(t.nabt_sell_amount, 0), t.collected_at "
            f"FROM program_trading t LEFT JOIN stock_masters m ON t.symbol = m.symbol "
            f"{where} ORDER BY t.dt DESC LIMIT :limit OFFSET :offset"
        )
        result = await session.execute(text(sql), params)
        rows = result.fetchall()

    items = [
        ProgramTradingRow(
            id=r[0], symbol=r[1], name=r[2], dt=_fmt_date(r[3]),
            pgm_buy_amount=r[4], pgm_sell_amount=r[5], pgm_net_amount=r[6],
            arbt_buy_amount=r[7], arbt_sell_amount=r[8],
            nabt_buy_amount=r[9], nabt_sell_amount=r[10],
            collected_at=_fmt_date(r[11]),
        ) for r in rows
    ]
    return PagedResponse(items=items, total=total, page=page, limit=limit)


@router.get("/news")
async def get_news(
    symbol: str | None = Query(None),
    start: date | None = Query(None),
    end: date | None = Query(None),
    page: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
) -> PagedResponse[NewsExplorerRow]:
    clauses: list[str] = []
    params: dict = {}
    if symbol:
        # symbols는 JSON 배열 → JSONB 캐스팅 + GIN 인덱스(@> 연산자) 활용
        clauses.append("n.symbols::jsonb @> :sym_json::jsonb")
        params["sym_json"] = json.dumps([symbol])
    if start:
        # published_at은 DateTime → 범위 비교로 인덱스 활용
        clauses.append("n.published_at >= :start")
        params["start"] = start
    if end:
        clauses.append("n.published_at < :end_next")
        params["end_next"] = end + timedelta(days=1)

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    offset = page * limit
    params["limit"] = limit
    params["offset"] = offset

    async with async_session() as session:
        total = await _count_query(session, "news_articles n", where, {k: v for k, v in params.items() if k not in ("limit", "offset")})

        sql = (
            f"SELECT n.id, n.symbols, n.source, n.title, n.url, n.published_at, "
            f"n.sentiment_score, n.market_impact "
            f"FROM news_articles n "
            f"{where} "
            f"ORDER BY n.published_at DESC LIMIT :limit OFFSET :offset"
        )
        result = await session.execute(text(sql), params)
        rows = result.fetchall()

        # stock_masters 이름 캐시 (한 번만 조회)
        name_result = await session.execute(text("SELECT symbol, name FROM stock_masters"))
        name_map: dict[str, str] = {r[0]: r[1] for r in name_result.fetchall()}

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


@router.get("/discussion")
async def get_discussion(
    symbol: str | None = Query(None),
    start: date | None = Query(None),
    end: date | None = Query(None),
    page: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
) -> PagedResponse[DiscussionExplorerRow]:
    """종토방 게시글 조회."""
    clauses: list[str] = []
    params: dict = {}
    if symbol:
        clauses.append("d.symbol = :symbol")
        params["symbol"] = symbol
    if start:
        clauses.append("d.published_at >= :start")
        params["start"] = start
    if end:
        clauses.append("d.published_at < :end_next")
        params["end_next"] = end + timedelta(days=1)

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    offset = page * limit
    params["limit"] = limit
    params["offset"] = offset

    async with async_session() as session:
        total = await _count_query(
            session, "discussion_posts d", where,
            {k: v for k, v in params.items() if k not in ("limit", "offset")},
        )

        sql = (
            f"SELECT d.id, d.symbol, d.title, d.content, d.author, "
            f"d.published_at, d.likes, d.dislikes, d.comment_count, d.sentiment_score "
            f"FROM discussion_posts d "
            f"{where} "
            f"ORDER BY d.published_at DESC LIMIT :limit OFFSET :offset"
        )
        result = await session.execute(text(sql), params)
        rows = result.fetchall()

        name_result = await session.execute(text("SELECT symbol, name FROM stock_masters"))
        name_map: dict[str, str] = {r[0]: r[1] for r in name_result.fetchall()}

    items = [
        DiscussionExplorerRow(
            id=str(r[0]), symbol=r[1], name=name_map.get(r[1]),
            title=r[2], content=(r[3] or "")[:200], author=r[4],
            published_at=_fmt_date(r[5]),
            likes=r[6] or 0, dislikes=r[7] or 0, comment_count=r[8] or 0,
            sentiment_score=r[9],
        )
        for r in rows
    ]
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


# ── 데이터 검증 (SSE) ──


_VERIFY_SOURCES: dict[str, dict] = {
    "daily_candle": {
        "display": "일봉 캔들",
        "date_sql": "SELECT DISTINCT dt::date AS d FROM stock_candles WHERE interval = '1d'",
        "count_sql": "SELECT COUNT(DISTINCT symbol) FROM stock_candles WHERE interval = '1d' AND dt::date = :d",
    },
    "minute_candle": {
        "display": "분봉 캔들",
        "date_sql": "SELECT DISTINCT dt::date AS d FROM stock_candles WHERE interval = '1m'",
        "count_sql": "SELECT COUNT(DISTINCT symbol) FROM stock_candles WHERE interval = '1m' AND dt::date = :d",
    },
    "margin_short": {
        "display": "신용잔고/공매도",
        "date_sql": "SELECT DISTINCT dt AS d FROM margin_short_daily",
        "count_sql": "SELECT COUNT(*) FROM margin_short_daily WHERE dt = :d AND (margin_balance > 0 OR short_volume > 0)",
    },
    "investor": {
        "display": "투자자별 매매동향",
        "date_sql": "SELECT DISTINCT dt::date AS d FROM investor_trading",
        "count_sql": "SELECT COUNT(*) FROM investor_trading WHERE dt::date = :d",
    },
    "program_trading": {
        "display": "프로그램 매매",
        "date_sql": "SELECT DISTINCT dt::date AS d FROM program_trading",
        "count_sql": "SELECT COUNT(*) FROM program_trading WHERE dt::date = :d AND (pgm_buy_amount > 0 OR pgm_sell_amount > 0)",
    },
}


def _sse_msg(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


@router.get("/verify/{source}")
async def verify_data(
    source: str,
    start_date: date | None = Query(None),
    end_date: date | None = Query(None),
    lookback_days: int = Query(90, ge=7, le=365),
):
    """데이터 갭 검증 — SSE 스트림으로 실시간 진행 로그 전송.

    /real-log 원칙: 매 스텝마다 진행률, 갭 즉시 보고, 에러 숨기지 않음.
    """
    if source not in _VERIFY_SOURCES:
        raise HTTPException(400, f"지원하지 않는 소스: {source}. 가능: {list(_VERIFY_SOURCES.keys())}")

    cfg = _VERIFY_SOURCES[source]

    async def _stream():
        yield _sse_msg({"type": "start", "source": source, "display": cfg["display"]})

        try:
            # 영업일 목록 (일봉 캔들이 있는 날짜를 기준으로)
            _end = end_date or date.today()
            _start = start_date or (_end - timedelta(days=lookback_days))

            async with async_session() as session:
                result = await session.execute(text(
                    "SELECT DISTINCT dt::date AS d FROM stock_candles "
                    "WHERE interval = '1d' AND dt::date >= :s AND dt::date <= :e ORDER BY d"
                ), {"s": _start, "e": _end})
                trading_days = [r[0] for r in result.fetchall()]

            if not trading_days:
                yield _sse_msg({"type": "error", "message": f"영업일 데이터 없음 ({_start} ~ {_end})"})
                yield _sse_msg({"type": "done", "verified_until": None, "gaps": [], "total_gaps": 0})
                return

            yield _sse_msg({
                "type": "progress", "pct": 0,
                "message": f"{cfg['display']} 검증 시작: {_start} ~ {_end} ({len(trading_days)} 영업일)",
            })

            # 해당 소스의 실제 데이터 날짜 조회
            async with async_session() as session:
                result = await session.execute(text(cfg["date_sql"]))
                existing_dates = {r[0] for r in result.fetchall()}

            gaps: list[dict] = []
            verified_until: date | None = None
            checked = 0

            for td in trading_days:
                checked += 1
                pct = int(checked / len(trading_days) * 100)

                if td not in existing_dates:
                    gaps.append({"date": td.isoformat(), "type": "missing", "message": f"{td} 데이터 없음"})
                    yield _sse_msg({"type": "gap", "date": td.isoformat(), "message": f"{td} 데이터 없음"})
                else:
                    # 데이터는 있지만 유효한지 (non-zero) 확인
                    async with async_session() as session:
                        result = await session.execute(text(cfg["count_sql"]), {"d": td})
                        cnt = result.scalar() or 0
                    if cnt == 0:
                        gaps.append({"date": td.isoformat(), "type": "empty", "message": f"{td} 데이터 0건 (빈 값)"})
                        yield _sse_msg({"type": "gap", "date": td.isoformat(), "message": f"{td} 데이터 0건 (빈 값)"})
                    else:
                        if not gaps or gaps[-1]["date"] != td.isoformat():
                            verified_until = td

                # 10일마다 진행 보고
                if checked % 10 == 0:
                    yield _sse_msg({
                        "type": "progress", "pct": pct,
                        "message": f"{checked}/{len(trading_days)} 영업일 검증 완료 (갭 {len(gaps)}건)",
                    })

                await asyncio.sleep(0)  # yield control

            yield _sse_msg({
                "type": "done",
                "verified_until": verified_until.isoformat() if verified_until else None,
                "gaps": gaps,
                "total_gaps": len(gaps),
                "total_checked": len(trading_days),
                "message": f"검증 완료: {len(trading_days)}일 중 {len(gaps)}건 갭 발견",
            })

        except Exception as e:
            yield _sse_msg({"type": "error", "message": f"검증 오류: {e}"})
            yield _sse_msg({"type": "done", "verified_until": None, "gaps": [], "total_gaps": -1})

    return StreamingResponse(_stream(), media_type="text/event-stream")


class RecollectRequest(BaseModel):
    dates: list[str]
    symbols: list[str] | None = None


@router.post("/recollect/{source}")
async def recollect_data(source: str, req: RecollectRequest):
    """특정 날짜의 데이터 재수집 — SSE 스트림으로 진행 로그.

    기존 수집기(pykrx, KIS API)를 활용하여 지정된 날짜만 재수집.
    """
    if source not in ("daily_candle", "margin_short", "investor"):
        raise HTTPException(400, f"재수집 지원 소스: daily_candle, margin_short, investor")

    if not req.dates:
        raise HTTPException(400, "재수집 날짜를 지정해주세요")

    async def _stream():
        yield _sse_msg({"type": "start", "source": source, "dates": req.dates})
        success = 0
        fail = 0

        for i, d_str in enumerate(req.dates):
            try:
                d = date.fromisoformat(d_str)
            except ValueError:
                yield _sse_msg({"type": "error", "message": f"날짜 형식 오류: {d_str}"})
                fail += 1
                continue

            pct = int((i + 1) / len(req.dates) * 100)

            try:
                if source == "daily_candle":
                    from app.scheduler.collectors.daily_candle import collect_daily_candles
                    cnt = await collect_daily_candles(d)
                    yield _sse_msg({"type": "progress", "pct": pct, "message": f"{d} 일봉 {cnt}건 수집"})
                    success += 1

                elif source == "margin_short":
                    from app.scheduler.collectors.margin_short import collect_margin_short
                    cnt = await collect_margin_short(d)
                    yield _sse_msg({"type": "progress", "pct": pct, "message": f"{d} 신용/공매도 {cnt}건 수집"})
                    success += 1

                elif source == "investor":
                    from app.scheduler.collectors.investor import collect_investor_trading
                    cnt = await collect_investor_trading(d)
                    yield _sse_msg({"type": "progress", "pct": pct, "message": f"{d} 투자자매매 {cnt}건 수집"})
                    success += 1

            except Exception as e:
                fail += 1
                yield _sse_msg({"type": "error", "message": f"{d} 수집 실패: {e}"})

            await asyncio.sleep(0.1)  # rate limit

        yield _sse_msg({
            "type": "done",
            "success": success,
            "fail": fail,
            "message": f"재수집 완료: {success}건 성공, {fail}건 실패",
        })

    return StreamingResponse(_stream(), media_type="text/event-stream")
