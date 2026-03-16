"""FastMCP 서버 — Stock Mate 도구 16개 + 리소스 1개.

기존 5개: execute_order, query_stock_data, fetch_sentiment, run_alpha_scan, get_portfolio_status
워크플로우 4개: get_workflow_status, get_best_factors, get_trading_review, submit_trading_feedback
보조 4개: get_system_health, trigger_workflow_phase, get_error_logs, get_factor_performance
마이닝 제어 3개: start_alpha_mining, stop_alpha_mining, get_mining_status
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date, timedelta

from fastmcp import FastMCP
from sqlalchemy import select

from app.core.database import async_session
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
    from app.trading.kis_client import get_kis_client
    from app.trading.kis_order import KISOrderExecutor

    is_mock = mode != "real"
    client = get_kis_client(is_mock=is_mock)
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


# ── Tool 6: get_workflow_status ───────────────────────────


@mcp.tool()
async def get_workflow_status() -> str:
    """현재 워크플로우 FSM 상태 + 오늘 실적 요약.

    반환: phase, status, 선택된 팩터, 매매 건수, PnL, 마이닝 상태.
    """
    start_ms = time.monotonic()
    from app.workflow.orchestrator import get_orchestrator

    orch = get_orchestrator()
    status = await orch.get_status()

    elapsed = int((time.monotonic() - start_ms) * 1000)
    await audit_log("get_workflow_status", {}, {"phase": status.get("phase")}, "success", execution_ms=elapsed)

    return json.dumps(status, ensure_ascii=False, default=str)


# ── Tool 7: get_best_factors ─────────────────────────────


@mcp.tool()
async def get_best_factors(
    limit: int = 5,
    min_sharpe: float = 0.0,
    min_ic: float = 0.0,
    require_causal: bool = False,
    interval: str = "",
) -> str:
    """매매 가능한 최상위 팩터 목록 (복합 점수 기준).

    Args:
        limit: 반환할 최대 팩터 수 (기본 5)
        min_sharpe: 최소 Sharpe 필터 (기본 0)
        min_ic: 최소 IC 필터 (기본 0)
        require_causal: 인과 검증 통과 팩터만 필터 (기본 false)
        interval: 데이터 인터벌 필터 (예: 1d, 비어있으면 기본값)

    반환: 팩터 이름, 수식, IC, Sharpe, MDD, 인과 검증, 복합 점수.
    """
    start_ms = time.monotonic()
    from app.workflow.auto_selector import select_best_factors

    async with async_session() as session:
        factors = await select_best_factors(
            session,
            limit=limit,
            min_ic=min_ic if min_ic > 0 else None,
            min_sharpe=min_sharpe if min_sharpe > 0 else None,
            require_causal=require_causal if require_causal else None,
            interval=interval if interval else None,
        )

    result = []
    for f in factors:
        factor = f["factor"]
        result.append({
            "name": factor.name,
            "expression": factor.expression_str[:100],
            "ic_mean": round(factor.ic_mean or 0, 6),
            "sharpe": round(factor.sharpe or 0, 4),
            "max_drawdown": round(getattr(factor, "max_drawdown", 0) or 0, 4),
            "causal_robust": factor.causal_robust,
            "composite_score": round(f["score"], 4),
            "breakdown": f["breakdown"],
        })

    elapsed = int((time.monotonic() - start_ms) * 1000)
    await audit_log("get_best_factors", {"limit": limit}, {"count": len(result)}, "success", execution_ms=elapsed)

    return json.dumps(result, ensure_ascii=False)


# ── Tool 8: get_trading_review ───────────────────────────


@mcp.tool()
async def get_trading_review(target_date: str = "") -> str:
    """특정 일자의 매매 리뷰 (성과 지표 + 거래 요약).

    Args:
        target_date: YYYY-MM-DD (비어있으면 오늘)

    반환: 총 수익률, 거래 수, 승률, 평균 보유시간, 개선 방향.
    """
    start_ms = time.monotonic()
    from datetime import date as d

    from app.workflow.models import WorkflowRun

    target = d.fromisoformat(target_date) if target_date else d.today()

    async with async_session() as session:
        stmt = select(WorkflowRun).where(WorkflowRun.date == target)
        result = await session.execute(stmt)
        run = result.scalar_one_or_none()

        if run is None:
            return json.dumps({"error": f"{target} 워크플로우 기록 없음"}, ensure_ascii=False)

        review = run.review_summary or {}
        data = {
            "date": str(target),
            "phase": run.phase,
            "status": run.status,
            "trade_count": run.trade_count,
            "pnl_pct": run.pnl_pct,
            "pnl_amount": float(run.pnl_amount) if run.pnl_amount else None,
            "review_summary": review,
            "mining_context": run.mining_context,
        }

    elapsed = int((time.monotonic() - start_ms) * 1000)
    await audit_log("get_trading_review", {"date": str(target)}, {"phase": data.get("phase")}, "success", execution_ms=elapsed)

    return json.dumps(data, ensure_ascii=False, default=str)


# ── Tool 8b: get_trade_details ────────────────────────────


@mcp.tool()
async def get_trade_details(target_date: str = "", limit: int = 50) -> str:
    """당일 개별 매매 상세 — Top 수익/손실, 보유시간 분포, 청산 유형별 집계.

    REVIEW 전에도 live_trades 테이블에서 직접 조회 가능.

    Args:
        target_date: YYYY-MM-DD (비어있으면 오늘)
        limit: 반환할 최대 거래 수

    반환: Top 수익/손실 거래, 보유시간 분포, 청산유형별 집계, 종합 요약.
    """
    start_ms = time.monotonic()
    from datetime import date as d

    from app.workflow.models import LiveTrade

    target = d.fromisoformat(target_date) if target_date else d.today()

    async with async_session() as session:
        from sqlalchemy import func as sqla_func

        # 당일 SELL 거래 조회 (PnL 정보가 있는 건)
        stmt = (
            select(LiveTrade)
            .where(
                sqla_func.date(LiveTrade.executed_at) == target,
                LiveTrade.side == "SELL",
            )
            .order_by(LiveTrade.executed_at)
            .limit(limit)
        )
        result = await session.execute(stmt)
        trades = list(result.scalars().all())

        if not trades:
            return json.dumps(
                {"date": str(target), "total_trades": 0, "message": "거래 없음"},
                ensure_ascii=False,
            )

        # 종합 요약
        pnls = [t.pnl_pct or 0 for t in trades]
        pnl_amounts = [float(t.pnl_amount) if t.pnl_amount else 0 for t in trades]
        holdings = [t.holding_minutes or 0 for t in trades]
        wins = [p for p in pnls if p > 0]

        summary = {
            "total_pnl_pct": round(sum(pnls), 4),
            "total_pnl_amount": round(sum(pnl_amounts), 2),
            "win_rate": round(len(wins) / len(trades) * 100, 2) if trades else 0,
            "avg_holding_minutes": round(sum(holdings) / len(holdings), 1) if holdings else 0,
            "avg_pnl_pct": round(sum(pnls) / len(pnls), 4) if pnls else 0,
        }

        # Top 3 수익/손실
        sorted_by_pnl = sorted(trades, key=lambda t: t.pnl_pct or 0, reverse=True)
        top_gains = [
            {
                "symbol": t.symbol,
                "name": t.name,
                "pnl_pct": round(t.pnl_pct or 0, 4),
                "pnl_amount": round(float(t.pnl_amount), 2) if t.pnl_amount else 0,
                "holding_minutes": round(t.holding_minutes or 0, 1),
                "step": t.step,
                "qty": t.qty,
            }
            for t in sorted_by_pnl[:3]
            if (t.pnl_pct or 0) > 0
        ]
        top_losses = [
            {
                "symbol": t.symbol,
                "name": t.name,
                "pnl_pct": round(t.pnl_pct or 0, 4),
                "pnl_amount": round(float(t.pnl_amount), 2) if t.pnl_amount else 0,
                "holding_minutes": round(t.holding_minutes or 0, 1),
                "step": t.step,
                "qty": t.qty,
            }
            for t in reversed(sorted_by_pnl)
            if (t.pnl_pct or 0) < 0
        ][:3]

        # 보유시간 분포
        dist = {"under_5m": 0, "5m_to_15m": 0, "15m_to_30m": 0, "30m_to_60m": 0, "over_60m": 0}
        for h in holdings:
            if h < 5:
                dist["under_5m"] += 1
            elif h < 15:
                dist["5m_to_15m"] += 1
            elif h < 30:
                dist["15m_to_30m"] += 1
            elif h < 60:
                dist["30m_to_60m"] += 1
            else:
                dist["over_60m"] += 1

        # 청산 유형별 집계
        exit_types: dict[str, int] = {}
        for t in trades:
            step = t.step or "UNKNOWN"
            exit_types[step] = exit_types.get(step, 0) + 1

        data = {
            "date": str(target),
            "total_trades": len(trades),
            "summary": summary,
            "top_gains": top_gains,
            "top_losses": top_losses,
            "holding_distribution": dist,
            "exit_type_breakdown": exit_types,
        }

    elapsed = int((time.monotonic() - start_ms) * 1000)
    await audit_log("get_trade_details", {"date": str(target)}, {"count": len(trades)}, "success", execution_ms=elapsed)

    return json.dumps(data, ensure_ascii=False, default=str)


# ── Tool 9: submit_trading_feedback ──────────────────────


@mcp.tool()
async def submit_trading_feedback(
    factor_id: str,
    review_text: str = "",
    improvement_suggestions: str = "",
    market_regime: str = "",
) -> str:
    """매매 피드백 제출 → 다음 마이닝 사이클에 반영.

    OpenClaw의 포스트마켓 분석 결과를 백엔드에 저장.

    Args:
        factor_id: 피드백 대상 팩터 UUID
        review_text: 분석 텍스트
        improvement_suggestions: 쉼표 구분 개선 제안
        market_regime: 시장 체제 (예: bull, bear, sideways)
    """
    start_ms = time.monotonic()
    from datetime import date as d

    from app.workflow.models import WorkflowRun

    async with async_session() as session:
        today = d.today()
        stmt = select(WorkflowRun).where(WorkflowRun.date == today)
        result = await session.execute(stmt)
        run = result.scalar_one_or_none()

        if run:
            existing = run.mining_context or ""
            feedback_text = f"\n[OpenClaw 피드백] {review_text}"
            if improvement_suggestions:
                feedback_text += f"\n개선 제안: {improvement_suggestions}"
            if market_regime:
                feedback_text += f"\n시장 체제: {market_regime}"
            run.mining_context = existing + feedback_text

        await session.commit()

    elapsed = int((time.monotonic() - start_ms) * 1000)
    await audit_log(
        "submit_trading_feedback",
        {"factor_id": factor_id, "regime": market_regime},
        {"applied": run is not None},
        "success",
        execution_ms=elapsed,
    )

    return json.dumps({
        "success": True,
        "message": "피드백이 다음 마이닝 컨텍스트에 반영됩니다.",
    }, ensure_ascii=False)


# ── Tool 10: get_system_health ───────────────────────────


@mcp.tool()
async def get_system_health() -> str:
    """시스템 헬스 상태 (DB, 스케줄러, 세션, 메모리).

    OpenClaw 야간 점검용.
    """
    start_ms = time.monotonic()

    try:
        import httpx
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get("http://localhost:8007/health/detailed")
            data = resp.text
    except Exception as e:
        data = json.dumps({"error": f"헬스 API 호출 실패: {e}"}, ensure_ascii=False)

    elapsed = int((time.monotonic() - start_ms) * 1000)
    await audit_log("get_system_health", {}, {}, "success", execution_ms=elapsed)

    return data


# ── Tool 11: trigger_workflow_phase ──────────────────────


@mcp.tool()
async def trigger_workflow_phase(phase: str) -> str:
    """워크플로우 페이즈 수동 트리거.

    Args:
        phase: pre_market, market_open, market_close, review, mining, emergency_stop, resume

    OpenClaw 긴급 제어용.
    """
    start_ms = time.monotonic()
    from app.workflow.orchestrator import get_orchestrator

    orch = get_orchestrator()

    handlers = {
        "pre_market": orch.handle_pre_market,
        "market_open": orch.handle_market_open,
        "market_close": orch.handle_market_close,
        "review": orch.handle_review,
        "mining": orch.handle_mining,
        "emergency_stop": orch.handle_emergency_stop,
        "resume": orch.handle_resume,
    }

    handler = handlers.get(phase)
    if not handler:
        return json.dumps({"error": f"알 수 없는 phase: {phase}"}, ensure_ascii=False)

    result = await handler()

    elapsed = int((time.monotonic() - start_ms) * 1000)
    await audit_log("trigger_workflow_phase", {"phase": phase}, result, "success", execution_ms=elapsed)

    return json.dumps(result, ensure_ascii=False, default=str)


# ── Tool 12: get_error_logs ──────────────────────────────


@mcp.tool()
async def get_error_logs(limit: int = 20) -> str:
    """최근 워크플로우 에러 로그 조회.

    Args:
        limit: 반환할 최대 에러 수 (기본 20)

    OpenClaw 버그 탐지용.
    """
    start_ms = time.monotonic()
    from app.workflow.models import WorkflowEvent

    async with async_session() as session:
        stmt = (
            select(WorkflowEvent)
            .where(WorkflowEvent.event_type.in_([
                "error", "emergency_stop", "trading_skip",
                "no_factor", "non_trading_day",
            ]))
            .order_by(WorkflowEvent.created_at.desc())
            .limit(limit)
        )
        result = await session.execute(stmt)
        events = result.scalars().all()

        data = [
            {
                "id": str(e.id),
                "phase": e.phase,
                "event_type": e.event_type,
                "message": e.message,
                "created_at": e.created_at.isoformat() if e.created_at else None,
            }
            for e in events
        ]

    elapsed = int((time.monotonic() - start_ms) * 1000)
    await audit_log("get_error_logs", {"limit": limit}, {"count": len(data)}, "success", execution_ms=elapsed)

    return json.dumps(data, ensure_ascii=False)


# ── Tool 13: get_factor_performance ──────────────────────


@mcp.tool()
async def get_factor_performance(
    factor_id: str = "",
    days: int = 7,
) -> str:
    """팩터별 실매매 성적 조회 (설계서 §4.1.1 #13).

    Args:
        factor_id: 팩터 UUID (비어있으면 최근 사용된 팩터 전체)
        days: 조회 기간 일수 (기본 7)

    반환: 팩터별 일별 PnL, 승률, 거래 수, 백테스트 IC 대비 실매매 IC 갭.
    """
    start_ms = time.monotonic()
    from datetime import date as d, timedelta
    from uuid import UUID

    from sqlalchemy import func

    from app.alpha.models import AlphaFactor
    from app.workflow.models import LiveFeedback, WorkflowRun

    cutoff = d.today() - timedelta(days=days)

    async with async_session() as session:
        if factor_id:
            factor_ids = [UUID(factor_id)]
        else:
            # 최근 N일간 사용된 팩터 조회
            stmt = (
                select(WorkflowRun.selected_factor_id)
                .where(
                    WorkflowRun.date >= cutoff,
                    WorkflowRun.selected_factor_id.isnot(None),
                )
                .distinct()
            )
            result = await session.execute(stmt)
            factor_ids = [row[0] for row in result.all()]

        if not factor_ids:
            elapsed = int((time.monotonic() - start_ms) * 1000)
            await audit_log("get_factor_performance", {"factor_id": factor_id, "days": days}, {"count": 0}, "success", execution_ms=elapsed)
            return json.dumps({"factors": [], "message": f"최근 {days}일 매매 실적 없음"}, ensure_ascii=False)

        performances = []
        for fid in factor_ids:
            # 팩터 기본 정보
            factor_result = await session.execute(
                select(AlphaFactor).where(AlphaFactor.id == fid)
            )
            factor = factor_result.scalar_one_or_none()
            if not factor:
                continue

            # 실매매 피드백
            fb_stmt = (
                select(LiveFeedback)
                .where(LiveFeedback.factor_id == fid, LiveFeedback.date >= cutoff)
                .order_by(LiveFeedback.date.desc())
            )
            fb_result = await session.execute(fb_stmt)
            feedbacks = fb_result.scalars().all()

            daily_pnl = []
            total_pnl = 0.0
            total_trades = 0
            total_wins = 0
            for fb in feedbacks:
                pnl = fb.realized_pnl_pct or 0
                trades = fb.trade_count or 0
                wins = round((fb.win_rate or 0) / 100 * trades) if trades > 0 else 0
                daily_pnl.append({
                    "date": str(fb.date),
                    "pnl_pct": round(pnl, 4),
                    "trade_count": trades,
                    "win_rate": round(fb.win_rate or 0, 1),
                })
                total_pnl += pnl
                total_trades += trades
                total_wins += wins

            avg_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0

            performances.append({
                "factor_id": str(fid),
                "name": factor.name,
                "expression": (factor.expression_str or "")[:80],
                "backtest_ic": round(factor.ic_mean or 0, 6),
                "backtest_sharpe": round(factor.sharpe or 0, 4),
                "causal_robust": factor.causal_robust,
                "live_days": len(feedbacks),
                "live_total_pnl_pct": round(total_pnl, 4),
                "live_avg_daily_pnl_pct": round(total_pnl / len(feedbacks), 4) if feedbacks else 0,
                "live_total_trades": total_trades,
                "live_win_rate": round(avg_win_rate, 1),
                "daily_pnl": daily_pnl,
            })

    elapsed = int((time.monotonic() - start_ms) * 1000)
    await audit_log(
        "get_factor_performance",
        {"factor_id": factor_id, "days": days},
        {"count": len(performances)},
        "success",
        execution_ms=elapsed,
    )

    return json.dumps({"factors": performances}, ensure_ascii=False)


# ── Tool 14: start_alpha_mining ─────────────────────────────


@mcp.tool()
async def start_alpha_mining(
    context: str = "",
    universe: str = "KOSPI200",
    data_interval: str = "5m",
    max_cycles: int = 10,
    interval_minutes: int = 0,
) -> str:
    """알파 팩토리 마이닝을 시작한다. 이미 실행 중이면 현재 상태를 반환한다.

    Args:
        context: 마이닝 컨텍스트 (예: "모멘텀 팩터 중심 탐색")
        universe: 종목 유니버스 (KOSPI200, KOSDAQ150, KRX300, ALL)
        data_interval: 데이터 간격 (1d, 5m, 1h 등)
        max_cycles: 최대 사이클 수 (0이면 무제한)
        interval_minutes: 사이클 간 대기(분) (0이면 연속 실행)
    """
    start_ms = time.monotonic()
    params = {
        "context": context,
        "universe": universe,
        "data_interval": data_interval,
        "max_cycles": max_cycles,
        "interval_minutes": interval_minutes,
    }

    from app.alpha.factory_client import get_factory_client

    client = get_factory_client()
    current = await client.get_status()

    if current["running"]:
        elapsed = int((time.monotonic() - start_ms) * 1000)
        result = {
            "success": False,
            "already_running": True,
            "message": (
                f"알파 팩토리가 이미 실행 중입니다. "
                f"현재 {current['cycles_completed']}사이클 완료, "
                f"{current['factors_discovered_total']}개 팩터 발견."
            ),
            "status": current,
        }
        await audit_log(
            "start_alpha_mining", params, result, "blocked",
            "already_running", execution_ms=elapsed,
        )
        return json.dumps(result, ensure_ascii=False, default=str)

    end_date = date.today()
    start_date = end_date - timedelta(days=365)

    fc_result = await client.start(
        context=context,
        universe=universe,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        data_interval=data_interval,
        interval_minutes=interval_minutes,
        max_cycles=max_cycles if max_cycles > 0 else None,
    )
    started = fc_result["started"]

    elapsed = int((time.monotonic() - start_ms) * 1000)

    if started:
        result = {
            "success": True,
            "message": "알파 팩토리 마이닝을 시작했습니다.",
            "config": {
                "context": context or "(기본)",
                "universe": universe,
                "data_interval": data_interval,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "max_cycles": max_cycles,
                "interval_minutes": interval_minutes,
            },
        }
        await audit_log(
            "start_alpha_mining", params, result, "success",
            execution_ms=elapsed,
        )
    else:
        result = {
            "success": False,
            "message": "알파 팩토리 시작에 실패했습니다.",
        }
        await audit_log(
            "start_alpha_mining", params, result, "error",
            execution_ms=elapsed,
        )

    return json.dumps(result, ensure_ascii=False, default=str)


# ── Tool 15: stop_alpha_mining ──────────────────────────────


@mcp.tool()
async def stop_alpha_mining() -> str:
    """실행 중인 알파 팩토리 마이닝을 중지한다.

    실행 중이 아니면 현재 상태를 반환한다.
    """
    start_ms = time.monotonic()

    from app.alpha.factory_client import get_factory_client

    client = get_factory_client()
    current = await client.get_status()

    if not current["running"]:
        elapsed = int((time.monotonic() - start_ms) * 1000)
        result = {
            "success": False,
            "not_running": True,
            "message": "알파 팩토리가 실행 중이 아닙니다.",
            "last_status": current,
        }
        await audit_log(
            "stop_alpha_mining", {}, result, "blocked",
            "not_running", execution_ms=elapsed,
        )
        return json.dumps(result, ensure_ascii=False, default=str)

    stop_result = await client.stop()
    final = stop_result["status"]
    elapsed = int((time.monotonic() - start_ms) * 1000)

    result = {
        "success": stopped,
        "message": (
            f"알파 팩토리 마이닝을 중지했습니다. "
            f"총 {final['cycles_completed']}사이클 완료, "
            f"{final['factors_discovered_total']}개 팩터 발견."
        ),
        "final_status": final,
    }
    await audit_log(
        "stop_alpha_mining", {}, result, "success",
        execution_ms=elapsed,
    )

    return json.dumps(result, ensure_ascii=False, default=str)


# ── Tool 16: get_mining_status ──────────────────────────────


@mcp.tool()
async def get_mining_status() -> str:
    """알파 팩토리 마이닝 현재 상태를 조회한다.

    실행 여부, 진행 사이클, 발견 팩터 수, 설정 등을 반환한다.
    """
    start_ms = time.monotonic()

    from app.alpha.factory_client import get_factory_client

    client = get_factory_client()
    status = await client.get_status()

    if status["running"]:
        max_c = (status.get("config") or {}).get("max_cycles")
        cycle_str = str(status["cycles_completed"])
        if max_c:
            cycle_str += f"/{max_c}"
        summary = (
            f"알파 팩토리 실행 중: {cycle_str} 사이클 완료, "
            f"{status['factors_discovered_total']}개 팩터 발견, "
            f"현재 사이클 진행률 {status['current_cycle_progress']}%"
        )
    elif status["cycles_completed"] > 0:
        summary = (
            f"알파 팩토리 중지됨. "
            f"마지막 실행: {status['cycles_completed']}사이클 완료, "
            f"{status['factors_discovered_total']}개 팩터 발견."
        )
    else:
        summary = "알파 팩토리가 아직 실행된 적 없습니다."

    result = {**status, "summary": summary}
    elapsed = int((time.monotonic() - start_ms) * 1000)

    await audit_log(
        "get_mining_status", {},
        {"running": status["running"], "cycles": status["cycles_completed"]},
        "success", execution_ms=elapsed,
    )

    return json.dumps(result, ensure_ascii=False, default=str)


# ── Tool 17: get_causal_validation_status ─────────────────


@mcp.tool()
async def get_causal_validation_status(
    job_id: str = "",
) -> str:
    """인과 검증 진행 상황을 조회한다.

    job_id를 지정하면 해당 잡, 비어 있으면 최근 잡의 상태를 반환한다.
    진행률(completed/total), 예상 잔여 시간, robust/mirage 수 등을 포함한다.

    Args:
        job_id: 검증 잡 ID (비어 있으면 최근 잡 자동 선택)
    """
    start_ms = time.monotonic()

    from app.alpha.factory_client import get_factory_client

    client = get_factory_client()

    if job_id:
        progress = await client.get_validation_progress(job_id)
        if not progress:
            result = {"success": False, "message": f"잡 {job_id}를 찾을 수 없습니다."}
            elapsed = int((time.monotonic() - start_ms) * 1000)
            await audit_log(
                "get_causal_validation_status", {"job_id": job_id},
                result, "error", "not_found", execution_ms=elapsed,
            )
            return json.dumps(result, ensure_ascii=False)
        result_data = {"job_id": job_id, **progress}
    else:
        latest = await client.get_latest_validation_job()
        if not latest:
            result = {"success": False, "message": "진행 중이거나 최근 완료된 인과 검증 잡이 없습니다."}
            elapsed = int((time.monotonic() - start_ms) * 1000)
            await audit_log(
                "get_causal_validation_status", {},
                result, "error", "no_jobs", execution_ms=elapsed,
            )
            return json.dumps(result, ensure_ascii=False)
        result_data = latest

    total = result_data.get("total", 0)
    done = result_data.get("completed", 0) + result_data.get("failed", 0)
    status = result_data.get("status", "unknown")
    remaining_ms = result_data.get("estimated_remaining_ms")
    remaining_str = (
        f"{remaining_ms / 1000:.0f}초" if remaining_ms and remaining_ms > 0 else "—"
    )

    summary = (
        f"인과 검증 {status}: {done}/{total} 완료 "
        f"(robust {result_data.get('robust', 0)}개, mirage {result_data.get('mirage', 0)}개, "
        f"실패 {result_data.get('failed', 0)}개). "
        f"예상 잔여: {remaining_str}"
    )
    result_data["summary"] = summary
    result_data["success"] = True

    elapsed = int((time.monotonic() - start_ms) * 1000)
    await audit_log(
        "get_causal_validation_status",
        {"job_id": result_data.get("job_id", "")},
        {"status": status, "done": done, "total": total},
        "success", execution_ms=elapsed,
    )

    return json.dumps(result_data, ensure_ascii=False, default=str)


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
