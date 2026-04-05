"""장중매매 리플레이 — 과거 거래일의 장중매매를 재현.

수집된 캔들 데이터를 활용하여 09:00~15:25 전체 봉을 워크포워드로
시뮬레이션한다. live_runner와 동일한 판단 로직(decision_logic)을
사용하되, DB/Redis/Telegram 등 사이드이펙트는 전부 스킵.

사용 예:
    result = await run_replay(target_date=date(2026, 3, 26))
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

import polars as pl

from app.backtest.cost_model import CostConfig, effective_buy_price, effective_sell_price
from app.backtest.engine import generate_signals
from app.core.config import settings
from app.core.stock_master import get_stock_name
from app.core.timezone import now_kst
from app.trading.context import TradingContext
from app.trading.decision_logic import (
    BuyDecision,
    evaluate_buy,
    evaluate_risk,
    evaluate_scale_in,
)
from app.trading.live_runner import (
    LivePosition,
    LiveScaleEntry,
    LiveSession,
    _collect_snapshot,
    _reduce_entries,
)

logger = logging.getLogger(__name__)


# ── 엔트리 ──────────────────────────────────────────────────


async def run_replay(
    factor_ids: list[str] | None = None,
    target_date: date | None = None,
    initial_capital: float = 100_000_000,
    max_positions: int = 10,
) -> dict:
    """장중매매 리플레이 실행.

    Parameters
    ----------
    factor_ids : 팩터 ID 목록. None이면 해당 날짜의 워크플로우 선택 팩터 사용.
    target_date : 리플레이 대상 날짜. None이면 최근 거래일 자동 결정.
    initial_capital : 세션별 초기 자본금.
    max_positions : 세션별 최대 동시 포지션 수.

    Returns
    -------
    dict : {target_date, factors, sessions, aggregate_metrics}
    """
    target = await _resolve_target_date(target_date)
    factors = await _resolve_factors(factor_ids, target)

    if not factors:
        return {
            "target_date": str(target),
            "factors": [],
            "sessions": [],
            "aggregate_metrics": {},
            "error": "해당 날짜에 사용 가능한 팩터가 없습니다.",
        }

    # 팩터별 리플레이 (병렬)
    tasks = []
    for f in factors:
        tasks.append(
            _replay_factor(
                factor_info=f,
                target_date=target,
                initial_capital=initial_capital,
                max_positions=max_positions,
            )
        )
    session_results = await asyncio.gather(*tasks, return_exceptions=True)

    sessions = []
    for f, res in zip(factors, session_results):
        if isinstance(res, Exception):
            logger.error("리플레이 실패 factor=%s: %s", f["name"], res, exc_info=res)
            sessions.append({
                "factor_id": f["factor_id"],
                "factor_name": f["name"],
                "interval": f.get("interval", "5m"),
                "error": str(res),
                "trade_count": 0,
                "trade_log": [],
                "equity_curve": [],
                "metrics": {},
                "final_cash": initial_capital,
            })
        else:
            sessions.append(res)

    # 집계 메트릭
    agg = _aggregate_metrics(sessions)

    return {
        "target_date": str(target),
        "factors": factors,
        "sessions": sessions,
        "aggregate_metrics": agg,
    }


# ── 날짜/팩터 해석 ──────────────────────────────────────────


async def _resolve_target_date(target_date: date | None) -> date:
    """최근 거래일(5분봉 데이터가 있는 날짜) 결정."""
    if target_date:
        return target_date
    from app.core.database import async_session
    from sqlalchemy import text

    async with async_session() as db:
        row = await db.execute(text(
            "SELECT MAX(dt::date) FROM stock_candles WHERE interval = '5m'"
        ))
        val = row.scalar()
        return val if val else date.today()


async def _resolve_factors(
    factor_ids: list[str] | None, target_date: date,
) -> list[dict]:
    """팩터 정보를 해석. None이면 워크플로우 선택 팩터 사용."""
    from app.core.database import async_session
    from sqlalchemy import select

    if factor_ids:
        # 직접 지정된 팩터 로드
        from app.alpha.models import AlphaFactor
        result = []
        async with async_session() as db:
            for fid in factor_ids:
                try:
                    factor = await db.get(AlphaFactor, uuid.UUID(fid))
                    if factor:
                        result.append({
                            "factor_id": str(factor.id),
                            "name": factor.name or str(factor.id)[:8],
                            "interval": factor.interval or "5m",
                            "expression_str": factor.expression_str,
                            "ic_mean": factor.ic_mean,
                        })
                except Exception as e:
                    logger.warning("팩터 %s 로드 실패: %s", fid, e)
        return result

    # 워크플로우에서 자동 선택
    from app.workflow.models import WorkflowRun
    async with async_session() as db:
        # 타겟 날짜의 워크플로우 → 없으면 최근
        stmt = (
            select(WorkflowRun)
            .where(WorkflowRun.date <= target_date)
            .order_by(WorkflowRun.date.desc())
            .limit(1)
        )
        row = await db.execute(stmt)
        run = row.scalar_one_or_none()

        if run and run.config:
            selected = run.config.get("selected_factors", [])
            if selected:
                # 팩터 상세 정보 보강
                from app.alpha.models import AlphaFactor
                enriched = []
                for sf in selected:
                    fid = sf.get("factor_id")
                    if not fid:
                        continue
                    try:
                        factor = await db.get(AlphaFactor, uuid.UUID(fid))
                        enriched.append({
                            "factor_id": fid,
                            "name": sf.get("name", str(fid)[:8]),
                            "interval": sf.get("interval", factor.interval if factor else "5m"),
                            "expression_str": factor.expression_str if factor else None,
                            "ic_mean": factor.ic_mean if factor else None,
                            "score": sf.get("score"),
                        })
                    except Exception:
                        enriched.append({
                            "factor_id": fid,
                            "name": sf.get("name", str(fid)[:8]),
                            "interval": sf.get("interval", "5m"),
                        })
                return enriched

    return []


# ── 컨텍스트 빌드 (DB 저장 없음) ──────────────────────────


def _build_replay_context(
    factor_info: dict,
    initial_capital: float,
    max_positions: int,
    symbols: list[str],
) -> TradingContext:
    """auto_selector.build_context_from_factor의 인메모리 버전."""
    short_id = factor_info["factor_id"][:8]
    indicator_name = f"alpha_{short_id}"
    ic_mean = factor_info.get("ic_mean") or 0.0
    conviction = min(1.0, ic_mean / 0.1) if ic_mean else 1.0
    position_size_pct = 1.0 / max_positions

    return TradingContext(
        id=str(uuid.uuid4()),
        mode="paper",
        created_at=now_kst().isoformat(),
        strategy={
            "factor_id": factor_info["factor_id"],
            "factor_name": factor_info.get("name", ""),
            "expression_str": factor_info.get("expression_str", ""),
            "interval": factor_info.get("interval", "5m"),
            "buy_conditions": [
                {"indicator": indicator_name, "params": {}, "op": ">", "value": 0.7},
            ],
            "sell_conditions": [
                {"indicator": indicator_name, "params": {}, "op": "<", "value": 0.3},
            ],
            "buy_logic": "AND",
            "sell_logic": "AND",
        },
        strategy_name=f"replay:{factor_info.get('name', short_id)}",
        position_sizing={
            "mode": "conviction",
            "conviction": round(conviction, 4),
        },
        risk_management={
            "stop_loss_pct": settings.WORKFLOW_STOP_LOSS_PCT,
            "trailing_stop_pct": 3.0,
            "max_drawdown_pct": settings.WORKFLOW_MAX_DRAWDOWN_PCT,
        },
        cost_config=CostConfig(
            buy_commission=0.00015,
            sell_commission=0.00215,
            slippage_pct=0.001,
        ),
        initial_capital=initial_capital,
        position_size_pct=position_size_pct,
        max_positions=max_positions,
        symbols=symbols,
        source_factor_id=factor_info["factor_id"],
    )


# ── 유니버스 해석 ──────────────────────────────────────────


async def _resolve_symbols_for_factor(factor_info: dict) -> list[str]:
    """팩터의 mining run에서 유니버스를 해석."""
    from app.core.database import async_session
    from sqlalchemy import select

    fid = factor_info["factor_id"]
    async with async_session() as db:
        from app.alpha.models import AlphaFactor, AlphaMiningRun
        factor = await db.get(AlphaFactor, uuid.UUID(fid))
        if factor and factor.mining_run_id:
            mining_run = await db.get(AlphaMiningRun, factor.mining_run_id)
            if mining_run and mining_run.config:
                universe_str = mining_run.config.get("universe", "KOSPI200")
                try:
                    from app.alpha.universe import Universe, resolve_universe
                    return await resolve_universe(Universe(universe_str))
                except Exception:
                    pass

    # Fallback
    try:
        from app.alpha.universe import Universe, resolve_universe
        return await resolve_universe(Universe.KOSPI200)
    except Exception:
        return []


# ── 팩터별 리플레이 오케스트레이터 ──────────────────────────


async def _replay_factor(
    factor_info: dict,
    target_date: date,
    initial_capital: float,
    max_positions: int,
) -> dict:
    """단일 팩터의 장중매매 리플레이."""
    interval = factor_info.get("interval", "5m")

    # 알파 팩터 등록 (인메모리)
    expr_str = factor_info.get("expression_str")
    if expr_str:
        from app.alpha.backtest_bridge import register_alpha_factor
        register_alpha_factor(factor_info["factor_id"], expr_str)

    # 유니버스 해석
    symbols = await _resolve_symbols_for_factor(factor_info)
    if not symbols:
        return {
            "factor_id": factor_info["factor_id"],
            "factor_name": factor_info.get("name", ""),
            "interval": interval,
            "error": "유니버스 해석 실패",
            "trade_count": 0,
            "trade_log": [],
            "equity_curve": [],
            "metrics": {},
            "final_cash": initial_capital,
        }

    ctx = _build_replay_context(factor_info, initial_capital, max_positions, symbols)

    if interval == "1d":
        return await _replay_1d(ctx, factor_info, target_date)
    else:
        return await _replay_5m(ctx, factor_info, target_date)


# ── 5분봉 리플레이 핵심 루프 ──────────────────────────────


async def _replay_5m(
    ctx: TradingContext,
    factor_info: dict,
    target_date: date,
) -> dict:
    """5분봉 워크포워드 리플레이.

    live_runner._paper_loop_tick()의 핵심 로직을 재현하되
    사이드이펙트(DB/Redis/Telegram) 없이 실행.
    """
    strategy = ctx.strategy
    cost = ctx.cost_config
    risk = ctx.risk_management or {}
    stop_loss_pct = risk.get("stop_loss_pct")
    trailing_stop_pct = risk.get("trailing_stop_pct")
    atr_stop_mult = risk.get("atr_stop_multiplier")
    scaling = ctx.scaling or {}
    scaling_enabled = scaling.get("enabled", False)
    ps_cfg = ctx.position_sizing or {}
    use_limit = settings.PAPER_USE_LIMIT_ORDERS

    # 데이터 로딩
    start = target_date - timedelta(days=120)
    end = target_date

    from app.backtest.data_loader import load_enriched_candles
    df = await load_enriched_candles(ctx.symbols, start, end, "5m")

    if df.is_empty():
        return _empty_result(factor_info, ctx.initial_capital)

    df = df.unique(subset=["symbol", "dt"], keep="first")

    # 시그널 생성 (CPU bound → to_thread)
    def _gen_signals():
        sym_frames: dict[str, list[dict]] = {}
        for sym in ctx.symbols:
            sym_df = df.filter(pl.col("symbol") == sym).sort("dt")
            if sym_df.height < 30:
                continue
            try:
                sym_df = generate_signals(sym_df, strategy)
                sym_frames[sym] = sym_df.to_dicts()
            except Exception:
                continue
        return sym_frames

    sym_frames_raw = await asyncio.to_thread(_gen_signals)

    # 타겟 날짜의 장중 봉만 필터
    market_open = datetime.combine(target_date, datetime.min.time().replace(hour=9, minute=0))

    sym_frames: dict[str, list[dict]] = {}
    for sym, all_rows in sym_frames_raw.items():
        rows = [r for r in all_rows if r.get("dt") is not None and r["dt"] >= market_open]
        if rows:
            sym_frames[sym] = rows

    if not sym_frames:
        return _empty_result(factor_info, ctx.initial_capital)

    # 시간순 정렬
    all_bars: list[tuple[str, dict]] = []
    for sym, rows in sym_frames.items():
        for r in rows:
            all_bars.append((sym, r))
    all_bars.sort(key=lambda x: str(x[1].get("dt", "")))

    # 세션 초기화
    cash = ctx.initial_capital
    positions: dict[str, LivePosition] = {}
    pending_orders: list[dict] = []
    trade_log: list[dict] = []
    equity_curve: list[dict] = []
    latest_prices: dict[str, float] = {}

    # 워크포워드 루프
    seen_bars: set[str] = set()

    for sym, row in all_bars:
        dt_val = row.get("dt")
        candle_dt = dt_val.isoformat() if hasattr(dt_val, "isoformat") else str(dt_val)
        close_price = row.get("close", 0)
        signal = row.get("signal", 0)

        if close_price <= 0:
            continue

        bar_key = f"{sym}:{candle_dt}"
        if bar_key in seen_bars:
            continue
        seen_bars.add(bar_key)

        latest_prices[sym] = close_price

        # ── pending order 체결 확인 ──
        if use_limit:
            cash, pending_orders = _process_pending(
                pending_orders, sym, row, cost, candle_dt,
                positions, trade_log, cash,
            )

        # ── 리스크 관리 (보유 포지션) ──
        if sym in positions:
            pos = positions[sym]
            if close_price > pos.highest_price:
                pos.highest_price = close_price

            risk_decision = evaluate_risk(
                avg_price=pos.avg_price,
                highest_price=pos.highest_price,
                current_price=close_price,
                qty=pos.qty,
                stop_loss_pct=stop_loss_pct,
                trailing_stop_pct=trailing_stop_pct,
                atr_val=row.get("atr_14"),
                atr_stop_mult=atr_stop_mult,
                partial_exit_gain_pct=scaling.get("partial_exit_gain_pct") if scaling_enabled else None,
                partial_exit_pct=scaling.get("partial_exit_pct", 0.5),
                has_partial_exited=pos.has_partial_exited,
                scaling_enabled=scaling_enabled,
            )

            if risk_decision:
                if risk_decision.action == "PARTIAL_EXIT":
                    sell_price = effective_sell_price(close_price, cost)
                    cash += sell_price * risk_decision.qty
                    _log_replay_trade(trade_log, sym, "SELL", "S-HALF",
                                      risk_decision.qty, sell_price, candle_dt,
                                      reason=risk_decision.reason, position=pos)
                    _reduce_entries(pos, risk_decision.qty)
                    pos.has_partial_exited = True
                else:
                    step = {"RISK_STOP": "S-STOP", "RISK_TRAIL": "S-TRAIL",
                            "RISK_ATR_STOP": "S-STOP"}.get(risk_decision.action, "S-STOP")
                    sell_price = effective_sell_price(close_price, cost)
                    cash += sell_price * pos.qty
                    _log_replay_trade(trade_log, sym, "SELL", step,
                                      pos.qty, sell_price, candle_dt,
                                      reason=risk_decision.reason, position=pos)
                    positions.pop(sym, None)
                continue

        # ── 매도 시그널 ──
        if signal == -1 and sym in positions:
            pos = positions[sym]
            sell_price = effective_sell_price(close_price, cost)
            cash += sell_price * pos.qty
            _log_replay_trade(trade_log, sym, "SELL", "",
                              pos.qty, sell_price, candle_dt,
                              reason="매도 시그널", position=pos)
            positions.pop(sym, None)
            continue

        # ── B2 추가매수 ──
        if scaling_enabled and sym in positions:
            pos = positions[sym]
            scale_decision = evaluate_scale_in(
                avg_price=pos.avg_price,
                current_price=close_price,
                current_qty=pos.qty,
                target_qty=pos.target_qty,
                scale_in_count=pos.scale_in_count,
                max_scale_in=scaling.get("max_scale_in", 1),
                scale_in_drop_pct=scaling.get("scale_in_drop_pct", 3.0),
            )
            if scale_decision is not None:
                buy_p = effective_buy_price(close_price, cost)
                remaining_qty = scale_decision.qty
                cost_amount = buy_p * remaining_qty
                if cost_amount > cash * 0.95:
                    remaining_qty = int(cash * 0.95 / buy_p)
                if remaining_qty > 0:
                    total_cost = buy_p * remaining_qty
                    cash -= total_cost
                    pos.entries.append(LiveScaleEntry(
                        date=candle_dt, price=buy_p,
                        qty=remaining_qty, step="B2",
                    ))
                    pos.scale_in_count += 1
                    _log_replay_trade(trade_log, sym, "BUY", "B2",
                                      remaining_qty, buy_p, candle_dt,
                                      reason=scale_decision.reason)

        # ── 매수 시그널 (B1) ──
        if signal == 1 and sym not in positions:
            # pending 중인 같은 종목 스킵
            if any(o["symbol"] == sym and o["side"] == "BUY" for o in pending_orders):
                continue

            buy_price = effective_buy_price(close_price, cost)
            pending_buy_count = sum(1 for o in pending_orders if o.get("side") == "BUY")
            buy_decision = evaluate_buy(
                signal=signal,
                symbol=sym,
                has_position=False,
                current_positions=len(positions) + pending_buy_count,
                max_positions=ctx.max_positions,
                cash=cash,
                initial_capital=ctx.initial_capital,
                position_size_pct=ctx.position_size_pct,
                close_price=close_price,
                buy_price=buy_price,
                row=row,
                strategy=strategy,
                ps_cfg=ps_cfg,
                scaling=scaling if scaling_enabled else None,
            )

            if buy_decision.action == "BUY":
                qty = buy_decision.qty
                total_cost = buy_price * qty

                if use_limit:
                    # pending order 생성
                    cash -= total_cost
                    pending_orders.append({
                        "symbol": sym, "side": "BUY",
                        "price": buy_price, "qty": qty, "step": "B1",
                        "conviction": buy_decision.conviction,
                        "target_qty": buy_decision.target_qty,
                        "created_at": candle_dt,
                        "ttl_bars": settings.PAPER_LIMIT_TTL_BARS,
                        "elapsed_bars": 0,
                        "reserved_cash": total_cost,
                    })
                else:
                    # 즉시 체결
                    cash -= total_cost
                    positions[sym] = LivePosition(
                        symbol=sym,
                        entries=[LiveScaleEntry(
                            date=candle_dt, price=buy_price, qty=qty, step="B1",
                        )],
                        highest_price=close_price,
                        conviction=buy_decision.conviction,
                        target_qty=buy_decision.target_qty,
                        entry_candle_dt=candle_dt,
                    )
                    _log_replay_trade(trade_log, sym, "BUY", "B1",
                                      qty, buy_price, candle_dt,
                                      reason="매수 시그널")

        # ── 봉별 equity 기록 ──
        # 동일 시각의 마지막 종목 처리 후 기록 (시간 변경 시)
        _record_equity_if_new_time(
            equity_curve, candle_dt, cash, positions, pending_orders, latest_prices,
        )

    # ── 잔여 포지션 + pending 청산 (장 마감) ──
    last_dt = all_bars[-1][1].get("dt", "") if all_bars else ""
    last_dt_str = last_dt.isoformat() if hasattr(last_dt, "isoformat") else str(last_dt)

    # pending BUY 취소 → 현금 반환
    for order in pending_orders:
        if order.get("side") == "BUY":
            cash += order.get("reserved_cash", 0)
    pending_orders.clear()

    # 잔여 포지션 청산
    for sym, pos in list(positions.items()):
        close_p = latest_prices.get(sym, pos.avg_price)
        sell_price = effective_sell_price(close_p, cost)
        cash += sell_price * pos.qty
        _log_replay_trade(trade_log, sym, "SELL", "S-CLOSE",
                          pos.qty, sell_price, last_dt_str,
                          reason="장 마감 청산", position=pos)
    positions.clear()

    # 최종 equity
    equity_curve.append({"dt": last_dt_str, "equity": round(cash, 2)})

    # 메트릭 계산
    metrics = _compute_replay_metrics(trade_log, equity_curve, ctx.initial_capital)

    return {
        "factor_id": factor_info["factor_id"],
        "factor_name": factor_info.get("name", ""),
        "interval": "5m",
        "trade_count": len(trade_log),
        "trade_log": trade_log,
        "equity_curve": equity_curve,
        "metrics": metrics,
        "final_cash": round(cash, 2),
    }


# ── 일봉 리플레이 ──────────────────────────────────────────


async def _replay_1d(
    ctx: TradingContext,
    factor_info: dict,
    target_date: date,
) -> dict:
    """일봉 스코어 기반 리밸런스 리플레이."""
    strategy = ctx.strategy
    cost = ctx.cost_config

    start = target_date - timedelta(days=150)
    end = target_date

    from app.backtest.data_loader import load_enriched_candles
    from app.backtest.engine import ensure_indicators

    df = await load_enriched_candles(ctx.symbols, start, end, "1d")
    if df.is_empty():
        return _empty_result(factor_info, ctx.initial_capital)

    # 지표 적용
    all_conds = strategy.get("buy_conditions", []) + strategy.get("sell_conditions", [])
    df = ensure_indicators(df, all_conds)

    indicator_name = strategy["buy_conditions"][0]["indicator"]
    if indicator_name not in df.columns:
        return _empty_result(factor_info, ctx.initial_capital, error="지표 컬럼 없음")

    # 종목별 마지막 행(타겟 날짜 종가) 스코어 추출
    latest = df.sort(["symbol", "dt"]).group_by("symbol").last()
    scores: dict[str, float] = {}
    close_prices: dict[str, float] = {}
    for row in latest.to_dicts():
        sym = row.get("symbol", "")
        val = row.get(indicator_name)
        scores[sym] = float(val) if val is not None and isinstance(val, (int, float)) else 0.5
        close_prices[sym] = row.get("close", 0)

    # 매매 실행
    cash = ctx.initial_capital
    positions: dict[str, LivePosition] = {}
    trade_log: list[dict] = []
    dt_str = str(target_date)

    buy_threshold = 0.7
    sell_threshold = 0.3
    buy_conds = strategy.get("buy_conditions", [])
    sell_conds = strategy.get("sell_conditions", [])
    if buy_conds:
        buy_threshold = buy_conds[0].get("value", 0.7)
    if sell_conds:
        sell_threshold = sell_conds[0].get("value", 0.3)

    # 매수: 스코어 상위 종목 (buy_threshold 초과)
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for sym, score in sorted_scores:
        if score <= buy_threshold:
            break
        if len(positions) >= ctx.max_positions:
            break
        if sym in positions:
            continue
        close_p = close_prices.get(sym, 0)
        if close_p <= 0:
            continue

        buy_price = effective_buy_price(close_p, cost)
        alloc = ctx.initial_capital * ctx.position_size_pct
        alloc = min(alloc, cash * 0.95)
        qty = int(alloc / buy_price) if buy_price > 0 else 0
        if qty <= 0:
            continue

        total_cost = buy_price * qty
        if total_cost > cash:
            continue

        cash -= total_cost
        positions[sym] = LivePosition(
            symbol=sym,
            entries=[LiveScaleEntry(date=dt_str, price=buy_price, qty=qty, step="B1")],
            highest_price=close_p,
            entry_candle_dt=dt_str,
        )
        _log_replay_trade(trade_log, sym, "BUY", "B1",
                          qty, buy_price, dt_str,
                          reason=f"일봉 스코어 {score:.3f} > {buy_threshold}")

    # 장 마감 청산
    for sym, pos in list(positions.items()):
        close_p = close_prices.get(sym, pos.avg_price)
        sell_price = effective_sell_price(close_p, cost)
        cash += sell_price * pos.qty
        _log_replay_trade(trade_log, sym, "SELL", "S-CLOSE",
                          pos.qty, sell_price, dt_str,
                          reason="장 마감 청산", position=pos)
    positions.clear()

    equity_curve = [
        {"dt": dt_str, "equity": ctx.initial_capital},
        {"dt": f"{dt_str} close", "equity": round(cash, 2)},
    ]
    metrics = _compute_replay_metrics(trade_log, equity_curve, ctx.initial_capital)

    return {
        "factor_id": factor_info["factor_id"],
        "factor_name": factor_info.get("name", ""),
        "interval": "1d",
        "trade_count": len(trade_log),
        "trade_log": trade_log,
        "equity_curve": equity_curve,
        "metrics": metrics,
        "final_cash": round(cash, 2),
    }


# ── pending order 처리 ──────────────────────────────────────


def _process_pending(
    pending_orders: list[dict],
    sym: str,
    row: dict,
    cost: CostConfig,
    candle_dt: str,
    positions: dict[str, LivePosition],
    trade_log: list[dict],
    cash: float,
) -> tuple[float, list[dict]]:
    """pending order 체결/만료 확인. (cash, updated_orders) 반환."""
    low = row.get("low", float("inf"))
    close = row.get("close", 0)

    still_pending: list[dict] = []
    for order in pending_orders:
        if order["symbol"] != sym:
            still_pending.append(order)
            continue

        order["elapsed_bars"] = order.get("elapsed_bars", 0) + 1

        if order["side"] == "BUY":
            if low <= order["price"]:
                # 체결
                _fill_pending(order, candle_dt, positions, trade_log)
                continue
            if order["elapsed_bars"] >= order.get("ttl_bars", 2):
                # TTL 만료 → 시장가 체결
                market_price = effective_buy_price(close, cost)
                reserved = order.get("reserved_cash", 0)
                cash_diff = market_price * order["qty"] - reserved
                if cash_diff > 0 and cash_diff > cash:
                    order["qty"] = int((reserved + cash * 0.95) / market_price)
                if order["qty"] > 0:
                    cash -= max(0, cash_diff)
                    order["price"] = market_price
                    _fill_pending(order, candle_dt, positions, trade_log)
                else:
                    cash += reserved  # 취소, 현금 반환
                continue
            still_pending.append(order)

        elif order["side"] == "SELL":
            high = row.get("high", 0)
            if high >= order["price"]:
                pos = positions.get(sym)
                if pos:
                    actual_qty = min(order["qty"], pos.qty)
                    proceeds = order["price"] * actual_qty
                    cash += proceeds
                    _log_replay_trade(trade_log, sym, "SELL", order.get("step", ""),
                                      actual_qty, order["price"], candle_dt,
                                      reason="지정가 매도 체결", position=pos)
                    positions.pop(sym, None)
                continue
            if order["elapsed_bars"] >= order.get("ttl_bars", 2):
                market_price = effective_sell_price(close, cost)
                pos = positions.get(sym)
                if pos:
                    actual_qty = min(order["qty"], pos.qty)
                    proceeds = market_price * actual_qty
                    cash += proceeds
                    _log_replay_trade(trade_log, sym, "SELL", order.get("step", ""),
                                      actual_qty, market_price, candle_dt,
                                      reason="매도 TTL 만료 체결", position=pos)
                    positions.pop(sym, None)
                continue
            still_pending.append(order)

    return cash, still_pending


def _fill_pending(
    order: dict,
    candle_dt: str,
    positions: dict[str, LivePosition],
    trade_log: list[dict],
) -> None:
    """pending 매수 체결 → 포지션 생성."""
    sym = order["symbol"]
    positions[sym] = LivePosition(
        symbol=sym,
        entries=[LiveScaleEntry(
            date=candle_dt, price=order["price"],
            qty=order["qty"], step=order.get("step", "B1"),
        )],
        highest_price=order["price"],
        conviction=order.get("conviction", 1.0),
        target_qty=order.get("target_qty", order["qty"]),
        entry_candle_dt=candle_dt,
    )
    _log_replay_trade(trade_log, sym, "BUY", order.get("step", "B1"),
                      order["qty"], order["price"], candle_dt,
                      reason="매수 시그널 체결")


# ── 트레이드 로깅 (메모리 전용) ──────────────────────────────


def _log_replay_trade(
    trade_log: list[dict],
    symbol: str,
    side: str,
    step: str,
    qty: int,
    price: float,
    candle_dt: str,
    reason: str = "",
    position: LivePosition | None = None,
) -> None:
    """인메모리 전용 매매 기록."""
    entry: dict[str, Any] = {
        "symbol": symbol,
        "name": get_stock_name(symbol),
        "side": side,
        "step": step,
        "qty": qty,
        "price": round(price, 2),
        "candle_dt": candle_dt,
        "reason": reason,
    }

    if side == "SELL" and position:
        avg = position.avg_price
        if avg > 0:
            pnl_pct = (price - avg) / avg * 100
            pnl_amount = (price - avg) * qty
            entry["pnl_pct"] = round(pnl_pct, 4)
            entry["pnl_amount"] = round(pnl_amount, 2)
            entry["avg_price"] = round(avg, 2)
            # 보유시간 (분)
            try:
                entry_dt = datetime.fromisoformat(position.entry_candle_dt) if position.entry_candle_dt else None
                sell_dt = datetime.fromisoformat(candle_dt) if candle_dt else None
                if entry_dt and sell_dt:
                    entry["holding_minutes"] = (sell_dt - entry_dt).total_seconds() / 60
            except (ValueError, TypeError):
                pass

    trade_log.append(entry)


# ── equity curve 기록 ──────────────────────────────────────


def _record_equity_if_new_time(
    equity_curve: list[dict],
    candle_dt: str,
    cash: float,
    positions: dict[str, LivePosition],
    pending_orders: list[dict],
    latest_prices: dict[str, float],
) -> None:
    """시간이 변경될 때 equity 기록 (봉별 1건)."""
    if equity_curve and equity_curve[-1].get("dt") == candle_dt:
        # 같은 시각 → 마지막 값만 업데이트
        equity_curve[-1]["equity"] = round(
            _calc_equity(cash, positions, pending_orders, latest_prices), 2
        )
        return

    equity_curve.append({
        "dt": candle_dt,
        "equity": round(
            _calc_equity(cash, positions, pending_orders, latest_prices), 2
        ),
    })


def _calc_equity(
    cash: float,
    positions: dict[str, LivePosition],
    pending_orders: list[dict],
    latest_prices: dict[str, float],
) -> float:
    """현재 포트폴리오 가치 = 현금 + 포지션(현재가) + pending 예약금."""
    pos_eval = sum(
        p.qty * latest_prices.get(sym, p.avg_price)
        for sym, p in positions.items()
    )
    pending_reserved = sum(
        o.get("reserved_cash", 0)
        for o in pending_orders
        if o.get("side") == "BUY"
    )
    return cash + pos_eval + pending_reserved


# ── 메트릭 계산 ──────────────────────────────────────────


def _compute_replay_metrics(
    trade_log: list[dict],
    equity_curve: list[dict],
    initial_capital: float,
) -> dict:
    """리플레이 결과 요약 메트릭."""
    sells = [t for t in trade_log if t["side"] == "SELL" and "pnl_pct" in t]
    buys = [t for t in trade_log if t["side"] == "BUY"]

    total_trades = len(sells)
    total_buys = len(buys)

    if not sells:
        return {
            "total_trades": 0,
            "total_buys": total_buys,
            "win_rate": 0,
            "total_pnl": 0,
            "total_pnl_pct": 0,
            "avg_pnl_pct": 0,
            "max_win": 0,
            "max_loss": 0,
        }

    wins = [t for t in sells if (t.get("pnl_amount") or 0) > 0]
    total_pnl = sum(t.get("pnl_amount", 0) for t in sells)

    final_equity = equity_curve[-1]["equity"] if equity_curve else initial_capital
    total_pnl_from_equity = final_equity - initial_capital

    pnl_pcts = [t.get("pnl_pct", 0) for t in sells]

    return {
        "total_trades": total_trades,
        "total_buys": total_buys,
        "win_rate": round(len(wins) / total_trades * 100, 2) if total_trades else 0,
        "total_pnl": round(total_pnl, 2),
        "total_pnl_pct": round(total_pnl_from_equity / initial_capital * 100, 4),
        "avg_pnl_pct": round(sum(pnl_pcts) / len(pnl_pcts), 4) if pnl_pcts else 0,
        "max_win": round(max(pnl_pcts), 4) if pnl_pcts else 0,
        "max_loss": round(min(pnl_pcts), 4) if pnl_pcts else 0,
        "avg_holding_minutes": round(
            sum(t.get("holding_minutes", 0) for t in sells) / total_trades, 1
        ) if total_trades else 0,
    }


def _aggregate_metrics(sessions: list[dict]) -> dict:
    """전체 세션 집계 메트릭."""
    total_pnl = 0
    total_trades = 0
    total_buys = 0
    for s in sessions:
        m = s.get("metrics", {})
        total_pnl += m.get("total_pnl", 0)
        total_trades += m.get("total_trades", 0)
        total_buys += m.get("total_buys", 0)

    return {
        "total_sessions": len(sessions),
        "total_trades": total_trades,
        "total_buys": total_buys,
        "total_pnl": round(total_pnl, 2),
    }


def _empty_result(factor_info: dict, initial_capital: float, error: str = "데이터 없음") -> dict:
    return {
        "factor_id": factor_info["factor_id"],
        "factor_name": factor_info.get("name", ""),
        "interval": factor_info.get("interval", "5m"),
        "error": error,
        "trade_count": 0,
        "trade_log": [],
        "equity_curve": [],
        "metrics": {},
        "final_cash": initial_capital,
    }
