"""실시간 전략 실행기 — 틱 감시 → 시그널 → 주문.

TradingContext 기반으로 전략을 실시간 실행한다.
- mode=paper: 자체 시뮬레이션 (KIS API 미사용, 캔들 종가 즉시 체결)
- mode=real: KIS API 실주문
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import polars as pl

from app.backtest.cost_model import CostConfig, effective_buy_price, effective_sell_price
from app.backtest.engine import generate_signals
from app.core.stock_master import get_stock_name
from app.services.ws_manager import manager
from .context import TradingContext

logger = logging.getLogger(__name__)

# 전체 지표 스냅샷 키 목록 — trade_log.snapshot에 모두 기록
_FULL_SNAP_KEYS = (
    "close", "open", "high", "low", "volume",
    "rsi", "macd_line", "macd_signal", "macd_hist",
    "bb_upper", "bb_middle", "bb_lower", "bb_width",
    "sma_5", "sma_20", "sma_60", "ema_20",
    "atr_14", "volume_ratio", "price_change_pct",
    # 알파 팩터/외부 데이터
    "foreign_net_norm", "inst_net_norm", "sentiment_score",
    "margin_rate", "short_balance_rate", "sector_return",
)


@dataclass
class LivePosition:
    """실시간 포지션 추적."""
    symbol: str
    qty: int
    avg_price: float
    highest_price: float = 0.0
    entry_date: str = ""        # 캔들 dt (시뮬 기준 시각)
    entry_candle_dt: str = ""   # 진입 봉의 원본 dt (보유시간 계산용)


@dataclass
class LiveSession:
    """실거래 세션 상태."""
    id: str
    context: TradingContext
    status: str = "stopped"  # "running" | "stopped" | "error"
    positions: dict[str, LivePosition] = field(default_factory=dict)
    pending_orders: list[dict] = field(default_factory=list)
    trade_log: list[dict] = field(default_factory=list)
    decision_log: list[dict] = field(default_factory=list)  # 판단 기록 (스킵 포함)
    error_message: str = ""
    started_at: str = ""
    stopped_at: str = ""
    # paper 모드 내부 현금
    _cash: float = 0.0

    # 내부 태스크
    _task: asyncio.Task | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "context_id": self.context.id,
            "mode": self.context.mode,
            "strategy_name": self.context.strategy_name,
            "status": self.status,
            "positions": {
                sym: {"symbol": p.symbol, "qty": p.qty, "avg_price": p.avg_price}
                for sym, p in self.positions.items()
            },
            "trade_count": len(self.trade_log),
            "error_message": self.error_message,
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
        }


# 세션 저장소
_sessions: dict[str, LiveSession] = {}


async def start_session(ctx: TradingContext) -> LiveSession:
    """전략 실거래 세션 시작."""
    session = LiveSession(
        id=ctx.id,
        context=ctx,
        status="running",
        started_at=datetime.now().isoformat(),
        _cash=ctx.initial_capital,
    )
    _sessions[session.id] = session

    # 백그라운드 러너 시작
    session._task = asyncio.create_task(_run_loop(session))
    logger.info("실거래 세션 시작: %s (mode=%s)", session.id, ctx.mode)

    await manager.broadcast("trading:status", {
        "session_id": session.id,
        "status": "running",
    })

    return session


async def stop_session(session_id: str) -> LiveSession | None:
    """세션 중지."""
    session = _sessions.get(session_id)
    if not session:
        return None

    session.status = "stopped"
    session.stopped_at = datetime.now().isoformat()

    if session._task and not session._task.done():
        session._task.cancel()
        try:
            await session._task
        except asyncio.CancelledError:
            pass

    # session_state 클리어 (재시작 시 자동 복구 방지)
    await _clear_session_state(session_id)

    logger.info("실거래 세션 중지: %s", session_id)

    await manager.broadcast("trading:status", {
        "session_id": session_id,
        "status": "stopped",
    })

    return session


def get_session(session_id: str) -> LiveSession | None:
    return _sessions.get(session_id)


def list_sessions() -> list[LiveSession]:
    return list(_sessions.values())


# ── 판단 기록 헬퍼 ──────────────────────────────────────────


def _collect_snapshot(row: dict) -> dict:
    """봉 데이터에서 전체 지표 스냅샷을 수집."""
    snap: dict[str, Any] = {}
    for k in _FULL_SNAP_KEYS:
        v = row.get(k)
        if v is not None:
            snap[k] = round(v, 4) if isinstance(v, float) else v
    # 알파 팩터 값 (alpha_로 시작하는 모든 컬럼)
    for k, v in row.items():
        if k.startswith("alpha_") and v is not None:
            snap[k] = round(v, 4) if isinstance(v, float) else v
    return snap


def _collect_conditions_detail(
    row: dict, strategy: dict, signal: int,
) -> dict:
    """매매 조건 충족 상세 기록.

    각 buy/sell condition이 실제 값과 비교하여 충족 여부를 기록한다.
    """
    detail: dict[str, Any] = {"signal": signal}

    for side_key in ("buy_conditions", "sell_conditions"):
        conds = strategy.get(side_key, [])
        side_detail = []
        for c in conds:
            ind = c.get("indicator", "")
            op = c.get("op", "")
            threshold = c.get("value")
            actual = row.get(ind)
            met = False
            if actual is not None and threshold is not None:
                if op == ">":
                    met = actual > threshold
                elif op == ">=":
                    met = actual >= threshold
                elif op == "<":
                    met = actual < threshold
                elif op == "<=":
                    met = actual <= threshold
                elif op == "==":
                    met = actual == threshold
            side_detail.append({
                "indicator": ind,
                "op": op,
                "threshold": threshold,
                "actual": round(actual, 6) if isinstance(actual, float) else actual,
                "met": met,
            })
        detail[side_key] = side_detail

    logic = strategy.get("buy_logic", "AND")
    detail["logic"] = logic
    return detail


def _log_decision(
    session: LiveSession,
    symbol: str,
    action: str,
    reason: str,
    *,
    signal: int = 0,
    conditions: dict | None = None,
    snapshot: dict | None = None,
    sizing: dict | None = None,
    risk: dict | None = None,
) -> None:
    """판단 기록 (매매 실행 여부와 무관하게 모든 판단을 남긴다)."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "symbol": symbol,
        "name": get_stock_name(symbol),
        "action": action,  # BUY, SELL, SKIP_BUY, SKIP_SELL, RISK_STOP, RISK_TRAIL 등
        "reason": reason,
        "signal": signal,
        "conditions": conditions,
        "snapshot": snapshot,
        "sizing": sizing,
        "risk": risk,
    }
    session.decision_log.append(entry)


# ── paper 모드: 자체 시뮬레이션 ──────────────────────────────


async def _run_loop(session: LiveSession) -> None:
    """전략 실행 메인 루프."""
    ctx = session.context
    is_paper = ctx.mode != "real"

    cost = CostConfig(
        buy_commission=ctx.cost_config.buy_commission,
        sell_commission=ctx.cost_config.sell_commission,
        slippage_pct=ctx.cost_config.slippage_pct,
    )

    strategy = ctx.strategy
    risk = ctx.risk_management or {}
    stop_loss_pct = risk.get("stop_loss_pct")
    trailing_stop_pct = risk.get("trailing_stop_pct")

    check_interval = 30  # 초

    try:
        while session.status == "running":
            try:
                if is_paper:
                    await _paper_loop_tick(
                        session, strategy, cost, ctx,
                        stop_loss_pct, trailing_stop_pct,
                    )
                else:
                    await _real_loop_tick(
                        session, strategy, cost, ctx,
                        stop_loss_pct, trailing_stop_pct,
                    )
                # 매 틱 후 세션 상태 DB 저장 (C2: 서버 재시작 복구)
                await _save_session_state(session)
            except Exception as e:
                logger.error("실거래 루프 에러: %s", e, exc_info=True)
                session.error_message = str(e)

            await asyncio.sleep(check_interval)

    except asyncio.CancelledError:
        logger.info("실거래 루프 취소: %s", session.id)
    except Exception as e:
        session.status = "error"
        session.error_message = str(e)
        logger.error("실거래 루프 치명적 에러: %s", e, exc_info=True)


async def _ensure_today_candles(symbols: list[str], interval: str) -> int:
    """당일 분봉이 DB에 없으면 KIS API로 수집하여 저장.

    KIS API는 1분봉만 제공하므로, 3m/5m 등은 1m을 리샘플링한다.
    Returns: 저장된 캔들 수.
    """
    from datetime import date as date_cls, timedelta, timezone as tz

    KST = tz(timedelta(hours=9))
    today = date_cls.today()
    today_str = today.strftime("%Y%m%d")

    # DB에 오늘 해당 interval 캔들이 있는지 빠르게 체크
    try:
        from app.core.database import async_session
        from sqlalchemy import text
        async with async_session() as db:
            row = await db.execute(
                text(
                    "SELECT count(*) FROM stock_candles "
                    "WHERE interval = :iv AND dt >= :start"
                ),
                {"iv": interval, "start": datetime.combine(today, datetime.min.time()).replace(tzinfo=KST)},
            )
            cnt = row.scalar() or 0
            if cnt > 0:
                return 0  # 이미 존재
    except Exception as e:
        logger.warning("당일 캔들 존재 체크 실패: %s", e)
        return 0

    # KIS 클라이언트로 1분봉 수집 (시세 조회는 실전 URL 필수)
    logger.info("당일 %s 캔들 없음 — KIS API 분봉 수집 시작 (%d종목)", interval, len(symbols[:50]))
    try:
        from app.trading.kis_client import get_kis_client
        client = get_kis_client(is_mock=False)  # 시세 조회는 실전 URL

        # interval에서 분 단위 추출 (1m→1, 3m→3, 5m→5)
        interval_min = int(interval.replace("m", "")) if interval.endswith("m") else 0
        if interval_min <= 0:
            return 0  # 일봉 등은 KIS 분봉으로 커버 불가

        total_written = 0
        now_kst = datetime.now(KST)
        hour_str = now_kst.strftime("%H%M%S")

        for symbol in symbols[:50]:  # rate limit 고려, 최대 50종목
            try:
                raw_candles, _, _ = await client.get_minute_candles(
                    symbol, today_str, hour_str,
                )
                if not raw_candles:
                    continue

                # KIS 응답 → 표준 dict 변환 (1분봉)
                one_min: list[dict] = []
                for c in raw_candles:
                    dt_str = c.get("stck_bsop_date", "") + c.get("stck_cntg_hour", "")
                    if len(dt_str) != 14:
                        continue
                    dt = datetime.strptime(dt_str, "%Y%m%d%H%M%S").replace(tzinfo=KST)
                    if dt.date() != today:
                        continue
                    one_min.append({
                        "dt": dt,
                        "open": float(c.get("stck_oprc", 0)),
                        "high": float(c.get("stck_hgpr", 0)),
                        "low": float(c.get("stck_lwpr", 0)),
                        "close": float(c.get("stck_prpr", 0)),
                        "volume": int(c.get("cntg_vol", 0)),
                    })

                if not one_min:
                    continue

                # 리샘플링 (1m → interval_min 분봉)
                if interval_min == 1:
                    resampled = one_min
                else:
                    resampled = _resample_candles(one_min, interval_min)

                # DB 저장
                from app.services.candle_writer import write_candles_bulk
                await write_candles_bulk(symbol, resampled, interval)
                total_written += len(resampled)

            except Exception as e:
                logger.warning("KIS 분봉 수집 실패 (%s): %s", symbol, e)
                continue

            # rate limit (15req/s → ~67ms/req, 여유 확보)
            await asyncio.sleep(0.1)

        if total_written > 0:
            logger.info("KIS 분봉 자동 수집: %d봉 (%s, %d종목)", total_written, interval, len(symbols[:50]))
        return total_written

    except Exception as e:
        logger.warning("KIS 분봉 수집 전체 실패: %s", e)
        return 0


def _resample_candles(candles_1m: list[dict], interval_min: int) -> list[dict]:
    """1분봉 리스트를 N분봉으로 리샘플링."""
    from datetime import timedelta

    if not candles_1m:
        return []

    # 시간순 정렬
    candles_1m.sort(key=lambda c: c["dt"])

    resampled: list[dict] = []
    bucket: list[dict] = []
    bucket_start: datetime | None = None

    for c in candles_1m:
        dt = c["dt"]
        # 현재 버킷의 시작 시각 계산 (09:00 기준으로 interval_min 단위 절삭)
        market_open_minutes = dt.hour * 60 + dt.minute - 9 * 60  # 09:00 기준 분 수
        if market_open_minutes < 0:
            market_open_minutes = 0
        slot = (market_open_minutes // interval_min) * interval_min
        slot_start = dt.replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(minutes=slot)

        if bucket_start is None or slot_start != bucket_start:
            # 새 버킷 — 이전 버킷 플러시
            if bucket:
                resampled.append({
                    "dt": bucket_start,
                    "open": bucket[0]["open"],
                    "high": max(b["high"] for b in bucket),
                    "low": min(b["low"] for b in bucket),
                    "close": bucket[-1]["close"],
                    "volume": sum(b["volume"] for b in bucket),
                })
            bucket = [c]
            bucket_start = slot_start
        else:
            bucket.append(c)

    # 마지막 버킷 플러시
    if bucket and bucket_start is not None:
        resampled.append({
            "dt": bucket_start,
            "open": bucket[0]["open"],
            "high": max(b["high"] for b in bucket),
            "low": min(b["low"] for b in bucket),
            "close": bucket[-1]["close"],
            "volume": sum(b["volume"] for b in bucket),
        })

    return resampled


async def _paper_loop_tick(
    session: LiveSession,
    strategy: dict,
    cost: CostConfig,
    ctx: TradingContext,
    stop_loss_pct: float | None,
    trailing_stop_pct: float | None,
) -> None:
    """Paper 모드: 장중 실시간 시뮬레이션.

    과거 캔들 데이터를 가지고 실제 장중 매매를 재현한다.
    - 첫 호출: 오늘 장 시작(09:00)부터 현재까지의 모든 봉을 시간순 워크스루
      (5분봉 기준 ~78봉/일, 매 봉마다 매수/매도 판단)
    - 이후 호출: _last_processed_dt 이후 새로 생성된 봉만 처리
    """
    from datetime import date, timedelta

    # 종목 리스트 결정
    universe = ctx.symbols or []
    scan_limit = max(ctx.max_positions * 5, 50)
    scan_symbols = list(set(list(session.positions.keys()) + universe[:scan_limit]))

    if not scan_symbols:
        return

    end = date.today()
    start = end - timedelta(days=120)
    interval = strategy.get("interval", "5m")

    # 당일 분봉이 없으면 KIS API로 자동 수집
    if interval.endswith("m"):
        await _ensure_today_candles(scan_symbols, interval)

    is_alpha = any(
        c.get("indicator", "").startswith("alpha_")
        for c in strategy.get("buy_conditions", [])
    )

    try:
        if is_alpha:
            from app.backtest.data_loader import load_enriched_candles
            df = await load_enriched_candles(scan_symbols, start, end, interval)
        else:
            from app.backtest.data_loader import load_candles
            df = await load_candles(scan_symbols, start, end, interval)
    except Exception as e:
        logger.warning("캔들 로딩 실패: %s", e)
        return

    if df.is_empty():
        return

    last_dt = getattr(session, "_last_processed_dt", None)

    # 오늘 장 시작 시각 (09:00 KST) — 오늘 봉만 추출하기 위한 기준
    from datetime import time as time_type, timezone as tz
    KST = tz(timedelta(hours=9))
    today_market_open = datetime.combine(end, time_type(9, 0), tzinfo=KST)

    logger.info(
        "paper_tick: 캔들 로딩 완료 (height=%d, interval=%s, symbols=%d, market_open=%s)",
        df.height, interval, len(scan_symbols), today_market_open.isoformat(),
    )

    # 종목별로 시그널 생성 → 오늘 장중 봉 추출
    sym_frames: dict[str, list[dict]] = {}
    _skip_data = 0
    _skip_error = 0
    _no_today = 0
    for sym in scan_symbols:
        sym_df = df.filter(pl.col("symbol") == sym).sort("dt")
        if sym_df.height < 30:
            _log_decision(session, sym, "SKIP_DATA",
                          f"데이터 부족: {sym_df.height}봉 (최소 30봉 필요)")
            _skip_data += 1
            continue
        try:
            sym_df = generate_signals(sym_df, strategy)
        except Exception as e:
            _log_decision(session, sym, "SKIP_ERROR",
                          f"시그널 생성 실패: {type(e).__name__}: {e}")
            _skip_error += 1
            continue

        all_rows = sym_df.to_dicts()

        if last_dt is None:
            # 첫 호출: 오늘 장 시작 이후 모든 봉을 워크스루
            # (5분봉 ~78봉, 1분봉 ~390봉)
            rows = []
            for r in all_rows:
                dt_val = r.get("dt")
                if dt_val is None:
                    continue
                # naive datetime이면 UTC로 간주
                if hasattr(dt_val, "tzinfo") and dt_val.tzinfo is None:
                    dt_val = dt_val.replace(tzinfo=tz(timedelta(0)))
                if dt_val >= today_market_open:
                    rows.append(r)
        else:
            # 이후 호출: 마지막 처리 봉 이후 새 봉만
            rows = [
                r for r in all_rows
                if str(r.get("dt", "")) > str(last_dt)
            ]

        if rows:
            sym_frames[sym] = rows
        else:
            _no_today += 1

    if not sym_frames:
        logger.info(
            "paper_tick: 오늘 봉 0개 — skip_data=%d, skip_error=%d, no_today=%d "
            "(interval=%s, market_open=%s, symbols=%d)",
            _skip_data, _skip_error, _no_today,
            interval, today_market_open.isoformat(), len(scan_symbols),
        )
        return

    logger.info(
        "paper_tick: %d종목 %d봉 처리 시작 (interval=%s)",
        len(sym_frames), sum(len(v) for v in sym_frames.values()), interval,
    )

    # 모든 봉을 시간순 정렬하여 워크포워드
    all_bars: list[tuple[str, dict]] = []
    for sym, rows in sym_frames.items():
        for r in rows:
            all_bars.append((sym, r))
    all_bars.sort(key=lambda x: str(x[1].get("dt", "")))

    for sym, row in all_bars:
        dt_val = row.get("dt")
        candle_dt_str = dt_val.isoformat() if hasattr(dt_val, "isoformat") else str(dt_val)
        close_price = row.get("close", 0)
        signal = row.get("signal", 0)

        if close_price <= 0:
            continue

        # ── 고가 갱신 + 리스크 관리 (보유 포지션) ──
        if sym in session.positions:
            pos = session.positions[sym]
            if close_price > pos.highest_price:
                pos.highest_price = close_price

            # 고정 손절
            if stop_loss_pct is not None and pos.avg_price > 0:
                loss_pct = (close_price - pos.avg_price) / pos.avg_price * 100
                if loss_pct <= -stop_loss_pct:
                    sell_price = effective_sell_price(close_price, cost)
                    risk_snap = {
                        "close": close_price, "avg_price": round(pos.avg_price),
                        "highest_price": round(pos.highest_price),
                        "loss_pct": round(loss_pct, 4),
                    }
                    _log_decision(
                        session, sym, "RISK_STOP",
                        f"손절 발동: 현재가 {close_price:,.0f} / 평단 {pos.avg_price:,.0f} = {loss_pct:+.2f}%",
                        risk={"type": "stop_loss", "loss_pct": round(loss_pct, 4), "threshold": stop_loss_pct},
                    )
                    _paper_sell(session, sym, pos.qty, close_price, sell_price, cost,
                                step="S-STOP",
                                reason=f"손절: 평단 대비 {loss_pct:+.2f}% (기준 -{stop_loss_pct}%)",
                                snapshot=risk_snap, candle_dt=candle_dt_str)
                    continue

            # 트레일링 스탑
            if trailing_stop_pct is not None and pos.highest_price > 0:
                drop_pct = (pos.highest_price - close_price) / pos.highest_price * 100
                if drop_pct >= trailing_stop_pct:
                    sell_price = effective_sell_price(close_price, cost)
                    risk_snap = {
                        "close": close_price, "highest_price": round(pos.highest_price),
                        "drop_pct": round(drop_pct, 4),
                    }
                    _log_decision(
                        session, sym, "RISK_TRAIL",
                        f"트레일링: 고점 {pos.highest_price:,.0f} → {close_price:,.0f} = -{drop_pct:.2f}%",
                        risk={"type": "trailing_stop", "drop_pct": round(drop_pct, 4), "threshold": trailing_stop_pct},
                    )
                    _paper_sell(session, sym, pos.qty, close_price, sell_price, cost,
                                step="S-TRAIL",
                                reason=f"트레일링: 고점 대비 -{drop_pct:.2f}% (기준 -{trailing_stop_pct}%)",
                                snapshot=risk_snap, candle_dt=candle_dt_str)
                    continue

        snapshot = _collect_snapshot(row)
        conditions = _collect_conditions_detail(row, strategy, signal)

        # ── 매수 시그널 ──
        if signal == 1 and sym not in session.positions:
            if len(session.positions) >= ctx.max_positions:
                _log_decision(session, sym, "SKIP_BUY",
                              f"최대 포지션: {len(session.positions)}/{ctx.max_positions}",
                              signal=signal, conditions=conditions, snapshot=snapshot)
                continue

            alloc = ctx.initial_capital * ctx.position_size_pct
            alloc = min(alloc, session._cash * 0.95)
            buy_price = effective_buy_price(close_price, cost)
            qty = int(alloc / buy_price)

            sizing = {
                "initial_capital": ctx.initial_capital,
                "position_size_pct": ctx.position_size_pct,
                "alloc_raw": ctx.initial_capital * ctx.position_size_pct,
                "alloc_cash_limited": alloc,
                "close_price": close_price,
                "buy_price_effective": round(buy_price, 2),
                "buy_commission": cost.buy_commission,
                "slippage_pct": cost.slippage_pct,
                "qty": qty,
                "total_cost": round(buy_price * qty, 2),
                "cash_before": round(session._cash, 2),
                "positions_count": len(session.positions),
                "max_positions": ctx.max_positions,
            }

            if qty <= 0:
                _log_decision(session, sym, "SKIP_BUY",
                              f"수량 0: 배분금 {alloc:,.0f} / 매수가 {buy_price:,.0f}",
                              signal=signal, conditions=conditions, snapshot=snapshot, sizing=sizing)
                continue

            total_cost = buy_price * qty
            if total_cost > session._cash:
                _log_decision(session, sym, "SKIP_BUY",
                              f"현금 부족: {total_cost:,.0f} > {session._cash:,.0f}",
                              signal=signal, conditions=conditions, snapshot=snapshot, sizing=sizing)
                continue

            session._cash -= total_cost
            session.positions[sym] = LivePosition(
                symbol=sym, qty=qty, avg_price=buy_price,
                highest_price=close_price,
                entry_date=candle_dt_str,
                entry_candle_dt=candle_dt_str,
            )
            result = {"success": True, "order_id": f"SIM-{uuid.uuid4().hex[:8]}",
                       "message": f"시뮬 매수 체결 @ {buy_price:,.0f}"}
            buy_reason = "매수 시그널: " + ", ".join(
                f"{c['indicator']}{c['op']}{c['threshold']} (실제={c['actual']})"
                for c in conditions.get("buy_conditions", []))
            _log_decision(session, sym, "BUY", buy_reason,
                          signal=signal, conditions=conditions, snapshot=snapshot, sizing=sizing)
            _log_trade(session, sym, "BUY", "B1", qty, buy_price, result,
                       reason=buy_reason, snapshot=snapshot,
                       conditions=conditions, sizing=sizing, candle_dt=candle_dt_str)

        # ── 매도 시그널 ──
        elif signal == -1 and sym in session.positions:
            pos = session.positions[sym]
            if pos.qty <= 0:
                continue
            sell_price = effective_sell_price(close_price, cost)
            sell_reason = "매도 시그널: " + ", ".join(
                f"{c['indicator']}{c['op']}{c['threshold']} (실제={c['actual']})"
                for c in conditions.get("sell_conditions", []))
            _log_decision(session, sym, "SELL", sell_reason,
                          signal=signal, conditions=conditions, snapshot=snapshot)
            _paper_sell(session, sym, pos.qty, close_price, sell_price, cost,
                        step="", reason=sell_reason,
                        snapshot=snapshot, conditions=conditions, candle_dt=candle_dt_str)

    # 마지막 처리 봉 기록
    if all_bars:
        last_dt_val = all_bars[-1][1].get("dt")
        session._last_processed_dt = (
            last_dt_val.isoformat() if hasattr(last_dt_val, "isoformat") else str(last_dt_val)
        )

    # H2: Paper 모드 포트폴리오 MDD 서킷브레이커
    from app.core.config import settings as _settings
    max_dd = _settings.WORKFLOW_MAX_DRAWDOWN_PCT
    # 포지션 평가금 추정 (마지막 close 기준)
    pos_eval = 0.0
    for sym, pos in session.positions.items():
        pos_eval += pos.avg_price * pos.qty  # 보수적 추정
    total_eval = session._cash + pos_eval

    if not hasattr(session, "_portfolio_peak"):
        session._portfolio_peak = ctx.initial_capital
    if total_eval > session._portfolio_peak:
        session._portfolio_peak = total_eval

    if session._portfolio_peak > 0 and total_eval > 0:
        dd = (session._portfolio_peak - total_eval) / session._portfolio_peak * 100
        if dd >= max_dd:
            logger.warning(
                "[paper] 포트폴리오 MDD 서킷브레이커: %.2f%% >= %.1f%%", dd, max_dd,
            )
            # 남은 포지션 전량 청산 (paper)
            for sym, pos in list(session.positions.items()):
                sell_price = effective_sell_price(pos.avg_price, cost)
                _paper_sell(
                    session, sym, pos.qty, pos.avg_price, sell_price, cost,
                    step="S-CIRCUIT",
                    reason=f"포트폴리오 MDD 서킷브레이커: -{dd:.2f}%",
                )
            session.status = "stopped"
            session.stopped_at = datetime.now().isoformat()
            session.error_message = f"포트폴리오 MDD 서킷브레이커: -{dd:.2f}%"
            return

    logger.info(
        "루프 체크 [paper]: cash=%s, positions=%d/%d, trades=%d, decisions=%d",
        f"{session._cash:,.0f}", len(session.positions), ctx.max_positions,
        len(session.trade_log), len(session.decision_log),
    )

    await manager.broadcast("trading:update", {
        "session_id": session.id,
        "positions": len(session.positions),
        "trades": len(session.trade_log),
        "cash": session._cash,
    })


def _paper_sell(
    session: LiveSession,
    symbol: str,
    qty: int,
    close_price: float,
    sell_price: float,
    cost: CostConfig,
    step: str,
    reason: str,
    snapshot: dict | None = None,
    conditions: dict | None = None,
    candle_dt: str | None = None,
) -> None:
    """Paper 모드 매도 처리: 포지션 제거 + 현금 회수."""
    pos = session.positions.get(symbol)
    if not pos:
        return

    proceeds = sell_price * qty
    session._cash += proceeds

    result = {
        "success": True,
        "order_id": f"SIM-{uuid.uuid4().hex[:8]}",
        "message": f"시뮬 매도 체결 @ {sell_price:,.0f}",
    }
    _log_trade(
        session, symbol, "SELL", step, qty, sell_price, result,
        reason=reason, snapshot=snapshot, position=pos,
        conditions=conditions,
        sizing={
            "close_price": close_price,
            "sell_price_effective": round(sell_price, 2),
            "sell_commission": cost.sell_commission,
            "slippage_pct": cost.slippage_pct,
            "proceeds": round(proceeds, 2),
            "cash_after": round(session._cash, 2),
        },
        candle_dt=candle_dt,
    )
    # 포지션 제거
    session.positions.pop(symbol, None)


# ── real 모드: KIS API ──────────────────────────────────────


async def _real_loop_tick(
    session: LiveSession,
    strategy: dict,
    cost: CostConfig,
    ctx: TradingContext,
    stop_loss_pct: float | None,
    trailing_stop_pct: float | None,
) -> None:
    """Real 모드 1회 루프: KIS API 실주문."""
    from .kis_client import get_kis_client
    from .kis_order import KISOrderExecutor

    client = get_kis_client(is_mock=False)
    executor = KISOrderExecutor(client)

    # H1: 미체결 주문 체결 확인
    await executor.check_pending_orders()

    balance_data = await client.inquire_balance()
    kis_positions = {p["symbol"]: p for p in balance_data["positions"]}
    cash = balance_data["account"]["cash"]
    total_eval = balance_data["account"].get("total_deposit", 0)

    # H2: 포트폴리오 MDD 서킷브레이커
    from app.core.config import settings
    max_dd_pct = settings.WORKFLOW_MAX_DRAWDOWN_PCT
    # 고점 추적 (세션 속성)
    if not hasattr(session, "_portfolio_peak"):
        session._portfolio_peak = total_eval
    if total_eval > session._portfolio_peak:
        session._portfolio_peak = total_eval

    if session._portfolio_peak > 0 and total_eval > 0:
        portfolio_dd = (session._portfolio_peak - total_eval) / session._portfolio_peak * 100
        if portfolio_dd >= max_dd_pct:
            logger.warning(
                "포트폴리오 MDD 서킷브레이커 발동: %.2f%% >= %.1f%% (고점 %s → 현재 %s)",
                portfolio_dd, max_dd_pct,
                f"{session._portfolio_peak:,.0f}", f"{total_eval:,.0f}",
            )
            # 전량 청산
            for sym, kp in kis_positions.items():
                qty = kp.get("qty", 0)
                if qty > 0:
                    result = await executor.sell(sym, qty, order_type="MARKET")
                    pos = session.positions.get(sym)
                    _log_trade(
                        session, sym, "SELL", "S-CIRCUIT", qty,
                        kp.get("current_price", 0), result,
                        reason=f"포트폴리오 MDD 서킷브레이커: -{portfolio_dd:.2f}% (기준 -{max_dd_pct}%)",
                        position=pos,
                    )
            # 워크플로우 이벤트 기록
            try:
                from app.core.database import async_session as get_db
                from app.workflow.models import WorkflowEvent
                import uuid as _uuid
                async with get_db() as db:
                    db.add(WorkflowEvent(
                        id=_uuid.uuid4(),
                        phase="TRADING",
                        event_type="portfolio_circuit_breaker",
                        message=f"포트폴리오 MDD -{portfolio_dd:.2f}% 발동. 전량 청산.",
                        data={
                            "peak": session._portfolio_peak,
                            "current": total_eval,
                            "drawdown_pct": round(portfolio_dd, 4),
                            "threshold_pct": max_dd_pct,
                        },
                    ))
                    await db.commit()
            except Exception as e:
                logger.warning("서킷브레이커 이벤트 기록 실패: %s", e)
            # 세션 중지
            session.status = "stopped"
            session.stopped_at = datetime.now().isoformat()
            session.error_message = f"포트폴리오 MDD 서킷브레이커 발동: -{portfolio_dd:.2f}%"
            await _clear_session_state(session.id)
            await manager.broadcast("trading:status", {
                "session_id": session.id,
                "status": "stopped",
                "reason": "circuit_breaker",
            })
            return

    for sym, kp in kis_positions.items():
        if sym in session.positions:
            session.positions[sym].qty = kp["qty"]
        else:
            session.positions[sym] = LivePosition(
                symbol=sym, qty=kp["qty"], avg_price=kp["avg_price"],
                entry_date=datetime.now().strftime("%Y-%m-%d"),
            )

    for sym in list(session.positions.keys()):
        if sym not in kis_positions:
            session.positions.pop(sym)

    for sym, pos in list(session.positions.items()):
        current_price = kis_positions.get(sym, {}).get("current_price", 0)
        if current_price <= 0 or pos.avg_price <= 0:
            continue
        if current_price > pos.highest_price:
            pos.highest_price = current_price

        if stop_loss_pct is not None:
            loss_pct = (current_price - pos.avg_price) / pos.avg_price * 100
            if loss_pct <= -stop_loss_pct:
                result = await executor.sell(sym, pos.qty, order_type="MARKET")
                _log_trade(
                    session, sym, "SELL", "S-STOP", pos.qty, current_price, result,
                    reason=f"손절: 평단 대비 {round(loss_pct, 2):+.2f}% (기준 -{stop_loss_pct}%)",
                    snapshot={"close": current_price, "avg_price": round(pos.avg_price)},
                    position=pos,
                )
                continue

        if trailing_stop_pct is not None and pos.highest_price > 0:
            drop_pct = (pos.highest_price - current_price) / pos.highest_price * 100
            if drop_pct >= trailing_stop_pct:
                result = await executor.sell(sym, pos.qty, order_type="MARKET")
                _log_trade(
                    session, sym, "SELL", "S-TRAIL", pos.qty, current_price, result,
                    reason=f"트레일링: 고점({round(pos.highest_price):,}) 대비 -{round(drop_pct, 2):.2f}%",
                    snapshot={"close": current_price, "highest_price": round(pos.highest_price)},
                    position=pos,
                )
                continue

    held_symbols = set(session.positions.keys())
    universe = ctx.symbols or [sym for sym in kis_positions]
    scan_limit = max(ctx.max_positions * 5, 50)
    scan_symbols = list(held_symbols | set(universe[:scan_limit]))

    if scan_symbols:
        await _real_check_signals(
            session, executor, strategy, scan_symbols, cash, cost, ctx
        )

    logger.info(
        "루프 체크 [real]: cash=%s, positions=%d, trades=%d",
        f"{cash:,.0f}", len(session.positions), len(session.trade_log),
    )

    await manager.broadcast("trading:update", {
        "session_id": session.id,
        "positions": len(session.positions),
        "trades": len(session.trade_log),
        "cash": cash,
    })


async def _real_check_signals(
    session: LiveSession,
    executor,
    strategy: dict,
    symbols: list[str],
    cash: float,
    cost: CostConfig,
    ctx: TradingContext,
) -> None:
    """Real 모드 시그널 체크 → KIS 주문."""
    from datetime import date, timedelta

    end = date.today()
    start = end - timedelta(days=120)
    interval = strategy.get("interval", "1d")

    is_alpha_strategy = any(
        c.get("indicator", "").startswith("alpha_")
        for c in strategy.get("buy_conditions", [])
    )

    try:
        if is_alpha_strategy:
            from app.backtest.data_loader import load_enriched_candles
            df = await load_enriched_candles(symbols, start, end, interval)
        else:
            from app.backtest.data_loader import load_candles
            df = await load_candles(symbols, start, end, interval)
    except Exception as e:
        logger.warning("캔들 로딩 실패: %s", e)
        return

    if df.is_empty():
        return

    for sym in symbols:
        sym_df = df.filter(pl.col("symbol") == sym).sort("dt")
        if sym_df.height < 30:
            continue

        try:
            sym_df = generate_signals(sym_df, strategy)
        except Exception as e:
            logger.debug("시그널 생성 실패 %s: %s", sym, e)
            continue

        last_row = sym_df.tail(1).to_dicts()[0]
        signal = last_row.get("signal", 0)
        snapshot = _collect_snapshot(last_row)
        conditions = _collect_conditions_detail(last_row, strategy, signal)

        # H1: 미체결 주문 있는 종목 스킵
        from .kis_order import has_pending_order
        if has_pending_order(sym):
            _log_decision(session, sym, "SKIP_PENDING",
                          "미체결 주문 존재 — 중복 주문 방지",
                          signal=signal, conditions=conditions, snapshot=snapshot)
            continue

        if signal == 1 and sym not in session.positions:
            if len(session.positions) >= ctx.max_positions:
                continue
            alloc = ctx.initial_capital * ctx.position_size_pct
            alloc = min(alloc, cash * 0.95)
            price = int(last_row.get("close", 0))
            if price <= 0:
                continue
            qty = int(alloc / price)
            if qty <= 0:
                continue

            result = await executor.buy(sym, qty, price, order_type="LIMIT")
            _log_trade(session, sym, "BUY", "B1", qty, price, result,
                       reason="매수 시그널", snapshot=snapshot, conditions=conditions)

        elif signal == -1 and sym in session.positions:
            pos = session.positions[sym]
            if pos.qty <= 0:
                continue
            result = await executor.sell(sym, pos.qty, order_type="MARKET")
            _log_trade(session, sym, "SELL", "", pos.qty, last_row.get("close", 0), result,
                       reason="매도 시그널", snapshot=snapshot, position=pos, conditions=conditions)


# ── 공통 유틸 ────────────────────────────────────────────────


def _log_trade(
    session: LiveSession,
    symbol: str,
    side: str,
    step: str,
    qty: int,
    price: float,
    result: dict,
    reason: str = "",
    snapshot: dict | None = None,
    position: LivePosition | None = None,
    conditions: dict | None = None,
    sizing: dict | None = None,
    candle_dt: str | None = None,
) -> None:
    """매매 기록.

    trade_log에 저장되는 필드:
      - symbol, name, side, step, qty, price: 기본 거래 정보
      - pnl_pct, pnl_amount, holding_minutes: SELL 시 자동 계산
      - reason: 사람이 읽을 수 있는 매매 사유 (조건식 + 실제값 포함)
      - snapshot: 봉 데이터 전체 지표 스냅샷
      - conditions: 매매 조건 충족 상세 (indicator, op, threshold, actual, met)
      - sizing: 포지션 사이징 상세 (alloc, cash, qty 산출 과정)
      - position_context: 매도 시 포지션 상태 (avg_price, entry_date, highest_price)
      - candle_dt: 체결 기준 봉의 시각 (캔들 타임스탬프)
    """
    pnl_pct = None
    pnl_amount = None
    holding_minutes = None
    position_context = None

    if side == "SELL" and position and position.avg_price > 0 and price > 0:
        pnl_pct = round((price - position.avg_price) / position.avg_price * 100, 4)
        pnl_amount = round((price - position.avg_price) * qty, 2)
        position_context = {
            "avg_price": round(position.avg_price, 2),
            "entry_date": position.entry_date,
            "entry_candle_dt": getattr(position, "entry_candle_dt", ""),
            "highest_price": round(position.highest_price, 2),
            "qty_held": position.qty,
        }
        # 보유시간: 캔들 dt 기반으로 계산 (시뮬레이션 시간이 아닌 실제 시장 시간)
        entry_dt_str = getattr(position, "entry_candle_dt", "") or position.entry_date
        sell_dt_str = candle_dt or ""
        if entry_dt_str and sell_dt_str:
            try:
                entry_dt = datetime.fromisoformat(str(entry_dt_str))
                sell_dt = datetime.fromisoformat(str(sell_dt_str))
                holding_minutes = round(
                    (sell_dt - entry_dt).total_seconds() / 60, 1
                )
            except (ValueError, TypeError):
                pass
        # 일봉인 경우 분 단위가 0이면 일수로 환산
        if holding_minutes is not None and holding_minutes == 0 and entry_dt_str and sell_dt_str:
            try:
                from datetime import date as date_type
                entry_d = datetime.fromisoformat(str(entry_dt_str)).date()
                sell_d = datetime.fromisoformat(str(sell_dt_str)).date()
                holding_days = (sell_d - entry_d).days
                if holding_days > 0:
                    # 일봉 기준 거래일수 × 장중 시간(390분)으로 환산
                    holding_minutes = round(holding_days * 390.0, 1)
            except (ValueError, TypeError):
                pass

    # 타임스탬프: 캔들 dt가 있으면 캔들 시각, 없으면 현재 시각
    ts = candle_dt if candle_dt else datetime.now().isoformat()

    entry = {
        "symbol": symbol,
        "name": get_stock_name(symbol),
        "side": side,
        "step": step,
        "qty": qty,
        "price": price,
        "pnl_pct": pnl_pct,
        "pnl_amount": pnl_amount,
        "holding_minutes": holding_minutes,
        "success": result.get("success", False),
        "order_id": result.get("order_id", ""),
        "message": result.get("message", ""),
        "timestamp": ts,
        "reason": reason,
        "snapshot": snapshot,
        "conditions": conditions,
        "sizing": sizing,
        "position_context": position_context,
    }
    session.trade_log.append(entry)

    # DB 영속화 (fire-and-forget — 실패해도 메모리 로그는 보존)
    asyncio.create_task(_persist_trade(session.context.id, entry))

    logger.info(
        "매매 기록: %s %s %s %d주 @ %s — %s (%s)%s",
        side, symbol, step, qty, f"{price:,.0f}",
        "성공" if result.get("success") else "실패",
        reason[:80] if reason else "-",
        f" PnL={pnl_pct:+.2f}%" if pnl_pct is not None else "",
    )


# ── DB 영속화 헬퍼 ────────────────────────────────────────────


async def _persist_trade(context_id: str, entry: dict) -> None:
    """매매 기록을 live_trades 테이블에 저장 (fire-and-forget)."""
    try:
        from app.core.database import async_session
        from app.workflow.models import LiveTrade

        ts_str = entry.get("timestamp", "")
        executed_at = None
        if ts_str:
            try:
                executed_at = datetime.fromisoformat(str(ts_str))
            except (ValueError, TypeError):
                pass

        async with async_session() as db:
            trade = LiveTrade(
                id=uuid.uuid4(),
                context_id=uuid.UUID(context_id),
                symbol=entry["symbol"],
                name=entry.get("name"),
                side=entry["side"],
                step=entry.get("step", ""),
                qty=entry["qty"],
                price=entry["price"],
                pnl_pct=entry.get("pnl_pct"),
                pnl_amount=entry.get("pnl_amount"),
                holding_minutes=entry.get("holding_minutes"),
                success=entry.get("success", False),
                order_id=entry.get("order_id", ""),
                reason=entry.get("reason"),
                snapshot=entry.get("snapshot"),
                conditions=entry.get("conditions"),
                sizing=entry.get("sizing"),
                position_context=entry.get("position_context"),
            )
            if executed_at:
                trade.executed_at = executed_at
            db.add(trade)
            await db.commit()
    except Exception as e:
        logger.warning("LiveTrade DB 저장 실패: %s", e)


async def _save_session_state(session: LiveSession) -> None:
    """세션 상태를 trading_contexts.session_state에 저장."""
    state = {
        "status": session.status,
        "positions": {
            sym: {
                "qty": p.qty,
                "avg_price": p.avg_price,
                "highest_price": p.highest_price,
                "entry_date": p.entry_date,
                "entry_candle_dt": p.entry_candle_dt,
            }
            for sym, p in session.positions.items()
        },
        "cash": session._cash,
        "last_processed_dt": getattr(session, "_last_processed_dt", None),
        "started_at": session.started_at,
        "trade_count": len(session.trade_log),
    }
    try:
        from app.core.database import async_session
        from app.workflow.models import TradingContextModel
        from sqlalchemy import update

        async with async_session() as db:
            stmt = (
                update(TradingContextModel)
                .where(TradingContextModel.id == uuid.UUID(session.context.id))
                .values(session_state=state)
            )
            await db.execute(stmt)
            await db.commit()
    except Exception as e:
        logger.warning("세션 상태 저장 실패: %s", e)


async def _clear_session_state(session_id: str) -> None:
    """세션 상태 클리어 (중지 시 재시작 복구 방지)."""
    try:
        from app.core.database import async_session
        from app.workflow.models import TradingContextModel
        from sqlalchemy import update

        async with async_session() as db:
            stmt = (
                update(TradingContextModel)
                .where(TradingContextModel.id == uuid.UUID(session_id))
                .values(session_state=None)
            )
            await db.execute(stmt)
            await db.commit()
    except Exception as e:
        logger.warning("세션 상태 클리어 실패: %s", e)


async def restore_sessions_from_db() -> int:
    """서버 재시작 시 active 세션 복구.

    load_active_contexts_from_db() 호출 이후에 실행해야 한다.
    """
    from app.trading.context import list_contexts

    count = 0
    for ctx in list_contexts():
        if not ctx.session_state or ctx.session_state.get("status") != "running":
            continue

        session = LiveSession(
            id=ctx.id,
            context=ctx,
            status="running",
            started_at=ctx.session_state.get("started_at", ""),
            _cash=ctx.session_state.get("cash", ctx.initial_capital),
        )

        # 포지션 복원
        for sym, pdata in ctx.session_state.get("positions", {}).items():
            session.positions[sym] = LivePosition(
                symbol=sym,
                qty=pdata["qty"],
                avg_price=pdata["avg_price"],
                highest_price=pdata.get("highest_price", 0),
                entry_date=pdata.get("entry_date", ""),
                entry_candle_dt=pdata.get("entry_candle_dt", ""),
            )

        session._last_processed_dt = ctx.session_state.get("last_processed_dt")

        # 알파 팩터 등록 (재시작 시 인메모리 등록 유실 복구)
        if ctx.source_factor_id:
            try:
                from app.alpha.backtest_bridge import register_alpha_factor
                from app.alpha.models import AlphaFactor
                from app.core.database import async_session as get_session
                from sqlalchemy import select
                async with get_session() as db:
                    factor = await db.get(AlphaFactor, ctx.source_factor_id)
                    if factor:
                        register_alpha_factor(str(factor.id), factor.expression_str)
                        logger.info("알파 팩터 등록 복구: %s (%s)", factor.name, factor.id)
            except Exception as e:
                logger.warning("알파 팩터 등록 복구 실패: %s", e)

        _sessions[session.id] = session

        # 백그라운드 러너 재개
        session._task = asyncio.create_task(_run_loop(session))
        count += 1
        logger.info(
            "세션 복구: %s (포지션 %d개, 현금 %s)",
            session.id, len(session.positions), f"{session._cash:,.0f}",
        )

    return count
