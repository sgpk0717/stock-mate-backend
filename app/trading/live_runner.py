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
class LiveScaleEntry:
    """개별 매수 건 (backtest _ScaleEntry 미러링)."""
    date: str
    price: float
    qty: int
    step: str  # "B1", "B2", "B-EXT"


@dataclass
class LivePosition:
    """실시간 포지션 추적 (entries 기반 분할매매 지원)."""
    symbol: str
    entries: list[LiveScaleEntry] = field(default_factory=list)
    highest_price: float = 0.0
    conviction: float = 0.0
    target_qty: int = 0               # 분할매수 목표 수량
    scale_in_count: int = 0           # B2 횟수
    has_partial_exited: bool = False   # S-HALF 1회 제한
    entry_candle_dt: str = ""         # 진입 봉의 원본 dt (보유시간 계산용)

    @property
    def qty(self) -> int:
        return sum(e.qty for e in self.entries)

    @property
    def avg_price(self) -> float:
        total_cost = sum(e.price * e.qty for e in self.entries)
        total_q = self.qty
        return total_cost / total_q if total_q > 0 else 0.0

    @property
    def entry_date(self) -> str:
        return self.entries[0].date if self.entries else ""


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
                sym: {
                    "symbol": p.symbol, "qty": p.qty, "avg_price": p.avg_price,
                    "entries_count": len(p.entries),
                    "conviction": p.conviction,
                    "scale_in_count": p.scale_in_count,
                    "has_partial_exited": p.has_partial_exited,
                    "target_qty": p.target_qty,
                }
                for sym, p in self.positions.items()
            },
            "trade_count": len(self.trade_log),
            "error_message": self.error_message,
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
        }


# 세션 저장소
_sessions: dict[str, LiveSession] = {}
_intraday_collector_task: asyncio.Task | None = None


async def _sync_session_to_redis(session: LiveSession) -> None:
    """세션 상태를 Redis Hash에 동기화 (Phase 1: 이중 기록)."""
    try:
        from app.core.redis import hset
        import json
        d = session.to_dict()
        await hset(f"sessions:{session.id}", {
            "id": d["id"],
            "context_id": d["context_id"],
            "mode": d["mode"],
            "strategy_name": d["strategy_name"],
            "status": d["status"],
            "positions": json.dumps(d["positions"], ensure_ascii=False),
            "trade_count": str(d["trade_count"]),
            "error_message": d["error_message"],
            "started_at": d["started_at"],
            "stopped_at": d["stopped_at"],
        })
    except Exception:
        pass  # Redis 실패해도 기존 동작에 영향 없음


async def _sync_sessions_index_to_redis() -> None:
    """활성 세션 ID 목록을 Redis Set에 동기화."""
    try:
        from app.core.redis import get_client
        r = get_client()
        ids = list(_sessions.keys())
        await r.delete("sessions:index")
        if ids:
            await r.sadd("sessions:index", *ids)
    except Exception:
        pass


async def start_session(ctx: TradingContext) -> LiveSession:
    """전략 실거래 세션 시작."""
    session = LiveSession(
        id=ctx.id,
        context=ctx,
        status="running",
        started_at=datetime.now().isoformat(),
        _cash=ctx.initial_capital,
    )

    # 오늘 이미 매매한 기록이 있으면 마지막 시각 복구 (재시작 시 중복 리플레이 방지)
    try:
        from app.core.database import async_session as get_session
        from sqlalchemy import text
        async with get_session() as db:
            row = await db.execute(text(
                "SELECT MAX(executed_at) FROM live_trades "
                "WHERE executed_at >= CURRENT_DATE"
            ))
            last_trade_dt = row.scalar()
            if last_trade_dt:
                session._last_processed_dt = str(last_trade_dt)
                logger.info("세션 %s: 오늘 마지막 매매 시각 복구 — %s", session.id[:8], last_trade_dt)
    except Exception as e:
        logger.debug("마지막 매매 시각 복구 실패: %s", e)

    _sessions[session.id] = session

    # 백그라운드 러너 시작
    session._task = asyncio.create_task(_run_loop(session))
    logger.info("실거래 세션 시작: %s (mode=%s)", session.id, ctx.mode)

    # Redis 동기화 (Phase 1)
    await _sync_session_to_redis(session)
    await _sync_sessions_index_to_redis()

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

    # Redis 동기화 (Phase 1)
    await _sync_session_to_redis(session)
    await _sync_sessions_index_to_redis()

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


# ── 분할매매 헬퍼 ────────────────────────────────────────────


def _reduce_entries(pos: LivePosition, sell_qty: int) -> None:
    """분할매도 시 entries에서 수량 차감 (FIFO, backtest engine 동일)."""
    remaining = sell_qty
    new_entries: list[LiveScaleEntry] = []
    for entry in pos.entries:
        if remaining <= 0:
            new_entries.append(entry)
            continue
        if entry.qty <= remaining:
            remaining -= entry.qty
        else:
            entry.qty -= remaining
            remaining = 0
            new_entries.append(entry)
    pos.entries = new_entries


def _calc_conviction_live(
    strategy: dict, ps_cfg: dict, row: dict | None = None,
) -> float:
    """확신도 계산 (backtest engine _calc_conviction 간소화)."""
    mode = ps_cfg.get("mode", "fixed")
    if mode == "fixed":
        return 1.0
    if mode == "conviction":
        weights = ps_cfg.get("weights") or {}
        buy_conds = strategy.get("buy_conditions", [])
        if not buy_conds or not weights:
            return 1.0
        total_w = sum(weights.values()) or 1.0
        met_w = sum(
            weights.get(c.get("indicator", ""), 1.0 / len(buy_conds))
            for c in buy_conds
        )
        return min(met_w / total_w, 1.0)
    if mode == "atr_target" and row is not None:
        atr = row.get("atr_14", 0) or 0
        price = row.get("close", 1) or 1
        atr_pct = atr / price if price > 0 else 0.1
        return min(1.0, 0.02 / max(atr_pct, 0.001))
    return 1.0


# ── paper 모드: 자체 시뮬레이션 ──────────────────────────────


async def _run_loop(session: LiveSession) -> None:
    """전략 실행 메인 루프."""
    logger.info("_run_loop 시작: %s (mode=%s)", session.id, session.context.mode)
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
    atr_stop_mult = risk.get("atr_stop_multiplier")

    scaling = ctx.scaling or {}
    scaling_enabled = scaling.get("enabled", False)
    ps_cfg = ctx.position_sizing or {}

    check_interval = 30  # 초

    try:
        while session.status == "running":
            logger.info("_run_loop tick 시작: %s", session.id[:8])
            try:
                if is_paper:
                    await _paper_loop_tick(
                        session, strategy, cost, ctx,
                        stop_loss_pct, trailing_stop_pct,
                        atr_stop_mult=atr_stop_mult,
                        scaling=scaling,
                        scaling_enabled=scaling_enabled,
                        ps_cfg=ps_cfg,
                    )
                else:
                    await _real_loop_tick(
                        session, strategy, cost, ctx,
                        stop_loss_pct, trailing_stop_pct,
                        atr_stop_mult=atr_stop_mult,
                        scaling=scaling,
                        scaling_enabled=scaling_enabled,
                        ps_cfg=ps_cfg,
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


async def start_intraday_candle_collector(symbols: list[str]) -> None:
    """장중 1분봉 수집기 시작 — live_runner와 독립적으로 실행.

    09:00 KST부터 현재까지 빈 구간을 앞쪽부터 채워나감.
    5분 주기로 반복, 15:30 이후 자동 종료.
    """
    global _intraday_collector_task
    if _intraday_collector_task and not _intraday_collector_task.done():
        logger.info("장중 분봉 수집기 이미 실행 중 — 스킵")
        return
    _intraday_collector_task = asyncio.create_task(
        _intraday_collect_loop(symbols)
    )
    logger.info("장중 분봉 수집기 시작 (%d종목)", len(symbols))


async def _intraday_collect_loop(symbols: list[str]) -> None:
    """5분 주기로 빈 구간 채우기 루프."""
    from datetime import date as date_cls, timedelta, timezone as tz

    KST = tz(timedelta(hours=9))

    while True:
        try:
            now_kst = datetime.now(KST)
            # 15:30 이후면 종료
            if now_kst.hour > 15 or (now_kst.hour == 15 and now_kst.minute >= 30):
                logger.info("장중 분봉 수집기 종료 (15:30 이후)")
                break

            today = now_kst.date()
            collected = await _collect_missing_candles(symbols, today, now_kst)
            if collected > 0:
                logger.info("장중 분봉 수집: %d봉 추가", collected)

        except Exception as e:
            logger.warning("장중 분봉 수집 루프 오류: %s", e)

        await asyncio.sleep(300)  # 5분 대기


async def _collect_missing_candles(
    symbols: list[str],
    today: "date",
    now_kst: datetime,
) -> int:
    """09:00~현재 사이 빈 구간을 앞쪽부터 채움."""
    from datetime import timedelta, timezone as tz
    from app.core.database import async_session as get_session
    from sqlalchemy import text

    KST = tz(timedelta(hours=9))
    today_str = today.strftime("%Y%m%d")
    market_open_kst = datetime(today.year, today.month, today.day, 9, 0, tzinfo=KST)

    # DB에서 오늘 1m 캔들의 마지막 시각 조회 (샘플 종목 기준)
    sample_sym = symbols[0] if symbols else "005930"
    try:
        async with get_session() as db:
            row = await db.execute(
                text(
                    "SELECT MAX(dt) FROM stock_candles "
                    "WHERE symbol = :sym AND interval = '1m' AND dt >= :start"
                ),
                {"sym": sample_sym, "start": market_open_kst.replace(tzinfo=None)},
            )
            last_dt = row.scalar()
    except Exception as e:
        logger.warning("마지막 수집 시각 조회 실패: %s", e)
        last_dt = None

    # 수집 시작점: DB에 있는 마지막 시각 이후, 또는 09:00
    if last_dt:
        # naive → KST aware
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=KST)
        collect_from = last_dt
    else:
        collect_from = market_open_kst

    # 이미 최신이면 스킵
    if collect_from >= now_kst - timedelta(minutes=2):
        return 0

    # KIS API로 수집 (collect_from ~ now_kst 구간)
    hour_str = now_kst.strftime("%H%M%S")

    try:
        from app.trading.kis_client import get_kis_client
        from app.services.candle_writer import write_candles_bulk

        client = get_kis_client(is_mock=False)
        total_written = 0

        for symbol in symbols:
            try:
                raw_candles, _, _ = await client.get_minute_candles(
                    symbol, today_str, hour_str,
                )
                if not raw_candles:
                    continue

                one_min: list[dict] = []
                for c in raw_candles:
                    dt_str = c.get("stck_bsop_date", "") + c.get("stck_cntg_hour", "")
                    if len(dt_str) != 14:
                        continue
                    dt = datetime.strptime(dt_str, "%Y%m%d%H%M%S").replace(tzinfo=KST)
                    if dt.date() != today:
                        continue
                    # collect_from 이후 데이터만 수집 (이미 있는 구간 스킵)
                    if dt <= collect_from:
                        continue
                    one_min.append({
                        "dt": dt,
                        "open": float(c.get("stck_oprc", 0)),
                        "high": float(c.get("stck_hgpr", 0)),
                        "low": float(c.get("stck_lwpr", 0)),
                        "close": float(c.get("stck_prpr", 0)),
                        "volume": int(c.get("cntg_vol", 0)),
                    })

                if one_min:
                    await write_candles_bulk(symbol, one_min, "1m")
                    total_written += len(one_min)

            except Exception as e:
                logger.warning("장중 분봉 수집 실패 (%s): %s", symbol, e)
                continue

            await asyncio.sleep(0.1)  # rate limit

        return total_written

    except Exception as e:
        logger.warning("장중 분봉 수집 전체 실패: %s", e)
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
    *,
    atr_stop_mult: float | None = None,
    scaling: dict | None = None,
    scaling_enabled: bool = False,
    ps_cfg: dict | None = None,
) -> None:
    """Paper 모드: 장중 실시간 시뮬레이션.

    과거 캔들 데이터를 가지고 실제 장중 매매를 재현한다.
    - 첫 호출: 오늘 장 시작(09:00)부터 현재까지의 모든 봉을 시간순 워크스루
      (5분봉 기준 ~78봉/일, 매 봉마다 매수/매도 판단)
    - 이후 호출: _last_processed_dt 이후 새로 생성된 봉만 처리
    """
    from datetime import date, timedelta

    # 종목 리스트 결정 (보유 종목 + 유니버스 전체)
    universe = ctx.symbols or []
    scan_symbols = list(set(list(session.positions.keys()) + universe))

    if not scan_symbols:
        return

    end = date.today()
    start = end - timedelta(days=120)
    interval = strategy.get("interval", "5m")

    # 분봉 수집은 별도 장중 수집기(intraday_collector)가 담당
    # live_runner는 DB에 있는 데이터로만 매매

    is_alpha = any(
        c.get("indicator", "").startswith("alpha_")
        for c in strategy.get("buy_conditions", [])
    )

    try:
        # to_thread: 데이터 로딩을 스레드풀에서 실행 → 메인 이벤트 루프 해방
        if is_alpha:
            from app.backtest.data_loader import load_enriched_candles_sync
            df = await asyncio.to_thread(load_enriched_candles_sync, scan_symbols, start, end, interval)
        else:
            from app.backtest.data_loader import load_candles_sync
            df = await asyncio.to_thread(load_candles_sync, scan_symbols, start, end, interval)
    except Exception as e:
        logger.warning("캔들 로딩 실패: %s", e)
        return

    if df.is_empty():
        return

    # 중복 행 감지 로그
    _total_rows = df.height
    _unique_rows = df.unique(subset=["symbol", "dt"]).height
    if _total_rows != _unique_rows:
        logger.warning(
            "paper_tick [%s]: 캔들 데이터에 중복 행 %d건 (total=%d, unique=%d)",
            session.id[:8], _total_rows - _unique_rows, _total_rows, _unique_rows,
        )
        # 중복 제거
        df = df.unique(subset=["symbol", "dt"], keep="first")

    last_dt = getattr(session, "_last_processed_dt", None)

    # 오늘 장 시작 시각 (09:00 KST, naive) — DB의 dt가 KST naive이므로 맞춤
    from datetime import time as time_type
    today_market_open = datetime.combine(end, time_type(9, 0))

    logger.info(
        "paper_tick: 캔들 로딩 완료 (height=%d, interval=%s, symbols=%d)",
        df.height, interval, len(scan_symbols),
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
            rows = []
            for r in all_rows:
                dt_val = r.get("dt")
                if dt_val is None:
                    continue
                try:
                    if dt_val >= today_market_open:
                        rows.append(r)
                except TypeError:
                    # 타입 불일치 시 문자열 비교 폴백
                    if str(dt_val) >= str(today_market_open):
                        rows.append(r)
        else:
            # 이후 호출: 마지막 처리 봉 이후 새 봉만 (datetime 비교로 통일)
            _last_dt_parsed = None
            if last_dt:
                try:
                    _last_dt_parsed = datetime.fromisoformat(str(last_dt))
                except (ValueError, TypeError):
                    pass

            if _last_dt_parsed:
                rows = []
                for r in all_rows:
                    dt_val = r.get("dt")
                    if dt_val is None:
                        continue
                    try:
                        if dt_val > _last_dt_parsed:
                            rows.append(r)
                    except TypeError:
                        # 타입 불일치 폴백
                        if str(dt_val).replace(" ", "T") > str(last_dt):
                            rows.append(r)
            else:
                rows = list(all_rows)

        if rows:
            sym_frames[sym] = rows
        else:
            _no_today += 1
            pass

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

    # 중복 봉 감지 (같은 symbol+dt가 2회 이상이면 WARNING)
    _seen_bars: set[str] = set()
    _dup_count = 0

    for sym, row in all_bars:
        dt_val = row.get("dt")
        candle_dt_str = dt_val.isoformat() if hasattr(dt_val, "isoformat") else str(dt_val)
        close_price = row.get("close", 0)
        signal = row.get("signal", 0)

        if close_price <= 0:
            continue

        # 중복 봉 스킵 (같은 종목+시각 2번 이상 처리 방지)
        bar_key = f"{sym}:{candle_dt_str}"
        if bar_key in _seen_bars:
            _dup_count += 1
            if _dup_count <= 5:
                logger.warning("중복 봉 스킵: %s (signal=%d)", bar_key, signal)
            continue
        _seen_bars.add(bar_key)

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

            # ATR 동적 손절
            if atr_stop_mult is not None and pos.avg_price > 0:
                atr_val = row.get("atr_14", 0) or 0
                if atr_val > 0:
                    stop_line = pos.avg_price - atr_val * atr_stop_mult
                    if close_price <= stop_line:
                        sell_price = effective_sell_price(close_price, cost)
                        risk_snap = {
                            "close": close_price, "avg_price": round(pos.avg_price),
                            "atr_14": round(atr_val, 2), "stop_line": round(stop_line),
                        }
                        _log_decision(
                            session, sym, "RISK_ATR_STOP",
                            f"ATR 스탑: {close_price:,.0f} <= {stop_line:,.0f} (평단-ATR×{atr_stop_mult})",
                            risk={"type": "atr_stop", "stop_line": round(stop_line), "atr": round(atr_val, 2)},
                        )
                        _paper_sell(session, sym, pos.qty, close_price, sell_price, cost,
                                    step="S-STOP",
                                    reason=f"ATR 스탑: {close_price:,.0f} <= {stop_line:,.0f}",
                                    snapshot=risk_snap, candle_dt=candle_dt_str)
                        continue

            # S-HALF 부분 익절
            if scaling_enabled and not pos.has_partial_exited and pos.qty > 1:
                _scaling = scaling or {}
                partial_gain_pct = _scaling.get("partial_exit_gain_pct", 5.0)
                partial_exit_ratio = _scaling.get("partial_exit_pct", 0.5)
                if pos.avg_price > 0:
                    gain_pct = (close_price - pos.avg_price) / pos.avg_price * 100
                    if gain_pct >= partial_gain_pct:
                        sell_qty = max(1, int(pos.qty * partial_exit_ratio))
                        if sell_qty >= pos.qty:
                            sell_qty = pos.qty - 1
                        if sell_qty > 0:
                            sell_price = effective_sell_price(close_price, cost)
                            _log_decision(
                                session, sym, "PARTIAL_EXIT",
                                f"부분 익절: +{gain_pct:.2f}% >= {partial_gain_pct}% → {sell_qty}주 매도",
                            )
                            _paper_sell(session, sym, sell_qty, close_price, sell_price, cost,
                                        step="S-HALF",
                                        reason=f"부분 익절: 평단 대비 +{gain_pct:.2f}%",
                                        snapshot={"close": close_price, "avg_price": round(pos.avg_price),
                                                  "gain_pct": round(gain_pct, 2)},
                                        candle_dt=candle_dt_str, partial=True)
                            pos.has_partial_exited = True

        snapshot = _collect_snapshot(row)
        conditions = _collect_conditions_detail(row, strategy, signal)

        # ── 매도 시그널 ──
        if signal == -1 and sym in session.positions:
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
            continue

        # ── B2 분할매수 ──
        if scaling_enabled and sym in session.positions:
            _scaling = scaling or {}
            pos = session.positions[sym]
            max_scale = _scaling.get("max_scale_in", 1)
            if pos.scale_in_count < max_scale and pos.qty > 0 and pos.avg_price > 0:
                drop_trigger = _scaling.get("scale_in_drop_pct", 3.0)
                drop_pct = (pos.avg_price - close_price) / pos.avg_price * 100
                if drop_pct >= drop_trigger:
                    remaining_qty = pos.target_qty - pos.qty
                    if remaining_qty > 0:
                        buy_p = effective_buy_price(close_price, cost)
                        cost_amount = buy_p * remaining_qty
                        # 현금 부족 시 수량 축소
                        if cost_amount > session._cash * 0.95:
                            remaining_qty = int(session._cash * 0.95 / buy_p)
                        if remaining_qty > 0:
                            total_cost = buy_p * remaining_qty
                            session._cash -= total_cost
                            pos.entries.append(LiveScaleEntry(
                                date=candle_dt_str, price=buy_p,
                                qty=remaining_qty, step="B2",
                            ))
                            pos.scale_in_count += 1
                            result = {"success": True, "order_id": f"SIM-{uuid.uuid4().hex[:8]}",
                                       "message": f"시뮬 추가매수 체결 @ {buy_p:,.0f}"}
                            _log_decision(
                                session, sym, "SCALE_IN",
                                f"B2 추가매수: 평단 대비 -{drop_pct:.2f}% (기준 -{drop_trigger}%) → {remaining_qty}주",
                                snapshot=snapshot,
                            )
                            _log_trade(session, sym, "BUY", "B2", remaining_qty, buy_p, result,
                                       reason=f"분할매수: 평단 대비 -{drop_pct:.2f}%",
                                       snapshot=snapshot, candle_dt=candle_dt_str,
                                       sizing={"drop_pct": round(drop_pct, 2), "target_qty": pos.target_qty,
                                               "current_qty": pos.qty, "scale_in_count": pos.scale_in_count})

        # ── 매수 시그널 (B1 신규 진입) ──
        if signal == 1 and sym not in session.positions:
            if len(session.positions) >= ctx.max_positions:
                _log_decision(session, sym, "SKIP_BUY",
                              f"최대 포지션: {len(session.positions)}/{ctx.max_positions}",
                              signal=signal, conditions=conditions, snapshot=snapshot)
                continue

            conviction = _calc_conviction_live(strategy, ps_cfg or {}, row)
            alloc = ctx.initial_capital * ctx.position_size_pct * conviction
            if scaling_enabled:
                initial_pct = (scaling or {}).get("initial_pct", 0.5)
            else:
                initial_pct = 1.0
            alloc *= initial_pct
            alloc = min(alloc, session._cash * 0.95)
            buy_price = effective_buy_price(close_price, cost)
            qty = int(alloc / buy_price)

            # target_qty: 분할매수 활성 시 전체 목표 수량
            if scaling_enabled and initial_pct > 0:
                target_qty = int(qty / initial_pct) if initial_pct < 1.0 else qty
            else:
                target_qty = qty

            sizing = {
                "initial_capital": ctx.initial_capital,
                "position_size_pct": ctx.position_size_pct,
                "conviction": round(conviction, 4),
                "initial_pct": initial_pct,
                "scaling_enabled": scaling_enabled,
                "alloc_raw": ctx.initial_capital * ctx.position_size_pct,
                "alloc_effective": round(alloc, 2),
                "close_price": close_price,
                "buy_price_effective": round(buy_price, 2),
                "qty": qty,
                "target_qty": target_qty,
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
                symbol=sym,
                entries=[LiveScaleEntry(
                    date=candle_dt_str, price=buy_price, qty=qty, step="B1",
                )],
                highest_price=close_price,
                conviction=conviction,
                target_qty=target_qty,
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

    # 중복 봉 요약
    if _dup_count > 0:
        logger.warning(
            "paper_tick [%s]: 중복 봉 %d건 스킵됨 (load_enriched_candles 중복 행 의심)",
            session.id[:8], _dup_count,
        )

    # 마지막 처리 봉 기록
    if all_bars:
        last_dt_val = all_bars[-1][1].get("dt")
        session._last_processed_dt = str(last_dt_val) if last_dt_val else None

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

    # Redis 동기화 (Phase 1: 매 tick마다)
    await _sync_session_to_redis(session)

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
    partial: bool = False,
) -> None:
    """Paper 모드 매도 처리.

    partial=True: _reduce_entries FIFO 차감, 포지션 유지 (S-HALF).
    partial=False: 포지션 완전 제거 (기존 동작).
    """
    pos = session.positions.get(symbol)
    if not pos:
        return

    actual_qty = min(qty, pos.qty)  # 안전: 보유량 초과 방지
    if actual_qty <= 0:
        return

    proceeds = sell_price * actual_qty
    session._cash += proceeds

    result = {
        "success": True,
        "order_id": f"SIM-{uuid.uuid4().hex[:8]}",
        "message": f"시뮬 매도 체결 @ {sell_price:,.0f}",
    }
    _log_trade(
        session, symbol, "SELL", step, actual_qty, sell_price, result,
        reason=reason, snapshot=snapshot, position=pos,
        conditions=conditions,
        sizing={
            "close_price": close_price,
            "sell_price_effective": round(sell_price, 2),
            "sell_commission": cost.sell_commission,
            "slippage_pct": cost.slippage_pct,
            "proceeds": round(proceeds, 2),
            "cash_after": round(session._cash, 2),
            "partial": partial,
        },
        candle_dt=candle_dt,
    )
    if partial:
        _reduce_entries(pos, actual_qty)
    else:
        session.positions.pop(symbol, None)


# ── real 모드: KIS API ──────────────────────────────────────


async def _real_loop_tick(
    session: LiveSession,
    strategy: dict,
    cost: CostConfig,
    ctx: TradingContext,
    stop_loss_pct: float | None,
    trailing_stop_pct: float | None,
    *,
    atr_stop_mult: float | None = None,
    scaling: dict | None = None,
    scaling_enabled: bool = False,
    ps_cfg: dict | None = None,
) -> None:
    """Real 모드 1회 루프: KIS API 실주문."""
    from .kis_client import get_kis_client
    from .kis_order import KISOrderExecutor

    client = get_kis_client(is_mock=False)
    executor = KISOrderExecutor(client)

    # H1: 미체결 주문 체결 확인 (OrderManager)
    from .order_manager import get_order_manager
    om = get_order_manager(executor)
    check_result = await om.check_orders()
    for filled in check_result.newly_filled:
        logger.info(
            "체결 확인: %s %s %d주 @ %.0f",
            filled.side, filled.symbol, filled.filled_qty, filled.filled_avg_price,
        )
        # entries 반영 (분할매매 추적)
        fsym = filled.symbol
        if filled.side == "BUY" and fsym in session.positions:
            pos = session.positions[fsym]
            step = getattr(filled, "meta", {}).get("step", "B1") if hasattr(filled, "meta") else "B1"
            pos.entries.append(LiveScaleEntry(
                date=datetime.now().isoformat(),
                price=filled.filled_avg_price,
                qty=filled.filled_qty,
                step=step,
            ))
            if step == "B2":
                pos.scale_in_count += 1
        elif filled.side == "BUY" and fsym not in session.positions:
            # 신규 B1 포지션 생성
            meta = getattr(filled, "meta", {}) if hasattr(filled, "meta") else {}
            session.positions[fsym] = LivePosition(
                symbol=fsym,
                entries=[LiveScaleEntry(
                    date=datetime.now().isoformat(),
                    price=filled.filled_avg_price,
                    qty=filled.filled_qty,
                    step=meta.get("step", "B1"),
                )],
                highest_price=filled.filled_avg_price,
                conviction=meta.get("conviction", 1.0),
                target_qty=meta.get("target_qty", filled.filled_qty),
                entry_candle_dt=datetime.now().isoformat(),
            )
        elif filled.side == "SELL" and fsym in session.positions:
            pos = session.positions[fsym]
            _reduce_entries(pos, filled.filled_qty)
            if pos.qty <= 0:
                session.positions.pop(fsym, None)

    for expired in check_result.expired:
        logger.info(
            "TTL 만료 취소: %s %s 잔량 %d주",
            expired.side, expired.symbol, expired.remaining_qty,
        )

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
            # 전량 청산 (OrderManager 경유)
            from .order_manager import get_order_manager
            om = get_order_manager(executor)
            # 미체결 전량 취소
            await om.cancel_all(reason="circuit_breaker")
            for sym, kp in kis_positions.items():
                qty = kp.get("qty", 0)
                if qty > 0:
                    managed = await om.submit_sell(
                        sym, qty, reason="circuit_breaker", urgent=True,
                    )
                    result = {"success": managed is not None, "order_id": managed.order_id if managed else ""}
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
        kis_qty = kp["qty"]
        kis_avg = kp["avg_price"]
        if sym in session.positions:
            pos = session.positions[sym]
            local_qty = pos.qty
            if kis_qty < local_qty:
                # 외부 매도 발생 — entries FIFO 차감
                _reduce_entries(pos, local_qty - kis_qty)
            elif kis_qty > local_qty:
                # 외부 매수 발생 — synthetic entry 추가
                pos.entries.append(LiveScaleEntry(
                    date=datetime.now().isoformat(),
                    price=kis_avg, qty=kis_qty - local_qty, step="B-EXT",
                ))
        else:
            session.positions[sym] = LivePosition(
                symbol=sym,
                entries=[LiveScaleEntry(
                    date=datetime.now().isoformat(),
                    price=kis_avg, qty=kis_qty, step="B-EXT",
                )],
                highest_price=kp.get("current_price", kis_avg),
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
                managed = await om.submit_sell(
                    sym, pos.qty, reason="stop_loss", urgent=True,
                )
                result = {"success": managed is not None, "order_id": managed.order_id if managed else ""}
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
                managed = await om.submit_sell(
                    sym, pos.qty, reason="trailing", urgent=True,
                )
                result = {"success": managed is not None, "order_id": managed.order_id if managed else ""}
                _log_trade(
                    session, sym, "SELL", "S-TRAIL", pos.qty, current_price, result,
                    reason=f"트레일링: 고점({round(pos.highest_price):,}) 대비 -{round(drop_pct, 2):.2f}%",
                    snapshot={"close": current_price, "highest_price": round(pos.highest_price)},
                    position=pos,
                )
                continue

        # ATR 동적 손절 (현재가 기반 — 캔들 ATR은 시그널 체크 시 확보)
        # Note: real 모드에서는 KIS current_price 기반이므로, ATR 값은 세션에 캐시 필요
        # → _real_check_signals에서 ATR 값을 세션 속성에 저장, 여기서 참조
        if atr_stop_mult is not None and pos.avg_price > 0:
            atr_val = getattr(session, "_atr_cache", {}).get(sym, 0)
            if atr_val > 0:
                stop_line = pos.avg_price - atr_val * atr_stop_mult
                if current_price <= stop_line:
                    managed = await om.submit_sell(
                        sym, pos.qty, reason="atr_stop", urgent=True,
                    )
                    result = {"success": managed is not None, "order_id": managed.order_id if managed else ""}
                    _log_trade(
                        session, sym, "SELL", "S-STOP", pos.qty, current_price, result,
                        reason=f"ATR 스탑: {current_price:,} <= {round(stop_line):,}",
                        snapshot={"close": current_price, "atr_14": round(atr_val, 2),
                                  "stop_line": round(stop_line)},
                        position=pos,
                    )
                    continue

    held_symbols = set(session.positions.keys())
    universe = ctx.symbols or [sym for sym in kis_positions]
    scan_limit = max(ctx.max_positions * 5, 50)
    scan_symbols = list(held_symbols | set(universe[:scan_limit]))

    if scan_symbols:
        await _real_check_signals(
            session, executor, strategy, scan_symbols, cash, cost, ctx,
            scaling=scaling, scaling_enabled=scaling_enabled, ps_cfg=ps_cfg,
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
    *,
    scaling: dict | None = None,
    scaling_enabled: bool = False,
    ps_cfg: dict | None = None,
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

        # ATR 캐시 (real 모드 리스크 체크에서 사용)
        atr_val = last_row.get("atr_14", 0) or 0
        if atr_val > 0:
            if not hasattr(session, "_atr_cache"):
                session._atr_cache = {}
            session._atr_cache[sym] = atr_val

        # H1: 미체결 주문 있는 종목 스킵 (OrderManager)
        from .order_manager import get_order_manager
        om = get_order_manager()
        if om.has_pending(sym):
            _log_decision(session, sym, "SKIP_PENDING",
                          "미체결 주문 존재 — 중복 주문 방지",
                          signal=signal, conditions=conditions, snapshot=snapshot)
            continue

        # S-HALF 부분 익절 (real 모드)
        if scaling_enabled and sym in session.positions:
            _scaling = scaling or {}
            pos = session.positions[sym]
            if not pos.has_partial_exited and pos.qty > 1 and pos.avg_price > 0:
                close_p = int(last_row.get("close", 0))
                if close_p > 0:
                    gain_pct = (close_p - pos.avg_price) / pos.avg_price * 100
                    partial_gain = _scaling.get("partial_exit_gain_pct", 5.0)
                    if gain_pct >= partial_gain:
                        partial_ratio = _scaling.get("partial_exit_pct", 0.5)
                        sell_qty = max(1, int(pos.qty * partial_ratio))
                        if sell_qty >= pos.qty:
                            sell_qty = pos.qty - 1
                        if sell_qty > 0:
                            managed = await om.submit_sell(
                                sym, sell_qty, close_p, reason="partial_exit", urgent=False,
                            )
                            if managed:
                                managed.meta = {"step": "S-HALF"}
                            result = {"success": managed is not None,
                                      "order_id": managed.order_id if managed else ""}
                            pos.has_partial_exited = True
                            _log_trade(session, sym, "SELL", "S-HALF", sell_qty, close_p, result,
                                       reason=f"부분 익절: +{gain_pct:.2f}%",
                                       snapshot=snapshot, position=pos, conditions=conditions)
                            continue

        # B2 분할매수 (real 모드)
        if scaling_enabled and sym in session.positions:
            _scaling = scaling or {}
            pos = session.positions[sym]
            max_scale = _scaling.get("max_scale_in", 1)
            if pos.scale_in_count < max_scale and pos.qty > 0 and pos.avg_price > 0:
                close_p = int(last_row.get("close", 0))
                if close_p > 0:
                    drop_pct = (pos.avg_price - close_p) / pos.avg_price * 100
                    drop_trigger = _scaling.get("scale_in_drop_pct", 3.0)
                    if drop_pct >= drop_trigger:
                        remaining_qty = pos.target_qty - pos.qty
                        if remaining_qty > 0:
                            if remaining_qty * close_p > cash * 0.95:
                                remaining_qty = int(cash * 0.95 / close_p)
                            if remaining_qty > 0:
                                managed = await om.submit_buy(sym, remaining_qty, close_p)
                                if managed:
                                    managed.meta = {"step": "B2"}
                                result = {"success": managed is not None,
                                          "order_id": managed.order_id if managed else ""}
                                _log_trade(session, sym, "BUY", "B2", remaining_qty, close_p, result,
                                           reason=f"분할매수: 평단 대비 -{drop_pct:.2f}%",
                                           snapshot=snapshot, conditions=conditions)
                                continue

        # 매도 시그널
        if signal == -1 and sym in session.positions:
            pos = session.positions[sym]
            if pos.qty <= 0:
                continue
            current_price = int(last_row.get("close", 0))
            managed = await om.submit_sell(
                sym, pos.qty, current_price, reason="signal", urgent=False,
            )
            result = {"success": managed is not None, "order_id": managed.order_id if managed else ""}
            _log_trade(session, sym, "SELL", "", pos.qty, current_price, result,
                       reason="매도 시그널", snapshot=snapshot, position=pos, conditions=conditions)
            continue

        # 매수 시그널 (B1 신규 진입)
        if signal == 1 and sym not in session.positions:
            if len(session.positions) >= ctx.max_positions:
                continue

            conviction = _calc_conviction_live(strategy, ps_cfg or {}, last_row)
            alloc = ctx.initial_capital * ctx.position_size_pct * conviction
            if scaling_enabled:
                initial_pct = (scaling or {}).get("initial_pct", 0.5)
            else:
                initial_pct = 1.0
            alloc *= initial_pct
            alloc = min(alloc, cash * 0.95)
            price = int(last_row.get("close", 0))
            if price <= 0:
                continue
            qty = int(alloc / price)
            if qty <= 0:
                continue

            # target_qty 계산
            if scaling_enabled and initial_pct > 0 and initial_pct < 1.0:
                target_qty = int(qty / initial_pct)
            else:
                target_qty = qty

            managed = await om.submit_buy(sym, qty, price)
            if managed:
                managed.meta = {"step": "B1", "conviction": conviction, "target_qty": target_qty}
            result = {"success": managed is not None, "order_id": managed.order_id if managed else ""}
            _log_trade(session, sym, "BUY", "B1", qty, price, result,
                       reason="매수 시그널", snapshot=snapshot, conditions=conditions,
                       sizing={"conviction": round(conviction, 4), "initial_pct": initial_pct,
                               "target_qty": target_qty, "scaling_enabled": scaling_enabled})


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
            "entries_count": len(position.entries),
            "conviction": position.conviction,
            "scale_in_count": position.scale_in_count,
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

    # 텔레그램 알림 (fire-and-forget — 매매 성능에 영향 없음)
    if result.get("success"):
        asyncio.create_task(_send_trade_telegram(entry))

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


async def _send_trade_telegram(entry: dict) -> None:
    """매매 체결 텔레그램 알림 (fire-and-forget)."""
    try:
        from app.telegram.bot import send_message

        symbol = entry.get("symbol", "")
        name = entry.get("name", symbol)
        side = entry.get("side", "")
        step = entry.get("step", "")
        qty = entry.get("qty", 0)
        price = entry.get("price", 0)
        total = round(price * qty, 0)
        ts = entry.get("timestamp", "")
        reason = entry.get("reason", "")

        if side == "BUY":
            sizing = entry.get("sizing") or {}
            cash_before = sizing.get("cash_before", 0)
            cash_after = cash_before - total if cash_before else 0
            alloc_pct = sizing.get("position_size_pct", 0) * 100

            msg = (
                f"\U0001f4c8 <b>매수</b> | {name} ({symbol})\n"
                f"  수량: {qty:,}주 @ {price:,.0f}원\n"
                f"  총액: {total:,.0f}원\n"
            )
            if alloc_pct:
                msg += f"  배분: 자본의 {alloc_pct:.0f}%"
                if cash_before:
                    msg += f" (현금 {cash_before:,.0f} \u2192 {cash_after:,.0f})"
                msg += "\n"
            if reason:
                msg += f"  사유: {reason[:80]}\n"
            msg += f"  시각: {ts[:19]}"

        elif side == "SELL":
            pnl_pct = entry.get("pnl_pct")
            pnl_amount = entry.get("pnl_amount")
            holding = entry.get("holding_minutes")
            pos_ctx = entry.get("position_context") or {}
            avg_price = pos_ctx.get("avg_price", 0)

            step_label = ""
            if step == "S-STOP":
                step_label = " [손절]"
            elif step == "S-TRAIL":
                step_label = " [트레일링]"
            elif step == "S-HALF":
                step_label = " [부분익절]"

            pnl_emoji = "\U0001f534" if (pnl_pct or 0) < 0 else "\U0001f7e2"

            msg = (
                f"\U0001f4c9 <b>매도</b> | {name} ({symbol}){step_label}\n"
                f"  수량: {qty:,}주 @ {price:,.0f}원\n"
                f"  총액: {total:,.0f}원\n"
            )
            if pnl_pct is not None:
                msg += f"  {pnl_emoji} PnL: {pnl_pct:+.2f}%"
                if pnl_amount is not None:
                    msg += f" ({pnl_amount:+,.0f}원)"
                msg += "\n"
            if avg_price:
                msg += f"  평단: {avg_price:,.0f}원 \u2192 매도 {price:,.0f}원\n"
            if holding is not None:
                if holding >= 60:
                    h, m = divmod(int(holding), 60)
                    msg += f"  보유: {h}시간 {m}분\n"
                else:
                    msg += f"  보유: {int(holding)}분\n"
            if reason:
                msg += f"  사유: {reason[:80]}\n"
            msg += f"  시각: {ts[:19]}"
        else:
            return

        await send_message(msg, category="trade", caller="live_runner")
    except Exception:
        pass  # 발송 실패가 매매에 영향 주면 안 됨


async def _save_session_state(session: LiveSession) -> None:
    """세션 상태를 trading_contexts.session_state에 저장."""
    state = {
        "status": session.status,
        "positions": {
            sym: {
                "entries": [
                    {"date": e.date, "price": e.price, "qty": e.qty, "step": e.step}
                    for e in p.entries
                ],
                "highest_price": p.highest_price,
                "conviction": p.conviction,
                "target_qty": p.target_qty,
                "scale_in_count": p.scale_in_count,
                "has_partial_exited": p.has_partial_exited,
                "entry_candle_dt": p.entry_candle_dt,
                # 하위 호환: 기존 소비자용 flat 필드
                "qty": p.qty,
                "avg_price": p.avg_price,
                "entry_date": p.entry_date,
            }
            for sym, p in session.positions.items()
        },
        "cash": session._cash,
        "last_processed_dt": getattr(session, "_last_processed_dt", None),
        "started_at": session.started_at,
        "trade_count": len(session.trade_log),
    }
    # OrderManager 미체결 주문 저장 (서버 재시작 시 복구용)
    try:
        from .order_manager import get_order_manager
        om = get_order_manager()
        state["pending_orders"] = om.to_state_dict()
    except Exception:
        pass
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

        # 포지션 복원 (entries 형식 + 레거시 flat 형식 호환)
        for sym, pdata in ctx.session_state.get("positions", {}).items():
            entries_data = pdata.get("entries")
            if entries_data:
                entries = [
                    LiveScaleEntry(
                        date=e.get("date", ""), price=e.get("price", 0),
                        qty=e.get("qty", 0), step=e.get("step", "B1"),
                    )
                    for e in entries_data
                ]
            else:
                # 레거시: flat qty/avg_price → single B1 entry
                entries = [LiveScaleEntry(
                    date=pdata.get("entry_date", ""),
                    price=pdata.get("avg_price", 0),
                    qty=pdata.get("qty", 0),
                    step="B1",
                )]
            session.positions[sym] = LivePosition(
                symbol=sym,
                entries=entries,
                highest_price=pdata.get("highest_price", 0),
                conviction=pdata.get("conviction", 0.0),
                target_qty=pdata.get("target_qty", 0),
                scale_in_count=pdata.get("scale_in_count", 0),
                has_partial_exited=pdata.get("has_partial_exited", False),
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
