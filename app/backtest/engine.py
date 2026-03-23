"""벡터화 백테스트 엔진 — 시그널 생성 + 단일 포트폴리오 시뮬레이션.

확신도 기반 비중 조절 및 분할매매(Scale-in/out) 지원.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Awaitable, Callable

import polars as pl

from .cost_model import CostConfig, effective_buy_price, effective_sell_price
from .data_loader import available_minute_symbols, load_candles
from .indicators import add_atr, ensure_indicators
from .metrics import Trade, compute_metrics

from app.core.stock_master import get_stock_name

# 뉴스 감성 통합 (지연 임포트로 순환 방지)
_sentiment_df_cache: pl.DataFrame | None = None

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], Awaitable[None]] | None


def _dt_to_key(dt_val) -> str:
    """dt 값 → 시뮬레이션 키 문자열. date='2024-01-15', datetime='2024-01-15 09:05:00'."""
    if isinstance(dt_val, datetime):
        return dt_val.strftime("%Y-%m-%d %H:%M:%S")
    return str(dt_val)


# ── 시그널 생성 ──────────────────────────────────────────


def _resolve_column(indicator: str, params: dict) -> str:
    """조건의 indicator 이름 → 실제 DataFrame 컬럼명."""
    mapping = {
        "rsi": "rsi",
        "sma": f"sma_{params.get('period', 20)}",
        "ema": f"ema_{params.get('period', 20)}",
        "macd_hist": "macd_hist",
        "macd_line": "macd_line",
        "macd_signal": "macd_signal",
        "bb_upper": "bb_upper",
        "bb_lower": "bb_lower",
        "bb_middle": "bb_middle",
        "volume_ratio": "volume_ratio",
        "price_change_pct": "price_change_pct",
        "golden_cross": "golden_cross",
        "dead_cross": "dead_cross",
        "atr": f"atr_{params.get('period', 14)}",
        "consec_decline": f"consec_decline_{params.get('days', 3)}",
        "open_gap_pct": "open_gap_pct",
        # 뉴스 감성 지표
        "sentiment_score": "sentiment_score",
        "article_count": "article_count",
        "event_score": "event_score",
    }
    # macd_cross: MACD가 시그널을 상향 돌파 (golden_cross 처럼 bool)
    if indicator == "macd_cross":
        return "macd_cross"
    return mapping.get(indicator, indicator)


def _build_condition_expr(cond: dict) -> pl.Expr:
    """단일 조건 dict → Polars boolean Expr."""
    ind = cond["indicator"]
    params = cond.get("params", {})
    op = cond.get("op", ">=")
    value = cond.get("value", 0)

    # alpha_* 접두사 지표: 동적으로 등록된 알파 팩터
    if ind.startswith("alpha_"):
        col_name = ind
    else:
        col_name = _resolve_column(ind, params)

    # bool 지표 (golden_cross, dead_cross, macd_cross)
    if ind in ("golden_cross", "dead_cross"):
        return pl.col(col_name) == True  # noqa: E712

    if ind == "macd_cross":
        # MACD 상향 돌파: 전봉 macd_hist<=0, 현봉 macd_hist>0
        return (pl.col("macd_hist") > 0) & (pl.col("macd_hist").shift(1) <= 0)

    ops = {
        ">": pl.col(col_name) > value,
        ">=": pl.col(col_name) >= value,
        "<": pl.col(col_name) < value,
        "<=": pl.col(col_name) <= value,
        "==": pl.col(col_name) == value,
        "!=": pl.col(col_name) != value,
    }
    return ops.get(op, pl.col(col_name) >= value)


def _combine_conditions(conditions: list[dict], logic: str = "AND") -> pl.Expr:
    """조건 리스트를 AND/OR 로 결합."""
    if not conditions:
        return pl.lit(False)

    exprs = [_build_condition_expr(c) for c in conditions]
    result = exprs[0]
    for e in exprs[1:]:
        if logic.upper() == "AND":
            result = result & e
        else:
            result = result | e
    return result


def _format_condition(cond: dict) -> str:
    """조건 dict → 사람이 읽기 쉬운 문자열."""
    ind = cond["indicator"]
    params = cond.get("params", {})
    op = cond.get("op", ">=")
    value = cond.get("value", 0)

    if ind in ("golden_cross", "dead_cross", "macd_cross"):
        return ind.replace("_", " ").title()

    if "period" in params:
        name = f"{ind.upper()}({params['period']})"
    elif "fast_period" in params:
        name = f"{ind}({params['fast_period']}/{params.get('slow_period', 20)})"
    elif "days" in params:
        name = f"{ind}({params['days']})"
    else:
        name = ind

    return f"{name} {op} {value}"


def generate_signals(df: pl.DataFrame, strategy: dict) -> pl.DataFrame:
    """전략 JSON → signal 컬럼 (1=매수, -1=매도, 0=대기) 추가.

    Look-ahead bias 방지: 시그널은 현재 봉 기준, 체결은 다음 봉 시가.
    개별 조건 결과를 _buy_cond_N / _sell_cond_N bool 컬럼으로 보존.
    """
    buy_conds = strategy.get("buy_conditions", [])
    sell_conds = strategy.get("sell_conditions", [])
    buy_logic = strategy.get("buy_logic", "AND")
    sell_logic = strategy.get("sell_logic", "OR")

    # 필요한 지표 추가
    all_conds = buy_conds + sell_conds
    df = ensure_indicators(df, all_conds)

    # 개별 조건 결과 bool 컬럼
    for i, cond in enumerate(buy_conds):
        df = df.with_columns(_build_condition_expr(cond).alias(f"_buy_cond_{i}"))
    for i, cond in enumerate(sell_conds):
        df = df.with_columns(_build_condition_expr(cond).alias(f"_sell_cond_{i}"))

    buy_expr = _combine_conditions(buy_conds, buy_logic)
    sell_expr = _combine_conditions(sell_conds, sell_logic)

    df = df.with_columns(
        pl.when(buy_expr).then(pl.lit(1))
        .when(sell_expr).then(pl.lit(-1))
        .otherwise(pl.lit(0))
        .alias("signal")
    )
    return df


# ── 포트폴리오 시뮬레이션 ────────────────────────────────


@dataclass
class _ScaleEntry:
    """개별 매수 건."""
    date: str
    price: float
    qty: int
    step: str  # "B1", "B2"


@dataclass
class _Position:
    symbol: str
    entries: list[_ScaleEntry] = field(default_factory=list)
    partial_exits: list[dict] = field(default_factory=list)
    highest_price: float = 0.0
    conviction: float = 0.0
    target_qty: int = 0
    scale_in_count: int = 0
    has_partial_exited: bool = False
    # 매수 이유 보존 (매도 시 Trade에 전파)
    entry_reason: list[dict] | None = None
    entry_snapshot: dict | None = None

    @property
    def total_qty(self) -> int:
        return sum(e.qty for e in self.entries)

    @property
    def avg_price(self) -> float:
        total_cost = sum(e.price * e.qty for e in self.entries)
        total_q = self.total_qty
        return total_cost / total_q if total_q > 0 else 0

    @property
    def entry_date(self) -> str:
        return self.entries[0].date if self.entries else ""


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


# ── 확신도 & 포지션 사이징 ────────────────────────────────


def _calc_conviction(
    strategy: dict,
    mode: str,
    weights: dict | None,
    row: dict | None = None,
) -> float:
    """확신도 0.0~1.0 계산."""
    if mode == "fixed":
        return 1.0

    if mode == "conviction":
        buy_conds = strategy.get("buy_conditions", [])
        if not buy_conds:
            return 1.0
        total_weight = sum((weights or {}).values()) or 1.0
        met_weight = 0.0
        for cond in buy_conds:
            ind = cond["indicator"]
            w = (weights or {}).get(ind, 1.0 / len(buy_conds))
            met_weight += w
        return min(met_weight / total_weight, 1.0)

    if mode == "atr_target" and row is not None:
        atr = row.get("atr_14", 0) or 0
        price = row.get("close", 1) or 1
        atr_pct = atr / price if price > 0 else 0.1
        return min(1.0, 0.02 / max(atr_pct, 0.001))

    return 1.0


def _calc_alloc(
    conviction: float,
    initial_capital: float,
    position_size_pct: float,
    cash: float,
    scaling: dict | None,
) -> float:
    """확신도 기반 배분 금액."""
    base_alloc = initial_capital * position_size_pct
    alloc = base_alloc * conviction

    if scaling and scaling.get("enabled"):
        alloc *= scaling.get("initial_pct", 0.5)

    alloc = min(alloc, cash * 0.95)
    return alloc


def _holding_days(entry_dt: str, exit_dt: str) -> int:
    try:
        d1 = datetime.strptime(entry_dt[:10], "%Y-%m-%d")
        d2 = datetime.strptime(exit_dt[:10], "%Y-%m-%d")
        return (d2 - d1).days
    except Exception:
        return 0


def _make_sell_trade(
    pos: _Position,
    sell_price: float,
    exit_date: str,
    qty: int,
    scale_step: str,
    exit_reason: str = "",
    exit_reason_detail: list[dict] | None = None,
    exit_snapshot: dict | None = None,
) -> Trade:
    """매도 Trade 레코드 생성."""
    avg_p = pos.avg_price
    pnl = (sell_price - avg_p) * qty
    pnl_pct = (sell_price - avg_p) / avg_p * 100 if avg_p > 0 else 0
    return Trade(
        symbol=pos.symbol,
        entry_date=pos.entry_date,
        entry_price=round(avg_p),
        exit_date=exit_date,
        exit_price=round(sell_price),
        qty=qty,
        pnl=round(pnl),
        pnl_pct=round(pnl_pct, 2),
        holding_days=_holding_days(pos.entry_date, exit_date),
        scale_step=scale_step,
        conviction=pos.conviction,
        name=get_stock_name(pos.symbol),
        entry_reason=pos.entry_reason,
        entry_snapshot=pos.entry_snapshot,
        exit_reason=exit_reason,
        exit_reason_detail=exit_reason_detail,
        exit_snapshot=exit_snapshot,
    )


LogCallback = Callable[[str], Awaitable[None]] | None


async def run_backtest(
    strategy: dict,
    symbols: list[str] | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    initial_capital: float = 100_000_000,
    max_positions: int = 10,
    position_size_pct: float = 0.1,
    cost_config: CostConfig | None = None,
    progress_cb: ProgressCallback = None,
    log_cb: LogCallback = None,
) -> BacktestResult:
    """전 종목 백테스트 실행.

    1. 데이터 로딩
    2. 종목별 시그널 생성
    3. 날짜순 포트폴리오 시뮬레이션
    4. 성과 지표 산출
    """
    cfg = cost_config or CostConfig()
    interval = strategy.get("timeframe", "1d")

    # 고급 설정 추출
    ps_cfg = strategy.get("position_sizing") or {}
    ps_mode = ps_cfg.get("mode", "fixed") if ps_cfg else "fixed"
    ps_weights = ps_cfg.get("weights") if ps_cfg else None

    scaling = strategy.get("scaling") or {}
    scaling_enabled = scaling.get("enabled", False) if scaling else False

    risk = strategy.get("risk_management") or {}
    stop_loss_pct = risk.get("stop_loss_pct") if risk else None
    trailing_stop_pct = risk.get("trailing_stop_pct") if risk else None
    atr_stop_mult = risk.get("atr_stop_multiplier") if risk else None

    needs_atr = (
        ps_mode == "atr_target"
        or atr_stop_mult is not None
    )

    # ── 0.5 분봉 시 종목 자동 제한 ──
    is_intraday = interval != "1d"
    if is_intraday and symbols is None:
        symbols = await available_minute_symbols()
        if not symbols:
            return BacktestResult(metrics={"error": "1분봉 데이터가 있는 종목이 없습니다."})
        logger.info("분봉 백테스트: %d개 종목 자동 선택", len(symbols))

    # ── 1. 데이터 로딩 (워밍업 버퍼 포함) ──
    # 일봉: ~200거래일 워밍업, 분봉: 30일 (봉 수가 이미 충분)
    warmup_days = 30 if is_intraday else 300
    warmup_start = (start_date - timedelta(days=warmup_days)) if start_date else None

    import time as _time
    _t0 = _time.monotonic()

    _period_str = f"{start_date or '?'}~{end_date or '?'}"
    if progress_cb:
        await progress_cb(0, 100, f"데이터 로딩 중... ({interval}, {_period_str})")
    if log_cb:
        await log_cb(f"데이터 로딩 시작 (interval={interval}, period={_period_str})")

    df = await load_candles(symbols, warmup_start, end_date, interval)
    if df.is_empty():
        return BacktestResult(metrics={"error": "데이터가 없습니다."})

    symbol_list = df["symbol"].unique().sort().to_list()
    total = len(symbol_list)
    _elapsed = _time.monotonic() - _t0

    if progress_cb:
        await progress_cb(5, 100, f"{total}개 종목, {df.height:,}행 로딩 완료 ({_elapsed:.1f}초)")
    if log_cb:
        await log_cb(f"{total}개 종목 × {df.height:,}행 로딩 ({_elapsed:.1f}초)")

    # ── 1.5. 뉴스 감성 데이터 로딩 (필요 시) ──
    sentiment_df: pl.DataFrame | None = None
    from app.news.backtest_integration import has_sentiment_conditions
    if has_sentiment_conditions(strategy):
        try:
            from app.news.backtest_integration import load_sentiment_data
            sentiment_df = await load_sentiment_data(symbols, warmup_start, end_date)
            if sentiment_df.is_empty():
                sentiment_df = None
            else:
                logger.info("감성 데이터 로딩: %d행", sentiment_df.height)
                if log_cb:
                    await log_cb(f"뉴스 감성 데이터 로딩: {sentiment_df.height:,}행")
        except Exception as e:
            logger.warning("감성 데이터 로딩 실패: %s", e)
            sentiment_df = None

    # ── 2. 종목별 시그널 생성 ──
    signal_dfs: dict[str, pl.DataFrame] = {}
    for i, sym in enumerate(symbol_list):
        sym_df = df.filter(pl.col("symbol") == sym).sort("dt")
        if sym_df.height < 30:
            continue
        try:
            # 감성 데이터 join (T+1 shift 적용됨)
            if sentiment_df is not None:
                sym_sentiment = sentiment_df.filter(pl.col("symbol") == sym).drop("symbol")
                if sym_sentiment.height > 0:
                    # dt 기준 left join
                    sym_df = sym_df.join(sym_sentiment, on="dt", how="left")
                    # null 값은 0으로 채움
                    for col in ["sentiment_score", "article_count", "event_score"]:
                        if col in sym_df.columns:
                            sym_df = sym_df.with_columns(pl.col(col).fill_null(0).alias(col))
                else:
                    # 감성 데이터 없는 종목: 0으로 컬럼 추가
                    for col in ["sentiment_score", "article_count", "event_score"]:
                        sym_df = sym_df.with_columns(pl.lit(0.0).alias(col))
            elif has_sentiment_conditions(strategy):
                # 감성 조건이 있지만 데이터가 없으면 0으로 채움
                for col in ["sentiment_score", "article_count", "event_score"]:
                    sym_df = sym_df.with_columns(pl.lit(0.0).alias(col))

            sym_df = generate_signals(sym_df, strategy)
            # ATR이 필요하면 추가
            if needs_atr and f"atr_{ps_cfg.get('atr_period', 14)}" not in sym_df.columns:
                atr_period = ps_cfg.get("atr_period", 14) if ps_cfg else 14
                sym_df = add_atr(sym_df, period=atr_period)
            signal_dfs[sym] = sym_df
        except Exception as e:
            logger.warning("Signal generation failed for %s: %s", sym, e)

        if progress_cb and ((i + 1) % 20 == 0 or i + 1 == total):
            pct = 5 + int(55 * (i + 1) / total)
            await progress_cb(pct, 100, f"시그널 생성 {i + 1}/{total}")
        if log_cb and (i == 0 or (i + 1) % 20 == 0 or i + 1 == total):
            await log_cb(f"시그널 생성 {i + 1}/{total} ({sym})")

    _signal_count = len(signal_dfs)
    if progress_cb:
        await progress_cb(60, 100, f"포트폴리오 시뮬레이션 시작 ({_signal_count}종목)")
    if log_cb:
        await log_cb(f"시그널 생성 완료 — {_signal_count}/{total}종목 유효")

    # ── 3. 날짜별 시그널 통합 + 포트폴리오 시뮬레이션 ──
    buy_conds = strategy.get("buy_conditions", [])
    sell_conds = strategy.get("sell_conditions", [])

    # 스냅샷에 포함할 지표 컬럼 사전 계산
    _snapshot_cols: set[str] = set()
    for cond in buy_conds + sell_conds:
        _snapshot_cols.add(_resolve_column(cond["indicator"], cond.get("params", {})))
    for col in ("rsi", "macd_hist", "macd_line", "volume_ratio",
                "sentiment_score", "article_count", "event_score"):
        _snapshot_cols.add(col)

    all_signals: list[dict] = []
    for sym, sdf in signal_dfs.items():
        sdf = sdf.with_columns(
            pl.col("open").shift(-1).alias("next_open"),
            pl.col("dt").shift(-1).alias("next_date"),
        )
        signals_only = sdf.filter(pl.col("signal") != 0)

        # 이 DataFrame에 실제 존재하는 스냅샷 컬럼만 필터
        available_snap_cols = [c for c in _snapshot_cols if c in sdf.columns]

        for row in signals_only.iter_rows(named=True):
            if row["next_open"] is not None:
                # 지표 스냅샷
                snapshot: dict = {"close": row["close"]}
                for col in available_snap_cols:
                    val = row.get(col)
                    if val is not None:
                        snapshot[col] = round(val, 4) if isinstance(val, float) else val

                # 매수 조건 평가 결과
                buy_cond_results = []
                for i, cond in enumerate(buy_conds):
                    col_name = _resolve_column(cond["indicator"], cond.get("params", {}))
                    buy_cond_results.append({
                        "condition": _format_condition(cond),
                        "column": col_name,
                        "actual": round(row.get(col_name, 0), 4) if isinstance(row.get(col_name), (int, float)) else row.get(col_name),
                        "met": bool(row.get(f"_buy_cond_{i}", False)),
                    })

                # 매도 조건 평가 결과
                sell_cond_results = []
                for i, cond in enumerate(sell_conds):
                    col_name = _resolve_column(cond["indicator"], cond.get("params", {}))
                    sell_cond_results.append({
                        "condition": _format_condition(cond),
                        "column": col_name,
                        "actual": round(row.get(col_name, 0), 4) if isinstance(row.get(col_name), (int, float)) else row.get(col_name),
                        "met": bool(row.get(f"_sell_cond_{i}", False)),
                    })

                all_signals.append({
                    "dt": row["dt"],
                    "symbol": row["symbol"],
                    "signal": row["signal"],
                    "next_open": row["next_open"],
                    "next_date": row.get("next_date"),
                    "close": row["close"],
                    "_snapshot": snapshot,
                    "_buy_cond_results": buy_cond_results,
                    "_sell_cond_results": sell_cond_results,
                    **{k: row.get(k) for k in row if k.startswith("atr_")},
                })

    all_signals.sort(key=lambda x: (x["dt"], -x["signal"]))

    # 시뮬레이션
    cash = initial_capital
    positions: dict[str, _Position] = {}
    trades: list[Trade] = []
    equity_by_date: dict[str, float] = {}

    # 날짜별 종가/ATR 캐시
    close_by_date_sym: dict[str, dict[str, float]] = defaultdict(dict)
    atr_by_date_sym: dict[str, dict[str, float]] = defaultdict(dict)
    for sym, sdf in signal_dfs.items():
        for row in sdf.iter_rows(named=True):
            dt_str = _dt_to_key(row["dt"])
            close_by_date_sym[dt_str][sym] = row["close"]
            # ATR 캐시
            for k in row:
                if isinstance(k, str) and k.startswith("atr_") and row[k] is not None:
                    atr_by_date_sym[dt_str][sym] = row[k]

    all_dates = sorted(close_by_date_sym.keys())

    # 워밍업 기간 제외: start_date 이후만 시뮬레이션
    if start_date:
        sim_start = start_date.isoformat()
        all_dates = [d for d in all_dates if d >= sim_start]

    signals_by_date: dict[str, list[dict]] = defaultdict(list)
    for sig in all_signals:
        dt_str = _dt_to_key(sig["dt"])
        signals_by_date[dt_str].append(sig)

    for date_idx, date_str in enumerate(all_dates):
        today_closes = close_by_date_sym.get(date_str, {})
        today_atrs = atr_by_date_sym.get(date_str, {})
        day_signals = signals_by_date.get(date_str, [])

        # ── (1) 고점 갱신 ──
        for sym, pos in positions.items():
            price = today_closes.get(sym, 0)
            if price > pos.highest_price:
                pos.highest_price = price

        # ── (2) 리스크 관리: 손절/트레일링 ──
        syms_to_remove: list[str] = []
        for sym, pos in list(positions.items()):
            if pos.total_qty <= 0:
                continue
            price = today_closes.get(sym, 0)
            if price <= 0:
                continue
            avg_p = pos.avg_price

            # 고정 손절
            if stop_loss_pct is not None:
                loss_pct = (price - avg_p) / avg_p * 100
                if loss_pct <= -stop_loss_pct:
                    sell_p = effective_sell_price(price, cfg)
                    qty = pos.total_qty
                    trades.append(_make_sell_trade(
                        pos, sell_p, date_str, qty, "S-STOP",
                        exit_reason=f"손절 (고정): 평단 대비 {round(loss_pct, 2):+.2f}%",
                        exit_snapshot={"close": price, "avg_price": round(avg_p), "loss_pct": round(loss_pct, 2)},
                    ))
                    cash += sell_p * qty
                    syms_to_remove.append(sym)
                    continue

            # 트레일링 스탑
            if trailing_stop_pct is not None and pos.highest_price > 0:
                drop_from_high = (pos.highest_price - price) / pos.highest_price * 100
                if drop_from_high >= trailing_stop_pct:
                    sell_p = effective_sell_price(price, cfg)
                    qty = pos.total_qty
                    trades.append(_make_sell_trade(
                        pos, sell_p, date_str, qty, "S-TRAIL",
                        exit_reason=f"트레일링 스탑: 고점({round(pos.highest_price):,}) 대비 -{round(drop_from_high, 2):.2f}%",
                        exit_snapshot={"close": price, "highest_price": round(pos.highest_price), "drop_pct": round(drop_from_high, 2)},
                    ))
                    cash += sell_p * qty
                    syms_to_remove.append(sym)
                    continue

            # ATR 동적 손절
            if atr_stop_mult is not None:
                atr_val = today_atrs.get(sym, 0)
                if atr_val > 0:
                    stop_line = avg_p - atr_val * atr_stop_mult
                    if price <= stop_line:
                        sell_p = effective_sell_price(price, cfg)
                        qty = pos.total_qty
                        trades.append(_make_sell_trade(
                            pos, sell_p, date_str, qty, "S-STOP",
                            exit_reason=f"ATR 스탑: 가격 {round(price):,} <= 스탑라인 {round(stop_line):,}",
                            exit_snapshot={"close": price, "atr": round(atr_val), "stop_line": round(stop_line), "avg_price": round(avg_p)},
                        ))
                        cash += sell_p * qty
                        syms_to_remove.append(sym)
                        continue

        for sym in syms_to_remove:
            positions.pop(sym, None)

        # ── (3) 분할매도: 부분 익절 ──
        if scaling_enabled:
            partial_gain = scaling.get("partial_exit_gain_pct", 5.0)
            partial_pct = scaling.get("partial_exit_pct", 0.5)

            for sym, pos in list(positions.items()):
                if pos.has_partial_exited or pos.total_qty <= 0:
                    continue
                price = today_closes.get(sym, 0)
                if price <= 0:
                    continue
                gain_pct = (price - pos.avg_price) / pos.avg_price * 100
                if gain_pct >= partial_gain:
                    sell_qty = max(1, int(pos.total_qty * partial_pct))
                    if sell_qty >= pos.total_qty:
                        sell_qty = pos.total_qty - 1
                    if sell_qty <= 0:
                        continue
                    sell_p = effective_sell_price(price, cfg)
                    trades.append(_make_sell_trade(
                        pos, sell_p, date_str, sell_qty, "S-HALF",
                        exit_reason=f"부분 익절: 평단 대비 +{round(gain_pct, 2):.2f}%",
                        exit_snapshot={"close": price, "avg_price": round(pos.avg_price), "gain_pct": round(gain_pct, 2)},
                    ))
                    cash += sell_p * sell_qty
                    pos.has_partial_exited = True
                    # entries에서 수량 차감
                    _reduce_entries(pos, sell_qty)

        # ── (4) 매도 시그널 처리 ──
        for sig in day_signals:
            if sig["signal"] == -1 and sig["symbol"] in positions:
                pos = positions[sig["symbol"]]
                if pos.total_qty <= 0:
                    continue
                if sig["next_open"] <= 0:
                    continue
                sell_p = effective_sell_price(sig["next_open"], cfg)
                qty = pos.total_qty
                # 체결일 = next_open이 속한 다음 거래일 (시그널일이 아닌 실제 체결일)
                exec_date = str(sig["next_date"]) if sig.get("next_date") else date_str
                trades.append(_make_sell_trade(
                    pos, sell_p, exec_date, qty, "",
                    exit_reason="SELL 시그널",
                    exit_reason_detail=sig.get("_sell_cond_results"),
                    exit_snapshot=sig.get("_snapshot"),
                ))
                cash += sell_p * qty
                positions.pop(sig["symbol"])

        # ── (5) 분할매수: 기존 포지션 추가 매수 ──
        if scaling_enabled:
            max_scale = scaling.get("max_scale_in", 1)
            drop_trigger = scaling.get("scale_in_drop_pct", 3.0)

            for sym, pos in list(positions.items()):
                if pos.scale_in_count >= max_scale:
                    continue
                if pos.total_qty <= 0:
                    continue
                price = today_closes.get(sym, 0)
                if price <= 0:
                    continue

                avg_p = pos.avg_price
                drop_pct = (avg_p - price) / avg_p * 100
                if drop_pct >= drop_trigger:
                    # 추가 매수: target_qty - current_qty
                    remaining_qty = pos.target_qty - pos.total_qty
                    if remaining_qty <= 0:
                        continue
                    buy_p = effective_buy_price(price, cfg)
                    cost = buy_p * remaining_qty
                    if cost > cash:
                        remaining_qty = int(cash / buy_p)
                    if remaining_qty <= 0:
                        continue
                    cost = buy_p * remaining_qty
                    if cost > cash:
                        continue

                    cash -= cost
                    pos.entries.append(_ScaleEntry(
                        date=date_str,
                        price=buy_p,
                        qty=remaining_qty,
                        step="B2",
                    ))
                    pos.scale_in_count += 1

                    trades.append(Trade(
                        symbol=sym,
                        entry_date=date_str,
                        entry_price=round(buy_p),
                        qty=remaining_qty,
                        scale_step="B2",
                        conviction=pos.conviction,
                        name=get_stock_name(sym),
                        entry_reason=[{"condition": f"추가매수: 평단 대비 -{round(drop_pct, 2):.2f}% 하락", "column": "price", "actual": round(price), "met": True}],
                        entry_snapshot={"close": price, "avg_price": round(avg_p), "drop_pct": round(drop_pct, 2)},
                    ))

        # ── (6) 신규 매수 시그널 ──
        for sig in day_signals:
            if sig["signal"] == 1 and sig["symbol"] not in positions:
                if len(positions) >= max_positions:
                    break
                if sig["next_open"] <= 0:
                    continue

                # 확신도 계산
                conviction = _calc_conviction(strategy, ps_mode, ps_weights, sig)

                # 배분 금액
                alloc = _calc_alloc(
                    conviction, initial_capital, position_size_pct, cash, scaling if scaling_enabled else None,
                )
                buy_p = effective_buy_price(sig["next_open"], cfg)
                if alloc < buy_p:
                    continue
                qty = int(alloc / buy_p)
                if qty <= 0:
                    continue
                cost = buy_p * qty
                if cost > cash:
                    continue

                # target_qty: 분할매수 활성 시 전체 목표 수량
                if scaling_enabled:
                    initial_pct = scaling.get("initial_pct", 0.5)
                    target_qty = int(qty / initial_pct) if initial_pct > 0 else qty
                else:
                    target_qty = qty

                cash -= cost
                # 체결일 = next_open이 속한 다음 거래일
                exec_date = str(sig["next_date"]) if sig.get("next_date") else date_str
                pos = _Position(
                    symbol=sig["symbol"],
                    entries=[_ScaleEntry(
                        date=exec_date,
                        price=buy_p,
                        qty=qty,
                        step="B1",
                    )],
                    highest_price=sig.get("close", buy_p),
                    conviction=conviction,
                    target_qty=target_qty,
                    entry_reason=sig.get("_buy_cond_results"),
                    entry_snapshot=sig.get("_snapshot"),
                )
                positions[sig["symbol"]] = pos

                trades.append(Trade(
                    symbol=sig["symbol"],
                    entry_date=exec_date,
                    entry_price=round(buy_p),
                    qty=qty,
                    scale_step="B1" if scaling_enabled else "",
                    conviction=conviction,
                    name=get_stock_name(sig["symbol"]),
                    entry_reason=sig.get("_buy_cond_results"),
                    entry_snapshot=sig.get("_snapshot"),
                ))

        # ── (7) equity 계산 ──
        pos_value = 0.0
        for sym, pos in positions.items():
            price = today_closes.get(sym, pos.avg_price)
            pos_value += price * pos.total_qty

        equity_by_date[date_str] = cash + pos_value

        if progress_cb and (date_idx + 1) % 20 == 0:
            pct = 60 + int(35 * (date_idx + 1) / len(all_dates))
            await progress_cb(min(pct, 95), 100, f"시뮬레이션 {date_idx + 1}/{len(all_dates)}일")

    # 미청산 포지션 정리
    last_date = all_dates[-1] if all_dates else ""
    last_closes = close_by_date_sym.get(last_date, {})
    for sym, pos in list(positions.items()):
        if pos.total_qty <= 0:
            continue
        price = last_closes.get(sym, pos.avg_price)
        if price <= 0 or pos.avg_price <= 0:
            continue
        sell_p = effective_sell_price(price, cfg)
        trades.append(_make_sell_trade(
            pos, sell_p, last_date, pos.total_qty, "",
            exit_reason="기간만료 청산",
            exit_snapshot={"close": price},
        ))

    # equity curve 정리
    equity_curve = [{"date": d, "equity": round(eq)} for d, eq in sorted(equity_by_date.items())]

    # ── 4. 성과 지표 ──
    if log_cb:
        await log_cb(f"성과 지표 계산 중... ({len(trades)}건 매매)")
    metrics = compute_metrics(trades, equity_curve, initial_capital)

    _total_elapsed = _time.monotonic() - _t0
    if progress_cb:
        await progress_cb(100, 100, "완료")
    if log_cb:
        await log_cb(f"백테스트 완료 — {len(trades)}건 매매, 총 {_total_elapsed:.1f}초 소요")

    return BacktestResult(
        trades=trades,
        equity_curve=equity_curve,
        metrics=metrics,
    )


def _reduce_entries(pos: _Position, sell_qty: int) -> None:
    """분할매도 시 entries에서 수량 차감 (FIFO)."""
    remaining = sell_qty
    new_entries = []
    for entry in pos.entries:
        if remaining <= 0:
            new_entries.append(entry)
            continue
        if entry.qty <= remaining:
            remaining -= entry.qty
            # 이 entry는 완전 소진
        else:
            entry.qty -= remaining
            remaining = 0
            new_entries.append(entry)
    pos.entries = new_entries
