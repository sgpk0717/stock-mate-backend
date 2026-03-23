"""Numba JIT 가속 팩터 백테스트 엔진.

기존 run_factor_backtest()와 동일한 입출력 인터페이스를 제공하되,
시뮬레이션 루프를 Numba @njit으로 컴파일하여 10-100배 속도 향상.

사용: run_factor_backtest(..., engine="vectorbt") → 이 모듈 호출.
"""

from __future__ import annotations

import logging
import math
from datetime import date, datetime, timedelta

import numpy as np
import numba as nb

from app.backtest.cost_model import CostConfig, effective_buy_price, effective_sell_price
from app.backtest.metrics import Trade
from app.backtest.engine import BacktestResult

logger = logging.getLogger(__name__)


@nb.njit(cache=True)
def _simulate_loop(
    close_matrix: np.ndarray,       # (n_bars, n_symbols) — close prices
    open_matrix: np.ndarray,        # (n_bars, n_symbols) — open prices
    volume_matrix: np.ndarray,      # (n_bars, n_symbols) — volumes
    rank_matrix: np.ndarray,        # (n_bars, n_symbols) — factor_rank (0-1)
    is_rebalance: np.ndarray,       # (n_bars,) — bool
    is_eod: np.ndarray,             # (n_bars,) — bool
    day_ids: np.ndarray,            # (n_bars,) — int, calendar day index
    initial_capital: float,
    top_pct: float,
    max_positions: int,
    stop_loss_pct: float,
    max_drawdown_pct: float,
    band_threshold: float,
    buy_cost_pct: float,            # buy commission + slippage
    sell_cost_pct: float,           # sell commission + slippage
    eod_liquidation: bool,
    vol_participation_limit: float,  # max fraction of bar volume
) -> tuple:
    """Numba JIT 시뮬레이션 코어.

    Returns
    -------
    (equity_curve, trade_log, counters)
    - equity_curve: (n_bars,) float64
    - trade_log: (max_trades, 9) float64
      columns: [sym_idx, entry_bar, entry_price, exit_bar, exit_price, qty, pnl, pnl_pct, exit_type]
      exit_type: 0=rebal, 1=stop_loss, 2=eod, 3=circuit_breaker, 4=final
    - counters: (5,) int64 — [total_buys, total_sells, rebal_count, stop_loss_count, eod_count]
    """
    n_bars, n_symbols = close_matrix.shape
    max_trades = n_bars * max_positions  # upper bound

    # Pre-allocate
    equity_curve = np.empty(n_bars, dtype=np.float64)
    trade_log = np.full((max_trades, 9), np.nan, dtype=np.float64)
    trade_idx = 0

    # Portfolio state
    holdings_qty = np.zeros(n_symbols, dtype=np.int64)
    holdings_avg_price = np.zeros(n_symbols, dtype=np.float64)
    holdings_entry_bar = np.full(n_symbols, -1, dtype=np.int64)
    holdings_high = np.zeros(n_symbols, dtype=np.float64)

    cash = initial_capital
    peak_equity = initial_capital
    circuit_breaker = False

    # Counters
    total_buys = 0
    total_sells = 0
    rebal_count = 0
    stop_loss_count = 0
    eod_count = 0
    band_saved = 0

    # Previous day rank for T-1 signal (accumulated per-symbol latest)
    prev_day_rank = np.full(n_symbols, np.nan, dtype=np.float64)
    accumulating_rank = np.full(n_symbols, np.nan, dtype=np.float64)
    prev_day_valid = False
    last_day_id = -1

    # Band exit threshold
    exit_rank_threshold = (1.0 - top_pct) * (1.0 - band_threshold) if band_threshold > 0 else 0.0

    for bar in range(n_bars):
        cur_day = day_ids[bar]

        # Day change → prev_day_rank = accumulated ranks from yesterday
        if cur_day != last_day_id and last_day_id >= 0:
            for s in range(n_symbols):
                if not np.isnan(accumulating_rank[s]):
                    prev_day_rank[s] = accumulating_rank[s]
                # else: keep previous prev_day_rank[s]
            prev_day_valid = True
            # Reset accumulator for new day
            for s in range(n_symbols):
                accumulating_rank[s] = np.nan
        last_day_id = cur_day

        # Accumulate today's ranks (overwrite per symbol → last bar wins)
        for s in range(n_symbols):
            r = rank_matrix[bar, s]
            if not np.isnan(r):
                accumulating_rank[s] = r

        if circuit_breaker:
            # Just track equity
            port_val = cash
            for s in range(n_symbols):
                if holdings_qty[s] > 0:
                    port_val += close_matrix[bar, s] * holdings_qty[s]
            equity_curve[bar] = port_val
            continue

        # ── Stop loss check ──
        if stop_loss_pct > 0:
            for s in range(n_symbols):
                if holdings_qty[s] <= 0:
                    continue
                price = close_matrix[bar, s]
                if np.isnan(price) or price <= 0:
                    continue
                if price > holdings_high[s]:
                    holdings_high[s] = price
                dd = (price - holdings_avg_price[s]) / holdings_avg_price[s]
                if dd <= -stop_loss_pct:
                    sell_price = price * (1.0 - sell_cost_pct)
                    pnl = (sell_price - holdings_avg_price[s]) * holdings_qty[s]
                    pnl_pct = (sell_price / holdings_avg_price[s] - 1.0) * 100.0
                    cash += sell_price * holdings_qty[s]
                    if trade_idx < max_trades:
                        trade_log[trade_idx, 0] = s
                        trade_log[trade_idx, 1] = holdings_entry_bar[s]
                        trade_log[trade_idx, 2] = holdings_avg_price[s]
                        trade_log[trade_idx, 3] = bar
                        trade_log[trade_idx, 4] = sell_price
                        trade_log[trade_idx, 5] = holdings_qty[s]
                        trade_log[trade_idx, 6] = pnl
                        trade_log[trade_idx, 7] = pnl_pct
                        trade_log[trade_idx, 8] = 1  # stop_loss
                        trade_idx += 1
                    holdings_qty[s] = 0
                    total_sells += 1
                    stop_loss_count += 1

        # ── Circuit breaker ──
        if max_drawdown_pct > 0:
            port_val = cash
            for s in range(n_symbols):
                if holdings_qty[s] > 0:
                    p = close_matrix[bar, s]
                    if np.isnan(p) or p <= 0:
                        p = holdings_avg_price[s]  # NaN → 매수가로 대체
                    port_val += p * holdings_qty[s]
            if port_val > peak_equity:
                peak_equity = port_val
            if port_val < peak_equity * (1.0 - max_drawdown_pct):
                circuit_breaker = True
                for s in range(n_symbols):
                    if holdings_qty[s] <= 0:
                        continue
                    p = close_matrix[bar, s]
                    if np.isnan(p) or p <= 0:
                        p = holdings_avg_price[s]
                    sell_price = p * (1.0 - sell_cost_pct)
                    pnl = (sell_price - holdings_avg_price[s]) * holdings_qty[s]
                    pnl_pct = (sell_price / holdings_avg_price[s] - 1.0) * 100.0
                    cash += sell_price * holdings_qty[s]
                    if trade_idx < max_trades:
                        trade_log[trade_idx, 0] = s
                        trade_log[trade_idx, 1] = holdings_entry_bar[s]
                        trade_log[trade_idx, 2] = holdings_avg_price[s]
                        trade_log[trade_idx, 3] = bar
                        trade_log[trade_idx, 4] = sell_price
                        trade_log[trade_idx, 5] = holdings_qty[s]
                        trade_log[trade_idx, 6] = pnl
                        trade_log[trade_idx, 7] = pnl_pct
                        trade_log[trade_idx, 8] = 3  # circuit_breaker
                        trade_idx += 1
                    holdings_qty[s] = 0
                    total_sells += 1
                equity_curve[bar] = cash
                continue

        # ── Rebalancing ──
        if is_rebalance[bar] and prev_day_valid and not circuit_breaker:
            # Determine target symbols from prev_day_rank
            # Sort by rank descending
            sorted_indices = np.argsort(-prev_day_rank)  # descending
            n_valid = 0
            for s in sorted_indices:
                if not np.isnan(prev_day_rank[s]):
                    n_valid += 1
            n_top = max(1, int(n_valid * top_pct))
            n_top = min(n_top, max_positions)
            _dummy = 0  # placeholder for Numba compatibility

            target_set = np.zeros(n_symbols, dtype=np.bool_)
            for i in range(n_top):
                s = sorted_indices[i]
                if not np.isnan(prev_day_rank[s]):
                    target_set[s] = True

            # Sell out-of-target holdings
            sell_list_count = 0
            for s in range(n_symbols):
                if holdings_qty[s] <= 0:
                    continue
                if target_set[s]:
                    continue
                # Band check
                if band_threshold > 0 and prev_day_rank[s] >= exit_rank_threshold:
                    band_saved += 1
                    continue
                # Sell at open
                op = open_matrix[bar, s]
                if np.isnan(op) or op <= 0:
                    continue
                sell_price = op * (1.0 - sell_cost_pct)
                pnl = (sell_price - holdings_avg_price[s]) * holdings_qty[s]
                pnl_pct = (sell_price / holdings_avg_price[s] - 1.0) * 100.0
                cash += sell_price * holdings_qty[s]
                if trade_idx < max_trades:
                    trade_log[trade_idx, 0] = s
                    trade_log[trade_idx, 1] = holdings_entry_bar[s]
                    trade_log[trade_idx, 2] = holdings_avg_price[s]
                    trade_log[trade_idx, 3] = bar
                    trade_log[trade_idx, 4] = sell_price
                    trade_log[trade_idx, 5] = holdings_qty[s]
                    trade_log[trade_idx, 6] = pnl
                    trade_log[trade_idx, 7] = pnl_pct
                    trade_log[trade_idx, 8] = 0  # rebal
                    trade_idx += 1
                holdings_qty[s] = 0
                total_sells += 1
                sell_list_count += 1

            # Count current holdings
            n_held = 0
            for s in range(n_symbols):
                if holdings_qty[s] > 0:
                    n_held += 1

            # Buy new targets
            buy_budget_count = max(0, max_positions - n_held)
            bought = 0
            for i in range(n_top):
                if bought >= buy_budget_count:
                    break
                s = sorted_indices[i]
                if holdings_qty[s] > 0:
                    continue
                op = open_matrix[bar, s]
                if np.isnan(op) or op <= 0:
                    continue
                buy_price = op * (1.0 + buy_cost_pct)
                if buy_price <= 0:
                    continue
                per_stock = cash / max(buy_budget_count - bought, 1)
                qty = int(per_stock / buy_price)
                # Volume participation limit
                bv = volume_matrix[bar, s]
                if not np.isnan(bv) and bv > 0:
                    max_qty = int(bv * vol_participation_limit)
                    if qty > max_qty:
                        qty = max_qty
                if qty <= 0:
                    continue
                cost = buy_price * qty
                if cost > cash:
                    qty = int(cash / buy_price)
                    cost = buy_price * qty
                if qty <= 0:
                    continue
                cash -= cost
                holdings_qty[s] = qty
                holdings_avg_price[s] = buy_price
                holdings_entry_bar[s] = bar
                holdings_high[s] = buy_price
                total_buys += 1
                bought += 1

            if sell_list_count > 0 or bought > 0:
                rebal_count += 1

        # ── EOD forced liquidation ──
        if eod_liquidation and is_eod[bar] and not circuit_breaker:
            for s in range(n_symbols):
                if holdings_qty[s] <= 0:
                    continue
                p = close_matrix[bar, s]
                if np.isnan(p) or p <= 0:
                    p = holdings_avg_price[s]
                sell_price = p * (1.0 - sell_cost_pct)
                pnl = (sell_price - holdings_avg_price[s]) * holdings_qty[s]
                pnl_pct = (sell_price / holdings_avg_price[s] - 1.0) * 100.0
                cash += sell_price * holdings_qty[s]
                if trade_idx < max_trades:
                    trade_log[trade_idx, 0] = s
                    trade_log[trade_idx, 1] = holdings_entry_bar[s]
                    trade_log[trade_idx, 2] = holdings_avg_price[s]
                    trade_log[trade_idx, 3] = bar
                    trade_log[trade_idx, 4] = sell_price
                    trade_log[trade_idx, 5] = holdings_qty[s]
                    trade_log[trade_idx, 6] = pnl
                    trade_log[trade_idx, 7] = pnl_pct
                    trade_log[trade_idx, 8] = 2  # eod
                    trade_idx += 1
                holdings_qty[s] = 0
                total_sells += 1
                eod_count += 1

        # ── Equity ──
        port_val = cash
        for s in range(n_symbols):
            if holdings_qty[s] > 0:
                p = close_matrix[bar, s]
                if np.isnan(p) or p <= 0:
                    p = holdings_avg_price[s]  # NaN → 매수가로 대체
                port_val += p * holdings_qty[s]
                if p > holdings_high[s]:
                    holdings_high[s] = p
        equity_curve[bar] = port_val
        if port_val > peak_equity:
            peak_equity = port_val

    # Final liquidation
    for s in range(n_symbols):
        if holdings_qty[s] <= 0:
            continue
        p = close_matrix[n_bars - 1, s]
        if np.isnan(p) or p <= 0:
            p = holdings_avg_price[s]
        sell_price = p * (1.0 - sell_cost_pct)
        pnl = (sell_price - holdings_avg_price[s]) * holdings_qty[s]
        pnl_pct = (sell_price / holdings_avg_price[s] - 1.0) * 100.0
        cash += sell_price * holdings_qty[s]
        if trade_idx < max_trades:
            trade_log[trade_idx, 0] = s
            trade_log[trade_idx, 1] = holdings_entry_bar[s]
            trade_log[trade_idx, 2] = holdings_avg_price[s]
            trade_log[trade_idx, 3] = n_bars - 1
            trade_log[trade_idx, 4] = sell_price
            trade_log[trade_idx, 5] = holdings_qty[s]
            trade_log[trade_idx, 6] = pnl
            trade_log[trade_idx, 7] = pnl_pct
            trade_log[trade_idx, 8] = 4  # final
            trade_idx += 1
        holdings_qty[s] = 0
        total_sells += 1

    counters = np.array([total_buys, total_sells, rebal_count, stop_loss_count, eod_count, band_saved], dtype=np.int64)
    return equity_curve, trade_log[:trade_idx], counters


# Exit type mapping
_EXIT_TYPE_MAP = {
    0: ("REBAL-SELL", "리밸런싱 매도"),
    1: ("STOP-LOSS", "손절"),
    2: ("S-EOD", "장 종료 강제 청산"),
    3: ("CIRCUIT-BREAKER", "서킷 브레이커"),
    4: ("FINAL", "백테스트 종료 청산"),
}


def run_factor_backtest_vbt(
    *,
    all_dates: list,
    date_data: dict,
    symbols: list[str],
    initial_capital: float,
    top_pct: float,
    max_positions: int,
    rebalance_dates_set: set,
    eod_bar_set: set,
    cost_config: CostConfig,
    stop_loss_pct: float,
    max_drawdown_pct: float,
    band_threshold: float,
    eod_liquidation: bool,
    intraday: bool,
    get_stock_name,
    interval: str = "5m",
) -> BacktestResult:
    """Numba JIT 가속 팩터 백테스트 실행.

    factor_backtest.py의 데이터 로딩/팩터 계산 결과를 받아
    시뮬레이션 루프만 Numba로 대체한다.
    """
    from app.backtest.metrics import compute_metrics

    n_bars = len(all_dates)
    if n_bars < 2:
        return BacktestResult(metrics={"error": "시뮬레이션 가능한 거래일이 부족합니다."})

    # Build symbol list from date_data
    sym_set: set[str] = set()
    for dt_data in date_data.values():
        sym_set.update(dt_data.keys())
    sym_list = sorted(sym_set)
    sym_to_idx = {s: i for i, s in enumerate(sym_list)}
    n_symbols = len(sym_list)

    if n_symbols == 0:
        return BacktestResult(metrics={"error": "종목 데이터가 없습니다."})

    # Build matrices
    close_matrix = np.full((n_bars, n_symbols), np.nan, dtype=np.float64)
    open_matrix = np.full((n_bars, n_symbols), np.nan, dtype=np.float64)
    volume_matrix = np.zeros((n_bars, n_symbols), dtype=np.float64)
    rank_matrix = np.full((n_bars, n_symbols), np.nan, dtype=np.float64)

    is_rebalance = np.zeros(n_bars, dtype=np.bool_)
    is_eod = np.zeros(n_bars, dtype=np.bool_)
    day_ids = np.zeros(n_bars, dtype=np.int64)

    day_map: dict = {}
    day_counter = 0

    for bar_idx, dt in enumerate(all_dates):
        today = date_data.get(dt, {})
        for sym, data in today.items():
            si = sym_to_idx.get(sym)
            if si is None:
                continue
            close_matrix[bar_idx, si] = data.get("close", np.nan)
            open_matrix[bar_idx, si] = data.get("open", np.nan)
            volume_matrix[bar_idx, si] = data.get("volume", 0)
            rank_matrix[bar_idx, si] = data.get("factor_rank", np.nan)

        if dt in rebalance_dates_set:
            is_rebalance[bar_idx] = True
        if dt in eod_bar_set:
            is_eod[bar_idx] = True

        d = dt.date() if isinstance(dt, datetime) else dt
        if d not in day_map:
            day_map[d] = day_counter
            day_counter += 1
        day_ids[bar_idx] = day_map[d]

    # Cost rates
    buy_cost = cost_config.slippage_pct + cost_config.buy_commission
    sell_cost = cost_config.slippage_pct + cost_config.sell_commission
    vol_limit = 0.10  # 10% participation

    logger.info(
        "VBT engine: %d bars x %d symbols, rebal_bars=%d, eod_bars=%d, days=%d",
        n_bars, n_symbols, int(np.sum(is_rebalance)), int(np.sum(is_eod)), len(day_map),
    )

    # Run JIT simulation
    equity_arr, trade_arr, counters = _simulate_loop(
        close_matrix, open_matrix, volume_matrix, rank_matrix,
        is_rebalance, is_eod, day_ids,
        initial_capital, top_pct, max_positions,
        stop_loss_pct, max_drawdown_pct, band_threshold,
        buy_cost, sell_cost, eod_liquidation, vol_limit,
    )

    logger.info("VBT engine completed: %d trades", len(trade_arr))

    # Convert to Trade objects
    def _dt_to_str(dt_val) -> str:
        return dt_val.isoformat() if isinstance(dt_val, (date, datetime)) else str(dt_val)

    def _calc_holding(entry_bar: int, exit_bar: int) -> float:
        if intraday:
            e = all_dates[int(entry_bar)]
            x = all_dates[int(exit_bar)]
            if isinstance(e, datetime) and isinstance(x, datetime):
                return max(0.01, round((x - e).total_seconds() / 86400, 2))
        return max(1, int(exit_bar - entry_bar))

    trades: list[Trade] = []
    for row in trade_arr:
        sym_idx = int(row[0])
        entry_bar = int(row[1])
        exit_bar = int(row[3])
        exit_type = int(row[8])

        sym = sym_list[sym_idx] if 0 <= sym_idx < n_symbols else "?"
        step, reason = _EXIT_TYPE_MAP.get(exit_type, ("?", "?"))

        trades.append(Trade(
            symbol=sym,
            name=get_stock_name(sym),
            entry_date=_dt_to_str(all_dates[entry_bar]),
            entry_price=row[2],
            exit_date=_dt_to_str(all_dates[exit_bar]),
            exit_price=row[4],
            qty=int(row[5]),
            pnl=row[6],
            pnl_pct=row[7],
            holding_days=_calc_holding(entry_bar, exit_bar),
            scale_step=step,
            exit_reason=reason,
        ))

    # Equity curve
    equity_curve = []
    for i, dt in enumerate(all_dates):
        equity_curve.append({
            "date": _dt_to_str(dt),
            "equity": float(equity_arr[i]),
        })

    # Metrics
    from app.alpha.interval import bars_per_year
    metrics = compute_metrics(
        trades, equity_curve, initial_capital,
        annualize=bars_per_year(interval),
        intraday=intraday,
    )

    metrics["total_buys"] = int(counters[0])
    metrics["total_sells"] = int(counters[1])
    metrics["rebalance_count"] = int(counters[2])
    metrics["stop_loss_count"] = int(counters[3])
    metrics["eod_close_count"] = int(counters[4])
    metrics["band_trades_saved"] = int(counters[5])
    metrics["eod_liquidation"] = eod_liquidation and intraday
    metrics["engine"] = "vectorbt"

    return BacktestResult(
        trades=trades,
        equity_curve=equity_curve,
        metrics=metrics,
    )
