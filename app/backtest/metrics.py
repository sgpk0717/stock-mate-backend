"""백테스트 성과 지표 계산."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Trade:
    symbol: str
    entry_date: str
    entry_price: float
    exit_date: str | None = None
    exit_price: float | None = None
    qty: int = 0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    holding_days: int = 0
    scale_step: str = ""
    conviction: float = 0.0
    # ── 검증/추적 필드 ──
    name: str = ""
    entry_reason: list[dict] | None = None
    exit_reason: str = ""
    exit_reason_detail: list[dict] | None = None
    entry_snapshot: dict | None = None
    exit_snapshot: dict | None = None


def compute_metrics(
    trades: list[Trade],
    equity_curve: list[dict],
    initial_capital: float,
    annualize: float = 252.0,
) -> dict:
    """trades 와 equity_curve 로부터 성과 지표를 산출한다."""

    if not equity_curve:
        return _empty_metrics()

    final_equity = equity_curve[-1]["equity"]
    total_return = (final_equity - initial_capital) / initial_capital * 100
    total_return_amount = final_equity - initial_capital

    # 연환산 수익률
    days = max(len(equity_curve), 1)
    years = days / annualize
    annualized = ((final_equity / initial_capital) ** (1 / max(years, 0.01)) - 1) * 100 if years > 0 else 0

    # MDD
    mdd, mdd_amount = _calc_mdd(equity_curve)

    # 승률
    closed = [t for t in trades if t.exit_date is not None]
    wins = [t for t in closed if t.pnl > 0]
    losses = [t for t in closed if t.pnl <= 0]
    total_trades = len(closed)
    win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0

    # 평균 수익/손실
    avg_win = sum(t.pnl_pct for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t.pnl_pct for t in losses) / len(losses) if losses else 0

    # Profit Factor
    gross_profit = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # 샤프 비율 (봉간 수익률 기반)
    sharpe = _calc_sharpe(equity_curve, annualize=annualize)

    # 평균 보유일
    avg_holding = sum(t.holding_days for t in closed) / total_trades if total_trades > 0 else 0

    # 최대 연승/연패
    max_wins, max_losses = _calc_streaks(closed)

    # 분할매매 통계
    avg_conviction = (
        round(sum(t.conviction for t in closed) / total_trades, 3)
        if total_trades > 0 else 0
    )
    scale_in_count = sum(1 for t in closed if t.scale_step.startswith("B2"))
    partial_exit_count = sum(1 for t in closed if t.scale_step == "S-HALF")
    stop_loss_count = sum(1 for t in closed if t.scale_step == "S-STOP")

    return {
        "total_return": round(total_return, 2),
        "total_return_amount": round(total_return_amount),
        "annualized_return": round(annualized, 2),
        "mdd": round(mdd, 2),
        "mdd_amount": round(mdd_amount),
        "win_rate": round(win_rate, 2),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else 999.99,
        "sharpe_ratio": round(sharpe, 2),
        "total_trades": total_trades,
        "avg_holding_days": round(avg_holding, 1),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "max_consecutive_wins": max_wins,
        "max_consecutive_losses": max_losses,
        "avg_conviction": avg_conviction,
        "scale_in_count": scale_in_count,
        "partial_exit_count": partial_exit_count,
        "stop_loss_count": stop_loss_count,
    }


def _empty_metrics() -> dict:
    return {
        "total_return": 0, "total_return_amount": 0, "annualized_return": 0,
        "mdd": 0, "mdd_amount": 0, "win_rate": 0, "profit_factor": 0,
        "sharpe_ratio": 0, "total_trades": 0, "avg_holding_days": 0,
        "avg_win": 0, "avg_loss": 0,
        "max_consecutive_wins": 0, "max_consecutive_losses": 0,
        "avg_conviction": 0, "scale_in_count": 0,
        "partial_exit_count": 0, "stop_loss_count": 0,
    }


def _calc_mdd(equity_curve: list[dict]) -> tuple[float, float]:
    peak = equity_curve[0]["equity"]
    max_dd_pct = 0.0
    max_dd_amt = 0.0
    for pt in equity_curve:
        eq = pt["equity"]
        if eq > peak:
            peak = eq
        dd_amt = peak - eq
        dd_pct = dd_amt / peak * 100 if peak > 0 else 0
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct
            max_dd_amt = dd_amt
    return max_dd_pct, max_dd_amt


def _calc_sharpe(
    equity_curve: list[dict],
    risk_free_rate: float = 0.03,
    annualize: float = 252.0,
) -> float:
    if len(equity_curve) < 2:
        return 0.0
    returns = []
    for i in range(1, len(equity_curve)):
        prev = equity_curve[i - 1]["equity"]
        cur = equity_curve[i]["equity"]
        returns.append((cur - prev) / prev if prev > 0 else 0)
    if not returns:
        return 0.0
    mean_r = sum(returns) / len(returns)
    std_r = math.sqrt(sum((r - mean_r) ** 2 for r in returns) / len(returns))
    if std_r == 0:
        return 0.0
    bar_rf = risk_free_rate / annualize
    return (mean_r - bar_rf) / std_r * math.sqrt(annualize)


def _calc_streaks(closed: list[Trade]) -> tuple[int, int]:
    max_w = max_l = cur_w = cur_l = 0
    for t in closed:
        if t.pnl > 0:
            cur_w += 1
            cur_l = 0
        else:
            cur_l += 1
            cur_w = 0
        max_w = max(max_w, cur_w)
        max_l = max(max_l, cur_l)
    return max_w, max_l
