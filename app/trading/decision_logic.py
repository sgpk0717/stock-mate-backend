"""매매 판단 로직 공통 모듈.

live_runner, sim_engine, backtest 모두 이 함수를 사용하여
동일한 매수/매도/리스크 판단을 수행한다.

핵심 원칙: 순수 함수 (pure function). 상태 변경 없음.
입력: 현재 상태 (가격, 포지션, 현금 등)
출력: 판단 결과 (BuyDecision / SellDecision)
실행(주문, 포지션 변경)은 호출자가 수행.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BuyDecision:
    """매수 판단 결과."""

    action: str  # "BUY", "SKIP_BUY", "NONE"
    qty: int = 0
    price: float = 0.0
    reason: str = ""
    conviction: float = 0.0
    alloc: float = 0.0
    target_qty: int = 0
    sizing: dict = field(default_factory=dict)


@dataclass
class RiskDecision:
    """리스크 판단 결과."""

    action: str  # "RISK_STOP", "RISK_TRAIL", "RISK_ATR_STOP", "PARTIAL_EXIT", None
    qty: int = 0
    reason: str = ""
    risk: dict = field(default_factory=dict)


def calc_conviction(
    strategy: dict,
    ps_cfg: dict,
    row: dict,
) -> float:
    """확신도 계산.

    Parameters
    ----------
    strategy : 전략 dict (buy_conditions 포함)
    ps_cfg : position_sizing config
    row : 현재 봉의 지표 dict

    Returns
    -------
    float : 0.0 ~ 1.0
    """
    mode = ps_cfg.get("mode", "fixed")
    if mode == "fixed":
        return ps_cfg.get("conviction", 1.0)

    if mode == "conviction":
        weights = ps_cfg.get("weights", {})
        if not weights:
            return 1.0
        total = 0.0
        for indicator, weight in weights.items():
            val = row.get(indicator)
            if val is not None:
                total += weight
        return min(max(total, 0.0), 1.0)

    if mode == "atr_target":
        atr_pct = row.get("atr_14", 0)
        close = row.get("close", 0)
        if close and close > 0 and atr_pct:
            atr_ratio = atr_pct / close
            return min(1.0, 0.02 / max(atr_ratio, 0.001))
        return 1.0

    return ps_cfg.get("conviction", 1.0)


def calc_alloc(
    initial_capital: float,
    position_size_pct: float,
    conviction: float,
    cash: float,
    scaling: dict | None = None,
) -> float:
    """배분금 계산.

    Returns
    -------
    float : 이 종목에 배분할 금액
    """
    alloc = initial_capital * position_size_pct * conviction
    if scaling and scaling.get("enabled"):
        initial_pct = scaling.get("initial_pct", 0.5)
        alloc *= initial_pct
    alloc = min(alloc, cash * 0.95)
    return alloc


def evaluate_buy(
    signal: int,
    symbol: str,
    has_position: bool,
    current_positions: int,
    max_positions: int,
    cash: float,
    initial_capital: float,
    position_size_pct: float,
    close_price: float,
    buy_price: float,
    row: dict,
    strategy: dict,
    ps_cfg: dict,
    scaling: dict | None = None,
) -> BuyDecision:
    """매수 판단 — 순수 함수.

    Parameters
    ----------
    signal : 시그널 값 (1=매수)
    buy_price : 수수료/슬리피지 적용 후 실효 매수가
    """
    if signal != 1 or has_position:
        return BuyDecision(action="NONE")

    if current_positions >= max_positions:
        return BuyDecision(
            action="SKIP_BUY",
            reason=f"최대 포지션: {current_positions}/{max_positions}",
        )

    conviction = calc_conviction(strategy, ps_cfg, row)
    alloc = calc_alloc(initial_capital, position_size_pct, conviction, cash, scaling)
    qty = int(alloc / buy_price) if buy_price > 0 else 0

    scaling_enabled = bool(scaling and scaling.get("enabled"))
    initial_pct = scaling.get("initial_pct", 0.5) if scaling_enabled else 1.0
    target_qty = int(qty / initial_pct) if scaling_enabled and initial_pct < 1.0 and initial_pct > 0 else qty

    sizing = {
        "initial_capital": initial_capital,
        "position_size_pct": position_size_pct,
        "conviction": round(conviction, 4),
        "initial_pct": initial_pct,
        "scaling_enabled": scaling_enabled,
        "alloc_raw": initial_capital * position_size_pct,
        "alloc_effective": round(alloc, 2),
        "close_price": close_price,
        "buy_price_effective": round(buy_price, 2),
        "qty": qty,
        "target_qty": target_qty,
        "total_cost": round(buy_price * qty, 2),
        "cash_before": round(cash, 2),
        "positions_count": current_positions,
        "max_positions": max_positions,
    }

    if qty <= 0:
        return BuyDecision(
            action="SKIP_BUY",
            reason=f"수량 0: 배분금 {alloc:,.0f} / 매수가 {buy_price:,.0f}",
            sizing=sizing,
        )

    total_cost = buy_price * qty
    if total_cost > cash:
        return BuyDecision(
            action="SKIP_BUY",
            reason=f"현금 부족: {total_cost:,.0f} > {cash:,.0f}",
            sizing=sizing,
        )

    return BuyDecision(
        action="BUY",
        qty=qty,
        price=buy_price,
        conviction=conviction,
        alloc=alloc,
        target_qty=target_qty,
        sizing=sizing,
    )


def evaluate_risk(
    avg_price: float,
    highest_price: float,
    current_price: float,
    qty: int,
    *,
    stop_loss_pct: float | None = None,
    trailing_stop_pct: float | None = None,
    atr_val: float | None = None,
    atr_stop_mult: float | None = None,
    partial_exit_gain_pct: float | None = None,
    partial_exit_pct: float = 0.5,
    has_partial_exited: bool = False,
    scaling_enabled: bool = False,
) -> RiskDecision | None:
    """리스크 체크 — 순수 함수.

    Returns
    -------
    RiskDecision | None : 리스크 발동 시 판단 결과, 아니면 None
    """
    # 1. 고정 손절
    if stop_loss_pct is not None and avg_price > 0:
        loss_pct = (current_price - avg_price) / avg_price * 100
        if loss_pct <= -stop_loss_pct:
            return RiskDecision(
                action="RISK_STOP",
                qty=qty,
                reason=f"손절: 현재가 {current_price:,.0f} / 평단 {avg_price:,.0f} = {loss_pct:+.2f}%",
                risk={"type": "stop_loss", "loss_pct": round(loss_pct, 4), "threshold": stop_loss_pct},
            )

    # 2. 트레일링 스탑
    if trailing_stop_pct is not None and highest_price > 0:
        drop_pct = (highest_price - current_price) / highest_price * 100
        if drop_pct >= trailing_stop_pct:
            return RiskDecision(
                action="RISK_TRAIL",
                qty=qty,
                reason=f"트레일링: 고점 {highest_price:,.0f} → {current_price:,.0f} = -{drop_pct:.2f}%",
                risk={"type": "trailing_stop", "drop_pct": round(drop_pct, 4), "threshold": trailing_stop_pct},
            )

    # 3. ATR 동적 손절
    if atr_stop_mult is not None and atr_val and atr_val > 0 and avg_price > 0:
        stop_line = avg_price - atr_val * atr_stop_mult
        if current_price <= stop_line:
            return RiskDecision(
                action="RISK_ATR_STOP",
                qty=qty,
                reason=f"ATR 스탑: {current_price:,.0f} <= {stop_line:,.0f} (평단-ATR×{atr_stop_mult})",
                risk={"type": "atr_stop", "stop_line": round(stop_line), "atr": round(atr_val, 2)},
            )

    # 4. 부분 익절
    if scaling_enabled and partial_exit_gain_pct and not has_partial_exited and avg_price > 0 and qty > 1:
        gain_pct = (current_price - avg_price) / avg_price * 100
        if gain_pct >= partial_exit_gain_pct:
            sell_qty = max(1, int(qty * partial_exit_pct))
            if sell_qty >= qty:
                sell_qty = qty - 1
            if sell_qty > 0:
                return RiskDecision(
                    action="PARTIAL_EXIT",
                    qty=sell_qty,
                    reason=f"부분 익절: +{gain_pct:.2f}% >= {partial_exit_gain_pct}% → {sell_qty}주 매도",
                    risk={"type": "partial_exit", "gain_pct": round(gain_pct, 4),
                          "threshold": partial_exit_gain_pct, "sell_ratio": partial_exit_pct},
                )

    return None


def evaluate_scale_in(
    avg_price: float,
    current_price: float,
    current_qty: int,
    target_qty: int,
    scale_in_count: int,
    max_scale_in: int,
    scale_in_drop_pct: float,
) -> BuyDecision | None:
    """추가매수(B2) 판단 — 순수 함수.

    Returns
    -------
    BuyDecision | None : 추가매수 판단 시 결과, 아니면 None
    """
    if current_qty >= target_qty:
        return None
    if scale_in_count >= max_scale_in:
        return None
    if avg_price <= 0:
        return None

    drop_pct = (avg_price - current_price) / avg_price * 100
    if drop_pct < scale_in_drop_pct:
        return None

    remaining_qty = target_qty - current_qty
    if remaining_qty <= 0:
        return None

    return BuyDecision(
        action="SCALE_IN",
        qty=remaining_qty,
        price=current_price,
        reason=f"B2 추가매수: 평단 대비 -{drop_pct:.2f}% (기준 -{scale_in_drop_pct}%) → {remaining_qty}주",
        sizing={"drop_pct": round(drop_pct, 2), "target_qty": target_qty,
                "current_qty": current_qty, "scale_in_count": scale_in_count},
    )
