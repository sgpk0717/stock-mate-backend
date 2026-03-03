"""비용 모델 — 수수료 + 슬리피지 + 세금."""

from pydantic import BaseModel


class CostConfig(BaseModel):
    buy_commission: float = 0.00015   # 0.015 % 증권사 수수료
    sell_commission: float = 0.00215  # 0.015 % 수수료 + 0.20 % 거래세
    slippage_pct: float = 0.001       # 0.1 % 슬리피지 (일봉 기준 보수적)


def effective_buy_price(price: float, cfg: CostConfig) -> float:
    """매수 시 실효 가격 (슬리피지 + 수수료 포함)."""
    slipped = price * (1 + cfg.slippage_pct)
    return slipped * (1 + cfg.buy_commission)


def effective_sell_price(price: float, cfg: CostConfig) -> float:
    """매도 시 실효 가격 (슬리피지 + 수수료+세금 차감)."""
    slipped = price * (1 - cfg.slippage_pct)
    return slipped * (1 - cfg.sell_commission)
