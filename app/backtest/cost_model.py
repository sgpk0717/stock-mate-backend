"""비용 모델 — 수수료 + 슬리피지 + 세금."""

from __future__ import annotations

from pydantic import BaseModel


class CostConfig(BaseModel):
    buy_commission: float = 0.00015   # 0.015 % 증권사 수수료
    sell_commission: float = 0.00215  # 0.015 % 수수료 + 0.20 % 거래세
    slippage_pct: float = 0.001       # 0.1 % 슬리피지 (일봉 기준 보수적)


def default_cost_config(interval: str = "1d") -> CostConfig:
    """인터벌별 기본 비용 설정.

    일봉: 슬리피지 0.10% → 왕복 0.43%
    분봉: 슬리피지 0.05% → 왕복 0.33% (보수적: 시장가성 지정가 + 미체결 추적)
    """
    if interval == "1d":
        return CostConfig()
    return CostConfig(
        buy_commission=0.00015,   # 0.015%
        sell_commission=0.00215,  # 0.015% + 거래세 0.20%
        slippage_pct=0.0005,     # 0.05%
    )


def effective_buy_price(price: float, cfg: CostConfig) -> float:
    """매수 시 실효 가격 (슬리피지 + 수수료 포함)."""
    slipped = price * (1 + cfg.slippage_pct)
    return slipped * (1 + cfg.buy_commission)


def effective_sell_price(price: float, cfg: CostConfig) -> float:
    """매도 시 실효 가격 (슬리피지 + 수수료+세금 차감)."""
    slipped = price * (1 - cfg.slippage_pct)
    return slipped * (1 - cfg.sell_commission)
