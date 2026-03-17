"""비용 모델 — 수수료 + 슬리피지 + 세금.

슬리피지 모델:
- fixed: 고정 비율 (기본값 0.1%)
- volumeshare: 거래량 참여율 기반 비선형 슬리피지
  slippage = min(order_qty / bar_volume, vs_volume_limit)² × vs_price_impact
  QuantConnect VolumeShareSlippageModel 기반.
"""

from __future__ import annotations

from pydantic import BaseModel


class CostConfig(BaseModel):
    buy_commission: float = 0.00015   # 0.015 % 증권사 수수료
    sell_commission: float = 0.00215  # 0.015 % 수수료 + 0.20 % 거래세
    slippage_pct: float = 0.001       # 0.1 % 슬리피지 (fixed 모드 기본값)
    slippage_model: str = "fixed"     # "fixed" | "volumeshare"
    vs_price_impact: float = 0.1      # VolumeShare: 가격 충격 계수
    vs_volume_limit: float = 0.025    # VolumeShare: 최대 참여율 (2.5%)


def default_cost_config(interval: str = "1d") -> CostConfig:
    """인터벌별 기본 비용 설정.

    일봉: 슬리피지 0.10% (fixed) → 왕복 0.43%
    분봉: VolumeShare 모델 → 거래량 기반 동적 슬리피지
    """
    if interval == "1d":
        return CostConfig()
    return CostConfig(
        buy_commission=0.00015,
        sell_commission=0.00215,
        slippage_pct=0.0005,          # fixed 폴백: 0.05%
        slippage_model="volumeshare",  # 분봉은 VolumeShare 기본 적용
    )


def _compute_volumeshare_slippage(
    order_qty: int,
    bar_volume: int,
    cfg: CostConfig,
) -> float:
    """VolumeShare 슬리피지 비율 계산.

    slippage = min(order_qty / bar_volume, volume_limit)² × price_impact
    """
    if bar_volume <= 0 or order_qty <= 0:
        return cfg.slippage_pct  # 폴백: 고정값
    share = min(order_qty / bar_volume, cfg.vs_volume_limit)
    return share * share * cfg.vs_price_impact


def effective_buy_price(
    price: float,
    cfg: CostConfig,
    order_qty: int = 0,
    bar_volume: int = 0,
) -> float:
    """매수 시 실효 가격 (슬리피지 + 수수료 포함)."""
    if cfg.slippage_model == "volumeshare" and bar_volume > 0 and order_qty > 0:
        slippage = _compute_volumeshare_slippage(order_qty, bar_volume, cfg)
    else:
        slippage = cfg.slippage_pct
    slipped = price * (1 + slippage)
    return slipped * (1 + cfg.buy_commission)


def effective_sell_price(
    price: float,
    cfg: CostConfig,
    order_qty: int = 0,
    bar_volume: int = 0,
) -> float:
    """매도 시 실효 가격 (슬리피지 + 수수료+세금 차감)."""
    if cfg.slippage_model == "volumeshare" and bar_volume > 0 and order_qty > 0:
        slippage = _compute_volumeshare_slippage(order_qty, bar_volume, cfg)
    else:
        slippage = cfg.slippage_pct
    slipped = price * (1 - slippage)
    return slipped * (1 - cfg.sell_commission)
