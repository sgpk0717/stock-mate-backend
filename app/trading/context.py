"""TradingContext — 전략 + 환경 통합 관리 객체.

백테스트 → 모의투자 → 실거래를 동일 Context로 심리스 전환.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class CostConfig:
    """수수료/슬리피지 설정."""
    buy_commission: float = 0.00015
    sell_commission: float = 0.00215
    slippage_pct: float = 0.001


@dataclass
class TradingContext:
    """전략 실행에 필요한 모든 설정을 통합."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    mode: str = "paper"  # "backtest" | "paper" | "real"

    # 전략
    strategy: dict = field(default_factory=dict)
    strategy_name: str = ""

    # 포지션 사이징
    position_sizing: dict = field(default_factory=dict)
    scaling: dict | None = None
    risk_management: dict | None = None

    # 비용
    cost_config: CostConfig = field(default_factory=CostConfig)

    # 자본/포지션
    initial_capital: float = 100_000_000
    position_size_pct: float = 0.1
    max_positions: int = 10

    # 종목 유니버스
    symbols: list[str] = field(default_factory=list)

    # 메타
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    source_backtest_id: str | None = None  # 원본 백테스트 run ID

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "mode": self.mode,
            "strategy": self.strategy,
            "strategy_name": self.strategy_name,
            "position_sizing": self.position_sizing,
            "scaling": self.scaling,
            "risk_management": self.risk_management,
            "cost_config": {
                "buy_commission": self.cost_config.buy_commission,
                "sell_commission": self.cost_config.sell_commission,
                "slippage_pct": self.cost_config.slippage_pct,
            },
            "initial_capital": self.initial_capital,
            "position_size_pct": self.position_size_pct,
            "max_positions": self.max_positions,
            "symbols": self.symbols,
            "created_at": self.created_at,
            "source_backtest_id": self.source_backtest_id,
        }

    @classmethod
    def from_backtest_run(cls, run: dict, mode: str = "paper") -> TradingContext:
        """백테스트 실행 결과에서 Context 생성."""
        strategy_json = run.get("strategy_json", {})

        cost_raw = strategy_json.get("cost_config") or run.get("cost_config") or {}
        cost = CostConfig(
            buy_commission=cost_raw.get("buy_commission", 0.00015),
            sell_commission=cost_raw.get("sell_commission", 0.00215),
            slippage_pct=cost_raw.get("slippage_pct", 0.001),
        )

        return cls(
            mode=mode,
            strategy=strategy_json.get("strategy", strategy_json),
            strategy_name=run.get("strategy_name", ""),
            position_sizing=strategy_json.get("position_sizing") or {},
            scaling=strategy_json.get("scaling"),
            risk_management=strategy_json.get("risk_management"),
            cost_config=cost,
            initial_capital=run.get("initial_capital", 100_000_000),
            position_size_pct=strategy_json.get("position_size_pct", 0.1),
            max_positions=strategy_json.get("max_positions", 10),
            symbols=strategy_json.get("symbols", []),
            source_backtest_id=run.get("id"),
        )


# 메모리 저장소 (단일 인스턴스)
_contexts: dict[str, TradingContext] = {}


def save_context(ctx: TradingContext) -> str:
    _contexts[ctx.id] = ctx
    return ctx.id


def get_context(ctx_id: str) -> TradingContext | None:
    return _contexts.get(ctx_id)


def delete_context(ctx_id: str) -> bool:
    return _contexts.pop(ctx_id, None) is not None


def list_contexts() -> list[TradingContext]:
    return list(_contexts.values())
