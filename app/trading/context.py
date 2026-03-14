"""TradingContext — 전략 + 환경 통합 관리 객체.

백테스트 → 모의투자 → 실거래를 동일 Context로 심리스 전환.
메모리 캐시 + DB 영속화 이중 저장.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


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
    source_factor_id: str | None = None  # 원본 알파 팩터 ID

    # 세션 상태 (DB 복원 시 LiveSession 재생성용)
    session_state: dict | None = None

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
            "source_factor_id": self.source_factor_id,
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

    @classmethod
    def from_db_model(cls, m) -> TradingContext:
        """DB 모델(TradingContextModel)에서 TradingContext 복원."""
        cost_raw = m.cost_config or {}
        cost = CostConfig(
            buy_commission=cost_raw.get("buy_commission", 0.00015),
            sell_commission=cost_raw.get("sell_commission", 0.00215),
            slippage_pct=cost_raw.get("slippage_pct", 0.001),
        )
        return cls(
            id=str(m.id),
            mode=m.mode,
            strategy=m.strategy or {},
            strategy_name=m.strategy_name or "",
            position_sizing=m.position_sizing or {},
            scaling=m.scaling,
            risk_management=m.risk_management,
            cost_config=cost,
            initial_capital=float(m.initial_capital),
            position_size_pct=m.position_size_pct,
            max_positions=m.max_positions,
            symbols=m.symbols or [],
            created_at=m.created_at.isoformat() if m.created_at else "",
            source_backtest_id=str(m.source_backtest_id) if m.source_backtest_id else None,
            source_factor_id=str(m.source_factor_id) if m.source_factor_id else None,
            session_state=m.session_state if hasattr(m, "session_state") else None,
        )

    def to_db_model(self):
        """TradingContext → DB 모델 변환."""
        from app.workflow.models import TradingContextModel
        return TradingContextModel(
            id=uuid.UUID(self.id),
            mode=self.mode,
            status="active",
            strategy=self.strategy,
            strategy_name=self.strategy_name,
            position_sizing=self.position_sizing,
            scaling=self.scaling,
            risk_management=self.risk_management,
            cost_config={
                "buy_commission": self.cost_config.buy_commission,
                "sell_commission": self.cost_config.sell_commission,
                "slippage_pct": self.cost_config.slippage_pct,
            },
            initial_capital=self.initial_capital,
            position_size_pct=self.position_size_pct,
            max_positions=self.max_positions,
            symbols=self.symbols,
            source_backtest_id=uuid.UUID(self.source_backtest_id) if self.source_backtest_id else None,
            source_factor_id=uuid.UUID(self.source_factor_id) if self.source_factor_id else None,
        )


# ── 메모리 캐시 + DB 영속화 ──

_contexts: dict[str, TradingContext] = {}


def save_context(ctx: TradingContext) -> str:
    """메모리 캐시에 저장. DB 영속화는 save_context_to_db() 사용."""
    _contexts[ctx.id] = ctx
    return ctx.id


async def save_context_to_db(ctx: TradingContext) -> str:
    """메모리 + DB 모두 저장."""
    _contexts[ctx.id] = ctx
    try:
        from app.core.database import async_session
        async with async_session() as session:
            from sqlalchemy import select
            from app.workflow.models import TradingContextModel
            # upsert: 이미 존재하면 업데이트
            stmt = select(TradingContextModel).where(
                TradingContextModel.id == uuid.UUID(ctx.id)
            )
            result = await session.execute(stmt)
            existing = result.scalar_one_or_none()
            if existing:
                existing.mode = ctx.mode
                existing.strategy = ctx.strategy
                existing.strategy_name = ctx.strategy_name
                existing.position_sizing = ctx.position_sizing
                existing.scaling = ctx.scaling
                existing.risk_management = ctx.risk_management
                existing.cost_config = {
                    "buy_commission": ctx.cost_config.buy_commission,
                    "sell_commission": ctx.cost_config.sell_commission,
                    "slippage_pct": ctx.cost_config.slippage_pct,
                }
                existing.initial_capital = ctx.initial_capital
                existing.position_size_pct = ctx.position_size_pct
                existing.max_positions = ctx.max_positions
                existing.symbols = ctx.symbols
            else:
                session.add(ctx.to_db_model())
            await session.commit()
    except Exception as e:
        logger.warning("TradingContext DB 저장 실패 (메모리에는 유지): %s", e)
    return ctx.id


def get_context(ctx_id: str) -> TradingContext | None:
    return _contexts.get(ctx_id)


def delete_context(ctx_id: str) -> bool:
    return _contexts.pop(ctx_id, None) is not None


async def delete_context_from_db(ctx_id: str) -> bool:
    """메모리 + DB 모두에서 삭제."""
    removed = _contexts.pop(ctx_id, None) is not None
    try:
        from app.core.database import async_session
        async with async_session() as session:
            from sqlalchemy import delete
            from app.workflow.models import TradingContextModel
            stmt = delete(TradingContextModel).where(
                TradingContextModel.id == uuid.UUID(ctx_id)
            )
            await session.execute(stmt)
            await session.commit()
    except Exception as e:
        logger.warning("TradingContext DB 삭제 실패: %s", e)
    return removed


def list_contexts() -> list[TradingContext]:
    return list(_contexts.values())


async def load_active_contexts_from_db() -> int:
    """서버 시작 시 DB에서 active 컨텍스트를 메모리에 복원.

    Returns:
        복원된 컨텍스트 수.
    """
    try:
        from app.core.database import async_session
        async with async_session() as session:
            from sqlalchemy import select
            from app.workflow.models import TradingContextModel
            stmt = select(TradingContextModel).where(
                TradingContextModel.status == "active"
            )
            result = await session.execute(stmt)
            models = result.scalars().all()
            count = 0
            for m in models:
                ctx = TradingContext.from_db_model(m)
                _contexts[ctx.id] = ctx
                count += 1
            if count:
                logger.info("DB에서 active TradingContext %d개 복원", count)
            return count
    except Exception as e:
        logger.warning("TradingContext DB 복원 실패: %s", e)
        return 0
