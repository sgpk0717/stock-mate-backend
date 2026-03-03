"""가상 거래소 — 다중 에이전트 시뮬레이션 환경."""

from __future__ import annotations

import logging
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

from app.simulation.agents import BaseAgent, LLMAgent
from app.simulation.orderbook import LimitOrderBook

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """시뮬레이션 결과."""

    price_series: list[float] = field(default_factory=list)
    volume_series: list[int] = field(default_factory=list)
    spread_series: list[float] = field(default_factory=list)
    depth_series: list[dict] = field(default_factory=list)
    agent_metrics: dict = field(default_factory=dict)
    strategy_pnl: float = 0.0
    events_injected: list[dict] = field(default_factory=list)


@dataclass
class SimulationMetrics:
    """시뮬레이션 성과 지표."""

    final_price: float = 0.0
    price_change_pct: float = 0.0
    max_drawdown: float = 0.0
    annualized_volatility: float = 0.0
    crash_depth: float = 0.0
    recovery_steps: int = 0
    strategy_pnl: float = 0.0
    strategy_pnl_pct: float = 0.0
    avg_spread: float = 0.0
    avg_volume: float = 0.0


class VirtualExchange:
    """다중 에이전트 시뮬레이션 환경."""

    def __init__(
        self,
        agents: list[BaseAgent],
        initial_price: float = 50000.0,
        tick_size: float = 10.0,
        seed: int | None = None,
    ) -> None:
        self._lob = LimitOrderBook(tick_size=tick_size)
        self._agents = agents
        self._initial_price = initial_price
        self._tick_size = tick_size
        self._rng = random.Random(seed)
        self._step = 0

        # Results accumulation
        self._price_series: list[float] = [initial_price]
        self._volume_series: list[int] = [0]
        self._spread_series: list[float] = [0.0]
        self._depth_snapshots: list[dict] = []
        self._events: list[dict] = []

        # Pending scenario injections: {step: (event_type, params)}
        self._scheduled_events: dict[int, tuple[str, dict]] = {}

        # Initialize LOB
        self._seed_initial_book(initial_price)

    def _seed_initial_book(self, price: float) -> None:
        """Place initial bids/asks around initial price to bootstrap the book."""
        tick = self._tick_size
        for i in range(1, 21):
            bid_price = price - i * tick
            ask_price = price + i * tick
            qty = max(1, self._rng.randint(5, 20))
            self._lob.limit_order("BUY", bid_price, qty)
            self._lob.limit_order("SELL", ask_price, qty)

    def schedule_event(self, step: int, event_type: str, params: dict) -> None:
        """Schedule a scenario event at a specific step."""
        self._scheduled_events[step] = (event_type, params)

    async def run_steps(
        self,
        n: int,
        progress_cb: Callable | None = None,
    ) -> SimulationResult:
        """Run n simulation steps."""
        for step in range(n):
            self._step = step

            # Check for scheduled events
            if step in self._scheduled_events:
                event_type, params = self._scheduled_events[step]
                self.inject_event(event_type, params)
                self._events.append(
                    {"step": step, "type": event_type, "params": params}
                )

            mid_price = self._lob.get_mid_price() or self._initial_price

            market_state = {
                "step": step,
                "mid_price": mid_price,
                "spread": self._lob.get_spread(),
                "depth": self._lob.get_depth(3),
                "recent_prices": self._price_series[-20:],
                "recent_volumes": self._volume_series[-20:],
                "step_volume": self._volume_series[-1] if self._volume_series else 0,
            }

            # Each agent decides and places orders
            step_volume = 0

            for agent in self._agents:
                try:
                    if isinstance(agent, LLMAgent):
                        actions = await agent.decide_async(step, mid_price, market_state)
                    else:
                        actions = agent.decide(step, mid_price, market_state)

                    for action in actions:
                        fills = self._execute_action(agent, action)
                        for fill in fills:
                            step_volume += fill.qty
                            agent.on_fill(fill)
                except Exception as e:
                    logger.debug("Agent %s error at step %d: %s", agent.id, step, e)

            # Record state
            current_price = self._lob.get_mid_price() or (
                self._price_series[-1] if self._price_series else self._initial_price
            )
            self._price_series.append(current_price)
            self._volume_series.append(step_volume)
            self._spread_series.append(self._lob.get_spread())

            # Depth snapshot every 50 steps
            if step % 50 == 0:
                self._depth_snapshots.append(self._lob.get_depth(5))

            # Progress callback every 10 steps
            if progress_cb and step % 10 == 0:
                await progress_cb(step, n, f"Step {step}/{n}")

        # Final progress
        if progress_cb:
            await progress_cb(n, n, f"시뮬레이션 완료 ({n}스텝)")

        return self._build_result()

    def inject_event(self, event_type: str, params: dict) -> None:
        """Inject a market event mid-simulation."""
        from app.simulation.agents import (
            ChartistAgent,
            FundamentalAgent,
            NoiseTrader,
        )

        logger.info("Injecting event: %s at step %d", event_type, self._step)

        if event_type == "rate_shock":
            value_impact = params.get("value_impact_pct", -5.0) / 100.0
            liquidity_drain = params.get("liquidity_drain_pct", 30) / 100.0
            for agent in self._agents:
                if isinstance(agent, FundamentalAgent):
                    agent.shift_intrinsic(value_impact)
                elif isinstance(agent, NoiseTrader):
                    if self._rng.random() < liquidity_drain:
                        agent.deactivate()

        elif event_type == "liquidity_crisis":
            noise_exit = params.get("noise_exit_pct", 80) / 100.0
            for agent in self._agents:
                if isinstance(agent, NoiseTrader):
                    if self._rng.random() < noise_exit:
                        agent.deactivate()

        elif event_type == "flash_crash":
            sell_multiple = params.get("sell_volume_multiple", 20)
            mid = self._lob.get_mid_price() or self._initial_price
            # Inject large market sell orders
            for _ in range(sell_multiple):
                self._lob.market_order("SELL", self._rng.randint(10, 50))

        elif event_type == "supply_chain":
            value_drop = params.get("value_drop_pct", -15.0) / 100.0
            momentum_boost = params.get("momentum_boost", 1.5)
            for agent in self._agents:
                if isinstance(agent, FundamentalAgent):
                    agent.shift_intrinsic(value_drop)
                elif isinstance(agent, ChartistAgent):
                    agent.boost_momentum(momentum_boost)

    def _execute_action(self, agent: BaseAgent, action: dict) -> list:
        """Execute agent action on LOB."""
        side = action["side"]
        qty = max(1, int(action.get("qty", 1)))
        order_type = action.get("type", "LIMIT")

        if order_type == "MARKET":
            return self._lob.market_order(side, qty)
        else:
            price = action.get("price", self._lob.get_mid_price() or self._initial_price)
            _order, fills = self._lob.limit_order(side, price, qty)
            return fills

    def _build_result(self) -> SimulationResult:
        """Build simulation result."""
        from app.simulation.agents import StrategyAgent

        # Aggregate agent metrics by type
        agent_pnls: dict[str, list[float]] = defaultdict(list)
        strategy_pnl = 0.0
        final_price = self._price_series[-1] if self._price_series else self._initial_price

        for agent in self._agents:
            pnl = agent.get_pnl(final_price)
            agent_type = type(agent).__name__
            agent_pnls[agent_type].append(pnl)
            if isinstance(agent, StrategyAgent):
                strategy_pnl = pnl

        agent_metrics = {}
        for atype, pnls in agent_pnls.items():
            agent_metrics[atype] = {
                "count": len(pnls),
                "avg_pnl": sum(pnls) / len(pnls) if pnls else 0,
                "total_pnl": sum(pnls),
            }

        return SimulationResult(
            price_series=self._price_series,
            volume_series=self._volume_series,
            spread_series=self._spread_series,
            depth_series=self._depth_snapshots,
            agent_metrics=agent_metrics,
            strategy_pnl=strategy_pnl,
            events_injected=self._events,
        )

    def compute_metrics(self) -> SimulationMetrics:
        """Compute simulation performance metrics."""
        prices = self._price_series
        if len(prices) < 2:
            return SimulationMetrics()

        final_price = prices[-1]
        initial_price = prices[0]
        price_change_pct = (final_price - initial_price) / initial_price * 100

        # Max drawdown
        peak = prices[0]
        max_dd = 0.0
        crash_depth = 0.0
        for p in prices:
            if p > peak:
                peak = p
            dd = (peak - p) / peak * 100
            if dd > max_dd:
                max_dd = dd
            if dd > crash_depth:
                crash_depth = dd

        # Recovery steps
        recovery_steps = 0
        crash_end_idx = 0
        peak_price = prices[0]
        for i, p in enumerate(prices):
            if p < peak_price * 0.95:  # 5% drawdown as "crash"
                crash_end_idx = i
        if crash_end_idx > 0:
            for i in range(crash_end_idx, len(prices)):
                if prices[i] >= peak_price * 0.99:
                    recovery_steps = i - crash_end_idx
                    break
            else:
                recovery_steps = len(prices) - crash_end_idx

        # Volatility
        returns = []
        for i in range(1, len(prices)):
            if prices[i - 1] > 0:
                returns.append((prices[i] - prices[i - 1]) / prices[i - 1])
        if returns:
            mean_ret = sum(returns) / len(returns)
            var = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
            daily_vol = math.sqrt(var)
            annualized_vol = daily_vol * math.sqrt(252) * 100
        else:
            annualized_vol = 0.0

        # Strategy P&L
        from app.simulation.agents import StrategyAgent

        strategy_pnl = 0.0
        for agent in self._agents:
            if isinstance(agent, StrategyAgent):
                strategy_pnl = agent.get_pnl(final_price)
                break

        initial_portfolio = 10_000_000.0 + 100 * initial_price
        strategy_pnl_pct = strategy_pnl / initial_portfolio * 100 if initial_portfolio > 0 else 0

        # Averages
        spreads = self._spread_series
        volumes = self._volume_series
        avg_spread = sum(spreads) / len(spreads) if spreads else 0
        avg_volume = sum(volumes) / len(volumes) if volumes else 0

        return SimulationMetrics(
            final_price=final_price,
            price_change_pct=price_change_pct,
            max_drawdown=max_dd,
            annualized_volatility=annualized_vol,
            crash_depth=crash_depth,
            recovery_steps=recovery_steps,
            strategy_pnl=strategy_pnl,
            strategy_pnl_pct=strategy_pnl_pct,
            avg_spread=avg_spread,
            avg_volume=avg_volume,
        )
