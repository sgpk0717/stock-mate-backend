"""이질적 시장 에이전트 6종.

FundamentalAgent, ChartistAgent, NoiseTrader, LLMAgent, StrategyAgent.
"""

from __future__ import annotations

import logging
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from app.simulation.orderbook import Fill

logger = logging.getLogger(__name__)


@dataclass
class BaseAgent(ABC):
    """에이전트 베이스."""

    id: str
    cash: float = 10_000_000.0
    shares: int = 100
    _avg_cost: float = 0.0

    @abstractmethod
    def decide(
        self, step: int, mid_price: float, market_state: dict
    ) -> list[dict]:
        """Returns list of {side, price, qty, type}."""

    def on_fill(self, fill: Fill) -> None:
        """Update portfolio on fill."""
        if fill.side == "BUY":
            total_cost = self._avg_cost * self.shares + fill.price * fill.qty
            self.shares += fill.qty
            self._avg_cost = total_cost / self.shares if self.shares > 0 else 0
            self.cash -= fill.price * fill.qty
        else:
            self.shares -= fill.qty
            self.cash += fill.price * fill.qty

    def get_pnl(self, current_price: float) -> float:
        """Unrealized + realized P&L."""
        return self.cash + self.shares * current_price - 10_000_000.0 - 100 * current_price


# ── Fundamental Agent ─────────────────────────────────────


class FundamentalAgent(BaseAgent):
    """내재가치 = SMA(200) ± noise. price < intrinsic → BUY."""

    def __init__(
        self,
        agent_id: str,
        noise_scale: float = 0.05,
        rng: random.Random | None = None,
        **kw,
    ):
        super().__init__(id=agent_id, **kw)
        self._noise_scale = noise_scale
        self._rng = rng or random.Random()
        self._price_history: list[float] = []
        self._intrinsic_offset = self._rng.gauss(0, noise_scale)

    def decide(
        self, step: int, mid_price: float, market_state: dict
    ) -> list[dict]:
        self._price_history.append(mid_price)
        if len(self._price_history) < 50:
            return []

        lookback = min(200, len(self._price_history))
        sma = sum(self._price_history[-lookback:]) / lookback
        intrinsic = sma * (1.0 + self._intrinsic_offset)

        actions: list[dict] = []
        if mid_price < intrinsic * 0.98 and self.cash > mid_price * 5:
            qty = max(1, int(self.cash * 0.05 / mid_price))
            actions.append(
                {"side": "BUY", "price": mid_price * 1.001, "qty": qty, "type": "LIMIT"}
            )
        elif mid_price > intrinsic * 1.02 and self.shares > 5:
            qty = max(1, self.shares // 10)
            actions.append(
                {"side": "SELL", "price": mid_price * 0.999, "qty": qty, "type": "LIMIT"}
            )
        return actions

    def shift_intrinsic(self, pct_change: float) -> None:
        """시나리오 주입: 내재가치 변경."""
        self._intrinsic_offset += pct_change


# ── Chartist Agent ────────────────────────────────────────


class ChartistAgent(BaseAgent):
    """모멘텀 추종: price > SMA(20) → BUY."""

    def __init__(
        self,
        agent_id: str,
        lookback: int = 20,
        rng: random.Random | None = None,
        **kw,
    ):
        super().__init__(id=agent_id, **kw)
        self._lookback = lookback
        self._rng = rng or random.Random()
        self._price_history: list[float] = []
        self._momentum_boost: float = 1.0

    def decide(
        self, step: int, mid_price: float, market_state: dict
    ) -> list[dict]:
        self._price_history.append(mid_price)
        if len(self._price_history) < self._lookback + 1:
            return []

        sma = sum(self._price_history[-self._lookback:]) / self._lookback
        deviation = (mid_price - sma) / sma

        actions: list[dict] = []
        threshold = 0.005 / self._momentum_boost

        if deviation > threshold and self.cash > mid_price * 5:
            qty = max(1, int(self.cash * 0.03 / mid_price))
            actions.append(
                {"side": "BUY", "price": mid_price * 1.002, "qty": qty, "type": "LIMIT"}
            )
        elif deviation < -threshold and self.shares > 5:
            qty = max(1, self.shares // 10)
            actions.append(
                {"side": "SELL", "price": mid_price * 0.998, "qty": qty, "type": "LIMIT"}
            )
        return actions

    def boost_momentum(self, factor: float) -> None:
        """시나리오 주입: 모멘텀 추종 강화."""
        self._momentum_boost *= factor


# ── Noise Trader ──────────────────────────────────────────


class NoiseTrader(BaseAgent):
    """랜덤 매매. intensity 파라미터로 활동량 조절."""

    def __init__(
        self,
        agent_id: str,
        intensity: float = 0.1,
        rng: random.Random | None = None,
        **kw,
    ):
        super().__init__(id=agent_id, **kw)
        self._intensity = intensity
        self._rng = rng or random.Random()
        self._active = True

    def decide(
        self, step: int, mid_price: float, market_state: dict
    ) -> list[dict]:
        if not self._active:
            return []
        if self._rng.random() > self._intensity:
            return []

        side = self._rng.choice(["BUY", "SELL"])
        spread_pct = self._rng.uniform(0.001, 0.005)

        if side == "BUY" and self.cash > mid_price * 2:
            price = mid_price * (1.0 - spread_pct)
            qty = max(1, self._rng.randint(1, 5))
            return [{"side": "BUY", "price": price, "qty": qty, "type": "LIMIT"}]
        elif side == "SELL" and self.shares > 2:
            price = mid_price * (1.0 + spread_pct)
            qty = max(1, self._rng.randint(1, min(3, self.shares)))
            return [{"side": "SELL", "price": price, "qty": qty, "type": "LIMIT"}]
        return []

    def deactivate(self) -> None:
        """시나리오 주입: 노이즈 트레이더 탈출."""
        self._active = False


# ── LLM Agent ─────────────────────────────────────────────


class LLMAgent(BaseAgent):
    """Claude 추론 기반 에이전트. 행동 편향 주입."""

    def __init__(
        self,
        agent_id: str,
        loss_aversion: float = 2.5,
        herding_factor: float = 0.3,
        call_interval: int = 20,
        rng: random.Random | None = None,
        **kw,
    ):
        super().__init__(id=agent_id, **kw)
        self._loss_aversion = loss_aversion
        self._herding_factor = herding_factor
        self._call_interval = call_interval
        self._rng = rng or random.Random()
        self._last_decision: dict | None = None
        self._last_call_step: int = -9999
        self._price_history: list[float] = []

    def decide(
        self, step: int, mid_price: float, market_state: dict
    ) -> list[dict]:
        """Synchronous wrapper. LLM calls are handled by exchange via decide_async."""
        self._price_history.append(mid_price)
        if step - self._last_call_step < self._call_interval:
            return self._replay_decision(mid_price)
        # Async call needed — exchange will call decide_async instead
        return []

    async def decide_async(
        self, step: int, mid_price: float, market_state: dict
    ) -> list[dict]:
        """Async version that actually calls Claude."""
        self._price_history.append(mid_price)
        if step - self._last_call_step < self._call_interval:
            return self._replay_decision(mid_price)

        self._last_call_step = step
        try:
            decision = await self._call_claude(mid_price, market_state)
            self._last_decision = decision
        except Exception as e:
            logger.warning("LLMAgent %s Claude call failed: %s", self.id, e)
            self._last_decision = {"action": "HOLD"}

        return self._replay_decision(mid_price)

    def _replay_decision(self, mid_price: float) -> list[dict]:
        """Execute last decision."""
        if not self._last_decision:
            return []

        action = self._last_decision.get("action", "HOLD")
        if action == "HOLD":
            return []

        qty = self._last_decision.get("qty", 1)
        price_offset = self._last_decision.get("price_offset_pct", 0.1) / 100.0

        if action == "BUY" and self.cash > mid_price * qty:
            return [
                {
                    "side": "BUY",
                    "price": mid_price * (1.0 + price_offset),
                    "qty": qty,
                    "type": "LIMIT",
                }
            ]
        elif action == "SELL" and self.shares >= qty:
            return [
                {
                    "side": "SELL",
                    "price": mid_price * (1.0 - price_offset),
                    "qty": qty,
                    "type": "LIMIT",
                }
            ]
        return []

    async def _call_claude(
        self, mid_price: float, market_state: dict
    ) -> dict:
        """Claude API 호출."""
        from app.core.llm import chat

        # 최근 가격 추이 요약
        recent = self._price_history[-20:] if len(self._price_history) >= 20 else self._price_history
        price_change = ((recent[-1] - recent[0]) / recent[0] * 100) if len(recent) > 1 else 0.0

        pnl = self.cash + self.shares * mid_price - 10_000_000.0 - 100 * mid_price
        pnl_pct = pnl / (10_000_000.0 + 100 * mid_price) * 100

        prompt = (
            f"당신은 가상 주식시장의 트레이더입니다.\n\n"
            f"현재 상태:\n"
            f"- 현재가: {mid_price:,.0f}\n"
            f"- 최근 {len(recent)}스텝 변화율: {price_change:+.2f}%\n"
            f"- 스프레드: {market_state.get('spread', 0):,.0f}\n"
            f"- 보유 현금: {self.cash:,.0f}\n"
            f"- 보유 주식: {self.shares}주\n"
            f"- 평가 손익: {pnl_pct:+.2f}%\n\n"
            f"행동 성향:\n"
            f"- 손실 회피 계수: {self._loss_aversion} (손실이 이익보다 {self._loss_aversion}배 고통스럽습니다)\n"
            f"- 군집 행동 강도: {self._herding_factor} (가격 추세를 따르는 경향)\n\n"
            f"JSON으로 응답하세요: {{\"action\": \"BUY\"|\"SELL\"|\"HOLD\", \"qty\": 수량(1-10), \"price_offset_pct\": 0.05-0.5, \"reasoning\": \"이유\"}}"
        )

        resp = await chat(
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
            caller="simulation.llm_agent",
        )

        import json

        text = resp.content[0].text.strip()
        # Extract JSON from possible markdown
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Fallback: find {...}
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                return json.loads(text[start : end + 1])
            return {"action": "HOLD"}


# ── Strategy Agent ────────────────────────────────────────


class StrategyAgent(BaseAgent):
    """사용자 백테스트 전략 조건을 시뮬레이션에서 실행.

    롤링 지표(SMA, RSI)를 순수 Python으로 계산.
    """

    def __init__(self, agent_id: str, strategy: dict, **kw):
        super().__init__(id=agent_id, **kw)
        self._strategy = strategy
        self._price_history: list[float] = []
        self._volume_history: list[int] = []
        self._has_position = False

    def decide(
        self, step: int, mid_price: float, market_state: dict
    ) -> list[dict]:
        self._price_history.append(mid_price)
        self._volume_history.append(market_state.get("step_volume", 0))

        if len(self._price_history) < 30:
            return []

        indicators = self._compute_indicators()

        buy_conditions = self._strategy.get("buy_conditions", [])
        sell_conditions = self._strategy.get("sell_conditions", [])

        buy_signal = self._evaluate_conditions(buy_conditions, indicators)
        sell_signal = self._evaluate_conditions(sell_conditions, indicators)

        actions: list[dict] = []

        if buy_signal and not self._has_position and self.cash > mid_price * 10:
            qty = max(1, int(self.cash * 0.1 / mid_price))
            actions.append(
                {"side": "BUY", "price": mid_price * 1.001, "qty": qty, "type": "LIMIT"}
            )
            self._has_position = True
        elif sell_signal and self._has_position and self.shares > 0:
            actions.append(
                {"side": "SELL", "price": mid_price * 0.999, "qty": self.shares, "type": "LIMIT"}
            )
            self._has_position = False

        return actions

    def _compute_indicators(self) -> dict:
        """순수 Python 롤링 지표."""
        prices = self._price_history
        n = len(prices)

        result: dict = {"close": prices[-1]}

        # SMA
        for period in [5, 10, 20, 50]:
            if n >= period:
                result[f"sma_{period}"] = sum(prices[-period:]) / period

        # RSI (14)
        if n >= 15:
            gains = []
            losses = []
            for i in range(-14, 0):
                diff = prices[i] - prices[i - 1]
                if diff > 0:
                    gains.append(diff)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(-diff)
            avg_gain = sum(gains) / 14
            avg_loss = sum(losses) / 14
            if avg_loss == 0:
                result["rsi"] = 100.0
            else:
                rs = avg_gain / avg_loss
                result["rsi"] = 100.0 - 100.0 / (1.0 + rs)

        # Volume ratio
        volumes = self._volume_history
        if len(volumes) >= 20 and sum(volumes[-20:]) > 0:
            avg_vol = sum(volumes[-20:]) / 20
            if avg_vol > 0:
                result["volume_ratio"] = volumes[-1] / avg_vol

        return result

    def _evaluate_conditions(
        self, conditions: list[dict], indicators: dict
    ) -> bool:
        """Simplified condition evaluation."""
        if not conditions:
            return False

        logic = "AND"
        results = []
        for cond in conditions:
            indicator_name = cond.get("indicator", "")
            operator = cond.get("operator", ">")
            value = cond.get("value", 0)

            # Resolve indicator value
            ind_val = indicators.get(indicator_name)
            if ind_val is None:
                # Try SMA pattern
                if indicator_name == "sma":
                    period = cond.get("params", {}).get("period", 20)
                    ind_val = indicators.get(f"sma_{period}")
                elif indicator_name == "rsi":
                    ind_val = indicators.get("rsi")
                elif indicator_name == "volume_ratio":
                    ind_val = indicators.get("volume_ratio")

            if ind_val is None:
                results.append(False)
                continue

            # Compare
            if operator == ">":
                results.append(ind_val > value)
            elif operator == ">=":
                results.append(ind_val >= value)
            elif operator == "<":
                results.append(ind_val < value)
            elif operator == "<=":
                results.append(ind_val <= value)
            elif operator == "==":
                results.append(ind_val == value)
            else:
                results.append(False)

        if logic == "AND":
            return all(results) if results else False
        return any(results) if results else False


# ── Factory ───────────────────────────────────────────────


def create_agents(
    agent_config: dict,
    seed: int | None = None,
) -> list[BaseAgent]:
    """AgentConfig dict에서 에이전트 리스트 생성."""
    rng = random.Random(seed)
    agents: list[BaseAgent] = []

    for i in range(agent_config.get("fundamental_count", 20)):
        agents.append(
            FundamentalAgent(
                agent_id=f"fundamental_{i}",
                noise_scale=0.05,
                rng=random.Random(rng.randint(0, 2**31)),
            )
        )

    for i in range(agent_config.get("chartist_count", 30)):
        agents.append(
            ChartistAgent(
                agent_id=f"chartist_{i}",
                lookback=rng.choice([10, 15, 20, 30]),
                rng=random.Random(rng.randint(0, 2**31)),
            )
        )

    for i in range(agent_config.get("noise_count", 100)):
        agents.append(
            NoiseTrader(
                agent_id=f"noise_{i}",
                intensity=rng.uniform(0.05, 0.15),
                rng=random.Random(rng.randint(0, 2**31)),
            )
        )

    for i in range(agent_config.get("llm_count", 5)):
        agents.append(
            LLMAgent(
                agent_id=f"llm_{i}",
                loss_aversion=rng.uniform(2.0, 3.0),
                herding_factor=rng.uniform(0.2, 0.5),
                call_interval=agent_config.get("llm_call_interval", 20),
                rng=random.Random(rng.randint(0, 2**31)),
            )
        )

    return agents
