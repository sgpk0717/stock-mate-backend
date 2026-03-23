"""카운터팩추얼 시나리오 프리셋 + Claude 커스텀 시나리오 생성."""

from __future__ import annotations

import json
import logging

from app.simulation.schemas import CustomScenarioResponse, ScenarioConfig, ScenarioPreset

logger = logging.getLogger(__name__)

PRESET_SCENARIOS: list[ScenarioPreset] = [
    ScenarioPreset(
        type="rate_shock",
        name="금리 충격",
        description="기준금리 급등 충격. 펀더멘탈 에이전트의 내재가치 하향, 유동성 감소.",
        default_params={
            "rate_change_bps": 50,
            "value_impact_pct": -5.0,
            "liquidity_drain_pct": 30,
        },
    ),
    ScenarioPreset(
        type="liquidity_crisis",
        name="유동성 위기",
        description="호가창 급격한 유동성 소멸. 노이즈 트레이더 80% 탈출, 스프레드 급등.",
        default_params={
            "noise_exit_pct": 80,
            "spread_multiplier": 5.0,
        },
    ),
    ScenarioPreset(
        type="flash_crash",
        name="플래시 크래시",
        description="대규모 시장가 매도 → 순간 가격 붕괴 → 점진적 회복.",
        default_params={
            "sell_volume_multiple": 20,
            "recovery_speed": 0.02,
        },
    ),
    ScenarioPreset(
        type="supply_chain",
        name="공급망 충격",
        description="공급망 이슈로 특정 섹터 내재가치 급락, 차티스트 추세 추종 강화.",
        default_params={
            "value_drop_pct": -15.0,
            "momentum_boost": 1.5,
        },
    ),
]


async def generate_custom_scenario(prompt: str) -> CustomScenarioResponse:
    """Claude로 커스텀 시나리오 생성."""
    from app.core.llm import chat

    system_prompt = (
        "당신은 금융 시뮬레이션 시나리오 설계자입니다.\n"
        "사용자의 설명을 바탕으로 시뮬레이션 시나리오를 생성하세요.\n\n"
        "사용 가능한 시나리오 타입:\n"
        "- rate_shock: value_impact_pct (float, -100~0), liquidity_drain_pct (0~100)\n"
        "- liquidity_crisis: noise_exit_pct (0~100)\n"
        "- flash_crash: sell_volume_multiple (int, 1~100)\n"
        "- supply_chain: value_drop_pct (float, -100~0), momentum_boost (float, 1~5)\n"
        "- custom: 위 타입 중 하나를 선택하되 파라미터를 조합할 수 있습니다.\n\n"
        "JSON 형식으로 응답하세요:\n"
        '{"scenario": {"type": "...", "params": {...}, "inject_at_step": 500}, "explanation": "..."}'
    )

    resp = await chat(
        max_tokens=500,
        system=system_prompt,
        messages=[{"role": "user", "content": prompt}],
        caller="simulation.scenarios",
    )

    text = resp.content[0].text.strip()

    # Extract JSON
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            parsed = json.loads(text[start : end + 1])
        else:
            parsed = {
                "scenario": {"type": "flash_crash", "params": {"sell_volume_multiple": 15}, "inject_at_step": 500},
                "explanation": "파싱 실패 — 기본 플래시 크래시 시나리오 적용",
            }

    scenario_data = parsed.get("scenario", {})
    return CustomScenarioResponse(
        scenario=ScenarioConfig(
            type=scenario_data.get("type", "flash_crash"),
            params=scenario_data.get("params", {}),
            inject_at_step=scenario_data.get("inject_at_step", 500),
        ),
        explanation=parsed.get("explanation", ""),
    )
