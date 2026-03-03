"""Risk Manager Agent — 리스크 관리 전문 에이전트.

전략의 리스크를 평가하고, 포지션 사이징과 손절 설정을 제안한다.
Manager Agent에서 ask_risk_manager 도구로 호출된다.
"""

from __future__ import annotations

import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
당신은 주식 트레이딩 리스크 관리(Risk Management) 전문가입니다.

## 역할
- 전략의 리스크를 평가합니다.
- 적절한 포지션 사이징을 제안합니다.
- 손절/익절 수준을 제안합니다.

## 리스크 평가 기준
1. 손절 설정 여부: 손절이 없으면 최대 손실이 무제한
2. 포지션 크기: 단일 종목에 과도한 비중은 위험
3. 최대 동시 포지션: 분산투자 수준
4. 분할매매: 한번에 전량 진입 vs 분할 진입
5. 변동성 고려: ATR 기반 동적 손절의 적절성

## 제안 원칙
- 보수적 접근을 기본으로 합니다.
- 단일 종목 손실이 전체 자산의 2% 이내가 되도록 권장합니다.
- 손절이 없으면 반드시 경고하고 추가를 권장합니다.
- 트레일링 스탑은 수익 보호에 효과적임을 안내합니다.

## 응답 형식
간결하게 요약:
- 리스크 등급 (낮음 / 보통 / 높음 / 매우 높음)
- 주요 리스크 요인
- 개선 제안 (구체적인 파라미터 포함)
"""


async def assess_strategy(strategy_draft: dict) -> dict:
    """전략의 리스크를 평가한다.

    Args:
        strategy_draft: 전략 초안 (StrategySchema dict)

    Returns:
        {"assessment": str, "risk_level": str, "suggestions": list[str]}
    """
    # 규칙 기반 리스크 평가
    issues: list[str] = []
    suggestions: list[str] = []
    risk_score = 0  # 0-10

    # 1. 손절 설정 확인
    risk = strategy_draft.get("risk_management") or {}
    has_stop_loss = risk.get("stop_loss_pct") is not None
    has_trailing = risk.get("trailing_stop_pct") is not None
    has_atr_stop = risk.get("atr_stop_multiplier") is not None

    if not has_stop_loss and not has_trailing and not has_atr_stop:
        issues.append("손절 설정이 없습니다. 최대 손실이 무제한입니다.")
        suggestions.append("stop_loss_pct: 5~7% 또는 trailing_stop_pct: 3~5% 설정을 권장합니다.")
        risk_score += 4
    elif has_stop_loss and risk.get("stop_loss_pct", 0) > 10:
        issues.append(f"손절폭이 {risk['stop_loss_pct']}%로 넓습니다.")
        suggestions.append("손절폭을 5~7%로 줄이는 것을 고려하세요.")
        risk_score += 2

    # 2. 포지션 사이징 확인
    ps = strategy_draft.get("position_sizing") or {}
    ps_mode = ps.get("mode", "fixed")

    if ps_mode == "fixed":
        # fixed 모드는 별도 위험은 아니지만, 변동성 무시
        suggestions.append("변동성이 큰 종목에서는 atr_target 모드로 포지션 크기를 자동 조절하는 것이 안전합니다.")

    # 3. 분할매매 확인
    scaling = strategy_draft.get("scaling") or {}
    scaling_enabled = scaling.get("enabled", False)

    if not scaling_enabled:
        suggestions.append("분할매매를 활성화하면 평균 매입가를 낮출 수 있습니다. initial_pct: 0.5 (50% 1차 진입) 권장.")
    else:
        max_scale = scaling.get("max_scale_in", 1)
        if max_scale > 3:
            issues.append(f"최대 추가매수 횟수가 {max_scale}회로 많습니다. 과도한 물타기 위험.")
            risk_score += 2

    # 4. 매수 조건 수 확인
    buy_conds = strategy_draft.get("buy_conditions", [])
    if len(buy_conds) < 2:
        issues.append("매수 조건이 1개뿐입니다. 복합 조건으로 노이즈를 줄이는 것이 좋습니다.")
        suggestions.append("거래량 비율(volume_ratio >= 1.5) 등 보조 필터를 추가하는 것을 권장합니다.")
        risk_score += 1

    # 5. 매도 조건 확인
    sell_conds = strategy_draft.get("sell_conditions", [])
    if not sell_conds:
        issues.append("매도 조건이 없습니다.")
        risk_score += 3
    elif len(sell_conds) == 1:
        sell_ind = sell_conds[0].get("indicator", "")
        if sell_ind == buy_conds[0].get("indicator", "") if buy_conds else "":
            suggestions.append("매수와 매도가 같은 지표의 반대 조건이면, 다른 지표를 추가하여 다각화하는 것이 좋습니다.")

    # 리스크 등급 결정
    if risk_score >= 7:
        risk_level = "매우 높음"
    elif risk_score >= 4:
        risk_level = "높음"
    elif risk_score >= 2:
        risk_level = "보통"
    else:
        risk_level = "낮음"

    # Claude API로 정교한 분석 (가능 시)
    api_key = settings.ANTHROPIC_API_KEY
    if api_key:
        try:
            assessment_text = await _claude_assessment(strategy_draft, issues, suggestions, risk_level)
            return {
                "assessment": assessment_text,
                "risk_level": risk_level,
                "suggestions": suggestions,
            }
        except Exception as e:
            logger.warning("Claude 리스크 분석 실패: %s", e)

    # 규칙 기반 결과
    parts = [f"**리스크 등급: {risk_level}**\n"]

    if issues:
        parts.append("주요 리스크:")
        for issue in issues:
            parts.append(f"- {issue}")

    if suggestions:
        parts.append("\n개선 제안:")
        for sug in suggestions:
            parts.append(f"- {sug}")

    if not issues:
        parts.append("전략의 리스크 관리가 적절합니다.")

    return {
        "assessment": "\n".join(parts),
        "risk_level": risk_level,
        "suggestions": suggestions,
    }


async def _claude_assessment(
    strategy: dict,
    issues: list[str],
    suggestions: list[str],
    risk_level: str,
) -> str:
    """Claude API로 리스크 평가를 수행한다."""
    import json

    from app.core.llm import chat

    user_content = f"전략 분석 요청:\n```json\n{json.dumps(strategy, ensure_ascii=False, indent=2)}\n```\n\n"
    user_content += f"규칙 기반 평가:\n- 리스크 등급: {risk_level}\n"
    if issues:
        user_content += f"- 이슈: {', '.join(issues)}\n"
    user_content += "\n위 전략의 리스크를 평가하고, 구체적인 개선 제안을 해주세요."

    response = await chat(
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )

    return response.content[0].text
