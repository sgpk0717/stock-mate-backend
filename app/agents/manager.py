"""Manager Agent — Claude Tool Use 기반 다중 턴 대화형 전략 수립.

사용자와 여러 번 대화하며 매매 전략을 점진적으로 구체화한다.
누락된 조건을 되묻고, 파라미터를 확인하며, 최종 StrategySchema를 생성한다.
"""

from __future__ import annotations

import logging
from typing import Any

from app.agents.session import ChatMessage, Session
from app.agents.tools import ASYNC_TOOLS, TOOLS, execute_tool
from app.core.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
당신은 주식 트레이딩 전략 설계를 도와주는 전문 어시스턴트입니다.

## 역할
- 사용자의 자연어 설명을 듣고, 정교한 매매 전략(StrategySchema)을 함께 만들어갑니다.
- 사용자가 불완전하게 설명하면, 누락된 조건이나 파라미터를 되물어 확인합니다.
- 단번에 전략을 만들지 말고, 대화를 통해 점진적으로 구체화하세요.

## 대화 원칙
1. 사용자의 의도를 먼저 파악하세요 (어떤 스타일의 매매인지: 스윙/단타/추세추종 등).
2. 매수 조건과 매도 조건을 반드시 모두 확인하세요.
3. 파라미터 값이 명시되지 않으면 기본값을 제안하고 확인받으세요.
4. 사용자가 "분할매수", "손절", "트레일링" 등을 언급하면 고급 설정을 함께 구성하세요.
5. 조건이 충분히 모이면 draft_strategy 도구로 전략 초안을 생성하고, 사용자에게 보여주세요.
6. 사용자가 수정을 요청하면 반영하여 다시 초안을 생성하세요.
7. 사용자가 "확정", "이대로", "좋아" 등 확인을 하면, 최종 전략임을 안내하세요.

## 응답 스타일
- 친절하고 간결하게 응답하세요.
- 전문 용어를 사용하되, 필요 시 쉽게 설명해주세요.
- 전략 초안을 보여줄 때는 조건을 읽기 쉽게 정리해서 설명하세요.

## bool 지표 규칙
golden_cross, dead_cross, macd_cross 같은 bool 지표는 op="==", value=1 로 설정합니다.

## 포지션 사이징 모드
- fixed: 고정 비중 (기본값, 별도 설정 불필요)
- conviction: 조건별 가중치를 부여하여 확신도가 높을수록 큰 포지션
- atr_target: ATR 기반, 변동성이 작을수록 큰 포지션
- kelly: 켈리 공식 기반 (실험적)

## 분할매매 설정
- initial_pct: 1차 진입 비율 (0.5 = 50%만 1차 매수)
- scale_in_drop_pct: 추가매수 트리거 (평단 대비 N% 하락 시)
- max_scale_in: 최대 추가매수 횟수
- partial_exit_pct: 부분익절 시 매도 비율
- partial_exit_gain_pct: 부분익절 트리거 (평단 대비 N% 수익 시)

## 리스크 관리
- stop_loss_pct: 고정 손절 (평단 대비 N% 하락 시 전량 매도)
- trailing_stop_pct: 고점 대비 N% 하락 시 전량 매도 (수익 보호)
- atr_stop_multiplier: ATR × N 만큼 평단 아래로 가면 손절

## 전문 에이전트 활용
- ask_technical_analyst: 특정 종목의 기술적 분석이 필요할 때 호출하세요.
- ask_risk_manager: 전략의 리스크를 평가하고 개선 사항을 확인할 때 호출하세요.
  특히 손절이 없는 전략을 만들었을 때 자동으로 리스크 검토를 하세요.
- search_sector_stocks: 사용자가 "반도체", "2차전지" 등 테마/섹터를 언급하면 관련 종목을 검색하세요.

## 뉴스 감성 지표
- sentiment_score: 뉴스 감성 스코어 (-1.0~+1.0)
- article_count: 뉴스 기사 수
- event_score: 이벤트 스코어 (감성×영향력×보도량 종합)
이 지표들은 뉴스 데이터가 수집된 종목에서만 사용 가능합니다.
"""

# 최대 tool use 재귀 깊이
MAX_TOOL_ROUNDS = 5


async def process_message(
    session: Session,
    user_message: str,
) -> ChatMessage:
    """사용자 메시지를 처리하고 에이전트 응답을 반환한다.

    내부적으로 Claude Tool Use 루프를 실행한다.
    """
    from app.core.llm import chat

    # 사용자 메시지를 세션에 추가
    user_msg = ChatMessage(role="user", content=user_message)
    session.messages.append(user_msg)
    session.touch()

    # Anthropic 메시지 형식으로 변환
    api_messages = _build_api_messages(session)

    max_tokens = getattr(settings, "AGENT_MAX_TOKENS", 4000)

    # Tool Use 루프
    for _round in range(MAX_TOOL_ROUNDS):
        response = await chat(
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=api_messages,
        )

        # 응답의 content 블록 처리
        assistant_text = ""
        tool_uses: list[dict[str, Any]] = []
        strategy_draft: dict[str, Any] | None = None

        for block in response.content:
            if block.type == "text":
                assistant_text += block.text
            elif block.type == "tool_use":
                tool_uses.append({
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

        if not tool_uses:
            # tool 호출 없이 텍스트만 반환 → 대화 응답 완료
            assistant_msg = ChatMessage(
                role="assistant",
                content=assistant_text,
                strategy_draft=session.strategy_draft,
            )
            session.messages.append(assistant_msg)
            session.touch()
            return assistant_msg

        # tool 호출 처리
        # 먼저 assistant 메시지 추가 (tool_use 포함)
        api_messages.append({"role": "assistant", "content": response.content})

        # 각 tool 실행 결과를 user 메시지로 추가
        tool_results = []
        for tu in tool_uses:
            if tu["name"] in ASYNC_TOOLS:
                # 비동기 도구 (orchestrator 경유)
                from app.agents.orchestrator import ASYNC_TOOL_HANDLERS
                handler = ASYNC_TOOL_HANDLERS.get(tu["name"])
                if handler:
                    result_str = await handler(tu["input"])
                else:
                    result_str = '{"error": "Unknown async tool"}'
            else:
                result_str = execute_tool(tu["name"], tu["input"])

            # draft_strategy의 성공 결과를 세션에 저장
            if tu["name"] == "draft_strategy":
                import json
                result_data = json.loads(result_str)
                if result_data.get("status") == "success":
                    session.strategy_draft = result_data.get("strategy")
                    strategy_draft = session.strategy_draft

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu["id"],
                "content": result_str,
            })

        api_messages.append({"role": "user", "content": tool_results})

    # MAX_TOOL_ROUNDS 초과 시
    fallback_text = assistant_text or "전략 수립 중 문제가 발생했습니다. 다시 시도해주세요."
    assistant_msg = ChatMessage(
        role="assistant",
        content=fallback_text,
        strategy_draft=session.strategy_draft,
    )
    session.messages.append(assistant_msg)
    session.touch()
    return assistant_msg


def _build_api_messages(session: Session) -> list[dict[str, Any]]:
    """세션의 메시지 히스토리를 Anthropic API 형식으로 변환한다."""
    messages: list[dict[str, Any]] = []
    for msg in session.messages:
        if msg.role in ("user", "assistant"):
            messages.append({
                "role": msg.role,
                "content": msg.content,
            })
    return messages
