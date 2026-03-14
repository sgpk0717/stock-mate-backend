"""Anthropic Claude API 프로바이더.

기존 app/core/llm.py 로직을 그대로 이동.
chat() 반환 타입은 anthropic.types.Message 유지 (Tool Use 호환).
"""

from __future__ import annotations

import anthropic
import httpx

from app.core.config import settings
from app.core.llm._types import LLMProvider, LLMResponse

_client: anthropic.AsyncAnthropic | None = None


def get_client() -> anthropic.AsyncAnthropic:
    """싱글턴 AsyncAnthropic 클라이언트 반환."""
    global _client
    if _client is None:
        api_key = settings.ANTHROPIC_API_KEY
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY가 설정되지 않았습니다. "
                ".env 파일에 ANTHROPIC_API_KEY=sk-ant-... 를 추가하세요."
            )
        _client = anthropic.AsyncAnthropic(
            api_key=api_key,
            timeout=httpx.Timeout(60.0, connect=10.0),
        )
    return _client


async def chat(
    *,
    messages: list[dict],
    system: str | None = None,
    max_tokens: int = 4000,
    tools: list | None = None,
) -> anthropic.types.Message:
    """Claude messages.create 래퍼.

    모델은 settings.AGENT_MODEL을 자동 사용한다.
    Tool Use 호출자(manager.py, factor_chat.py)를 위해
    반환 타입을 anthropic.types.Message로 유지.
    """
    client = get_client()
    kwargs: dict = {
        "model": settings.AGENT_MODEL,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system is not None:
        kwargs["system"] = system
    if tools is not None:
        kwargs["tools"] = tools
    return await client.messages.create(**kwargs)


async def chat_simple(
    *,
    messages: list[dict],
    system: str | None = None,
    max_tokens: int = 4000,
) -> LLMResponse:
    """chat() 래퍼 — 프로바이더 독립적 LLMResponse 반환.

    텍스트만 필요한 호출자용 (Tool Use 불필요).
    """
    response = await chat(
        messages=messages,
        system=system,
        max_tokens=max_tokens,
    )
    return LLMResponse(
        text=response.content[0].text,
        model=response.model,
        provider=LLMProvider.ANTHROPIC,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )
