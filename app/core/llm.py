"""LLM 클라이언트 — Anthropic Claude API 호출 통합.

모든 Claude API 호출은 이 모듈의 get_client() 또는 chat()을 사용한다.
모델/API키 설정은 settings에서 중앙 관리.
"""

from __future__ import annotations

import anthropic
import httpx

from app.core.config import settings

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
