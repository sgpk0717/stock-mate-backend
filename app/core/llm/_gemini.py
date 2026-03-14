"""Google Gemini API 프로바이더.

google-genai SDK 사용 (레거시 google-generativeai 아님).
뉴스 감성분석 등 비용 민감한 배치 작업에 사용.
"""

from __future__ import annotations

import logging

from google import genai
from google.genai import types as genai_types

from app.core.config import settings
from app.core.llm._types import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)

_client: genai.Client | None = None


def get_gemini_client() -> genai.Client:
    """싱글턴 Gemini 클라이언트 반환."""
    global _client
    if _client is None:
        api_key = settings.GEMINI_API_KEY
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY가 설정되지 않았습니다. "
                ".env 파일에 GEMINI_API_KEY=... 를 추가하세요."
            )
        _client = genai.Client(api_key=api_key)
    return _client


async def chat_gemini(
    *,
    messages: list[dict],
    system: str | None = None,
    max_tokens: int = 2000,
    temperature: float | None = None,
    json_mode: bool = False,
    json_schema: dict | None = None,
) -> LLMResponse:
    """Gemini chat completion — LLMResponse 반환.

    Args:
        messages: [{"role": "user"|"assistant", "content": str}]
        system: 시스템 프롬프트 (config.system_instruction으로 전달)
        max_tokens: 최대 출력 토큰
        temperature: 생성 온도 (None이면 모델 기본값)
        json_schema: JSON 스키마 (설정 시 JSON 모드 활성화)
    """
    client = get_gemini_client()

    # 메시지 → Gemini contents 변환
    contents: list[genai_types.Content] = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(
            genai_types.Content(
                role=role,
                parts=[genai_types.Part(text=msg["content"])],
            )
        )

    # GenerateContentConfig 구성
    config_kwargs: dict = {"max_output_tokens": max_tokens}
    if temperature is not None:
        config_kwargs["temperature"] = temperature
    if system is not None:
        config_kwargs["system_instruction"] = system
    if json_schema is not None:
        config_kwargs["response_mime_type"] = "application/json"
        config_kwargs["response_json_schema"] = json_schema
    elif json_mode:
        config_kwargs["response_mime_type"] = "application/json"

    config = genai_types.GenerateContentConfig(**config_kwargs)

    response = await client.aio.models.generate_content(
        model=settings.GEMINI_MODEL,
        contents=contents,
        config=config,
    )

    # 토큰 사용량 추출
    usage = response.usage_metadata
    input_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
    output_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0

    return LLMResponse(
        text=response.text or "",
        model=settings.GEMINI_MODEL,
        provider=LLMProvider.GEMINI,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
