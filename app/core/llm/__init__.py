"""LLM 클라이언트 통합 모듈.

기존 인터페이스 (하위호환):
    from app.core.llm import chat, get_client

새 인터페이스:
    from app.core.llm import chat_simple, chat_gemini
    from app.core.llm import LLMProvider, LLMResponse
"""

from app.core.llm._anthropic import chat, chat_simple, get_client
from app.core.llm._gemini import chat_gemini, get_gemini_client
from app.core.llm._types import LLMProvider, LLMResponse

__all__ = [
    # Anthropic (하위호환)
    "chat",
    "get_client",
    "chat_simple",
    # Gemini
    "chat_gemini",
    "get_gemini_client",
    # 타입
    "LLMProvider",
    "LLMResponse",
]
