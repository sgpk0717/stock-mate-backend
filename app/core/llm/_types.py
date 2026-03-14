"""LLM 프로바이더 공통 타입."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class LLMProvider(Enum):
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


@dataclass
class LLMResponse:
    """프로바이더에 독립적인 LLM 응답."""

    text: str
    model: str
    provider: LLMProvider
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
