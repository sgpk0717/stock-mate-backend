"""LLM 사용량 비동기 로거.

모든 LLM 호출을 llm_usage_logs 테이블에 자동 기록한다.
asyncio.create_task()로 fire-and-forget 호출되므로 LLM 응답 지연에 영향 없음.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# 모델별 가격표 (USD per 1M tokens): (input, output)
_PRICE_TABLE: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-4-20250514": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-haiku-4-5": (1.0, 5.0),
    # Gemini
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.0),
    "gemini-3.1-flash-lite-preview": (0.25, 1.50),
    "gemini-3.1-pro-preview": (2.0, 12.0),
}


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """모델별 가격표로 비용 추정 (USD)."""
    prices = _PRICE_TABLE.get(model)
    if prices is None:
        # 부분 매칭 시도 (모델명에 버전 접미사가 붙는 경우)
        for key, val in _PRICE_TABLE.items():
            if key in model or model in key:
                prices = val
                break
    if prices is None:
        return None
    input_cost = input_tokens * prices[0] / 1_000_000
    output_cost = output_tokens * prices[1] / 1_000_000
    return round(input_cost + output_cost, 6)


async def log_llm_usage(
    *,
    caller: str,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    duration_ms: int | None = None,
    status: str = "success",
    error: str | None = None,
) -> None:
    """LLM 사용량을 DB에 비동기 기록.

    asyncio.create_task()로 호출되므로 예외가 발생해도
    원래 LLM 호출에 영향을 주지 않는다.
    """
    try:
        from app.core.database import async_session
        from app.core.llm._models import LLMUsageLog

        cost = _estimate_cost(model, input_tokens, output_tokens)
        total = input_tokens + output_tokens

        async with async_session() as session:
            log = LLMUsageLog(
                caller=caller,
                provider=provider,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total,
                cost_usd=cost,
                status=status,
                error=error,
                duration_ms=duration_ms,
            )
            session.add(log)
            await session.commit()
    except Exception:
        logger.debug("LLM usage log failed", exc_info=True)
