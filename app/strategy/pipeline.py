"""전략 파이프라인 — 시그널 필터 체인.

팩터 시그널(1/-1/0)을 받아 전략 필터를 순차 적용한 후
실행 여부를 결정한다. 리스크 관리(손절/트레일링)는 건드리지 않는다.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from app.core.config import settings
from .filters import time_filter, volume_filter, trade_limit_filter

logger = logging.getLogger(__name__)

FilterFn = Callable[[int, dict, Any, dict], dict[str, Any]]


class StrategyPipeline:
    """시그널 필터 체인.

    필터는 등록된 순서대로 실행되며, 하나라도 skip=True를 반환하면
    이후 필터는 건너뛴다.
    """

    def __init__(self) -> None:
        self.filters: list[FilterFn] = []
        self.state: dict[str, Any] = {}
        self._enabled = settings.STRATEGY_PIPELINE_ENABLED

    @classmethod
    def default(cls) -> "StrategyPipeline":
        """기본 필터 체인 생성 (시간 → 거래량 → 매매 횟수)."""
        pipe = cls()
        pipe.filters = [
            time_filter,
            volume_filter,
            trade_limit_filter,
        ]
        return pipe

    def evaluate(
        self,
        signal: int,
        row: dict[str, Any],
        context: Any,
    ) -> dict[str, Any]:
        """시그널 평가. 동기 함수 (필터들이 I/O 없음).

        Returns:
            {"skip": False, "signal": signal}  — 통과
            {"skip": True, "filter": ..., "reason": ...}  — 거부
        """
        if not self._enabled:
            return {"skip": False, "signal": signal}

        for fn in self.filters:
            try:
                result = fn(signal, row, context, self.state)
            except Exception as e:
                logger.warning("필터 %s 예외: %s — 통과 처리", fn.__name__, e)
                continue

            if result.get("skip"):
                return result

        return {"skip": False, "signal": signal}

    def record_buy(self) -> None:
        """매수 체결 기록 (일일 카운터 증가)."""
        self.state["daily_buy_count"] = self.state.get("daily_buy_count", 0) + 1

    def reset_daily(self) -> None:
        """일일 상태 초기화 (매일 장 시작 시 호출)."""
        self.state["daily_buy_count"] = 0

    def get_stats(self) -> dict[str, Any]:
        """현재 파이프라인 상태 반환."""
        return {
            "enabled": self._enabled,
            "filter_count": len(self.filters),
            "daily_buy_count": self.state.get("daily_buy_count", 0),
            "filters": [fn.__name__ for fn in self.filters],
        }
