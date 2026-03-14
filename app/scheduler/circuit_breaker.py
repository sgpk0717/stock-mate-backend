"""외부 API 서킷 브레이커.

CLOSED → OPEN → HALF_OPEN 상태 전이로 장애 전파를 차단한다.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


class CircuitBreakerOpen(Exception):
    """서킷이 OPEN 상태일 때 호출 시 발생."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"Circuit breaker '{name}' is OPEN")


class CircuitBreaker:
    """비동기 서킷 브레이커.

    Args:
        name: 식별 이름 (로그용).
        failure_threshold: 연속 실패 N회 시 OPEN.
        reset_timeout: OPEN 후 대기 시간(초) → HALF_OPEN 전환.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
    ) -> None:
        self.name = name
        self._failure_threshold = failure_threshold
        self._reset_timeout = reset_timeout
        self._state: str = "CLOSED"
        self._failure_count: int = 0
        self._last_failure_at: float = 0.0

    @property
    def state(self) -> str:
        return self._state

    async def call(self, func, *args, **kwargs):
        """func를 서킷 브레이커 안에서 실행.

        Raises:
            CircuitBreakerOpen: OPEN 상태이고 reset_timeout 미경과.
        """
        if self._state == "OPEN":
            if time.monotonic() - self._last_failure_at > self._reset_timeout:
                self._state = "HALF_OPEN"
                logger.info(
                    "Circuit '%s' → HALF_OPEN (reset_timeout=%.0fs 경과)",
                    self.name,
                    self._reset_timeout,
                )
            else:
                raise CircuitBreakerOpen(self.name)

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception:
            self._on_failure()
            raise

    def _on_success(self) -> None:
        if self._state == "HALF_OPEN":
            logger.info("Circuit '%s' → CLOSED (복구 확인)", self.name)
        self._failure_count = 0
        self._state = "CLOSED"

    def _on_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_at = time.monotonic()
        if self._failure_count >= self._failure_threshold:
            self._state = "OPEN"
            logger.warning(
                "Circuit '%s' → OPEN (%d회 연속 실패, %.0fs 대기)",
                self.name,
                self._failure_count,
                self._reset_timeout,
            )

    def reset(self) -> None:
        """수동 리셋."""
        self._state = "CLOSED"
        self._failure_count = 0
