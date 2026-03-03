"""Token Bucket Rate Limiter — asyncio 기반.

KIS API: 초당 15건 제한 준수.
"""

from __future__ import annotations

import asyncio
import time


class TokenBucket:
    """비동기 토큰 버킷.

    Args:
        rate: 초당 토큰 보충 속도 (기본 15)
        capacity: 최대 토큰 수 (기본 15)
    """

    def __init__(self, rate: float = 15.0, capacity: float = 15.0):
        self._rate = rate
        self._capacity = capacity
        self._tokens = capacity
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._last_refill = now

    async def acquire(self, tokens: float = 1.0) -> None:
        """토큰을 소비한다. 부족하면 대기."""
        async with self._lock:
            while True:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                # 필요한 토큰이 채워질 때까지 대기
                deficit = tokens - self._tokens
                wait_time = deficit / self._rate
                await asyncio.sleep(wait_time)


# KIS API 전역 Rate Limiter (15 req/s)
kis_rate_limiter = TokenBucket(rate=15.0, capacity=15.0)
