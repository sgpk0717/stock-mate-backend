"""Redis 연결 풀 + IPC 래퍼.

서비스 분리 시 API ↔ Worker ↔ MCP 간 통신에 사용.
모든 쓰기 함수: 3회 재시도 + WARNING 로깅 (silent failure 방지).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import redis.asyncio as aioredis

from app.core.config import settings

logger = logging.getLogger(__name__)

_pool: aioredis.ConnectionPool | None = None

# 재시도 설정
_MAX_RETRIES = 3
_RETRY_DELAYS = (0.1, 0.3, 0.7)  # 지수 백오프 (총 ~1.1초)


def get_pool() -> aioredis.ConnectionPool:
    """싱글턴 Redis 연결 풀."""
    global _pool
    if _pool is None:
        _pool = aioredis.ConnectionPool.from_url(
            settings.REDIS_URL,
            max_connections=20,
            decode_responses=True,
            socket_connect_timeout=10,
            socket_timeout=10,
        )
    return _pool


def get_client() -> aioredis.Redis:
    """Redis 클라이언트 (연결 풀 사용)."""
    return aioredis.Redis(connection_pool=get_pool())


async def close() -> None:
    """연결 풀 종료 (lifespan shutdown)."""
    global _pool
    if _pool:
        await _pool.aclose()
        _pool = None


async def _retry_write(op_name: str, coro_factory, *args, **kwargs) -> bool:
    """Redis 쓰기 연산 재시도 (3회, 지수 백오프).

    Returns:
        True = 성공, False = 모든 재시도 실패
    """
    last_err = None
    for attempt, delay in enumerate(_RETRY_DELAYS, 1):
        try:
            await coro_factory(*args, **kwargs)
            return True
        except (aioredis.ConnectionError, aioredis.TimeoutError,
                ConnectionRefusedError, OSError) as e:
            last_err = e
            if attempt < _MAX_RETRIES:
                await asyncio.sleep(delay)
        except Exception as e:
            # 직렬화 에러 등 재시도 무의미한 에러
            logger.warning("Redis %s 실패 (재시도 불가): %s", op_name, e)
            return False

    logger.warning(
        "Redis %s 실패 (%d회 재시도 소진): %s",
        op_name, _MAX_RETRIES, last_err,
    )
    return False


# ── 상태 캐시 (Hash) ──


async def hset(key: str, mapping: dict[str, Any], ttl: int | None = None) -> bool:
    """Redis Hash에 상태 저장 (재시도 포함).

    Args:
        key: Redis 키
        mapping: 저장할 데이터
        ttl: TTL 초 (None이면 TTL 미설정)

    Returns:
        True = 성공, False = 실패
    """
    r = get_client()
    # dict 값을 문자열로 직렬화
    serialized = {}
    for k, v in mapping.items():
        if isinstance(v, (dict, list)):
            serialized[k] = json.dumps(v, ensure_ascii=False, default=str)
        elif v is None:
            serialized[k] = ""
        else:
            serialized[k] = str(v)

    async def _do():
        await r.hset(key, mapping=serialized)
        if ttl:
            await r.expire(key, ttl)

    return await _retry_write(f"hset({key})", _do)


async def hgetall(key: str) -> dict[str, str]:
    """Redis Hash 전체 읽기."""
    r = get_client()
    try:
        return await r.hgetall(key)
    except Exception as e:
        logger.warning("Redis hgetall 실패 (%s): %s", key, e)
        return {}


async def delete(key: str) -> None:
    """Redis 키 삭제."""
    r = get_client()
    try:
        await r.delete(key)
    except Exception as e:
        logger.warning("Redis delete 실패 (%s): %s", key, e)


# ── 판단 로그 (List) ──


async def rpush(key: str, *values: str) -> int | None:
    """Redis List에 엔트리 추가 (RPUSH)."""
    r = get_client()
    try:
        return await r.rpush(key, *values)
    except Exception as e:
        logger.warning("Redis rpush 실패 (%s): %s", key, e)
        return None


async def lrange(key: str, start: int = 0, end: int = -1) -> list[str]:
    """Redis List 범위 조회 (LRANGE)."""
    r = get_client()
    try:
        return await r.lrange(key, start, end)
    except Exception as e:
        logger.warning("Redis lrange 실패 (%s): %s", key, e)
        return []


# ── 명령 큐 (Stream) ──


async def xadd(stream: str, fields: dict[str, str], maxlen: int | None = None) -> str | None:
    """Redis Stream에 메시지 추가."""
    r = get_client()
    try:
        kwargs: dict = {}
        if maxlen:
            kwargs["maxlen"] = maxlen
            kwargs["approximate"] = True
        return await r.xadd(stream, fields, **kwargs)
    except Exception as e:
        logger.warning("Redis xadd 실패 (%s): %s", stream, e)
        return None


async def xread(streams: dict[str, str], count: int = 10, block: int = 5000) -> list:
    """Redis Stream에서 메시지 읽기 (block ms 대기)."""
    r = get_client()
    try:
        return await r.xread(streams, count=count, block=block)
    except Exception as e:
        logger.debug("Redis xread 실패: %s", e)
        return []


async def ensure_consumer_group(stream: str, group: str) -> None:
    """Consumer Group 생성 (이미 있으면 무시, stream 없으면 자동 생성)."""
    r = get_client()
    try:
        await r.xgroup_create(stream, group, id="0", mkstream=True)
    except Exception as e:
        if "BUSYGROUP" not in str(e):
            logger.debug("Redis xgroup_create 실패 (%s/%s): %s", stream, group, e)


async def xreadgroup(
    group: str, consumer: str, streams: dict[str, str],
    count: int = 10, block: int = 5000,
) -> list:
    """Consumer Group에서 메시지 읽기."""
    r = get_client()
    try:
        return await r.xreadgroup(group, consumer, streams, count=count, block=block)
    except Exception as e:
        logger.debug("Redis xreadgroup 실패: %s", e)
        return []


async def xack(stream: str, group: str, *msg_ids: str) -> int:
    """메시지 처리 완료 ACK."""
    r = get_client()
    try:
        return await r.xack(stream, group, *msg_ids)
    except Exception as e:
        logger.debug("Redis xack 실패: %s", e)
        return 0


# ── Pub/Sub ──


async def publish(channel: str, message: dict | str) -> None:
    """Redis 채널에 메시지 발행."""
    r = get_client()
    try:
        data = json.dumps(message, ensure_ascii=False, default=str) if isinstance(message, dict) else message
        await r.publish(channel, data)
    except Exception as e:
        logger.warning("Redis publish 실패 (%s): %s", channel, e)


# ── 유틸 ──


async def ping() -> bool:
    """Redis 연결 확인."""
    r = get_client()
    try:
        return await r.ping()
    except Exception:
        return False
