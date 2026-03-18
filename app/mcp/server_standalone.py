"""MCP SSE 서버 독립 실행 엔트리포인트.

docker-compose의 stockmate-mcp 서비스에서 사용.
app 컨테이너와 별도로 실행되어, app 재시작 시에도 MCP 연결 유지.

사용법:
    python -m app.mcp.server_standalone
"""

from __future__ import annotations

import asyncio
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)
logger = logging.getLogger(__name__)


async def main() -> None:
    """MCP 서버 독립 실행."""
    from app.core.config import settings

    logger.info("MCP Standalone 서버 시작 (port=%d)", settings.MCP_SSE_PORT)

    # Redis 연결 확인
    try:
        from app.core.redis import ping
        ok = await ping()
        logger.info("Redis 연결: %s", "OK" if ok else "FAIL")
    except Exception as e:
        logger.warning("Redis 연결 실패: %s — 폴백 모드로 동작", e)

    # MCP 서버 실행
    from app.mcp.server import mcp
    await mcp.run_async(transport="sse", host="0.0.0.0", port=settings.MCP_SSE_PORT)


if __name__ == "__main__":
    asyncio.run(main())
