"""FastAPI ↔ MCP 서버 공존 — SSE transport 기반 브릿지."""

from __future__ import annotations

import asyncio
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

_mcp_task: asyncio.Task | None = None


async def start_mcp_server() -> None:
    """MCP 서버 백그라운드 시작 (SSE transport)."""
    global _mcp_task
    if not settings.MCP_ENABLED:
        logger.info("MCP server disabled (MCP_ENABLED=false)")
        return

    from app.mcp.server import mcp

    async def _run() -> None:
        try:
            await mcp.run_async(transport="sse", port=settings.MCP_SSE_PORT)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("MCP server error: %s", e)

    _mcp_task = asyncio.create_task(_run())
    logger.info("MCP SSE server started on port %d", settings.MCP_SSE_PORT)


async def stop_mcp_server() -> None:
    """MCP 서버 중지."""
    global _mcp_task
    if _mcp_task and not _mcp_task.done():
        _mcp_task.cancel()
        try:
            await _mcp_task
        except asyncio.CancelledError:
            pass
    _mcp_task = None
    logger.info("MCP server stopped")


def is_mcp_running() -> bool:
    """MCP 서버 실행 여부."""
    return _mcp_task is not None and not _mcp_task.done()
