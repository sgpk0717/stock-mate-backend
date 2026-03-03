"""Phase 4: MCP 관리 라우터."""

from __future__ import annotations

import logging
from dataclasses import asdict

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.mcp.bridge import is_mcp_running
from app.mcp.governance import get_rules, update_rules
from app.simulation.models import McpAuditLog
from app.simulation.schemas import (
    GovernanceRulesUpdate,
    McpAuditLogResponse,
    McpStatusResponse,
    McpToolResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/mcp", tags=["mcp"])


@router.get("/status", response_model=McpStatusResponse)
async def mcp_status():
    """MCP 서버 상태."""
    return McpStatusResponse(
        running=is_mcp_running(),
        sse_port=settings.MCP_SSE_PORT,
        governance=asdict(get_rules()),
    )


@router.get("/tools", response_model=list[McpToolResponse])
async def mcp_tools():
    """등록된 MCP 도구 목록."""
    from app.mcp.server import mcp as mcp_server

    tools = []
    for tool in mcp_server._tool_manager.list_tools():
        tools.append(
            McpToolResponse(
                name=tool.name,
                description=tool.description or "",
            )
        )
    return tools


@router.get("/audit", response_model=list[McpAuditLogResponse])
async def mcp_audit(
    limit: int = 50,
    tool_name: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    """감사 로그 조회 (페이지네이션)."""
    query = (
        select(McpAuditLog)
        .order_by(McpAuditLog.created_at.desc())
        .limit(limit)
    )
    if tool_name:
        query = query.where(McpAuditLog.tool_name == tool_name)

    result = await db.execute(query)
    logs = result.scalars().all()

    return [
        McpAuditLogResponse(
            id=str(log.id),
            tool_name=log.tool_name,
            input_params=log.input_params,
            output=log.output,
            status=log.status,
            blocked_reason=log.blocked_reason,
            execution_ms=log.execution_ms,
            created_at=log.created_at.isoformat(),
        )
        for log in logs
    ]


@router.put("/governance", response_model=dict)
async def update_governance_rules(data: GovernanceRulesUpdate):
    """거버넌스 규칙 업데이트."""
    update_kwargs = {
        k: v for k, v in data.model_dump().items() if v is not None
    }
    rules = update_rules(**update_kwargs)
    return asdict(rules)
