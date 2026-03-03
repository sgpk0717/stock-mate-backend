"""MCP 거버넌스 — 도구 호출 전/후 검증 + 감사 로깅."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

from app.core.database import async_session
from app.simulation.models import McpAuditLog

logger = logging.getLogger(__name__)


@dataclass
class GovernanceRules:
    max_order_qty: int = 1000
    allowed_actions: list[str] = field(default_factory=lambda: ["BUY", "SELL"])
    require_human_approval_real: bool = True
    enabled: bool = True


# Module-level singleton
_rules = GovernanceRules()


def get_rules() -> GovernanceRules:
    return _rules


def update_rules(**kwargs) -> GovernanceRules:
    global _rules
    for k, v in kwargs.items():
        if hasattr(_rules, k) and v is not None:
            setattr(_rules, k, v)
    return _rules


class GovernanceCheck:
    """MCP 도구 호출 전 검증."""

    @staticmethod
    def pre_check(tool_name: str, params: dict) -> tuple[bool, str | None]:
        """Returns (allowed, blocked_reason)."""
        rules = get_rules()
        if not rules.enabled:
            return True, None

        if tool_name == "execute_order":
            qty = params.get("qty", 0)
            if qty > rules.max_order_qty:
                return False, f"수량 {qty}가 최대 허용치 {rules.max_order_qty}을 초과"

            action = str(params.get("action", "")).upper()
            if action not in rules.allowed_actions:
                return False, f"액션 '{action}'은 허용되지 않음 (허용: {rules.allowed_actions})"

            mode = params.get("mode", "paper")
            if mode == "real" and rules.require_human_approval_real:
                return False, "실전 모드 주문은 사람 승인 필요"

        return True, None


async def audit_log(
    tool_name: str,
    input_params: dict,
    output: dict | None,
    status: str,
    blocked_reason: str | None = None,
    execution_ms: int | None = None,
) -> None:
    """MCP 감사 로그 DB 기록."""
    try:
        async with async_session() as db:
            log = McpAuditLog(
                tool_name=tool_name,
                input_params=input_params,
                output=output,
                status=status,
                blocked_reason=blocked_reason,
                execution_ms=execution_ms,
            )
            db.add(log)
            await db.commit()
    except Exception as e:
        logger.warning("Failed to write MCP audit log: %s", e)
