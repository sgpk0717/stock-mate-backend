"""대화 세션 관리.

메모리 dict 기반, TTL 자동 만료.
향후 Redis 전환 가능.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class ChatMessage:
    """단일 대화 메시지."""

    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    tool_use: dict[str, Any] | None = None
    strategy_draft: dict[str, Any] | None = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
        }
        if self.tool_use:
            d["tool_use"] = self.tool_use
        if self.strategy_draft:
            d["strategy_draft"] = self.strategy_draft
        return d


@dataclass
class Session:
    """대화 세션."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: list[ChatMessage] = field(default_factory=list)
    strategy_draft: dict[str, Any] | None = None
    status: str = "active"  # "active" | "finalized" | "expired"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "session_id": self.id,
            "messages": [m.to_dict() for m in self.messages],
            "strategy_draft": self.strategy_draft,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class SessionStore:
    """메모리 기반 세션 저장소 (TTL 자동 만료)."""

    def __init__(self, ttl_minutes: int = 30) -> None:
        self._sessions: dict[str, Session] = {}
        self._ttl_minutes = ttl_minutes

    def create(self) -> Session:
        session = Session()
        self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        session = self._sessions.get(session_id)
        if session is None:
            return None
        if self._is_expired(session):
            session.status = "expired"
            del self._sessions[session_id]
            return None
        return session

    def delete(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    def cleanup_expired(self) -> int:
        """만료된 세션을 정리하고 삭제된 개수를 반환한다."""
        expired_ids = [
            sid
            for sid, s in self._sessions.items()
            if self._is_expired(s)
        ]
        for sid in expired_ids:
            del self._sessions[sid]
        return len(expired_ids)

    def _is_expired(self, session: Session) -> bool:
        updated = datetime.fromisoformat(session.updated_at)
        now = datetime.now(timezone.utc)
        return (now - updated).total_seconds() > self._ttl_minutes * 60


# 싱글톤 인스턴스 (main.py에서 TTL 설정 가능)
session_store = SessionStore()
