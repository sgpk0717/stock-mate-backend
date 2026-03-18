"""텔레그램 로그 Pydantic 스키마."""

from __future__ import annotations

from pydantic import BaseModel


class TelegramLogResponse(BaseModel):
    id: str
    category: str
    caller: str
    text: str
    chat_id: str
    status: str
    error_message: str | None = None
    telegram_message_id: int | None = None
    created_at: str


class TelegramLogRequest(BaseModel):
    """외부(OpenClaw 등)에서 발송한 텔레그램 메시지 기록 요청."""

    text: str
    category: str = "openclaw"
    caller: str = "openclaw"
    telegram_message_id: int | None = None
