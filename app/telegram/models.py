"""텔레그램 메시지 발송 로그 DB 모델."""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class TelegramMessageLog(Base):
    __tablename__ = "telegram_message_logs"
    __table_args__ = (
        Index("ix_telegram_logs_category", "category"),
        Index("ix_telegram_logs_created", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    category: Mapped[str] = mapped_column(
        String(50), nullable=False, server_default="'system'"
    )
    caller: Mapped[str] = mapped_column(String(100), nullable=False, server_default="''")
    text: Mapped[str] = mapped_column(Text, nullable=False)
    chat_id: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default="'success'"
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    telegram_message_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
