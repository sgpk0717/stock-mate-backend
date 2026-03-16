"""텔레그램 발송 내역 조회 API."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.telegram.models import TelegramMessageLog
from app.telegram.schemas import TelegramLogResponse

router = APIRouter(prefix="/telegram", tags=["telegram"])


@router.get("/logs", response_model=list[TelegramLogResponse])
async def get_telegram_logs(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    category: str | None = Query(None),
    status: str | None = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """텔레그램 발송 내역 조회."""
    query = (
        select(TelegramMessageLog)
        .order_by(TelegramMessageLog.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    if category:
        query = query.where(TelegramMessageLog.category == category)
    if status:
        query = query.where(TelegramMessageLog.status == status)

    result = await db.execute(query)
    rows = result.scalars().all()
    return [
        TelegramLogResponse(
            id=str(r.id),
            category=r.category,
            caller=r.caller,
            text=r.text,
            chat_id=r.chat_id,
            status=r.status,
            error_message=r.error_message,
            telegram_message_id=r.telegram_message_id,
            created_at=r.created_at.isoformat() if r.created_at else "",
        )
        for r in rows
    ]
