"""텔레그램 발송 내역 조회 API."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.telegram.models import TelegramMessageLog
from app.telegram.schemas import TelegramLogRequest, TelegramLogResponse

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


@router.post("/log", response_model=dict)
async def log_external_message(
    body: TelegramLogRequest,
    db: AsyncSession = Depends(get_db),
):
    """외부(OpenClaw 등)에서 발송한 텔레그램 메시지를 DB에 기록.

    OpenClaw 크론잡 등에서 텔레그램 발송 후 이 엔드포인트를 호출하여
    telegram_message_logs 테이블에 히스토리를 남긴다.
    """
    from app.core.config import settings

    log = TelegramMessageLog(
        category=body.category,
        caller=body.caller,
        text=body.text[:4000],
        chat_id=settings.TELEGRAM_CHAT_ID or "",
        status="success",
        telegram_message_id=body.telegram_message_id,
    )
    db.add(log)
    await db.commit()
    return {"ok": True, "id": str(log.id)}
