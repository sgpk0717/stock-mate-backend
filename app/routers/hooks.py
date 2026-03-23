"""OpenClaw 크론잡 webhook 수신 엔드포인트.

크론잡 완료 시 OpenClaw가 POST로 결과를 전달하면,
백엔드가 텔레그램 발송 + DB 로그 기록을 100% 보장한다.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter(prefix="/hooks", tags=["hooks"])
logger = logging.getLogger(__name__)

# jobName → (category, caller) 매핑
_JOB_MAP: dict[str, tuple[str, str]] = {
    "morning_brief": ("openclaw_morning", "openclaw.cron.morning_brief"),
    "pre_market_check": ("openclaw_pre_market", "openclaw.cron.pre_market_check"),
    "midday_check": ("openclaw_midday", "openclaw.cron.midday_check"),
    "post_market_analysis": ("openclaw_post_market", "openclaw.cron.post_market_analysis"),
    "mining_start_check": ("openclaw_mining_start", "openclaw.cron.mining_start_check"),
    "mining_review": ("openclaw_mining", "openclaw.cron.mining_review"),
    "project_improvement": ("openclaw_improvement", "openclaw.cron.project_improvement"),
    "overnight_check": ("openclaw_overnight", "openclaw.cron.overnight_check"),
}


class CronWebhookPayload(BaseModel):
    """OpenClaw 크론잡 완료 webhook payload."""
    jobId: str = ""
    jobName: str = ""
    status: str = ""  # success | failure | timeout
    timestamp: str = ""
    agentResult: dict[str, Any] = {}
    executionTime: int = 0


@router.post("/openclaw-cron")
async def receive_openclaw_cron(payload: CronWebhookPayload) -> dict:
    """OpenClaw 크론잡 완료 시 텔레그램 발송 + DB 로그 기록."""
    job_name = payload.jobName
    status = payload.status
    agent_text = payload.agentResult.get("text", "").strip()

    logger.info(
        "OpenClaw webhook 수신: job=%s status=%s text_len=%d",
        job_name, status, len(agent_text),
    )

    if not agent_text:
        logger.warning("OpenClaw webhook: agentResult.text 비어있음 (job=%s)", job_name)
        return {"received": True, "sent": False, "reason": "empty_text"}

    # jobName → category/caller 매핑
    category, caller = _JOB_MAP.get(
        job_name,
        (f"openclaw_{job_name}", f"openclaw.cron.{job_name}"),
    )

    # 실패/타임아웃 시 prefix 추가
    if status == "failure":
        agent_text = f"[크론잡 실패] {job_name}\n\n{agent_text}"
    elif status == "timeout":
        agent_text = f"[크론잡 타임아웃] {job_name}\n\n{agent_text}"

    # 텔레그램 발송 + DB 로그 (send_message가 둘 다 처리)
    try:
        from app.telegram.bot import send_message
        await send_message(agent_text, category=category, caller=caller)
        logger.info("OpenClaw webhook → 텔레그램 발송 완료: job=%s", job_name)
        return {"received": True, "sent": True, "jobName": job_name}
    except Exception as e:
        logger.error("OpenClaw webhook → 텔레그램 발송 실패: %s", e)
        return {"received": True, "sent": False, "reason": str(e)[:200]}
