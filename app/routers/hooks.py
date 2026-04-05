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

# OpenClaw jobId → jobName 매핑 (evt에 jobName이 없으므로)
_JOB_ID_MAP: dict[str, str] = {
    "39e1e9ba-9ed0-4e50-b577-51c235946ef1": "morning_brief",
    "8a914e56-708d-418a-b4d2-7059568c4e0b": "pre_market_check",
    "2365e0be-2ba3-4836-b81a-b9f306319243": "midday_check",
    "91382f28-6c48-43e5-a4cc-c686e95844d5": "post_market_analysis",
    "6a8ff852-06f1-4497-bde5-934dbbd953e9": "mining_start_check",
    "8fd7cb3c-634e-48e2-b711-396ebd21c6df": "mining_review",
    "c2572226-0967-4d0d-8b9d-84e2989bfa68": "project_improvement",
    "cee80b37-8c70-4f79-bdef-c7b3b43b7f18": "overnight_check",
}


class CronWebhookPayload(BaseModel):
    """OpenClaw 크론잡 완료 webhook payload.

    OpenClaw는 evt 객체 전체를 payload로 전달한다.
    주요 필드: jobId, jobName, status, summary, durationMs, model, usage 등.
    summary에 에이전트 응답 텍스트가 담긴다.
    """
    model_config = {"extra": "allow"}

    jobId: str = ""
    jobName: str = ""
    status: str = ""  # ok | error
    summary: str = ""
    durationMs: int = 0
    model: str = ""
    # 하위 호환: 이전 포맷
    agentResult: dict[str, Any] = {}


@router.post("/openclaw-cron")
async def receive_openclaw_cron(payload: CronWebhookPayload) -> dict:
    """OpenClaw 크론잡 완료 시 텔레그램 발송 + DB 로그 기록."""
    # jobName 또는 jobId→name 매핑으로 job name 결정
    job_name = payload.jobName or _JOB_ID_MAP.get(payload.jobId, "")
    status = payload.status
    # OpenClaw 실제 포맷: summary 필드에 텍스트
    # 하위 호환: agentResult.text 폴백
    agent_text = (
        payload.summary.strip()
        or payload.agentResult.get("text", "").strip()
    )

    logger.info(
        "OpenClaw webhook 수신: job=%s status=%s text_len=%d",
        job_name, status, len(agent_text),
    )

    if not agent_text:
        logger.warning("OpenClaw webhook: summary 비어있음 (job=%s)", job_name)
        return {"received": True, "sent": False, "reason": "empty_text"}

    # jobName → category/caller 매핑
    category, caller = _JOB_MAP.get(
        job_name,
        (f"openclaw_{job_name}", f"openclaw.cron.{job_name}"),
    )

    # 실패/타임아웃 시 prefix 추가
    if status in ("failure", "error"):
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
