"""일일 배치 스케줄러 REST 엔드포인트."""

from __future__ import annotations

from fastapi import APIRouter

from app.scheduler.daily_scheduler import get_daily_scheduler
from app.scheduler.schemas import TriggerRequest, TriggerResponse

router = APIRouter(prefix="/scheduler", tags=["scheduler"])


@router.get("/status")
async def scheduler_status():
    """스케줄러 전체 상태 + 잡별 진행률."""
    return get_daily_scheduler().get_status()


@router.post("/start")
async def scheduler_start():
    """자동 스케줄링 시작."""
    ok = await get_daily_scheduler().start()
    if ok:
        return {"message": "스케줄러 시작됨"}
    return {"message": "이미 실행 중"}


@router.post("/stop")
async def scheduler_stop():
    """자동 스케줄링 중지."""
    ok = await get_daily_scheduler().stop()
    if ok:
        return {"message": "스케줄러 중지됨"}
    return {"message": "실행 중이 아님"}


@router.post("/trigger", response_model=TriggerResponse)
async def scheduler_trigger(req: TriggerRequest):
    """수동 트리거. 스케줄러 중지 상태에서도 사용 가능."""
    triggered, message = await get_daily_scheduler().trigger_job(
        job_name=req.job, date=req.date,
    )
    return TriggerResponse(triggered=triggered, message=message)
