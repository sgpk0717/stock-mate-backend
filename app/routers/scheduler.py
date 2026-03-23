"""일일 배치 스케줄러 REST 엔드포인트."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException

from app.scheduler.daily_scheduler import get_daily_scheduler
from app.scheduler.schemas import (
    ActiveJob,
    ManualTriggerRequest,
    ManualTriggerResponse,
    TriggerRequest,
    TriggerResponse,
)

logger = logging.getLogger(__name__)

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


# ── 수동 수집 (ManualJobRunner) ──


@router.post("/collect", response_model=ManualTriggerResponse)
async def scheduler_collect(req: ManualTriggerRequest):
    """수동 데이터 수집 트리거. 날짜/기간 지정 가능."""
    from app.core.config import settings
    from app.scheduler.manual_runner import VALID_COLLECTORS, _resolve_dates

    # 공통 검증 (모드 무관)
    if req.collector not in VALID_COLLECTORS:
        raise HTTPException(400, f"알 수 없는 수집기: {req.collector}")
    try:
        dates = _resolve_dates(req)
    except ValueError as e:
        raise HTTPException(400, str(e))

    # 중복 실행 체크 (Redis에서 활성 잡 확인)
    existing_jobs = await _read_jobs_from_redis()
    for ej in existing_jobs:
        if ej.collector == req.collector and ej.status in ("running", "cancelling"):
            raise HTTPException(
                409,
                f"'{req.collector}' 수집이 이미 진행 중입니다 (job_id={ej.job_id[:8]})",
            )

    if settings.WORKER_MODE == "external":
        # Worker가 별도 프로세스 → Redis Stream으로 명령 전달
        from app.core.redis import xadd
        fields: dict[str, str] = {
            "action": "start_collect",
            "collector": req.collector,
            "mode": req.mode,
        }
        if req.date:
            fields["date"] = req.date
        if req.date_from:
            fields["date_from"] = req.date_from
        if req.date_to:
            fields["date_to"] = req.date_to
        if req.recent_days is not None:
            fields["recent_days"] = str(req.recent_days)

        msg_id = await xadd("commands:scheduler", fields, maxlen=100)
        if not msg_id:
            raise HTTPException(500, "Redis 명령 전송 실패")

        return ManualTriggerResponse(
            job_id="pending",
            collector=req.collector,
            dates=dates,
            message="수집 명령 전송됨 (Worker에서 실행)",
        )

    # inline/worker 모드: 직접 실행
    from app.scheduler.manual_runner import get_manual_runner
    runner = get_manual_runner()
    try:
        job = await runner.start_job(req)
    except ValueError as e:
        raise HTTPException(400, str(e))

    return ManualTriggerResponse(
        job_id=job.job_id,
        collector=job.collector,
        dates=job.dates,
        message="수집 시작됨",
    )


@router.get("/jobs", response_model=list[ActiveJob])
async def scheduler_jobs():
    """활성 + 최근 완료 수집 작업 목록."""
    from app.core.config import settings

    if settings.WORKER_MODE == "external":
        # Redis에서 읽기
        return await _read_jobs_from_redis()

    from app.scheduler.manual_runner import get_manual_runner
    return get_manual_runner().list_jobs()


@router.post("/jobs/{job_id}/cancel")
async def scheduler_cancel_job(job_id: str):
    """수집 작업 중단."""
    from app.core.config import settings

    if settings.WORKER_MODE == "external":
        from app.core.redis import xadd
        msg_id = await xadd("commands:scheduler", {
            "action": "cancel_job",
            "job_id": job_id,
        }, maxlen=100)
        if not msg_id:
            raise HTTPException(500, "Redis 명령 전송 실패")
        return {"success": True, "message": "중단 명령 전송됨"}

    from app.scheduler.manual_runner import get_manual_runner
    ok = await get_manual_runner().cancel_job(job_id)
    if not ok:
        raise HTTPException(404, "작업을 찾을 수 없거나 이미 완료됨")
    return {"success": True, "message": "중단 요청됨"}


async def _read_jobs_from_redis() -> list[ActiveJob]:
    """Redis에서 활성 잡 목록을 읽는다."""
    try:
        from app.core.redis import get_client, hgetall
        r = get_client()

        # active set에서 job_id 목록
        active_ids: set[str] = set()
        try:
            raw_ids = await r.smembers("scheduler:active_job_ids")
            active_ids = {mid.decode() if isinstance(mid, bytes) else mid for mid in raw_ids}
        except Exception:
            pass

        # 추가로 scheduler:jobs:* 키 스캔 (완료 잡 포함)
        try:
            cursor = 0
            while True:
                cursor, keys = await r.scan(cursor, match="scheduler:jobs:*", count=50)
                for key in keys:
                    k = key.decode() if isinstance(key, bytes) else key
                    jid = k.replace("scheduler:jobs:", "")
                    active_ids.add(jid)
                if cursor == 0:
                    break
        except Exception:
            pass

        jobs: list[ActiveJob] = []
        for jid in active_ids:
            raw = await hgetall(f"scheduler:jobs:{jid}")
            if not raw:
                continue
            try:
                dates_raw = raw.get("dates", "[]")
                dates = json.loads(dates_raw) if dates_raw else []
                jobs.append(ActiveJob(
                    job_id=raw.get("job_id", jid),
                    collector=raw.get("collector", ""),
                    status=raw.get("status", "unknown"),
                    dates=dates,
                    current_date=raw.get("current_date") or None,
                    date_progress=int(raw.get("date_progress", 0)),
                    date_total=int(raw.get("date_total", 0)),
                    total=int(raw.get("total", 0)),
                    completed=int(raw.get("completed", 0)),
                    failed=int(raw.get("failed", 0)),
                    started_at=raw.get("started_at") or None,
                    completed_at=raw.get("completed_at") or None,
                    error=raw.get("error") or None,
                    source=raw.get("source", "manual"),
                    logs=json.loads(raw["logs"]) if raw.get("logs") else [],
                ))
            except Exception as e:
                logger.debug("Redis 잡 파싱 실패 (%s): %s", jid, e)

        # started_at 역순 정렬
        jobs.sort(key=lambda j: j.started_at or "", reverse=True)
        return jobs

    except Exception as e:
        logger.debug("Redis 잡 목록 조회 실패: %s", e)
        return []
