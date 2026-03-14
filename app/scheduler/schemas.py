"""일일 배치 스케줄러 Pydantic 스키마."""

from __future__ import annotations

from pydantic import BaseModel


class JobStatus(BaseModel):
    """개별 잡 진행 상태."""

    name: str
    status: str = "pending"  # pending | running | completed | failed | skipped
    total: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    last_symbol: str | None = None
    duration_seconds: float | None = None


class SchedulerStatus(BaseModel):
    """스케줄러 전체 상태."""

    running: bool
    current_job: str | None = None
    jobs: dict[str, JobStatus] = {}
    last_run_date: str | None = None
    next_run_at: str | None = None


class TriggerRequest(BaseModel):
    """수동 트리거 요청."""

    job: str | None = None  # None = 전체 사이클
    date: str | None = None  # YYYYMMDD, None = 오늘


class TriggerResponse(BaseModel):
    """수동 트리거 응답."""

    triggered: bool
    message: str


class CollectionResult(BaseModel):
    """수집 결과."""

    job: str
    total: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    error: str | None = None
