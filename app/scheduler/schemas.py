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


# ── 수동 수집 트리거 ──


class ManualTriggerRequest(BaseModel):
    """수동 수집 트리거 요청."""

    collector: str  # daily_candle | minute_candle | news | margin_short | investor | dart_financial
    mode: str = "single"  # single | range | recent
    date: str | None = None  # YYYYMMDD (mode=single)
    date_from: str | None = None  # YYYYMMDD (mode=range)
    date_to: str | None = None  # YYYYMMDD (mode=range)
    recent_days: int | None = None  # N (mode=recent, max 365)


class ManualTriggerResponse(BaseModel):
    """수동 수집 트리거 응답."""

    job_id: str
    collector: str
    dates: list[str]
    message: str


class ActiveJob(BaseModel):
    """활성 수집 작업."""

    job_id: str
    collector: str
    status: str = "running"  # running | cancelling | cancelled | completed | failed
    dates: list[str] = []
    current_date: str | None = None
    date_progress: int = 0
    date_total: int = 0
    total: int = 0
    completed: int = 0
    failed: int = 0
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    source: str = "manual"  # manual | auto
    logs: list[str] = []  # 최근 작업 로그 (최대 50줄)
