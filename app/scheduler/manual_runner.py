"""수동 데이터 수집 러너.

개별 수집기를 날짜/기간 지정으로 수동 트리거하고,
진행 상황을 Redis에 동기화하여 API가 읽을 수 있게 한다.
자동 스케줄러(DailyScheduler) 작업도 여기에 등록하여 통합 모니터링.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from app.scheduler.circuit_breaker import CircuitBreaker, CircuitBreakerOpen
from app.scheduler.schemas import ActiveJob, CollectionResult, ManualTriggerRequest

logger = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9))

# DailyScheduler의 JOB_NAMES와 동일
VALID_COLLECTORS = (
    "daily_candle", "minute_candle", "news",
    "margin_short", "investor", "dart_financial",
)

_BREAKER_MAP: dict[str, str] = {
    "daily_candle": "pykrx",
    "minute_candle": "kis",
    "news": "claude",
    "margin_short": "kis",
    "investor": "kis",
    "dart_financial": "dart",
}

_COLLECTOR_LABELS: dict[str, str] = {
    "daily_candle": "일봉",
    "minute_candle": "분봉",
    "news": "뉴스",
    "margin_short": "신용/공매도",
    "investor": "투자자 수급",
    "dart_financial": "DART 재무",
}

# 완료된 잡 유지 시간 (초)
_COMPLETED_TTL = 300


def _now_kst() -> datetime:
    return datetime.now(KST)


def _today_str() -> str:
    return _now_kst().strftime("%Y%m%d")


def _resolve_dates(req: ManualTriggerRequest) -> list[str]:
    """요청에서 날짜 리스트를 해석."""
    if req.mode == "single":
        d = req.date or _today_str()
        return [d]

    if req.mode == "range":
        if not req.date_from or not req.date_to:
            raise ValueError("기간 모드에는 date_from, date_to가 필요합니다")
        start = datetime.strptime(req.date_from, "%Y%m%d")
        end = datetime.strptime(req.date_to, "%Y%m%d")
        if start > end:
            start, end = end, start
        dates = []
        cur = start
        while cur <= end:
            dates.append(cur.strftime("%Y%m%d"))
            cur += timedelta(days=1)
        if len(dates) > 365:
            raise ValueError("최대 365일까지 지정 가능합니다")
        return dates

    if req.mode == "recent":
        n = min(req.recent_days or 7, 365)
        if n < 1:
            n = 1
        today = _now_kst().date()
        return [(today - timedelta(days=i)).strftime("%Y%m%d") for i in range(n)]

    raise ValueError(f"알 수 없는 모드: {req.mode}")


_MAX_LOGS = 100  # 잡당 최대 로그 줄수


@dataclass
class _RunningJob:
    task: asyncio.Task | None
    cancel_event: asyncio.Event
    job: ActiveJob
    completed_at_mono: float | None = None  # monotonic time of completion

    def log(self, msg: str) -> None:
        """작업 로그 추가 (타임스탬프 자동 삽입)."""
        ts = _now_kst().strftime("%H:%M:%S")
        entry = f"[{ts}] {msg}"
        self.job.logs.append(entry)
        if len(self.job.logs) > _MAX_LOGS:
            self.job.logs = self.job.logs[-_MAX_LOGS:]


class ManualJobRunner:
    """수동 수집 작업 관리자."""

    def __init__(self) -> None:
        self._jobs: dict[str, _RunningJob] = {}
        self._breakers: dict[str, CircuitBreaker] = {
            "pykrx": CircuitBreaker("pykrx", failure_threshold=10, reset_timeout=120.0),
            "kis": CircuitBreaker("kis", failure_threshold=5, reset_timeout=60.0),
            "claude": CircuitBreaker("claude", failure_threshold=3, reset_timeout=120.0),
            "dart": CircuitBreaker("dart", failure_threshold=10, reset_timeout=120.0),
        }

    # ── 공개 API ──

    async def start_job(self, req: ManualTriggerRequest) -> ActiveJob:
        """수동 수집 작업 시작. 동일 수집기 중복 실행 방지."""
        if req.collector not in VALID_COLLECTORS:
            raise ValueError(f"알 수 없는 수집기: {req.collector}")

        # 중복 체크
        for rj in self._jobs.values():
            if (rj.job.collector == req.collector
                    and rj.job.status in ("running", "cancelling")):
                raise ValueError(
                    f"'{_COLLECTOR_LABELS.get(req.collector, req.collector)}' "
                    f"수집이 이미 진행 중입니다"
                )

        dates = _resolve_dates(req)
        job_id = str(uuid.uuid4())

        job = ActiveJob(
            job_id=job_id,
            collector=req.collector,
            status="running",
            dates=dates,
            date_total=len(dates),
            started_at=_now_kst().isoformat(),
            source="manual",
        )

        cancel_event = asyncio.Event()
        task = asyncio.create_task(self._run_job(job_id))

        self._jobs[job_id] = _RunningJob(
            task=task, cancel_event=cancel_event, job=job,
        )

        await self._sync_to_redis(job_id)
        logger.info(
            "수동 수집 시작: %s (%s) %d일 [%s]",
            req.collector, job_id[:8], len(dates),
            f"{dates[0]}~{dates[-1]}" if len(dates) > 1 else dates[0],
        )
        return job

    async def cancel_job(self, job_id: str) -> bool:
        """작업 중단 요청."""
        rj = self._jobs.get(job_id)
        if not rj or rj.job.status not in ("running",):
            return False
        rj.cancel_event.set()
        rj.job.status = "cancelling"
        await self._sync_to_redis(job_id)
        logger.info("수동 수집 중단 요청: %s (%s)", rj.job.collector, job_id[:8])
        return True

    def list_jobs(self) -> list[ActiveJob]:
        """활성 + 최근 완료 작업 목록."""
        self._cleanup_expired()
        return [rj.job for rj in self._jobs.values()]

    async def register_auto_job(self, collector: str, date: str) -> str:
        """자동 스케줄러 잡을 활성 목록에 등록."""
        job_id = f"auto-{collector}-{date}"

        # 이미 등록되어 있으면 무시
        if job_id in self._jobs:
            return job_id

        job = ActiveJob(
            job_id=job_id,
            collector=collector,
            status="running",
            dates=[date],
            date_total=1,
            current_date=date,
            started_at=_now_kst().isoformat(),
            source="auto",
        )
        self._jobs[job_id] = _RunningJob(
            task=None, cancel_event=asyncio.Event(), job=job,
        )
        await self._sync_to_redis(job_id)
        return job_id

    async def update_auto_job(
        self, job_id: str, *,
        status: str | None = None,
        total: int | None = None,
        completed: int | None = None,
        failed: int | None = None,
        error: str | None = None,
    ) -> None:
        """자동 스케줄러 잡 상태 업데이트."""
        rj = self._jobs.get(job_id)
        if not rj:
            return
        if status:
            rj.job.status = status
        if total is not None:
            rj.job.total = total
        if completed is not None:
            rj.job.completed = completed
        if failed is not None:
            rj.job.failed = failed
        if error:
            rj.job.error = error
        if status in ("completed", "failed", "cancelled"):
            rj.job.completed_at = _now_kst().isoformat()
            rj.job.date_progress = rj.job.date_total
            import time
            rj.completed_at_mono = time.monotonic()
        await self._sync_to_redis(job_id)

    async def start_command_consumer(self) -> None:
        """Redis Stream에서 API의 수집 명령을 소비."""
        from app.core.redis import get_client

        # 스트림의 현재 마지막 ID를 가져와서 이전 메시지 스킵
        r = get_client()
        try:
            info = await r.xinfo_stream("commands:scheduler")
            last_id = info.get("last-generated-id", "0-0")
            logger.info("수집 명령 소비자: last_id=%s부터 시작 (이전 메시지 스킵)", last_id)
        except Exception:
            last_id = "0-0"

        while True:
            try:
                results = await r.xread(
                    {"commands:scheduler": last_id},
                    count=10, block=5000,
                )
                if not results:
                    continue

                for _stream, messages in results:
                    for msg_id, fields in messages:
                        last_id = msg_id if isinstance(msg_id, str) else msg_id.decode()
                        # bytes→str 변환
                        str_fields = {
                            (k.decode() if isinstance(k, bytes) else k): (v.decode() if isinstance(v, bytes) else v)
                            for k, v in fields.items()
                        }
                        await self._handle_command(str_fields)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("수집 명령 소비 에러: %s", e)
                await asyncio.sleep(2)

    # ── 내부 ──

    async def _handle_command(self, fields: dict) -> None:
        """Redis Stream 명령 처리."""
        action = fields.get("action", "")

        if action == "start_collect":
            try:
                req = ManualTriggerRequest(
                    collector=fields.get("collector", ""),
                    mode=fields.get("mode", "single"),
                    date=fields.get("date") or None,
                    date_from=fields.get("date_from") or None,
                    date_to=fields.get("date_to") or None,
                    recent_days=int(fields["recent_days"]) if fields.get("recent_days") else None,
                )
                await self.start_job(req)
            except Exception as e:
                logger.error("수집 명령 처리 실패: %s", e)

        elif action == "cancel_job":
            job_id = fields.get("job_id", "")
            if job_id:
                await self.cancel_job(job_id)

    async def _run_job(self, job_id: str) -> None:
        """수집 작업 실행 루프 — 날짜별 순차 실행."""
        import time

        rj = self._jobs.get(job_id)
        if not rj:
            return

        job = rj.job
        collector = job.collector
        label = _COLLECTOR_LABELS.get(collector, collector)
        breaker_key = _BREAKER_MAP.get(collector, "pykrx")
        cb = self._breakers[breaker_key]

        total_completed = 0
        total_failed = 0
        date_range = (
            f"{job.dates[-1]}~{job.dates[0]}" if len(job.dates) > 1
            else job.dates[0] if job.dates else "?"
        )
        rj.log(f"{label} 수집 시작 — {len(job.dates)}일 ({date_range})")

        try:
            for i, date in enumerate(job.dates):
                # 중단 체크
                if rj.cancel_event.is_set():
                    rj.log("사용자 중단 요청 감지 — 수집 중단")
                    job.status = "cancelled"
                    job.completed_at = _now_kst().isoformat()
                    break

                job.current_date = date
                job.date_progress = i
                job.total = 0
                job.completed = 0
                rj.log(f"[{i+1}/{len(job.dates)}] {date} 수집 시작")
                await self._sync_to_redis(job_id)
                await self._broadcast_progress(job)

                try:
                    result = await self._dispatch_collector(
                        collector, date, job, rj, cb,
                    )
                    total_completed += result.completed
                    total_failed += result.failed
                    rj.log(
                        f"[{i+1}/{len(job.dates)}] {date} 완료 — "
                        f"성공 {result.completed}, 실패 {result.failed}, "
                        f"스킵 {result.skipped}"
                    )
                except CircuitBreakerOpen as e:
                    rj.log(f"[{i+1}/{len(job.dates)}] {date} 서킷 브레이커 OPEN: {e.name}")
                    job.error = f"서킷 브레이커 OPEN: {e.name}"
                    total_failed += 1
                except Exception as e:
                    err_msg = str(e)[:200]
                    rj.log(f"[{i+1}/{len(job.dates)}] {date} 실패: {err_msg}")
                    logger.error("수동 수집 날짜 %s 실패: %s", date, e, exc_info=True)
                    total_failed += 1

                await self._sync_to_redis(job_id)

            # 최종 상태
            if job.status == "running":
                job.status = "completed"
                job.completed_at = _now_kst().isoformat()

            job.date_progress = i + 1 if job.dates else 0
            job.completed = total_completed
            job.failed = total_failed
            rj.log(f"수집 종료 [{job.status}] — 총 성공 {total_completed}, 실패 {total_failed}")

        except asyncio.CancelledError:
            job.status = "cancelled"
            job.completed_at = _now_kst().isoformat()
            rj.log("태스크 취소됨 (CancelledError)")
        except Exception as e:
            job.status = "failed"
            job.error = str(e)[:500]
            job.completed_at = _now_kst().isoformat()
            rj.log(f"치명적 오류: {str(e)[:300]}")
            logger.error("수동 수집 작업 실패: %s", e, exc_info=True)
        finally:
            rj.completed_at_mono = time.monotonic()
            await self._sync_to_redis(job_id)
            await self._broadcast_progress(job)

    async def _dispatch_collector(
        self, collector: str, date: str, job: ActiveJob,
        rj: _RunningJob, cb: CircuitBreaker,
    ) -> CollectionResult:
        """수집기 호출 (DailyScheduler._dispatch_job과 동일 로직)."""
        _last_log_count = 0

        async def _progress_cb(total: int, completed: int, last_symbol: str):
            nonlocal _last_log_count
            job.total = total
            job.completed = completed
            # 매 100건마다 로그 + Redis 동기화
            if completed - _last_log_count >= 100 or completed == total:
                _last_log_count = completed
                rj.log(f"  진행 {completed}/{total} (최근: {last_symbol})")
                await self._sync_to_redis(job.job_id)
            await self._broadcast_progress(job)

        if collector == "daily_candle":
            from app.scheduler.collectors.daily_candle import collect_daily_candles
            return await collect_daily_candles(date, progress_cb=_progress_cb, cb=cb)

        if collector == "minute_candle":
            from app.scheduler.collectors.minute_candle import collect_minute_candles
            return await collect_minute_candles(date, progress_cb=_progress_cb, cb=cb)

        if collector == "news":
            from app.scheduler.collectors.news_batch import collect_news
            return await collect_news(date, progress_cb=_progress_cb, cb=cb)

        if collector == "margin_short":
            from app.scheduler.collectors.margin_short import collect_margin_short
            return await collect_margin_short(date, progress_cb=_progress_cb, cb=cb)

        if collector == "investor":
            from app.scheduler.collectors.investor import collect_investor
            return await collect_investor(date, progress_cb=_progress_cb, cb=cb)

        if collector == "dart_financial":
            from app.scheduler.collectors.dart_financial import collect_dart_financials
            return await collect_dart_financials(date, progress_cb=_progress_cb, cb=cb)

        return CollectionResult(job=collector)

    async def _sync_to_redis(self, job_id: str) -> None:
        """잡 상태를 Redis에 동기화."""
        try:
            from app.core.redis import hset
            rj = self._jobs.get(job_id)
            if not rj:
                return
            j = rj.job
            await hset(f"scheduler:jobs:{job_id}", {
                "job_id": j.job_id,
                "collector": j.collector,
                "status": j.status,
                "dates": json.dumps(j.dates),
                "current_date": j.current_date or "",
                "date_progress": str(j.date_progress),
                "date_total": str(j.date_total),
                "total": str(j.total),
                "completed": str(j.completed),
                "failed": str(j.failed),
                "started_at": j.started_at or "",
                "completed_at": j.completed_at or "",
                "error": j.error or "",
                "source": j.source,
                "logs": json.dumps(j.logs[-_MAX_LOGS:], ensure_ascii=False),
            })
            # active set에 추가/제거
            from app.core.redis import get_client
            r = get_client()
            if j.status in ("running", "cancelling"):
                await r.sadd("scheduler:active_job_ids", job_id)
            else:
                # 완료된 잡도 TTL 동안 유지
                await r.expire(f"scheduler:jobs:{job_id}", _COMPLETED_TTL)
        except Exception as e:
            logger.debug("Redis 잡 동기화 실패: %s", e)

    async def _broadcast_progress(self, job: ActiveJob) -> None:
        """WebSocket으로 진행 상황 브로드캐스트."""
        try:
            from app.services.ws_manager import manager
            await manager.broadcast("scheduler:jobs", {
                "type": "job_progress",
                "job_id": job.job_id,
                "collector": job.collector,
                "status": job.status,
                "date_progress": job.date_progress,
                "date_total": job.date_total,
                "total": job.total,
                "completed": job.completed,
                "current_date": job.current_date,
            })
        except Exception:
            pass

    def _cleanup_expired(self) -> None:
        """TTL 지난 완료 잡 제거."""
        import time
        now = time.monotonic()
        expired = [
            jid for jid, rj in self._jobs.items()
            if rj.completed_at_mono and now - rj.completed_at_mono > _COMPLETED_TTL
        ]
        for jid in expired:
            del self._jobs[jid]


# ── 싱글턴 ──

_runner: ManualJobRunner | None = None


def get_manual_runner() -> ManualJobRunner:
    global _runner
    if _runner is None:
        _runner = ManualJobRunner()
    return _runner
