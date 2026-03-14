"""일일 배치 데이터 수집 스케줄러.

매일 장 마감 후 일봉/분봉/뉴스를 자동 수집한다.
장중에는 틱 순환 스케줄 JSON을 생성하여 Data Pump에 전달한다.

싱글턴 패턴 + asyncio.Task 기반.
AlphaFactoryScheduler(app/alpha/scheduler.py) 패턴을 따른다.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from app.core.config import settings
from app.scheduler.circuit_breaker import CircuitBreaker, CircuitBreakerOpen
from app.scheduler.schemas import CollectionResult, JobStatus
from app.services.ws_manager import manager

logger = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9))

JOB_NAMES = ("daily_candle", "minute_candle", "news", "margin_short", "investor", "dart_financial")


# ── 상태 ────────────────────────────────────────────────


@dataclass
class _SchedulerState:
    running: bool = False
    current_job: str | None = None
    jobs: dict[str, JobStatus] = field(default_factory=dict)
    last_run_date: str | None = None
    next_run_at: str | None = None


# ── 유틸 ────────────────────────────────────────────────


def _now_kst() -> datetime:
    return datetime.now(KST)


def _today_str() -> str:
    return _now_kst().strftime("%Y%m%d")


async def _is_trading_day(date_str: str) -> bool:
    """pykrx로 거래일 여부 확인."""
    def _check():
        from pykrx import stock as krx
        tickers = krx.get_market_ticker_list(date_str, market="ALL")
        return len(tickers) > 0

    try:
        return await asyncio.to_thread(_check)
    except Exception as e:
        logger.warning("거래일 확인 실패 (기본 True): %s", e)
        return True


def _calc_next_run(hour: int, minute: int) -> datetime:
    """다음 실행 시각 계산. 이미 지났으면 내일."""
    now = _now_kst()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return target


# ── 메인 스케줄러 ───────────────────────────────────────


class DailyScheduler:
    """일일 배치 데이터 수집 스케줄러."""

    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self._tick_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._state = _SchedulerState()
        self._breakers: dict[str, CircuitBreaker] = {
            "pykrx": CircuitBreaker("pykrx", failure_threshold=10, reset_timeout=120.0),
            "kis": CircuitBreaker("kis", failure_threshold=5, reset_timeout=60.0),
            "claude": CircuitBreaker("claude", failure_threshold=3, reset_timeout=120.0),
            "dart": CircuitBreaker("dart", failure_threshold=10, reset_timeout=120.0),
        }

    # ── 제어 ──

    async def start(self) -> bool:
        """스케줄러 시작. 이미 실행 중이면 False."""
        async with self._lock:
            if self._state.running:
                return False
            self._state.running = True
            self._task = asyncio.create_task(self._daily_loop())
            self._tick_task = asyncio.create_task(self._tick_schedule_loop())
            logger.info("Daily scheduler started")
            return True

    async def stop(self) -> bool:
        """스케줄러 중지."""
        async with self._lock:
            if not self._state.running:
                return False
            self._state.running = False
            if self._task:
                self._task.cancel()
                self._task = None
            if self._tick_task:
                self._tick_task.cancel()
                self._tick_task = None
            logger.info("Daily scheduler stopped")
            return True

    def get_status(self) -> dict:
        """REST/WebSocket용 상태 반환."""
        return {
            "running": self._state.running,
            "current_job": self._state.current_job,
            "jobs": {
                name: job.model_dump() for name, job in self._state.jobs.items()
            },
            "last_run_date": self._state.last_run_date,
            "next_run_at": self._state.next_run_at,
        }

    async def trigger_job(
        self, job_name: str | None = None, date: str | None = None,
    ) -> tuple[bool, str]:
        """수동 트리거. 스케줄러 중지 상태에서도 호출 가능.

        Args:
            job_name: None이면 전체 사이클.
            date: YYYYMMDD. None이면 오늘.
        """
        if self._state.current_job is not None:
            return False, f"이미 '{self._state.current_job}' 실행 중"

        target_date = date or _today_str()

        if job_name is None:
            asyncio.create_task(self._run_daily_cycle(target_date))
            return True, f"전체 사이클 시작 (date={target_date})"

        if job_name not in JOB_NAMES:
            return False, f"알 수 없는 잡: {job_name} (가능: {JOB_NAMES})"

        asyncio.create_task(self._run_single_job(job_name, target_date))
        return True, f"'{job_name}' 시작 (date={target_date})"

    # ── 메인 루프 ──

    async def _already_collected_today(self) -> bool:
        """DB에서 오늘 일봉 수집 기록 존재 여부 확인."""
        from app.core.database import async_session
        from sqlalchemy import text

        today = _now_kst().strftime("%Y-%m-%d")
        try:
            async with async_session() as session:
                result = await session.execute(
                    text(
                        "SELECT 1 FROM stock_candles "
                        "WHERE dt::date = :d AND interval = '1d' LIMIT 1"
                    ),
                    {"d": today},
                )
                return result.scalar() is not None
        except Exception as e:
            logger.warning("수집 여부 확인 실패 (기본 False): %s", e)
            return False

    async def _daily_loop(self) -> None:
        """이벤트 기반 메인 루프 — 매일 지정 시각에 실행."""
        # ── Catch-up: 앱 시작이 수집 시각 이후면 즉시 실행 ──
        try:
            now = _now_kst()
            collect_time = now.replace(
                hour=settings.DAILY_COLLECT_HOUR,
                minute=settings.DAILY_COLLECT_MINUTE,
                second=0, microsecond=0,
            )
            if now > collect_time:
                target_date = _today_str()
                if not await self._already_collected_today():
                    logger.info("Catch-up: 당일 수집 미완료 (date=%s), 즉시 실행", target_date)
                    await self._run_daily_cycle(target_date)
                else:
                    logger.info("Catch-up: 당일 수집 이미 완료 (date=%s), 스킵", target_date)
        except Exception as e:
            logger.error("Catch-up 체크 에러: %s", e, exc_info=True)

        while self._state.running:
            try:
                next_run = _calc_next_run(
                    settings.DAILY_COLLECT_HOUR,
                    settings.DAILY_COLLECT_MINUTE,
                )
                self._state.next_run_at = next_run.isoformat()
                delta = (next_run - _now_kst()).total_seconds()
                logger.info(
                    "Next daily run at %s (%.0fs later)",
                    next_run.strftime("%Y-%m-%d %H:%M KST"),
                    delta,
                )
                await asyncio.sleep(delta)

                if not self._state.running:
                    break

                target_date = _today_str()

                if not await _is_trading_day(target_date):
                    logger.info("비거래일 (%s) — 스킵", target_date)
                    continue

                await self._run_daily_cycle(target_date)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Daily loop 에러: %s", e, exc_info=True)
                await asyncio.sleep(60)

    async def _tick_schedule_loop(self) -> None:
        """장 시작 전 틱 순환 스케줄 JSON 생성."""
        while self._state.running:
            try:
                # 다음 장 시작 전 08:50 KST
                next_gen = _calc_next_run(8, 50)
                delta = (next_gen - _now_kst()).total_seconds()
                await asyncio.sleep(max(delta, 0))

                if not self._state.running:
                    break

                target_date = _today_str()
                if not await _is_trading_day(target_date):
                    continue

                from app.scheduler.collectors.tick_rotation import (
                    generate_tick_schedule,
                )

                await generate_tick_schedule()
                logger.info("틱 순환 스케줄 생성 완료")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Tick schedule loop 에러: %s", e, exc_info=True)
                await asyncio.sleep(60)

    # ── 실행 ──

    async def _run_daily_cycle(self, date: str) -> None:
        """일봉 → 분봉 → 뉴스 순차 실행."""
        logger.info("=== 일일 수집 사이클 시작 (date=%s) ===", date)
        self._state.last_run_date = date

        for job_name in JOB_NAMES:
            if not self._state.running:
                break
            await self._run_single_job(job_name, date)

        self._state.current_job = None
        logger.info("=== 일일 수집 사이클 완료 (date=%s) ===", date)
        await self._broadcast({"type": "cycle_complete", "date": date})

    async def _run_single_job(self, job_name: str, date: str) -> None:
        """개별 잡 실행."""
        job = JobStatus(
            name=job_name,
            status="running",
            started_at=_now_kst().isoformat(),
        )
        self._state.jobs[job_name] = job
        self._state.current_job = job_name

        try:
            result = await self._dispatch_job(job_name, date, job)
            job.status = "completed"
            job.completed = result.completed
            job.failed = result.failed
            job.skipped = result.skipped
            job.total = result.total
        except CircuitBreakerOpen as e:
            job.status = "failed"
            job.error = f"서킷 브레이커 OPEN: {e.name}"
            logger.error("Job '%s' 실패 — 서킷 OPEN: %s", job_name, e.name)
        except Exception as e:
            job.status = "failed"
            job.error = str(e)[:500]
            logger.error("Job '%s' 실패: %s", job_name, e, exc_info=True)
        finally:
            job.completed_at = _now_kst().isoformat()
            if job.started_at:
                start = datetime.fromisoformat(job.started_at)
                end = datetime.fromisoformat(job.completed_at)
                job.duration_seconds = (end - start).total_seconds()
            self._state.current_job = None
            await self._broadcast({
                "type": "job_complete",
                "job": job.model_dump(),
            })

    async def _dispatch_job(
        self, job_name: str, date: str, job: JobStatus,
    ) -> CollectionResult:
        """잡 이름으로 적절한 collector 호출."""

        async def _progress_cb(total: int, completed: int, last_symbol: str):
            job.total = total
            job.completed = completed
            job.last_symbol = last_symbol
            await self._broadcast({
                "type": "job_progress",
                "job": job_name,
                "total": total,
                "completed": completed,
                "last_symbol": last_symbol,
            })

        if job_name == "daily_candle":
            from app.scheduler.collectors.daily_candle import collect_daily_candles

            return await collect_daily_candles(
                date, progress_cb=_progress_cb, cb=self._breakers["pykrx"],
            )

        if job_name == "minute_candle":
            from app.scheduler.collectors.minute_candle import (
                collect_minute_candles,
            )

            return await collect_minute_candles(
                date, progress_cb=_progress_cb, cb=self._breakers["kis"],
            )

        if job_name == "news":
            from app.scheduler.collectors.news_batch import collect_news

            return await collect_news(
                date, progress_cb=_progress_cb, cb=self._breakers["claude"],
            )

        if job_name == "margin_short":
            from app.scheduler.collectors.margin_short import collect_margin_short

            return await collect_margin_short(
                date, progress_cb=_progress_cb, cb=self._breakers["kis"],
            )

        if job_name == "investor":
            from app.scheduler.collectors.investor import collect_investor

            return await collect_investor(
                date, progress_cb=_progress_cb, cb=self._breakers["kis"],
            )

        if job_name == "dart_financial":
            from app.scheduler.collectors.dart_financial import (
                collect_dart_financials,
            )

            return await collect_dart_financials(
                date, progress_cb=_progress_cb, cb=self._breakers["dart"],
            )

        return CollectionResult(job=job_name)

    async def _broadcast(self, data: dict) -> None:
        """WebSocket 브로드캐스트."""
        try:
            await manager.broadcast("scheduler:daily", data)
        except Exception:
            pass  # WS 실패는 무시


# ── 싱글턴 ──────────────────────────────────────────────

_scheduler: DailyScheduler | None = None


def get_daily_scheduler() -> DailyScheduler:
    """싱글턴 스케줄러 인스턴스."""
    global _scheduler
    if _scheduler is None:
        _scheduler = DailyScheduler()
    return _scheduler
