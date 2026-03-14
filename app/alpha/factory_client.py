"""팩토리 클라이언트 추상화.

WORKER_MODE에 따라 inline(기존 동작) 또는 external(DB 경유 워커 위임) 선택.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod

from app.core.config import settings

logger = logging.getLogger(__name__)


class FactoryClient(ABC):
    """알파 팩토리 + 인과검증 접근 인터페이스."""

    @abstractmethod
    async def start(self, **kwargs) -> dict:
        """팩토리 시작. 성공 시 status dict 반환."""

    @abstractmethod
    async def stop(self) -> dict:
        """팩토리 중지. 최종 status dict 반환."""

    @abstractmethod
    async def get_status(self) -> dict:
        """팩토리 현재 상태 조회."""

    @abstractmethod
    async def start_validation_batch(self, factor_ids: list, job_id: str, total: int) -> None:
        """배치 인과 검증 시작."""

    @abstractmethod
    async def get_validation_progress(self, job_id: str) -> dict | None:
        """인과 검증 잡 진행 상황 조회."""

    @abstractmethod
    async def get_latest_validation_job(self) -> dict | None:
        """가장 최근 검증 잡 조회."""


class InlineFactoryClient(FactoryClient):
    """기존 동작: API 프로세스 내에서 직접 실행."""

    async def start(self, **kwargs) -> dict:
        from app.alpha.scheduler import get_scheduler

        scheduler = get_scheduler()
        started = await scheduler.start(**kwargs)
        return {"started": started, "status": scheduler.get_status()}

    async def stop(self) -> dict:
        from app.alpha.scheduler import get_scheduler

        scheduler = get_scheduler()
        stopped = await scheduler.stop()
        return {"stopped": stopped, "status": scheduler.get_status()}

    async def get_status(self) -> dict:
        from app.alpha.scheduler import get_scheduler

        return get_scheduler().get_status()

    async def start_validation_batch(self, factor_ids: list, job_id: str, total: int) -> None:
        from app.alpha.causal_runner import start_validation_job, validate_factors_by_ids

        start_validation_job(job_id, total)
        asyncio.create_task(validate_factors_by_ids(factor_ids, job_id))

    async def get_validation_progress(self, job_id: str) -> dict | None:
        from app.alpha.causal_runner import get_validation_progress

        return get_validation_progress(job_id)

    async def get_latest_validation_job(self) -> dict | None:
        from app.alpha.causal_runner import get_latest_validation_job

        return get_latest_validation_job()


class ExternalFactoryClient(FactoryClient):
    """워커 분리 모드: DB 명령큐를 통해 워커에 위임."""

    async def start(self, **kwargs) -> dict:
        from app.core.database import async_session
        from app.models.base import WorkerCommand, WorkerState
        from sqlalchemy import select

        # 현재 상태 먼저 확인
        status = await self.get_status()
        if status.get("running"):
            return {"started": False, "status": status}

        async with async_session() as session:
            cmd = WorkerCommand(
                command="factory_start",
                payload=kwargs,
            )
            session.add(cmd)
            await session.commit()

        return {"started": True, "status": status}

    async def stop(self) -> dict:
        from app.core.database import async_session
        from app.models.base import WorkerCommand

        status = await self.get_status()
        if not status.get("running"):
            return {"stopped": False, "status": status}

        async with async_session() as session:
            cmd = WorkerCommand(
                command="factory_stop",
                payload={},
            )
            session.add(cmd)
            await session.commit()

        return {"stopped": True, "status": status}

    async def get_status(self) -> dict:
        from app.core.database import async_session
        from app.models.base import WorkerState
        from sqlalchemy import select

        async with async_session() as session:
            result = await session.execute(select(WorkerState).where(WorkerState.id == 1))
            state = result.scalar_one_or_none()

        if not state or not state.factory_status:
            return {
                "running": False,
                "cycles_completed": 0,
                "factors_discovered_total": 0,
                "current_cycle_progress": 0,
                "current_cycle_message": "",
                "last_cycle_at": None,
                "started_at": None,
                "config": None,
                "population_size": 0,
                "elite_count": 0,
                "generation": 0,
                "operator_stats": None,
                "last_funnel": None,
            }

        return state.factory_status

    async def start_validation_batch(self, factor_ids: list, job_id: str, total: int) -> None:
        from app.core.database import async_session
        from app.models.base import WorkerCommand

        async with async_session() as session:
            cmd = WorkerCommand(
                command="validate_batch",
                payload={
                    "factor_ids": [str(fid) for fid in factor_ids],
                    "job_id": job_id,
                    "total": total,
                },
            )
            session.add(cmd)
            await session.commit()

    async def get_validation_progress(self, job_id: str) -> dict | None:
        from app.core.database import async_session
        from app.models.base import WorkerState
        from sqlalchemy import select

        async with async_session() as session:
            result = await session.execute(select(WorkerState).where(WorkerState.id == 1))
            state = result.scalar_one_or_none()

        if not state or not state.causal_jobs:
            return None

        return state.causal_jobs.get(job_id)

    async def get_latest_validation_job(self) -> dict | None:
        from app.core.database import async_session
        from app.models.base import WorkerState
        from sqlalchemy import select

        async with async_session() as session:
            result = await session.execute(select(WorkerState).where(WorkerState.id == 1))
            state = result.scalar_one_or_none()

        if not state or not state.causal_jobs:
            return None

        jobs = state.causal_jobs
        if not jobs:
            return None

        latest_id = max(jobs, key=lambda k: jobs[k].get("started_at", 0))
        return {"job_id": latest_id, **jobs[latest_id]}


def get_factory_client() -> FactoryClient:
    """WORKER_MODE에 따라 적절한 클라이언트 반환."""
    if settings.WORKER_MODE == "external":
        return ExternalFactoryClient()
    return InlineFactoryClient()
