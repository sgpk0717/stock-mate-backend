"""Stock Mate 알파 팩토리 워커 프로세스.

API 서버와 분리되어 CPU-heavy 작업을 전담한다:
- 알파 팩토리 스케줄러 (진화 엔진 + 마이닝 사이클)
- 인과 검증 배치 (DoWhy 4단계)
- 인과 검증 자동 스케줄러 (1시간 주기)

API→워커 통신은 PostgreSQL worker_commands 테이블을 통해 이루어진다.
워커→API 상태 전달은 worker_state 테이블을 통해 이루어진다.

실행: python -m worker
"""

from __future__ import annotations

import asyncio
import logging
import signal
import time
import uuid

from sqlalchemy import select, update, text
from sqlalchemy.ext.asyncio import AsyncSession

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [WORKER] %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("worker")

# 폴링 간격
COMMAND_POLL_INTERVAL = 3  # 명령큐 폴링 (초)
STATE_SYNC_INTERVAL = 5  # 상태 동기화 (초)
HEARTBEAT_INTERVAL = 30  # 하트비트 (초)


async def sync_state(session: AsyncSession) -> None:
    """워커 상태를 worker_state 테이블에 동기화."""
    from app.alpha.scheduler import get_scheduler
    from app.alpha.causal_runner import _validation_jobs
    from app.models.base import WorkerState

    factory_status = get_scheduler().get_status()
    causal_jobs = dict(_validation_jobs)

    await session.execute(
        update(WorkerState)
        .where(WorkerState.id == 1)
        .values(
            factory_status=factory_status,
            causal_jobs=causal_jobs,
        )
    )
    await session.commit()


async def update_heartbeat(session: AsyncSession) -> None:
    """하트비트 타임스탬프 갱신."""
    from app.models.base import WorkerState
    from sqlalchemy import func

    await session.execute(
        update(WorkerState)
        .where(WorkerState.id == 1)
        .values(heartbeat_at=func.now())
    )
    await session.commit()


async def pick_command(session: AsyncSession) -> dict | None:
    """pending 명령 하나를 FOR UPDATE SKIP LOCKED로 안전하게 가져온다."""
    result = await session.execute(
        text(
            "SELECT id, command, payload FROM worker_commands "
            "WHERE status = 'pending' "
            "ORDER BY created_at "
            "LIMIT 1 "
            "FOR UPDATE SKIP LOCKED"
        )
    )
    row = result.fetchone()
    if not row:
        return None

    cmd_id, command, payload = row
    await session.execute(
        text(
            "UPDATE worker_commands SET status = 'picked', picked_at = NOW() "
            "WHERE id = :id"
        ),
        {"id": cmd_id},
    )
    await session.commit()

    return {"id": cmd_id, "command": command, "payload": payload}


async def complete_command(
    session: AsyncSession, cmd_id, *, status: str = "done", result: dict | None = None,
) -> None:
    """명령 완료 표시."""
    await session.execute(
        text(
            "UPDATE worker_commands SET status = :status, result = :result, "
            "completed_at = NOW() WHERE id = :id"
        ),
        {"id": cmd_id, "status": status, "result": result},
    )
    await session.commit()


async def handle_factory_start(payload: dict) -> dict:
    """factory_start 명령 처리."""
    from app.alpha.scheduler import get_scheduler

    scheduler = get_scheduler()
    started = await scheduler.start(**payload)
    return {"started": started, "status": scheduler.get_status()}


async def handle_factory_stop() -> dict:
    """factory_stop 명령 처리."""
    from app.alpha.scheduler import get_scheduler

    scheduler = get_scheduler()
    stopped = await scheduler.stop()
    return {"stopped": stopped, "status": scheduler.get_status()}


async def handle_validate_batch(payload: dict) -> dict:
    """validate_batch 명령 처리."""
    from app.alpha.causal_runner import start_validation_job, validate_factors_by_ids

    job_id = payload["job_id"]
    factor_ids = [uuid.UUID(fid) for fid in payload["factor_ids"]]
    total = payload["total"]

    start_validation_job(job_id, total)
    # 비동기 태스크로 실행 (워커 이벤트 루프에서)
    asyncio.create_task(validate_factors_by_ids(factor_ids, job_id))

    return {"job_id": job_id, "total": total}


async def command_loop() -> None:
    """명령큐 폴링 루프."""
    from app.core.database import async_session

    while True:
        try:
            async with async_session() as session:
                cmd = await pick_command(session)

            if cmd:
                logger.info("명령 수신: %s (id=%s)", cmd["command"], str(cmd["id"])[:8])
                try:
                    if cmd["command"] == "factory_start":
                        result = await handle_factory_start(cmd["payload"])
                    elif cmd["command"] == "factory_stop":
                        result = await handle_factory_stop()
                    elif cmd["command"] == "validate_batch":
                        result = await handle_validate_batch(cmd["payload"])
                    else:
                        result = {"error": f"Unknown command: {cmd['command']}"}

                    async with async_session() as session:
                        await complete_command(session, cmd["id"], result=result)

                    logger.info("명령 완료: %s", cmd["command"])
                except Exception as e:
                    logger.error("명령 실행 실패 %s: %s", cmd["command"], e)
                    async with async_session() as session:
                        await complete_command(
                            session, cmd["id"],
                            status="failed",
                            result={"error": str(e)[:500]},
                        )

        except Exception as e:
            logger.error("명령 루프 에러: %s", e)

        await asyncio.sleep(COMMAND_POLL_INTERVAL)


async def state_sync_loop() -> None:
    """주기적 상태 동기화 루프."""
    from app.core.database import async_session

    while True:
        try:
            async with async_session() as session:
                await sync_state(session)
        except Exception as e:
            logger.error("상태 동기화 에러: %s", e)

        await asyncio.sleep(STATE_SYNC_INTERVAL)


async def heartbeat_loop() -> None:
    """주기적 하트비트 루프."""
    from app.core.database import async_session

    while True:
        try:
            async with async_session() as session:
                await update_heartbeat(session)
        except Exception as e:
            logger.error("하트비트 에러: %s", e)

        await asyncio.sleep(HEARTBEAT_INTERVAL)


async def main() -> None:
    """워커 메인 엔트리포인트."""
    from app.core.config import settings

    logger.info("=" * 60)
    logger.info("Stock Mate Worker 시작 (WORKER_MODE=%s)", settings.WORKER_MODE)
    logger.info("=" * 60)

    # 종목 마스터 캐시 로드
    from app.core.database import async_session
    from app.core.stock_master import load_stock_cache

    async with async_session() as session:
        await load_stock_cache(session)
    logger.info("종목 마스터 캐시 로드 완료")

    tasks: list[asyncio.Task] = []

    # 알파 팩토리 자동 시작
    if settings.ALPHA_FACTORY_AUTO_START:
        from app.alpha.scheduler import get_scheduler

        factory = get_scheduler()
        await factory.start(
            context="자동 알파 팩토리 (워커)",
            interval_minutes=settings.ALPHA_FACTORY_INTERVAL_MINUTES,
            max_iterations=settings.ALPHA_FACTORY_MAX_ITERATIONS,
            enable_crossover=settings.ALPHA_FACTORY_CROSSOVER_ENABLED,
        )
        logger.info("알파 팩토리 자동 시작")

    # 인과 검증 자동 스케줄러
    if settings.CAUSAL_AUTO_VALIDATE:
        from app.alpha.causal_scheduler import start_causal_scheduler

        tasks.append(start_causal_scheduler())
        logger.info("인과 검증 스케줄러 시작 (1h 주기)")

    # 백그라운드 루프 시작
    tasks.append(asyncio.create_task(command_loop()))
    tasks.append(asyncio.create_task(state_sync_loop()))
    tasks.append(asyncio.create_task(heartbeat_loop()))

    logger.info("명령큐 폴링(%ds), 상태 동기화(%ds), 하트비트(%ds) 시작",
                COMMAND_POLL_INTERVAL, STATE_SYNC_INTERVAL, HEARTBEAT_INTERVAL)

    # 종료 시그널 처리
    stop_event = asyncio.Event()

    def _signal_handler():
        logger.info("종료 시그널 수신")
        stop_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            # Windows에서는 signal handler가 제한적
            pass

    try:
        await stop_event.wait()
    except (KeyboardInterrupt, SystemExit):
        pass

    logger.info("워커 종료 시작...")

    # 팩토리 중지
    try:
        from app.alpha.scheduler import get_scheduler as _get_scheduler

        factory = _get_scheduler()
        if factory.get_status()["running"]:
            await factory.stop()
            logger.info("알파 팩토리 중지 완료")
    except Exception as e:
        logger.warning("팩토리 중지 실패: %s", e)

    # 인과 검증 스케줄러 중지
    try:
        from app.alpha.causal_scheduler import stop_causal_scheduler
        stop_causal_scheduler()
    except Exception:
        pass

    # 태스크 취소
    for t in tasks:
        t.cancel()

    logger.info("워커 종료 완료")


if __name__ == "__main__":
    asyncio.run(main())
