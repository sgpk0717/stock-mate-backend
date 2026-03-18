"""Stock Mate Worker 엔트리포인트.

docker-compose의 stockmate-worker 서비스에서 사용.
API 서버와 별도 프로세스로 실행되어, API 재시작 시에도 매매가 유지됨.

실행할 작업:
- live_runner (매매 세션)
- APScheduler (워크플로우 크론잡 8개)
- 장중 분봉 수집기
- 프로그램 매매 수집기
- 알파 팩토리 스케줄러
- 인과 검증 스케줄러
- Redis 명령 소비자 (commands:workflow, commands:trading)

사용법:
    python -m app.worker
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import date

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)
logger = logging.getLogger(__name__)


async def main() -> None:
    """Worker 메인 루프."""
    from app.core.config import settings

    logger.info("=== Stock Mate Worker 시작 ===")

    # Redis 연결 확인
    try:
        from app.core.redis import ping
        ok = await ping()
        logger.info("Redis 연결: %s", "OK" if ok else "FAIL")
    except Exception as e:
        logger.warning("Redis 연결 실패: %s", e)

    # stock_masters 메모리 캐시 로딩 (유니버스 DB 폴백용)
    try:
        from app.core.database import async_session
        from app.core.stock_master import load_stock_cache
        async with async_session() as db:
            await load_stock_cache(db)
        from app.core.stock_master import get_all_stocks
        logger.info("stock_masters 캐시 로드: %d종목", len(get_all_stocks()))
    except Exception as e:
        logger.warning("stock_masters 캐시 로드 실패: %s", e)

    # DB 좀비 정리 (main.py와 동일)
    try:
        from app.core.database import async_session
        from sqlalchemy import text

        async with async_session() as db:
            # 좀비 트레이딩 세션 정리
            result = await db.execute(text(
                "UPDATE trading_contexts "
                "SET session_state = jsonb_set(COALESCE(session_state::jsonb, '{}'::jsonb), '{status}', '\"stopped\"') "
                "WHERE status = 'active' AND session_state IS NOT NULL "
                "AND session_state::jsonb->>'status' = 'running'"
            ))
            if result.rowcount:
                logger.info("좀비 세션 %d건 정리", result.rowcount)
            await db.commit()
    except Exception as e:
        logger.warning("좀비 정리 실패: %s", e)

    tasks: list[asyncio.Task] = []

    # TradingContext + 세션 복구
    if settings.WORKFLOW_ENABLED:
        try:
            from app.trading.context import load_active_contexts_from_db
            await load_active_contexts_from_db()

            from app.trading.live_runner import restore_sessions_from_db
            restored = await restore_sessions_from_db()
            if restored:
                logger.info("활성 매매 세션 %d개 복구됨", restored)
        except Exception as e:
            logger.warning("세션 복구 실패: %s", e)

    # 워크플로우 오케스트레이터 + APScheduler
    if settings.WORKFLOW_ENABLED:
        try:
            from app.workflow.orchestrator import get_orchestrator
            wf = get_orchestrator()
            await wf.setup_scheduler()
            logger.info("APScheduler 크론잡 등록 완료")

            # Redis 명령 소비자
            tasks.append(asyncio.create_task(wf.start_command_consumer()))
        except Exception as e:
            logger.error("워크플로우 시작 실패: %s", e)

    # 텔레그램 Redis Stream consumer (at-least-once 보장)
    try:
        from app.telegram.bot import start_telegram_consumer
        tasks.append(asyncio.create_task(start_telegram_consumer()))
        logger.info("텔레그램 Redis consumer 시작")
    except Exception as e:
        logger.warning("텔레그램 consumer 시작 실패: %s", e)

    # 팩토리 상태 DB 동기화 (ExternalFactoryClient용)
    async def _sync_factory_status() -> None:
        """알파 팩토리 상태를 worker_state 테이블에 5초마다 기록."""
        from app.alpha.scheduler import get_scheduler
        from app.models.base import WorkerState
        from sqlalchemy import update as sa_update

        while True:
            try:
                status = get_scheduler().get_status()
                async with async_session() as db:
                    await db.execute(
                        sa_update(WorkerState)
                        .where(WorkerState.id == 1)
                        .values(factory_status=status)
                    )
                    await db.commit()
            except Exception as e:
                logger.debug("Factory status sync failed: %s", e)
            await asyncio.sleep(5)

    tasks.append(asyncio.create_task(_sync_factory_status()))
    logger.info("팩토리 상태 DB 동기화 시작 (5초)")

    # 프로그램 매매 수집기
    if settings.PGM_TRADING_ENABLED:
        try:
            from app.services.program_trading_collector import start_collector
            tasks.append(asyncio.create_task(start_collector()))
            logger.info("프로그램 매매 수집기 시작")
        except Exception as e:
            logger.warning("프로그램 매매 수집기 실패: %s", e)

    # 알파 팩토리 스케줄러
    if settings.WORKER_MODE in ("inline", "worker"):
        try:
            from app.alpha.scheduler import get_scheduler
            scheduler = get_scheduler()
            if hasattr(scheduler, "auto_start") and scheduler.auto_start:
                tasks.append(asyncio.create_task(scheduler.start()))
                logger.info("알파 팩토리 스케줄러 시작")
        except Exception as e:
            logger.warning("알파 팩토리 시작 실패: %s", e)

    # 인과 검증 스케줄러
    if settings.WORKFLOW_ENABLED and settings.WORKER_MODE in ("inline", "worker"):
        try:
            from app.alpha.causal_scheduler import start_causal_scheduler
            tasks.append(asyncio.create_task(start_causal_scheduler()))
            logger.info("인과 검증 스케줄러 시작")
        except Exception as e:
            logger.warning("인과 검증 스케줄러 실패: %s", e)

    # 일일 배치 스케줄러
    if settings.DAILY_SCHEDULER_ENABLED:
        try:
            from app.scheduler.daily_scheduler import get_daily_scheduler
            ds = get_daily_scheduler()
            await ds.start()
            logger.info("일일 배치 스케줄러 시작")
        except Exception as e:
            logger.warning("일일 스케줄러 실패: %s", e)

    logger.info("=== Worker 실행 중 (tasks=%d) ===", len(tasks))

    # 무한 대기 (Ctrl+C로 종료)
    try:
        await asyncio.gather(*tasks) if tasks else await asyncio.Future()
    except asyncio.CancelledError:
        pass
    finally:
        logger.info("=== Worker 종료 ===")
        # Redis 연결 정리
        try:
            from app.core.redis import close
            await close()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
