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
    from concurrent.futures import ThreadPoolExecutor
    from app.core.config import settings

    # 전역 스레드 풀: Polars CPU 바운드 작업용 (GIL 해제됨)
    # max_workers=2: 8코어 기준, Polars Rayon 내부 스레드와 경합 방지
    _executor = ThreadPoolExecutor(max_workers=2)
    asyncio.get_event_loop().set_default_executor(_executor)

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

    # 임베딩 모델 사전 로딩 (to_thread — 마이닝 중 블로킹 방지)
    try:
        def _preload_embedding():
            from app.sector.embedder import _get_model
            _get_model()
        await asyncio.to_thread(_preload_embedding)
        logger.info("임베딩 모델 사전 로딩 완료")
    except Exception as e:
        logger.warning("임베딩 모델 사전 로딩 실패 (마이닝 시 로딩): %s", e)

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

    # WorkerState singleton row 보장 (id=1) — UPDATE가 no-op이 되는 것 방지
    try:
        from app.models.base import WorkerState
        from sqlalchemy import select as sa_select

        async with async_session() as db:
            existing = await db.execute(sa_select(WorkerState).where(WorkerState.id == 1))
            if existing.scalar_one_or_none() is None:
                db.add(WorkerState(id=1))
                await db.commit()
                logger.info("WorkerState id=1 생성 완료")
    except Exception as e:
        logger.warning("WorkerState 초기화 실패: %s", e)

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
        """알파 팩토리 상태를 worker_state 테이블에 5초마다 기록.

        모든 인터벌 스케줄러 중 running인 것을 우선 반환.
        """
        from app.alpha.scheduler import get_all_schedulers, get_scheduler
        from app.models.base import WorkerState
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        while True:
            try:
                # 모든 활성 스케줄러 중 running인 것 찾기
                all_scheds = get_all_schedulers()
                status = None
                for _iv, _sched in all_scheds.items():
                    _s = _sched.get_status()
                    if _s.get("running"):
                        status = _s
                        break
                if status is None:
                    # running인 스케줄러 없음 → 아무거나 (기본 5m)
                    status = get_scheduler().get_status()

                # Redis user_stopped 플래그를 status에 반영
                # (Worker 재시작 시 인메모리는 리셋되지만 Redis 플래그는 유지됨)
                try:
                    from app.core.redis import get_client as _get_redis_flag
                    _rf = _get_redis_flag()
                    _flag = await _rf.get("alpha:factory:user_stopped")
                    if _flag and str(_flag).lower() == "true":
                        status["user_stopped"] = True
                except Exception:
                    pass

                async with async_session() as db:
                    stmt = pg_insert(WorkerState).values(
                        id=1, factory_status=status,
                    ).on_conflict_do_update(
                        index_elements=["id"],
                        set_={"factory_status": status},
                    )
                    await db.execute(stmt)
                    await db.commit()

                # workflow:status Redis Hash도 동기화
                try:
                    from app.core.redis import get_client as _get_redis
                    _wr = _get_redis()
                    await _wr.hset("workflow:status", mapping={
                        "mining_running": str(status.get("running", False)),
                        "mining_cycles": str(status.get("cycles_completed", 0)),
                        "mining_factors": str(status.get("factors_discovered_total", 0)),
                    })
                except Exception:
                    pass
            except Exception as e:
                logger.debug("Factory status sync failed: %s", e)
            await asyncio.sleep(5)

    tasks.append(asyncio.create_task(_sync_factory_status()))
    logger.info("팩토리 상태 DB 동기화 시작 (5초)")

    # ── 팩토리 명령 소비자 (Redis Stream) ──
    async def _consume_factory_commands() -> None:
        """API에서 보낸 factory start/stop 명령을 소비."""
        import json as _json
        from app.alpha.scheduler import get_scheduler
        from app.core.redis import get_client
        from app.models.base import WorkerState
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        r = get_client()
        last_id = "$"  # 재시작 시 새 명령만 처리 (이전 명령 리플레이 방지)
        logger.info("팩토리 명령 소비자 시작 (commands:factory)")

        while True:
            try:
                results = await r.xread(
                    {"commands:factory": last_id},
                    count=5, block=3000,
                )
                if not results:
                    continue
                for _, messages in results:
                    for msg_id, fields in messages:
                        last_id = msg_id
                        action = fields.get("action", "")
                        payload_str = fields.get("payload", "{}")
                        try:
                            payload = _json.loads(payload_str) if payload_str else {}
                        except Exception:
                            payload = {}

                        try:
                            interval = payload.get("data_interval", "5m")
                            scheduler = get_scheduler(interval)

                            if action == "factory_start":
                                ok = await scheduler.start(**payload)
                                logger.info("팩토리 명령: start(%s) → %s", interval, ok)
                            elif action == "factory_stop":
                                # 모든 interval 스케줄러 일괄 중지 + user_stopped 마킹
                                from app.alpha.scheduler import get_all_schedulers
                                stopped_any = False
                                for iv, sched in get_all_schedulers().items():
                                    if sched.get_status().get("running"):
                                        ok = await sched.stop()
                                        logger.info("팩토리 명령: stop(%s) → %s", iv, ok)
                                        if ok:
                                            stopped_any = True
                                            scheduler = sched  # DB 동기화용
                                    else:
                                        # 비활성 스케줄러도 user_stopped 마킹 (watchdog 재시작 방지)
                                        sched._state.user_stopped = True
                                if not stopped_any:
                                    logger.info("팩토리 명령: stop — 실행 중인 스케줄러 없음")

                            # 결과를 즉시 DB 동기화 (UPSERT)
                            status = scheduler.get_status()
                            async with async_session() as db:
                                stmt = pg_insert(WorkerState).values(
                                    id=1, factory_status=status,
                                ).on_conflict_do_update(
                                    index_elements=["id"],
                                    set_={"factory_status": status},
                                )
                                await db.execute(stmt)
                                await db.commit()

                            # workflow:status Redis Hash도 동기화
                            # (워크플로우 페이지의 "마이닝 진행 중" 표시와 일치시킴)
                            try:
                                from app.core.redis import get_client as _get_redis
                                _wr = _get_redis()
                                await _wr.hset("workflow:status", mapping={
                                    "mining_running": str(status.get("running", False)),
                                    "mining_cycles": str(status.get("cycles_completed", 0)),
                                    "mining_factors": str(status.get("factors_discovered_total", 0)),
                                })
                            except Exception:
                                pass
                        except Exception as e:
                            logger.error("팩토리 명령 실패 (%s): %s", action, e)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug("팩토리 명령 소비 에러: %s", e)
                await asyncio.sleep(3)

    tasks.append(asyncio.create_task(_consume_factory_commands()))

    # 프로그램 매매 수집기
    if settings.PGM_TRADING_ENABLED:
        try:
            from app.services.program_trading_collector import start_collector
            tasks.append(asyncio.create_task(start_collector()))
            logger.info("프로그램 매매 수집기 시작")
        except Exception as e:
            logger.warning("프로그램 매매 수집기 실패: %s", e)

    # 종토방 24시간 수집기
    try:
        from app.scheduler.collectors.discussion_collector import start_discussion_collector
        await start_discussion_collector()
        logger.info("종토방 수집기 시작 (24시간)")
    except Exception as e:
        logger.warning("종토방 수집기 시작 실패: %s", e)

    # 공시+뉴스 API 24시간 수집기
    try:
        from app.scheduler.collectors.news_api_collector import start_news_api_collector
        await start_news_api_collector()
        logger.info("공시+뉴스 API 수집기 시작 (24시간)")
    except Exception as e:
        logger.warning("공시+뉴스 API 수집기 시작 실패: %s", e)

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

    # 알파 스코어 엔진 — orchestrator.py의 market_open에서 시작 (이중 호출 방지)
    # worker.py에서는 시작하지 않음
    logger.info("알파 스코어 엔진: orchestrator market_open에서 시작 예정")

    # 일일 배치 스케줄러
    if settings.DAILY_SCHEDULER_ENABLED:
        try:
            from app.scheduler.daily_scheduler import get_daily_scheduler
            ds = get_daily_scheduler()
            await ds.start()
            logger.info("일일 배치 스케줄러 시작")
        except Exception as e:
            logger.warning("일일 스케줄러 실패: %s", e)

    # 수동 수집 러너 (Redis 명령 소비자)
    try:
        from app.scheduler.manual_runner import get_manual_runner
        runner = get_manual_runner()
        tasks.append(asyncio.create_task(runner.start_command_consumer()))
        logger.info("수동 수집 명령 소비자 시작")
    except Exception as e:
        logger.warning("수동 수집 러너 실패: %s", e)

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
