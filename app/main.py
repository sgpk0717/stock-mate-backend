import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.database import async_session
from app.core.stock_master import load_stock_cache
from app.routers import accounts, agents, alpha, backtest, data_explorer, health, mcp, news, orders, paper, positions, scheduler, sector, simulation, stocks, telegram, trading, workflow, ws

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load stock master cache from DB
    async with async_session() as session:
        await load_stock_cache(session)

    # 좀비 RUNNING/PENDING 백테스트 정리 (서버 재시작으로 task 소실)
    try:
        from sqlalchemy import text
        async with async_session() as db:
            result = await db.execute(text(
                "UPDATE backtest_runs SET status='FAILED', error_message='서버 재시작으로 중단됨' "
                "WHERE status IN ('RUNNING', 'PENDING')"
            ))
            count = result.rowcount
            await db.commit()
            if count:
                logger.info("좀비 백테스트 %d건 정리 (RUNNING/PENDING → FAILED)", count)
    except Exception as e:
        logger.warning("좀비 백테스트 정리 실패: %s", e)

    # Agent 세션 TTL 설정
    from app.agents.session import session_store
    session_store._ttl_minutes = settings.AGENT_SESSION_TTL_MINUTES

    # 틱 저장 백그라운드 태스크 시작
    from app.services.tick_writer import start_writer, stop_writer

    tasks: list[asyncio.Task] = []
    zmq_sub = None

    tasks.append(asyncio.create_task(start_writer()))

    if settings.USE_SIMULATOR:
        # 개발 모드: 가짜 틱/호가 데이터 생성
        from app.services.tick_simulator import simulate_orderbook, simulate_ticks

        logger.info("실시간 데이터 소스: 시뮬레이터 (USE_SIMULATOR=true)")
        tasks.append(asyncio.create_task(simulate_ticks()))
        tasks.append(asyncio.create_task(simulate_orderbook()))
    else:
        # 프로덕션 모드: ZMQ에서 키움 Data Pump 데이터 수신
        from app.services.zmq_handler import handle_zmq_message
        from app.services.zmq_subscriber import ZmqSubscriber

        logger.info("실시간 데이터 소스: ZMQ (키움 Data Pump 대기)")
        zmq_sub = ZmqSubscriber()
        zmq_sub.on_message(handle_zmq_message)
        tasks.append(asyncio.create_task(zmq_sub.start()))

    # Phase 3: 알파 팩토리 자동 시작 (inline 모드에서만)
    if settings.ALPHA_FACTORY_AUTO_START and settings.WORKER_MODE == "inline":
        from app.alpha.scheduler import get_scheduler

        factory = get_scheduler()
        await factory.start(
            context="자동 알파 팩토리",
            interval_minutes=settings.ALPHA_FACTORY_INTERVAL_MINUTES,
            max_iterations=settings.ALPHA_FACTORY_MAX_ITERATIONS,
            enable_crossover=settings.ALPHA_FACTORY_CROSSOVER_ENABLED,
        )
        logger.info("Alpha factory auto-started")

    # Daily Scheduler (일일 배치 수집)
    if settings.DAILY_SCHEDULER_ENABLED:
        from app.scheduler.daily_scheduler import get_daily_scheduler

        daily = get_daily_scheduler()
        await daily.start()
        logger.info("Daily scheduler auto-started")

    # 프로그램 매매 수집기
    if settings.PGM_TRADING_ENABLED:
        from app.services.program_trading_collector import start_collector

        tasks.append(asyncio.create_task(start_collector()))
        logger.info("Program trading collector started")

    # 인과 검증 스케줄러 (1시간 주기, 미검증 상위 30% 팩터) — inline 모드에서만
    if settings.WORKER_MODE == "inline":
        from app.alpha.causal_scheduler import start_causal_scheduler

        tasks.append(start_causal_scheduler())
        logger.info("Causal validation scheduler started (1h interval)")

    # TradingContext DB 복원
    from app.trading.context import load_active_contexts_from_db
    await load_active_contexts_from_db()

    # 활성 매매 세션 복구 (C2: 서버 재시작 시 LiveSession 자동 재개)
    from app.trading.live_runner import restore_sessions_from_db
    restored = await restore_sessions_from_db()
    if restored:
        logger.info("활성 매매 세션 %d개 복구됨", restored)

    # 워크플로우 오케스트레이터
    if settings.WORKFLOW_ENABLED:
        # DB에서 EMERGENCY_STOP 상태 확인 — 토글 OFF 시 재시작해도 비활성 유지
        _wf_skip = False
        try:
            from app.core.database import async_session as _wf_session
            from app.workflow.models import WorkflowRun
            from sqlalchemy import select as _sel, func as _fn
            async with _wf_session() as _db:
                _latest = await _db.execute(
                    _sel(WorkflowRun.phase)
                    .order_by(WorkflowRun.date.desc())
                    .limit(1)
                )
                _phase = _latest.scalar_one_or_none()
                if _phase == "EMERGENCY_STOP":
                    logger.info("워크플로우: DB에 EMERGENCY_STOP 상태 → 오케스트레이터 시작 안 함")
                    _wf_skip = True
        except Exception as _e:
            logger.warning("워크플로우 상태 확인 실패 (정상 시작): %s", _e)

        if not _wf_skip:
            from app.workflow.orchestrator import get_orchestrator
            wf = get_orchestrator()
            await wf.setup_scheduler()
            logger.info("Workflow orchestrator started (APScheduler)")

    # Phase 4: MCP 서버 시작
    if settings.MCP_ENABLED:
        from app.mcp.bridge import start_mcp_server

        await start_mcp_server()

    # Telegram Bot (명령어 핸들러 + 인라인 키보드)
    from app.telegram.bot import start_bot as start_telegram_bot
    await start_telegram_bot()

    yield

    # Shutdown
    # 워크플로우 스케줄러 중지
    if settings.WORKFLOW_ENABLED:
        try:
            from app.workflow.orchestrator import get_orchestrator as _get_wf
            await _get_wf().shutdown_scheduler()
        except Exception as e:
            logger.warning("Workflow scheduler stop failed: %s", e)

    # Phase 3: 알파 팩토리 중지 (inline 모드에서만 직접 중지)
    if settings.WORKER_MODE == "inline":
        try:
            from app.alpha.scheduler import get_scheduler as _get_scheduler

            factory = _get_scheduler()
            if factory.get_status()["running"]:
                await factory.stop()
        except Exception as e:
            logger.warning("Alpha factory stop failed during shutdown: %s", e)

    # Daily Scheduler 중지
    try:
        from app.scheduler.daily_scheduler import get_daily_scheduler as _get_daily

        daily = _get_daily()
        if daily.get_status()["running"]:
            await daily.stop()
    except Exception as e:
        logger.warning("Daily scheduler stop failed during shutdown: %s", e)

    # Phase 4: MCP 서버 중지
    try:
        from app.mcp.bridge import stop_mcp_server

        await stop_mcp_server()
    except Exception as e:
        logger.warning("MCP server stop failed during shutdown: %s", e)

    # Telegram Bot 중지
    try:
        from app.telegram.bot import stop_bot as stop_telegram_bot
        await stop_telegram_bot()
    except Exception as e:
        logger.warning("Telegram bot stop failed: %s", e)

    # 인과 검증 스케줄러 중지 (inline 모드에서만)
    if settings.WORKER_MODE == "inline":
        try:
            from app.alpha.causal_scheduler import stop_causal_scheduler

            stop_causal_scheduler()
        except Exception as e:
            logger.warning("Causal scheduler stop failed: %s", e)

    # 프로그램 매매 수집기 중지
    if settings.PGM_TRADING_ENABLED:
        from app.services.program_trading_collector import stop_collector

        stop_collector()

    await stop_writer()

    if settings.USE_SIMULATOR:
        from app.services.tick_simulator import stop_simulator

        stop_simulator()
    elif zmq_sub:
        await zmq_sub.stop()

    for t in tasks:
        t.cancel()


app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(accounts.router)
app.include_router(positions.router)
app.include_router(orders.router)
app.include_router(stocks.router)
app.include_router(ws.router)
app.include_router(paper.router)
app.include_router(backtest.router)
app.include_router(alpha.router)
app.include_router(agents.router)
app.include_router(news.router)
app.include_router(sector.router)
app.include_router(trading.router)
app.include_router(simulation.router)
app.include_router(mcp.router)
app.include_router(scheduler.router)
app.include_router(workflow.router)
app.include_router(data_explorer.router)
app.include_router(telegram.router)


@app.get("/")
async def root():
    return {"message": "Stock Mate API"}
