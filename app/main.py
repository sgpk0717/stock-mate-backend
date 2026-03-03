import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.database import async_session
from app.core.stock_master import load_stock_cache
from app.routers import accounts, agents, alpha, backtest, health, mcp, news, orders, paper, positions, sector, simulation, stocks, trading, ws

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load stock master cache from DB
    async with async_session() as session:
        await load_stock_cache(session)

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

    # Phase 3: 알파 팩토리 자동 시작
    if settings.ALPHA_FACTORY_AUTO_START:
        from app.alpha.scheduler import get_scheduler

        factory = get_scheduler()
        await factory.start(
            context="자동 알파 팩토리",
            interval_minutes=settings.ALPHA_FACTORY_INTERVAL_MINUTES,
            max_iterations=settings.ALPHA_FACTORY_MAX_ITERATIONS,
            enable_crossover=settings.ALPHA_FACTORY_CROSSOVER_ENABLED,
        )
        logger.info("Alpha factory auto-started")

    # Phase 4: MCP 서버 시작
    if settings.MCP_ENABLED:
        from app.mcp.bridge import start_mcp_server

        await start_mcp_server()

    yield

    # Shutdown
    # Phase 3: 알파 팩토리 중지
    try:
        from app.alpha.scheduler import get_scheduler as _get_scheduler

        factory = _get_scheduler()
        if factory.get_status()["running"]:
            await factory.stop()
    except Exception as e:
        logger.warning("Alpha factory stop failed during shutdown: %s", e)

    # Phase 4: MCP 서버 중지
    try:
        from app.mcp.bridge import stop_mcp_server

        await stop_mcp_server()
    except Exception as e:
        logger.warning("MCP server stop failed during shutdown: %s", e)

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


@app.get("/")
async def root():
    return {"message": "Stock Mate API"}
