"""WebSocket 라우터.

ws://host/ws/ticks/{symbol}         → 틱 데이터 스트림
ws://host/ws/orderbook/{symbol}     → 호가 데이터 스트림
ws://host/ws/alpha/factory          → 알파 팩토리 실시간 로그
ws://host/ws/alpha/{run_id}         → 알파 마이닝 실시간 로그
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.ws_manager import manager

router = APIRouter(prefix="/ws", tags=["websocket"])


@router.websocket("/ticks/{symbol}")
async def ws_ticks(websocket: WebSocket, symbol: str):
    await websocket.accept()
    channel = f"ticks:{symbol}"
    await manager.subscribe(websocket, channel)
    try:
        while True:
            # 클라이언트 메시지 대기 (keepalive / ping)
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await manager.unsubscribe(websocket, channel)


@router.websocket("/orderbook/{symbol}")
async def ws_orderbook(websocket: WebSocket, symbol: str):
    await websocket.accept()
    channel = f"orderbook:{symbol}"
    await manager.subscribe(websocket, channel)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await manager.unsubscribe(websocket, channel)


@router.websocket("/backtest/{run_id}")
async def ws_backtest_progress(websocket: WebSocket, run_id: str):
    await websocket.accept()
    channel = f"backtest:{run_id}"
    await manager.subscribe(websocket, channel)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await manager.unsubscribe(websocket, channel)


@router.websocket("/simulation/{run_id}")
async def ws_simulation_progress(websocket: WebSocket, run_id: str):
    await websocket.accept()
    channel = f"simulation:{run_id}"
    await manager.subscribe(websocket, channel)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await manager.unsubscribe(websocket, channel)


# ── 알파 마이닝 WebSocket ──
# factory 라우트가 path param 라우트보다 먼저 선언되어야 충돌 방지


@router.websocket("/alpha/factory")
async def ws_alpha_factory(websocket: WebSocket):
    await websocket.accept()
    channel = "alpha:factory"
    await manager.subscribe(websocket, channel)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await manager.unsubscribe(websocket, channel)


@router.websocket("/alpha/{run_id}")
async def ws_alpha_mining(websocket: WebSocket, run_id: str):
    await websocket.accept()
    channel = f"alpha:{run_id}"
    await manager.subscribe(websocket, channel)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await manager.unsubscribe(websocket, channel)
