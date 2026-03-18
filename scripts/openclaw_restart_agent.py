"""OpenClaw Restart Agent — Windows 호스트에서 실행.

Docker 백엔드(orchestrator.py)가 오픈클로 헬스체크 실패를 감지하면
http://host.docker.internal:18790/restart 를 POST 호출한다.
이 스크립트가 그 요청을 받아 'openclaw gateway start' 를 실행한다.

실행 방법 (Windows 터미널, 백그라운드):
    pip install aiohttp
    pythonw scripts/openclaw_restart_agent.py

또는 포그라운드:
    python scripts/openclaw_restart_agent.py
"""
import logging
import subprocess
from datetime import datetime, timezone, timedelta

from aiohttp import web

PORT = 18790
_KST = timezone(timedelta(hours=9))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("openclaw_restart_agent")


async def handle_restart(request: web.Request) -> web.Response:
    """POST /restart — openclaw gateway start 실행."""
    ts = datetime.now(_KST).strftime("%Y-%m-%d %H:%M:%S KST")
    logger.info("[%s] OpenClaw 재시작 요청 수신", ts)
    try:
        # CREATE_NO_WINDOW: cmd 창이 뜨지 않도록 함
        CREATE_NO_WINDOW = 0x08000000
        proc = subprocess.Popen(
            "openclaw gateway start",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=CREATE_NO_WINDOW,
        )
        logger.info("openclaw gateway start 실행 (pid=%s)", proc.pid)
        return web.json_response({"ok": True, "pid": proc.pid, "ts": ts})
    except Exception as e:
        logger.error("openclaw 실행 실패: %s", e)
        return web.json_response({"ok": False, "error": str(e)}, status=500)


async def handle_health(request: web.Request) -> web.Response:
    """GET /health — 에이전트 생존 확인."""
    return web.json_response({"status": "ok", "port": PORT})


app = web.Application()
app.router.add_post("/restart", handle_restart)
app.router.add_get("/health", handle_health)

if __name__ == "__main__":
    logger.info("OpenClaw Restart Agent 시작 (포트 %d)", PORT)
    logger.info("Docker 백엔드에서 http://host.docker.internal:%d/restart 로 호출", PORT)
    web.run_app(app, host="0.0.0.0", port=PORT, access_log=None)
