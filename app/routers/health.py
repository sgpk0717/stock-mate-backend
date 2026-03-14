"""시스템 헬스 체크 (설계서 §4.1.4)."""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(db: AsyncSession = Depends(get_db)):
    """기본 헬스체크 — DB 연결 확인."""
    try:
        await db.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception:
        db_status = "disconnected"
    return HealthResponse(status="healthy", database=db_status)


@router.get("/health/detailed")
async def detailed_health_check(db: AsyncSession = Depends(get_db)):
    """상세 시스템 상태 (OpenClaw get_system_health용).

    DB, 스케줄러, 세션, 메모리 전부 보고.
    """
    checks: dict = {}

    # 1. DB 연결
    try:
        await db.execute(text("SELECT 1"))
        checks["db"] = {"status": "ok"}
    except Exception as e:
        checks["db"] = {"status": "error", "error": str(e)}

    # 2. 워크플로우 상태
    try:
        from app.workflow.orchestrator import get_orchestrator
        orch = get_orchestrator()
        wf_status = await orch.get_status()
        checks["workflow"] = {
            "status": "ok",
            "phase": wf_status.get("phase"),
            "scheduler_running": orch._scheduler is not None,
        }
    except Exception as e:
        checks["workflow"] = {"status": "error", "error": str(e)}

    # 3. 알파 팩토리
    try:
        from app.alpha.scheduler import get_scheduler
        factory = get_scheduler()
        f_status = factory.get_status()
        checks["alpha_factory"] = {
            "status": "ok",
            "running": f_status["running"],
            "cycles": f_status.get("cycles_completed", 0),
            "factors": f_status.get("factors_discovered_total", 0),
        }
    except Exception as e:
        checks["alpha_factory"] = {"status": "error", "error": str(e)}

    # 4. LiveSession
    try:
        from app.trading.live_runner import list_sessions
        sessions = list_sessions()
        active = [s for s in sessions if s.status == "running"]
        checks["trading_sessions"] = {
            "status": "ok",
            "total": len(sessions),
            "active": len(active),
        }
    except Exception as e:
        checks["trading_sessions"] = {"status": "error", "error": str(e)}

    # 5. 시스템 리소스
    try:
        import psutil
        mem = psutil.virtual_memory()
        checks["system"] = {
            "memory_used_pct": round(mem.percent, 1),
            "memory_available_mb": round(mem.available / 1024 / 1024, 0),
            "cpu_percent": psutil.cpu_percent(interval=0.1),
        }
    except ImportError:
        checks["system"] = {"status": "ok", "note": "psutil 미설치"}

    # 6. 종합 판정
    all_ok = all(
        c.get("status") == "ok"
        for c in checks.values()
        if isinstance(c, dict) and "status" in c
    )

    return {
        "status": "ok" if all_ok else "degraded",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
    }
