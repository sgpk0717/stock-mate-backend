"""시스템 토폴로지 API — 컨테이너 상태 + 메트릭 + 최근 이벤트.

프론트엔드 /system 페이지에서 5초 폴링으로 호출.
성능 영향 최소: Redis INFO (~0.1ms) + pg_stat (~1ms) + 5초 캐싱.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/system", tags=["system"])

# 캐시 (5초 TTL)
_topology_cache: dict | None = None
_topology_cache_at: float = 0
_CACHE_TTL = 5.0


@router.get("/topology")
async def get_topology():
    """시스템 토폴로지 — 노드 상태 + 메트릭 + 최근 이벤트."""
    global _topology_cache, _topology_cache_at

    now = time.monotonic()
    if _topology_cache and now - _topology_cache_at < _CACHE_TTL:
        return _topology_cache

    KST = timezone(timedelta(hours=9))
    now_kst = datetime.now(KST)

    nodes = {}
    edges = []
    events = []

    # ── 노드: Redis ──
    try:
        from app.core.redis import get_client
        r = get_client()
        info = await r.info(section="memory")
        stats = await r.info(section="stats")
        keyspace = await r.info(section="keyspace")
        db_keys = 0
        if keyspace:
            for db_info in keyspace.values():
                if isinstance(db_info, dict):
                    db_keys += db_info.get("keys", 0)
        nodes["redis"] = {
            "status": "healthy",
            "memory_mb": round(info.get("used_memory", 0) / 1024 / 1024, 1),
            "keys": db_keys,
            "ops_per_sec": stats.get("instantaneous_ops_per_sec", 0),
            "connected_clients": stats.get("connected_clients", 0),
        }
    except Exception as e:
        nodes["redis"] = {"status": "unhealthy", "error": str(e)[:100]}

    # ── 노드: PostgreSQL ──
    try:
        from app.core.database import async_session
        from sqlalchemy import text
        async with async_session() as db:
            # 활성 연결 수
            r = await db.execute(text("SELECT count(*) FROM pg_stat_activity WHERE state = 'active'"))
            active = r.scalar() or 0
            # 전체 연결 수
            r2 = await db.execute(text("SELECT count(*) FROM pg_stat_activity"))
            total = r2.scalar() or 0
            # DB 사이즈
            r3 = await db.execute(text("SELECT pg_database_size(current_database())"))
            db_size = r3.scalar() or 0
        nodes["db"] = {
            "status": "healthy",
            "active_queries": active,
            "total_connections": total,
            "db_size_mb": round(db_size / 1024 / 1024, 1),
        }
    except Exception as e:
        nodes["db"] = {"status": "unhealthy", "error": str(e)[:100]}

    # ── 노드: API ──
    nodes["api"] = {
        "status": "healthy",
        "worker_mode": "external",
        "uptime": now_kst.isoformat(),
    }

    # ── 노드: Worker ──
    try:
        from app.core.redis import hgetall
        wf = await hgetall("workflow:status")
        # 세션 수
        r = get_client()
        session_ids = await r.smembers("sessions:index")
        session_count = len(session_ids) if session_ids else 0

        # 각 세션의 trade_count 합산
        total_trades = 0
        for sid in (session_ids or []):
            sess = await r.hgetall(f"sessions:{sid}")
            total_trades += int(sess.get("trade_count", 0))

        nodes["worker"] = {
            "status": "healthy" if wf.get("phase") else "unknown",
            "phase": wf.get("phase", "?"),
            "workflow_status": wf.get("status", "?"),
            "sessions": session_count,
            "total_trades": total_trades,
            "mining_running": wf.get("mining_running", "false").lower() == "true",
        }
    except Exception as e:
        nodes["worker"] = {"status": "unknown", "error": str(e)[:100]}

    # ── 노드: MCP ──
    # MCP: TCP 연결 가능 여부로 판단 (SSE는 스트리밍이라 HTTP 체크 불가)
    import socket
    nodes["mcp"] = {"status": "unknown"}
    for mcp_host in ["mcp", "stockmate-mcp"]:
        try:
            s = socket.create_connection((mcp_host, 8009), timeout=2)
            s.close()
            nodes["mcp"] = {"status": "healthy"}
            break
        except Exception:
            continue

    # MCP audit 최근 활동
    try:
        from app.core.database import async_session
        from sqlalchemy import text
        async with async_session() as db:
            r = await db.execute(text(
                "SELECT count(*) FROM mcp_audit_logs WHERE created_at >= :since"
            ), {"since": now_kst - timedelta(hours=1)})
            mcp_calls_1h = r.scalar() or 0
            nodes["mcp"]["calls_1h"] = mcp_calls_1h
    except Exception:
        pass

    # ── 노드: OpenClaw ──
    openclaw_urls = ["http://host.docker.internal:18789/", "http://127.0.0.1:18789/"]
    nodes["openclaw"] = {"status": "unknown"}
    for oc_url in openclaw_urls:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=2) as client:
                resp = await client.get(oc_url)
                if resp.status_code == 200:
                    nodes["openclaw"] = {"status": "healthy"}
                    break
        except Exception:
            continue

    # ── 최근 이벤트 (기존 데이터 활용) ──
    try:
        from app.core.database import async_session
        from sqlalchemy import text
        async with async_session() as db:
            # 텔레그램 최근 5건
            r = await db.execute(text(
                "SELECT category, caller, status, created_at "
                "FROM telegram_message_logs ORDER BY created_at DESC LIMIT 5"
            ))
            for row in r.fetchall():
                events.append({
                    "type": "telegram",
                    "category": row[0],
                    "caller": row[1],
                    "status": row[2],
                    "ts": row[3].isoformat() if row[3] else "",
                })

            # MCP audit 최근 5건
            r2 = await db.execute(text(
                "SELECT tool_name, status, execution_ms, created_at "
                "FROM mcp_audit_logs ORDER BY created_at DESC LIMIT 5"
            ))
            for row in r2.fetchall():
                events.append({
                    "type": "mcp",
                    "tool": row[0],
                    "status": row[1],
                    "duration_ms": row[2],
                    "ts": row[3].isoformat() if row[3] else "",
                })

            # 매매 최근 5건
            r3 = await db.execute(text(
                "SELECT symbol, side, qty, price, executed_at "
                "FROM live_trades ORDER BY executed_at DESC LIMIT 5"
            ))
            for row in r3.fetchall():
                events.append({
                    "type": "trade",
                    "symbol": row[0],
                    "side": row[1],
                    "qty": row[2],
                    "price": float(row[3]) if row[3] else 0,
                    "ts": row[4].isoformat() if row[4] else "",
                })
    except Exception as e:
        logger.debug("이벤트 수집 실패: %s", e)

    # 이벤트 시간순 정렬
    events.sort(key=lambda e: e.get("ts", ""), reverse=True)

    # ── 엣지 정의 ──
    edges = [
        {"from": "api", "to": "redis", "label": "상태 읽기", "type": "read"},
        {"from": "worker", "to": "redis", "label": "매매 결과 저장", "type": "write"},
        {"from": "redis", "to": "worker", "label": "작업 지시", "type": "command"},
        {"from": "mcp", "to": "redis", "label": "상태 확인", "type": "read"},
        {"from": "worker", "to": "db", "label": "거래 데이터 저장", "type": "write"},
        {"from": "api", "to": "db", "label": "데이터 조회", "type": "read"},
        {"from": "mcp", "to": "db", "label": "로그 기록", "type": "write"},
        {"from": "worker", "to": "api", "label": "알림 전송", "type": "event"},
        {"from": "openclaw", "to": "mcp", "label": "실시간 연결", "type": "sse"},
    ]

    result = {
        "timestamp": now_kst.isoformat(),
        "nodes": nodes,
        "edges": edges,
        "events": events[:15],
    }

    _topology_cache = result
    _topology_cache_at = now
    return result
