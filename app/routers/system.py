"""시스템 토폴로지 + 수집기 모니터링 API.

프론트엔드 /system, /workflow 페이지에서 5~10초 폴링으로 호출.
성능 영향 최소: Redis HGETALL (~0.1ms) + scheduler get_status (~0ms) + 캐싱.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/system", tags=["system"])

# 캐시 (5초 TTL)
_topology_cache: dict | None = None
_topology_cache_at: float = 0
_CACHE_TTL = 5.0

# 수집기 캐시 (5초 TTL)
_collectors_cache: dict | None = None
_collectors_cache_at: float = 0


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


# ── 수집기 모니터링 ─────────────────────────────────────────────────────────


@router.get("/collectors")
async def get_collectors():
    """모든 데이터 수집기의 현재 상태를 반환.

    소스:
    - 장중 분봉 / 프로그램 매매: Redis (collector:intraday, collector:program_trading)
    - 일일 배치 잡 6개: DailyScheduler.get_status()
    """
    global _collectors_cache, _collectors_cache_at

    now = time.monotonic()
    if _collectors_cache and now - _collectors_cache_at < _CACHE_TTL:
        return _collectors_cache

    KST = timezone(timedelta(hours=9))
    collectors: dict[str, dict] = {}

    # ── 장중 분봉 수집기 (Redis) ──
    try:
        from app.core.redis import hgetall
        raw = await hgetall("collector:intraday")
        if raw:
            collectors["intraday_candle"] = {
                "name": "장중 분봉",
                "type": "realtime",
                "interval": "5분",
                "status": raw.get("status", "unknown"),
                "last_at": raw.get("last_at", ""),
                "last_count": int(raw.get("last_count", 0)),
                "symbols_total": int(raw.get("symbols_total", 0)),
                "next_at": raw.get("next_at", ""),
                "error": raw.get("error", ""),
            }
        else:
            collectors["intraday_candle"] = {
                "name": "장중 분봉",
                "type": "realtime",
                "interval": "5분",
                "status": "unknown",
            }
    except Exception:
        collectors["intraday_candle"] = {
            "name": "장중 분봉",
            "type": "realtime",
            "interval": "5분",
            "status": "unknown",
        }

    # ── 프로그램 매매 수집기 (Redis) ──
    try:
        from app.core.redis import hgetall
        raw = await hgetall("collector:program_trading")
        if raw:
            collectors["program_trading"] = {
                "name": "프로그램 매매",
                "type": "realtime",
                "interval": "5분",
                "status": raw.get("status", "unknown"),
                "last_at": raw.get("last_at", ""),
                "success": int(raw.get("success", 0)),
                "failed": int(raw.get("failed", 0)),
                "symbols_total": int(raw.get("symbols_total", 0)),
                "daily_rounds": int(raw.get("daily_rounds", 0)),
                "next_at": raw.get("next_at", ""),
                "error": raw.get("error", ""),
            }
        else:
            collectors["program_trading"] = {
                "name": "프로그램 매매",
                "type": "realtime",
                "interval": "5분",
                "status": "unknown",
            }
    except Exception:
        collectors["program_trading"] = {
            "name": "프로그램 매매",
            "type": "realtime",
            "interval": "5분",
            "status": "unknown",
        }

    # ── 일일 배치 스케줄러 잡 6개 (Redis 우선 → 로컬 폴백) ──
    scheduler_running = False
    sched_status: dict = {}

    # 1) Redis에서 Worker의 스케줄러 상태 읽기
    try:
        from app.core.redis import hgetall
        import json as _json
        redis_sched = await hgetall("scheduler:daily")
        if redis_sched and redis_sched.get("running"):
            scheduler_running = redis_sched.get("running") == "1"
            jobs_raw = redis_sched.get("jobs", "{}")
            try:
                sched_status = {
                    "running": scheduler_running,
                    "jobs": _json.loads(jobs_raw) if isinstance(jobs_raw, str) else jobs_raw,
                    "last_run_date": redis_sched.get("last_run_date", ""),
                    "next_run_at": redis_sched.get("next_run_at", ""),
                }
            except (_json.JSONDecodeError, TypeError):
                sched_status = {}
    except Exception as e:
        logger.debug("Redis 스케줄러 상태 조회 실패: %s", e)

    # 2) Redis 미사용 시 로컬 싱글턴 폴백
    if not sched_status:
        try:
            from app.scheduler.daily_scheduler import get_daily_scheduler
            sched = get_daily_scheduler()
            sched_status = sched.get_status()
            scheduler_running = sched_status.get("running", False)
        except Exception as e:
            logger.debug("로컬 스케줄러 상태 조회 실패: %s", e)

    _JOB_LABELS = {
        "daily_candle": "일봉",
        "minute_candle": "분봉 (배치)",
        "news": "뉴스",
        "margin_short": "신용/공매도",
        "investor": "투자자 수급",
        "dart_financial": "DART 재무",
    }

    for job_name, label in _JOB_LABELS.items():
        job_data = sched_status.get("jobs", {}).get(job_name, {})
        collectors[job_name] = {
            "name": label,
            "type": "batch",
            "interval": "일 1회",
            "status": job_data.get("status", "waiting"),
            "total": job_data.get("total", 0),
            "completed": job_data.get("completed", 0),
            "failed": job_data.get("failed", 0),
            "last_run_date": sched_status.get("last_run_date", ""),
            "next_run_at": sched_status.get("next_run_at", ""),
            "error": job_data.get("error_message", job_data.get("error", "")),
        }

    result = {
        "timestamp": datetime.now(KST).isoformat(),
        "scheduler_running": scheduler_running,
        "collectors": collectors,
    }

    _collectors_cache = result
    _collectors_cache_at = now
    return result


@router.post("/collectors/{collector_id}/restart")
async def restart_collector(collector_id: str):
    """개별 수집기 재시작.

    - intraday_candle: 장중 분봉 수집기 cancel → 재시작
    - program_trading: stop → start
    - 일일 배치 잡: /scheduler/trigger 위임
    """
    if collector_id == "intraday_candle":
        from app.trading.live_runner import (
            _intraday_collector_task,
            start_intraday_candle_collector,
        )
        # 기존 태스크 취소
        if _intraday_collector_task and not _intraday_collector_task.done():
            _intraday_collector_task.cancel()
            try:
                await _intraday_collector_task
            except Exception:
                pass

        # 종목 목록 로드 후 재시작
        try:
            from app.core.redis import get_client
            r = get_client()
            cached_symbols = await r.smembers("universe:symbols")
            symbols = list(cached_symbols) if cached_symbols else []
        except Exception:
            symbols = []

        if not symbols:
            # DB fallback
            from app.core.database import async_session
            from sqlalchemy import text
            async with async_session() as db:
                rows = await db.execute(text("SELECT symbol FROM stock_masters LIMIT 200"))
                symbols = [r[0] for r in rows.fetchall()]

        if not symbols:
            raise HTTPException(400, "수집 대상 종목을 찾을 수 없음")

        await start_intraday_candle_collector(symbols)
        return {"success": True, "message": f"장중 분봉 수집기 재시작 ({len(symbols)}종목)"}

    elif collector_id == "program_trading":
        import asyncio as _aio
        from app.services.program_trading_collector import (
            stop_collector,
            start_collector,
            set_task,
        )
        stop_collector()
        await _aio.sleep(0.5)
        task = _aio.create_task(start_collector())
        set_task(task)
        return {"success": True, "message": "프로그램 매매 수집기 재시작"}

    elif collector_id in (
        "daily_candle", "minute_candle", "news",
        "margin_short", "investor", "dart_financial",
    ):
        from app.scheduler.daily_scheduler import get_daily_scheduler
        sched = get_daily_scheduler()
        triggered, msg = await sched.trigger_job(job_name=collector_id)
        if not triggered:
            raise HTTPException(409, msg)
        return {"success": True, "message": msg}

    else:
        raise HTTPException(
            400,
            f"알 수 없는 수집기: {collector_id}. "
            f"허용: intraday_candle, program_trading, daily_candle, "
            f"minute_candle, news, margin_short, investor, dart_financial",
        )


# ── LLM 사용량 조회 ─────────────────────────────────────────


@router.get("/llm-usage")
async def get_llm_usage(
    days: int = 7,
    caller: str | None = None,
    provider: str | None = None,
    group_by: str = "caller",
):
    """LLM 사용량 집계 조회.

    group_by: caller | provider | model | date | caller_date
    """
    from sqlalchemy import Date,cast, func, select
    from sqlalchemy.ext.asyncio import AsyncSession

    from app.core.database import async_session
    from app.core.llm._models import LLMUsageLog

    since = datetime.now(timezone.utc) - timedelta(days=days)

    async with async_session() as session:
        # group_by 컬럼 결정
        if group_by == "provider":
            group_cols = [LLMUsageLog.provider]
        elif group_by == "model":
            group_cols = [LLMUsageLog.model]
        elif group_by == "date":
            group_cols = [cast(LLMUsageLog.created_at, Date).label("date")]
        elif group_by == "caller_date":
            group_cols = [
                LLMUsageLog.caller,
                cast(LLMUsageLog.created_at, Date).label("date"),
            ]
        else:  # default: caller
            group_cols = [LLMUsageLog.caller]

        stmt = (
            select(
                *group_cols,
                func.count().label("total_calls"),
                func.sum(LLMUsageLog.input_tokens).label("total_input_tokens"),
                func.sum(LLMUsageLog.output_tokens).label("total_output_tokens"),
                func.sum(LLMUsageLog.total_tokens).label("total_tokens"),
                func.sum(LLMUsageLog.cost_usd).label("total_cost_usd"),
                func.avg(LLMUsageLog.duration_ms).label("avg_duration_ms"),
            )
            .where(LLMUsageLog.created_at >= since)
            .where(LLMUsageLog.status == "success")
        )

        if caller:
            stmt = stmt.where(LLMUsageLog.caller == caller)
        if provider:
            stmt = stmt.where(LLMUsageLog.provider == provider)

        stmt = stmt.group_by(*group_cols).order_by(
            func.sum(LLMUsageLog.cost_usd).desc()
        )

        result = await session.execute(stmt)
        rows = result.all()

    return [
        {
            **{col.key if hasattr(col, "key") else str(col): getattr(row, col.key if hasattr(col, "key") else str(col), None) for col in group_cols},
            "total_calls": row.total_calls,
            "total_input_tokens": row.total_input_tokens or 0,
            "total_output_tokens": row.total_output_tokens or 0,
            "total_tokens": row.total_tokens or 0,
            "total_cost_usd": round(float(row.total_cost_usd or 0), 4),
            "avg_duration_ms": int(row.avg_duration_ms or 0),
        }
        for row in rows
    ]


@router.get("/llm-usage/recent")
async def get_llm_usage_recent(limit: int = 50):
    """최근 LLM 호출 로그 (디버깅용)."""
    from sqlalchemy import select

    from app.core.database import async_session
    from app.core.llm._models import LLMUsageLog

    async with async_session() as session:
        stmt = (
            select(LLMUsageLog)
            .order_by(LLMUsageLog.created_at.desc())
            .limit(min(limit, 200))
        )
        result = await session.execute(stmt)
        rows = result.scalars().all()

    return [
        {
            "id": str(r.id),
            "caller": r.caller,
            "provider": r.provider,
            "model": r.model,
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "total_tokens": r.total_tokens,
            "cost_usd": r.cost_usd,
            "status": r.status,
            "error": r.error,
            "duration_ms": r.duration_ms,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]
