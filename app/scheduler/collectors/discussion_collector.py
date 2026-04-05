"""종토방 24시간 상시 수집기.

장중(09:00~15:30): 5분 간격
장외: 10분 간격
감성분석: Gemini 경량 모델 (기존 analyzer.py)
상세 로그: Redis List (collector:discussion:logs) — 프론트 실시간 표시
"""

from __future__ import annotations

import asyncio
import logging

from app.core.timezone import now_kst

logger = logging.getLogger(__name__)

_task: asyncio.Task | None = None


async def _append_log(message: str) -> None:
    """종토방 수집기 로그를 Redis List에 추가 (프론트 실시간 표시용)."""
    try:
        from app.core.redis import get_client
        r = get_client()
        ts = now_kst().strftime("%H:%M:%S")
        entry = f"[{ts}] {message}"
        key = "collector:discussion:logs"
        await r.rpush(key, entry)
        await r.ltrim(key, -500, -1)
        await r.expire(key, 86400)
    except Exception:
        pass


async def _discussion_collect_loop() -> None:
    """24시간 상시 수집 루프."""
    from app.news.discussion_scorer import (
        collect_and_store,
        analyze_unscored,
        aggregate_hourly,
        _get_last_nid,
    )

    logger.info("종토방 수집기 시작 (24시간 상시)")

    while True:
        try:
            now = now_kst()
            is_market_hours = (
                now.weekday() < 5
                and 9 <= now.hour < 16
            )
            interval = 300 if is_market_hours else 600  # 장중 5분, 장외 10분

            # ── 사이클 시작 ──
            await _update_status("collecting")
            await _append_log(f"━━━ 수집 사이클 시작 ━━━")

            # 1. 수집
            last_nid = await _get_last_nid()
            await _append_log(f"· 증분 수집 시작 (last_nid={last_nid[:8] + '...' if last_nid else 'None'})")

            result = await collect_and_store(
                max_pages=100,
                log_cb=_append_log,
            )
            collected = result["collected"]
            stored = result["stored"]
            dupes = result["duplicates"]

            await _append_log(f"✔ 수집 완료: {collected}건 조회, {stored}건 신규 저장, {dupes}건 중복 스킵")

            # 2. 감성분석 — 미분석 0건 될 때까지 반복
            analyzed = 0
            round_num = 0
            # 미분석 잔여 건수 조회
            try:
                from app.core.database import async_session as _as
                from app.news.models import DiscussionPost as _DP
                from sqlalchemy import select as _sel, func as _fn
                async with _as() as _db:
                    _unscored_total = (await _db.execute(
                        _sel(_fn.count(_DP.id)).where(_DP.sentiment_score.is_(None))
                    )).scalar() or 0
                await _append_log(f"· 감성분석 시작 (미분석 {_unscored_total:,}건 잔여, 500건/배치)")
            except Exception:
                _unscored_total = 0
                await _append_log("· 감성분석 시작")

            while True:
                round_num += 1
                batch_analyzed = await analyze_unscored(
                    batch_size=500,
                    max_items=5000,
                    log_cb=_append_log,
                )
                analyzed += batch_analyzed
                if batch_analyzed == 0:
                    break
                remaining = max(0, _unscored_total - analyzed)
                await _append_log(f"  ✔ 감성분석 라운드 {round_num}: {batch_analyzed}건 (누적 {analyzed}건, 잔여 ~{remaining:,}건)")
            if analyzed > 0:
                await _append_log(f"✔ 감성분석 전체 완료: {analyzed}건 분석 (미분석 소진)")
            else:
                await _append_log("· 미분석 게시글 없음")

            # 3. 시간별 집계
            await _append_log("· 시간별 집계 시작 (최근 2시간)")
            aggregated = await aggregate_hourly(hours_back=2, log_cb=_append_log)
            await _append_log(f"✔ 시간별 집계: {aggregated}건 업서트")

            # 4. DB 통계
            try:
                from app.core.database import async_session
                from app.news.models import DiscussionPost
                from sqlalchemy import select, func
                async with async_session() as db:
                    total_result = await db.execute(select(func.count(DiscussionPost.id)))
                    total = total_result.scalar() or 0
                    analyzed_result = await db.execute(
                        select(func.count(DiscussionPost.id)).where(
                            DiscussionPost.sentiment_score.isnot(None)
                        )
                    )
                    total_analyzed = analyzed_result.scalar() or 0
                    symbols_result = await db.execute(
                        select(func.count(func.distinct(DiscussionPost.symbol)))
                    )
                    total_symbols = symbols_result.scalar() or 0
                await _append_log(
                    f"· DB 현황: 총 {total:,}건, 분석완료 {total_analyzed:,}건, {total_symbols}종목"
                )
            except Exception:
                pass

            next_at = (now_kst() + __import__("datetime").timedelta(seconds=interval)).strftime("%H:%M")
            await _append_log(
                f"✔━━━ 사이클 완료: 수집={stored}, 분석={analyzed}, 집계={aggregated}, 다음={next_at} ━━━"
            )

            await _update_status(
                "idle",
                last_count=stored,
                symbols_total=0,  # 전체 피드라 특정 종목 수 없음
                next_at=next_at,
            )

        except Exception as e:
            logger.error("종토방 수집 사이클 오류: %s", e, exc_info=True)
            await _append_log(f"✘ 오류: {str(e)[:200]}")
            await _update_status("error", error=str(e)[:200])

        await asyncio.sleep(interval)


async def _update_status(
    status: str,
    last_count: int = 0,
    symbols_total: int = 0,
    next_at: str = "",
    error: str = "",
) -> None:
    """Redis에 수집기 상태 기록."""
    try:
        from app.core.redis import hset, get_client
        _now = now_kst()
        await hset("collector:discussion", {
            "status": status,
            "last_at": _now.strftime("%H:%M:%S"),
            "last_date": _now.strftime("%Y%m%d"),
            "last_count": str(last_count),
            "symbols_total": str(symbols_total),
            "next_at": next_at,
            "error": error,
        })
        r = get_client()
        await r.expire("collector:discussion", 86400)
    except Exception as e:
        logger.warning("Redis 종토방 상태 업데이트 실패: %s", e)


async def start_discussion_collector() -> None:
    """종토방 수집기 시작 (백그라운드 태스크)."""
    global _task
    if _task and not _task.done():
        logger.info("종토방 수집기 이미 실행 중")
        return
    _task = asyncio.create_task(_discussion_collect_loop())
    logger.info("종토방 수집기 태스크 생성")


async def stop_discussion_collector() -> None:
    """종토방 수집기 중지."""
    global _task
    if _task and not _task.done():
        _task.cancel()
        try:
            await _task
        except asyncio.CancelledError:
            pass
    _task = None
    await _update_status("stopped")
    logger.info("종토방 수집기 중지")
