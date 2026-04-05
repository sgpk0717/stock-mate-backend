"""공시 + 실시간 뉴스 24시간 수집기.

stock.naver.com JSON API 기반 (크롤링 없음).
장중 5분, 장외 15분 간격.
"""

from __future__ import annotations

import asyncio
import logging

from app.core.timezone import now_kst

logger = logging.getLogger(__name__)

_task: asyncio.Task | None = None


async def _append_log(message: str) -> None:
    try:
        from app.core.redis import get_client
        r = get_client()
        ts = now_kst().strftime("%H:%M:%S")
        key = "collector:news_api:logs"
        await r.rpush(key, f"[{ts}] {message}")
        await r.ltrim(key, -500, -1)
        await r.expire(key, 86400)
    except Exception:
        pass


async def _update_status(status: str, **kwargs) -> None:
    try:
        from app.core.redis import hset, get_client
        _now = now_kst()
        fields = {
            "status": status,
            "last_at": _now.strftime("%H:%M:%S"),
            "last_date": _now.strftime("%Y%m%d"),
            **{k: str(v) for k, v in kwargs.items()},
        }
        await hset("collector:news_api", fields)
        r = get_client()
        await r.expire("collector:news_api", 86400)
    except Exception:
        pass


async def _news_api_collect_loop() -> None:
    from app.news.news_api_scorer import run_news_api_pipeline

    logger.info("공시+뉴스 API 수집기 시작 (24시간)")

    while True:
        try:
            now = now_kst()
            is_market = now.weekday() < 5 and 9 <= now.hour < 16
            interval = 300 if is_market else 900  # 장중 5분, 장외 15분

            await _update_status("collecting")
            await _append_log("━━━ 공시+뉴스 수집 시작 ━━━")

            result = await run_news_api_pipeline(log_cb=_append_log)

            notices = result["notices"]
            flash = result["flash"]
            analyzed = result["analyzed"]

            next_at = (now_kst() + __import__("datetime").timedelta(seconds=interval)).strftime("%H:%M")
            await _append_log(
                f"✔━━━ 완료: 공시 {notices['stored']}건, 뉴스 {flash['stored']}건, "
                f"분석 {analyzed}건, 다음={next_at} ━━━"
            )
            await _update_status("idle", next_at=next_at,
                                 last_count=str(notices["stored"] + flash["stored"]))

        except Exception as e:
            logger.error("공시+뉴스 수집 오류: %s", e, exc_info=True)
            await _append_log(f"✘ 오류: {str(e)[:200]}")
            await _update_status("error", error=str(e)[:200])

        await asyncio.sleep(interval)


async def start_news_api_collector() -> None:
    global _task
    if _task and not _task.done():
        logger.info("공시+뉴스 수집기 이미 실행 중")
        return
    _task = asyncio.create_task(_news_api_collect_loop())
    logger.info("공시+뉴스 수집기 태스크 생성")


async def stop_news_api_collector() -> None:
    global _task
    if _task and not _task.done():
        _task.cancel()
        try:
            await _task
        except asyncio.CancelledError:
            pass
    _task = None
    await _update_status("stopped")
