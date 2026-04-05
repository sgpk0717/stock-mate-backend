"""공시 + 실시간 뉴스 JSON API 수집 → 감성분석 → DB 저장 파이프라인."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from sqlalchemy import select

from app.core.database import async_session
from app.core.timezone import now_kst
from app.news.models import NewsArticle
from app.news.collectors.naver_api import (
    NoticeRaw, FlashNewsRaw,
    collect_notices, collect_flash_news,
)

logger = logging.getLogger(__name__)


# ── Redis 증분 키 ──

async def _get_redis_key(key: str) -> str | None:
    try:
        from app.core.redis import get_client
        r = get_client()
        val = await r.get(key)
        return str(val) if val else None
    except Exception:
        return None


async def _set_redis_key(key: str, value: str) -> None:
    try:
        from app.core.redis import get_client
        r = get_client()
        await r.set(key, value)
    except Exception:
        pass


# ── 공시 수집 + 저장 ──

async def collect_and_store_notices(*, max_pages: int = 50, log_cb=None) -> dict:
    """공시 증분 수집 → news_articles 저장."""
    last_no = await _get_redis_key("news:last_notice_no")
    items = await collect_notices(last_no=last_no, max_pages=max_pages, log_cb=log_cb)

    if not items:
        return {"collected": 0, "stored": 0}

    stored = 0
    async with async_session() as db:
        # URL 기반 중복 방지
        urls = [f"naver_notice://{it.no}" for it in items]
        existing = set()
        for i in range(0, len(urls), 500):
            batch = urls[i:i + 500]
            result = await db.execute(
                select(NewsArticle.url).where(NewsArticle.url.in_(batch))
            )
            existing.update(r[0] for r in result)

        seen = set()
        for it in items:
            url = f"naver_notice://{it.no}"
            if url in existing or url in seen:
                continue
            seen.add(url)
            symbols = [it.symbol] if it.symbol else None
            db.add(NewsArticle(
                source="naver_notice",
                title=f"[{it.notice_type}] {it.title}",
                content=it.content[:2000] if it.content else None,
                url=url,
                published_at=it.published_at,
                symbols=symbols,
            ))
            stored += 1
        await db.commit()

    if items:
        await _set_redis_key("news:last_notice_no", items[0].no)

    if log_cb:
        await log_cb(f"✔ 공시: {len(items)}건 조회, {stored}건 저장")
    return {"collected": len(items), "stored": stored}


# ── 실시간 뉴스 수집 + 저장 ──

async def collect_and_store_flash_news(*, max_pages: int = 50, log_cb=None) -> dict:
    """실시간 뉴스 증분 수집 → news_articles 저장."""
    last_id = await _get_redis_key("news:last_flash_id")
    items = await collect_flash_news(last_article_id=last_id, max_pages=max_pages, log_cb=log_cb)

    if not items:
        return {"collected": 0, "stored": 0}

    stored = 0
    async with async_session() as db:
        urls = [f"naver_flash://{it.article_id}" for it in items]
        existing = set()
        for i in range(0, len(urls), 500):
            batch = urls[i:i + 500]
            result = await db.execute(
                select(NewsArticle.url).where(NewsArticle.url.in_(batch))
            )
            existing.update(r[0] for r in result)

        seen = set()
        for it in items:
            url = f"naver_flash://{it.article_id}"
            if url in existing or url in seen:
                continue
            seen.add(url)
            db.add(NewsArticle(
                source="naver_flash",
                title=it.title,
                content=it.subcontent[:2000] if it.subcontent else None,
                url=url,
                published_at=it.published_at,
                symbols=None,  # 감성분석 시 LLM이 추출
            ))
            stored += 1
        await db.commit()

    if items:
        await _set_redis_key("news:last_flash_id", items[0].article_id)

    if log_cb:
        await log_cb(f"✔ 뉴스: {len(items)}건 조회, {stored}건 저장")
    return {"collected": len(items), "stored": stored}


# ── 미분석 뉴스/공시 감성분석 ──

async def analyze_unscored_api_news(*, max_items: int = 500, log_cb=None) -> int:
    """naver_notice + naver_flash 미분석 건 감성분석."""
    from app.news.analyzer import analyze_notices_batch, analyze_news_batch

    analyzed = 0
    async with async_session() as db:
        # 공시 미분석
        result = await db.execute(
            select(NewsArticle)
            .where(NewsArticle.sentiment_score.is_(None))
            .where(NewsArticle.source == "naver_notice")
            .order_by(NewsArticle.published_at.desc())
            .limit(max_items)
        )
        notices = list(result.scalars().all())

        if notices:
            batch_dicts = [
                {"title": n.title, "content": (n.content or "")[:200], "notice_type": "공시"}
                for n in notices
            ]
            for i in range(0, len(batch_dicts), 100):
                batch = batch_dicts[i:i + 100]
                batch_articles = notices[i:i + 100]
                try:
                    pairs = await analyze_notices_batch(batch)
                    for j, (score, impact) in enumerate(pairs):
                        if j < len(batch_articles):
                            batch_articles[j].sentiment_score = score
                            batch_articles[j].market_impact = impact
                            batch_articles[j].sentiment_magnitude = abs(score)
                            batch_articles[j].analyzed_at = datetime.now(timezone.utc)
                            analyzed += 1
                except Exception as e:
                    logger.warning("공시 감성분석 실패: %s", e)
            if log_cb:
                await log_cb(f"  · 공시 감성분석: {analyzed}건")

        # 뉴스 미분석
        result = await db.execute(
            select(NewsArticle)
            .where(NewsArticle.sentiment_score.is_(None))
            .where(NewsArticle.source == "naver_flash")
            .order_by(NewsArticle.published_at.desc())
            .limit(max_items)
        )
        flash_articles = list(result.scalars().all())

        if flash_articles:
            news_analyzed = 0
            batch_dicts = [
                {"title": n.title, "subcontent": (n.content or "")[:150], "office": ""}
                for n in flash_articles
            ]
            for i in range(0, len(batch_dicts), 200):
                batch = batch_dicts[i:i + 200]
                batch_articles = flash_articles[i:i + 200]
                try:
                    triples = await analyze_news_batch(batch)
                    for j, (score, impact, symbols_csv) in enumerate(triples):
                        if j < len(batch_articles):
                            batch_articles[j].sentiment_score = score
                            batch_articles[j].market_impact = impact
                            batch_articles[j].sentiment_magnitude = abs(score)
                            batch_articles[j].analyzed_at = datetime.now(timezone.utc)
                            # LLM이 추출한 종목코드 반영
                            if symbols_csv:
                                syms = [s.strip() for s in symbols_csv.split(",") if s.strip()]
                                if syms:
                                    existing_syms = set(batch_articles[j].symbols or [])
                                    existing_syms.update(syms)
                                    batch_articles[j].symbols = list(existing_syms)
                            news_analyzed += 1
                            analyzed += 1
                except Exception as e:
                    logger.warning("뉴스 감성분석 실패: %s", e)
            if log_cb:
                await log_cb(f"  · 뉴스 감성분석: {news_analyzed}건")

        await db.commit()

    logger.info("API 뉴스/공시 감성분석 완료: %d건", analyzed)
    return analyzed


# ── 전체 파이프라인 ──

async def run_news_api_pipeline(*, log_cb=None) -> dict:
    """공시 + 뉴스 수집 → 감성분석 전체 파이프라인."""
    notice_result = await collect_and_store_notices(log_cb=log_cb)
    flash_result = await collect_and_store_flash_news(log_cb=log_cb)

    analyzed = 0
    total_stored = notice_result["stored"] + flash_result["stored"]
    if total_stored > 0:
        analyzed = await analyze_unscored_api_news(log_cb=log_cb)

    return {
        "notices": notice_result,
        "flash": flash_result,
        "analyzed": analyzed,
    }
