"""종토방 수집 → 감성분석 → 시간별 집계 파이프라인."""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

from sqlalchemy import Integer as SAInteger, case, select, func, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import async_session
import re

from app.core.timezone import KST, now_kst
from app.news.models import DiscussionPost, DiscussionSentimentHourly
from app.news.collectors.discussion import DiscussionPostRaw, collect_all_new_posts

# ── 홍보/스팸 필터 ──

_SPAM_KEYWORDS = [
    # 리딩방/카톡방/텔레그램 홍보
    "오픈카톡", "open.kakao", "카톡방", "카카오톡방", "단톡방",
    "텔레그램방", "t.me/", "텔레방",
    "리딩방", "무료리딩", "선취매", "수익인증",
    # 유료 서비스 홍보
    "무료추천", "무료종목", "종목추천", "VIP방",
    "월수익", "일수익", "수익률 인증", "실시간 종목",
    # 외부 링크 스팸
    "tistory.com", "naver.me", "blog.naver",
    "bit.ly", "url.kr",
]

_SPAM_PATTERNS = re.compile(
    "|".join(re.escape(kw) for kw in _SPAM_KEYWORDS),
    re.IGNORECASE,
)


def is_spam_post(title: str, content: str | None) -> bool:
    """제목+본문에서 홍보/스팸 여부 판별."""
    text = f"{title} {content or ''}"
    return bool(_SPAM_PATTERNS.search(text))

logger = logging.getLogger(__name__)


async def _get_last_nid() -> str | None:
    """Redis에서 마지막 수집 nid 로드."""
    try:
        from app.core.redis import get_client
        r = get_client()
        val = await r.get("discussion:last_nid")
        return str(val) if val else None
    except Exception:
        return None


async def _set_last_nid(nid: str) -> None:
    """Redis에 마지막 수집 nid 저장."""
    try:
        from app.core.redis import get_client
        r = get_client()
        await r.set("discussion:last_nid", nid)
    except Exception as e:
        logger.warning("Redis discussion:last_nid 저장 실패: %s", e)


async def collect_and_store(
    *,
    max_pages: int = 200,
    log_cb=None,
) -> dict:
    """종토방 수집 → DB 저장 (증분).

    Returns:
        {"collected": N, "stored": N, "duplicates": N}
    """
    last_nid = await _get_last_nid()
    posts = await collect_all_new_posts(last_nid=last_nid, max_pages=max_pages, log_cb=log_cb)

    if not posts:
        return {"collected": 0, "stored": 0, "duplicates": 0}

    stored = 0
    duplicates = 0

    async with async_session() as db:
        # 기존 nid 조회 (중복 방지)
        nids = [p.nid for p in posts]
        existing = set()
        for i in range(0, len(nids), 500):
            batch = nids[i:i + 500]
            result = await db.execute(
                select(DiscussionPost.nid).where(DiscussionPost.nid.in_(batch))
            )
            existing.update(r[0] for r in result)

        for p in posts:
            if p.nid in existing:
                duplicates += 1
                continue
            spam = is_spam_post(p.title, p.content)
            db.add(DiscussionPost(
                symbol=p.symbol,
                nid=p.nid,
                title=p.title,
                content=p.content,
                author=p.author,
                published_at=p.published_at,
                likes=p.likes,
                dislikes=p.dislikes,
                comment_count=p.comment_count,
                is_spam=spam,
            ))
            stored += 1

        await db.commit()

    # 최신 nid 저장 (첫 번째 = 가장 최신)
    if posts:
        await _set_last_nid(posts[0].nid)

    logger.info("종토방 저장: %d건 신규, %d건 중복", stored, duplicates)
    return {"collected": len(posts), "stored": stored, "duplicates": duplicates}


async def analyze_unscored(
    *,
    batch_size: int = 500,
    max_items: int = 1000,
    log_cb=None,
) -> int:
    """미분석 종토방 게시글 경량 감성분석.

    50건씩 제목만 보내고 점수 배열만 받음 (토큰 최소화).

    Returns:
        분석 완료 건수.
    """
    from app.news.analyzer import analyze_discussion_batch

    analyzed = 0
    async with async_session() as db:
        result = await db.execute(
            select(DiscussionPost)
            .where(DiscussionPost.sentiment_score.is_(None))
            .where(DiscussionPost.is_spam.is_(False))
            .order_by(DiscussionPost.published_at.desc())
            .limit(max_items)
        )
        unscored = list(result.scalars().all())

        if not unscored:
            return 0

        for i in range(0, len(unscored), batch_size):
            batch = unscored[i:i + batch_size]
            titles = [p.title for p in batch]
            try:
                pairs = await analyze_discussion_batch(titles)
                for j, (score, magnitude) in enumerate(pairs):
                    if j < len(batch):
                        batch[j].sentiment_score = score
                        batch[j].sentiment_magnitude = magnitude
                        batch[j].analyzed_at = datetime.now(timezone.utc)
                        analyzed += 1
                if log_cb:
                    await log_cb(
                        f"  · 감성분석 배치 {i // batch_size + 1}: "
                        f"{len(pairs)}건 완료 (누적 {analyzed}건)"
                    )
            except Exception as e:
                logger.warning("종토방 감성분석 배치 실패: %s", e)
                if log_cb:
                    await log_cb(f"  ✘ 감성분석 배치 실패: {str(e)[:100]}")
                continue

        await db.commit()

    if log_cb:
        await log_cb(f"✔ 감성분석 완료: {analyzed}/{len(unscored)}건")
    logger.info("종토방 감성분석 완료: %d건", analyzed)
    return analyzed


async def aggregate_hourly(
    *,
    hours_back: int = 2,
    log_cb=None,
) -> int:
    """discussion_posts → discussion_sentiment_hourly 시간별 집계.

    Returns:
        업서트된 행 수.
    """
    since = now_kst() - timedelta(hours=hours_back)

    async with async_session() as db:
        # 시간별 종목별 집계 쿼리
        hour_trunc = func.date_trunc("hour", DiscussionPost.published_at)
        result = await db.execute(
            select(
                DiscussionPost.symbol,
                hour_trunc.label("dt"),
                func.count().label("post_count"),
                func.avg(DiscussionPost.sentiment_score).label("avg_sentiment"),
                func.sum(case(
                    (DiscussionPost.sentiment_score > 0.2, 1), else_=0
                )).label("pos_count"),
                func.sum(case(
                    (DiscussionPost.sentiment_score < -0.2, 1), else_=0
                )).label("neg_count"),
                func.sum(DiscussionPost.likes).label("total_likes"),
                func.sum(DiscussionPost.dislikes).label("total_dislikes"),
            )
            .where(DiscussionPost.published_at >= since)
            .where(DiscussionPost.is_spam.is_(False))
            .group_by(DiscussionPost.symbol, hour_trunc)
        )
        rows = result.all()

        upserted = 0
        for row in rows:
            count = row.post_count or 1
            pos = row.pos_count or 0
            neg = row.neg_count or 0

            stmt = pg_insert(DiscussionSentimentHourly).values(
                symbol=row.symbol,
                dt=row.dt,
                post_count=count,
                avg_sentiment=row.avg_sentiment,
                positive_ratio=pos / count if count else None,
                negative_ratio=neg / count if count else None,
                total_likes=row.total_likes or 0,
                total_dislikes=row.total_dislikes or 0,
            ).on_conflict_do_update(
                constraint="uq_disc_hourly_symbol_dt",
                set_={
                    "post_count": count,
                    "avg_sentiment": row.avg_sentiment,
                    "positive_ratio": pos / count if count else None,
                    "negative_ratio": neg / count if count else None,
                    "total_likes": row.total_likes or 0,
                    "total_dislikes": row.total_dislikes or 0,
                    "updated_at": func.now(),
                },
            )
            await db.execute(stmt)
            upserted += 1

        await db.commit()

    if log_cb:
        await log_cb(f"종토방 시간별 집계: {upserted}건 업서트")
    logger.info("종토방 시간별 집계: %d건", upserted)
    return upserted


async def run_discussion_pipeline(
    *,
    max_pages: int = 200,
    analyze: bool = True,
    aggregate: bool = True,
    log_cb=None,
) -> dict:
    """수집 → 감성분석 → 시간별 집계 전체 파이프라인."""
    result = await collect_and_store(max_pages=max_pages, log_cb=log_cb)

    analyzed = 0
    if analyze and result["stored"] > 0:
        analyzed = await analyze_unscored(log_cb=log_cb)

    aggregated = 0
    if aggregate:
        aggregated = await aggregate_hourly(log_cb=log_cb)

    result["analyzed"] = analyzed
    result["aggregated"] = aggregated
    return result
