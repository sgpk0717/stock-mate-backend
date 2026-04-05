"""네이버 종토방 수집기 — stock.naver.com JSON API 기반.

전 종목 통합 피드를 커서 페이지네이션으로 수집.
크롤링 불필요 (JSON 직접 파싱).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

import httpx

logger = logging.getLogger(__name__)

_BASE_URL = "https://stock.naver.com/api/community/discussion/posts"
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://stock.naver.com/discussion/feed/all",
    "Accept": "application/json",
    "Accept-Language": "ko-KR,ko;q=0.9",
}
_KST = timezone(timedelta(hours=9))
_PAGE_SIZE = 100
_REQUEST_DELAY = 0.5  # 초 (rate limit 매우 관대하지만 안전 마진)

_client: httpx.AsyncClient | None = None


def _get_client(timeout: float = 15.0) -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            headers=_HEADERS,
            timeout=httpx.Timeout(timeout),
            follow_redirects=True,
        )
    return _client


@dataclass
class DiscussionPostRaw:
    """종토방 게시글 원시 데이터."""
    nid: str
    symbol: str
    symbol_name: str
    title: str
    content: str | None
    author: str | None
    published_at: datetime  # KST, 초 단위
    likes: int
    dislikes: int
    comment_count: int


async def fetch_feed_page(
    *,
    offset: str | None = None,
    page_size: int = _PAGE_SIZE,
    item_code: str | None = None,
) -> tuple[list[DiscussionPostRaw], str | None]:
    """종토방 피드 1페이지 수집.

    Returns:
        (게시글 리스트, lastOffset 커서) — lastOffset이 None이면 마지막 페이지.
    """
    client = _get_client()
    params: dict[str, str] = {"pageSize": str(page_size)}
    if offset:
        params["offset"] = offset
    if item_code:
        params["itemCode"] = item_code

    resp = await client.get(_BASE_URL, params=params)

    if resp.status_code == 429:
        logger.warning("종토방 API 429 — 30초 대기")
        await asyncio.sleep(30)
        return [], None
    if resp.status_code != 200:
        logger.warning("종토방 API %d: %s", resp.status_code, resp.text[:200])
        return [], None

    data = resp.json()
    posts_raw = data.get("posts", [])
    last_offset = data.get("lastOffset")

    posts: list[DiscussionPostRaw] = []
    for p in posts_raw:
        try:
            # writtenAt: "2026-04-01T05:22:27" (KST naive)
            written_at = datetime.fromisoformat(p["writtenAt"])
            if written_at.tzinfo is None:
                written_at = written_at.replace(tzinfo=_KST)

            writer = p.get("writer") or {}
            posts.append(DiscussionPostRaw(
                nid=str(p["id"]),
                symbol=p.get("itemCode", ""),
                symbol_name=p.get("itemName", ""),
                title=p.get("title", ""),
                content=p.get("contentSwReplaced") or None,
                author=writer.get("nickname"),
                published_at=written_at,
                likes=p.get("recommendCount", 0) or 0,
                dislikes=p.get("notRecommendCount", 0) or 0,
                comment_count=p.get("commentCount", 0) or 0,
            ))
        except Exception as e:
            logger.debug("종토방 게시글 파싱 실패: %s", e)
            continue

    return posts, str(last_offset) if last_offset else None


async def collect_all_new_posts(
    *,
    last_nid: str | None = None,
    max_pages: int = 200,
    log_cb=None,
) -> list[DiscussionPostRaw]:
    """전체 피드에서 새 게시글만 수집 (증분).

    Args:
        last_nid: 마지막으로 수집한 게시글 ID. 이 ID가 나오면 중단.
        max_pages: 최대 페이지 수 (안전 장치).
        log_cb: 로그 콜백 (async callable).

    Returns:
        새 게시글 리스트 (최신순).
    """
    all_posts: list[DiscussionPostRaw] = []
    offset: str | None = None
    stop = False

    for page in range(1, max_pages + 1):
        posts, next_offset = await fetch_feed_page(offset=offset)
        if not posts:
            break

        for p in posts:
            if last_nid and p.nid == last_nid:
                stop = True
                break
            all_posts.append(p)

        if log_cb:
            await log_cb(f"종토방 피드 page {page}: {len(posts)}건 (누적 {len(all_posts)}건)")

        if stop or not next_offset:
            break

        offset = next_offset
        await asyncio.sleep(_REQUEST_DELAY)

    logger.info("종토방 수집 완료: %d건 (pages=%d)", len(all_posts), page)
    return all_posts
