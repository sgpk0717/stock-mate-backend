"""네이버 뉴스/공시 JSON API 수집기.

종토방과 동일 패턴 — stock.naver.com JSON API 직접 호출, 크롤링 불필요.
전 종목 통합 피드로 종목별 순회 없음.

API:
  - 공시: GET /api/domestic/news/noticeList?page={n}&pageSize=100
  - 실시간 뉴스: GET /api/domestic/news/list?page={n}&pageSize=100
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

import httpx

logger = logging.getLogger(__name__)

_BASE = "https://stock.naver.com/api/domestic/news"
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://stock.naver.com/news/flashnews",
    "Accept": "application/json",
}
_KST = timezone(timedelta(hours=9))
_DELAY = 0.3

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(headers=_HEADERS, timeout=15.0, follow_redirects=True)
    return _client


def _strip_html(html: str | None, max_len: int = 500) -> str:
    """HTML → 텍스트 (간단 strip)."""
    if not html:
        return ""
    import re
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_len]


# ── 데이터 클래스 ──

@dataclass
class NoticeRaw:
    """공시 원시 데이터."""
    no: str  # 공시 고유번호
    symbol: str
    symbol_name: str
    title: str
    content: str  # HTML → 텍스트 변환
    notice_type: str  # 수시공시, 시장조치 등
    published_at: datetime


@dataclass
class FlashNewsRaw:
    """실시간 뉴스 원시 데이터."""
    article_id: str  # officeId_articleId
    office: str  # 언론사
    title: str
    subcontent: str  # 요약
    published_at: datetime


# ── 공시 수집 ──

async def collect_notices(
    *,
    last_no: str | None = None,
    max_pages: int = 50,
    log_cb=None,
) -> list[NoticeRaw]:
    """공시 통합 피드 증분 수집."""
    client = _get_client()
    all_items: list[NoticeRaw] = []
    stop = False

    for page in range(1, max_pages + 1):
        resp = await client.get(f"{_BASE}/noticeList", params={"page": str(page), "pageSize": "100"})
        if resp.status_code != 200:
            logger.warning("공시 API %d: %s", resp.status_code, resp.text[:100])
            break

        data = resp.json()
        items = data.get("content", [])
        if not items:
            break

        for item in items:
            no = str(item.get("no", ""))
            if last_no and no == last_no:
                stop = True
                break

            dt_str = item.get("datetime", "")
            try:
                dt = datetime.fromisoformat(dt_str)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=_KST)
            except Exception:
                continue

            all_items.append(NoticeRaw(
                no=no,
                symbol=str(item.get("itemcode", "")),
                symbol_name=item.get("itemName", ""),
                title=item.get("title", ""),
                content=_strip_html(item.get("contents"), 500),
                notice_type=item.get("noticeTypeName", ""),
                published_at=dt,
            ))

        if log_cb and page % 5 == 0:
            await log_cb(f"  공시 page {page}: {len(items)}건 (누적 {len(all_items)}건)")

        if stop or data.get("last", False):
            break
        await asyncio.sleep(_DELAY)

    logger.info("공시 수집: %d건 (pages=%d)", len(all_items), page)
    return all_items


# ── 실시간 뉴스 수집 ──

async def collect_flash_news(
    *,
    last_article_id: str | None = None,
    max_pages: int = 50,
    log_cb=None,
) -> list[FlashNewsRaw]:
    """실시간 뉴스 통합 피드 증분 수집."""
    client = _get_client()
    all_items: list[FlashNewsRaw] = []
    stop = False

    for page in range(1, max_pages + 1):
        resp = await client.get(f"{_BASE}/list", params={"page": str(page), "pageSize": "100"})
        if resp.status_code != 200:
            logger.warning("뉴스 API %d: %s", resp.status_code, resp.text[:100])
            break

        data = resp.json()
        articles = data.get("articles", [])
        if not articles:
            break

        for art in articles:
            aid = f"{art.get('officeId', '')}_{art.get('articleId', '')}"
            if last_article_id and aid == last_article_id:
                stop = True
                break

            dt_str = art.get("datetime", "")
            try:
                dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                dt = dt.replace(tzinfo=_KST)
            except Exception:
                continue

            all_items.append(FlashNewsRaw(
                article_id=aid,
                office=art.get("officeHname", ""),
                title=art.get("title", ""),
                subcontent=art.get("subcontent", ""),
                published_at=dt,
            ))

        if log_cb and page % 5 == 0:
            await log_cb(f"  뉴스 page {page}: {len(articles)}건 (누적 {len(all_items)}건)")

        if stop:
            break
        await asyncio.sleep(_DELAY)

    logger.info("실시간 뉴스 수집: %d건 (pages=%d)", len(all_items), page)
    return all_items
