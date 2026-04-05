"""네이버 금융 뉴스 수집기.

네이버 금융 종목 뉴스 페이지에서 기사 제목/링크/날짜를 크롤링한다.
차단 방지: 요청 간 3-6초 랜덤 지터 + 429/403 감지 + 지수 백오프.
"""

from __future__ import annotations

from app.core.timezone import now_kst

import asyncio
import logging
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

NAVER_FINANCE_NEWS_URL = "https://finance.naver.com/news/news_list.naver"
NAVER_STOCK_NEWS_URL = "https://finance.naver.com/item/news_news.naver"

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://finance.naver.com",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
    "DNT": "1",
    "Connection": "keep-alive",
}

# 요청 간 최소/최대 딜레이 (초) — 차단 방지
_MIN_DELAY = 3.0
_MAX_DELAY = 6.0

# 429 감지 시 초기 대기 (초)
_RATE_LIMIT_BASE_WAIT = 30


@dataclass
class RawArticle:
    """수집된 기사 원시 데이터."""

    source: str
    title: str
    url: str
    published_at: datetime
    content: str | None = None
    symbols: list[str] | None = None


# ── 공유 클라이언트 (Connection keep-alive 재사용) ──

_client: httpx.AsyncClient | None = None


def _get_client(timeout: float = 10.0) -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=timeout,
            headers=_HEADERS,
            follow_redirects=True,
        )
    return _client


async def _safe_delay() -> None:
    """랜덤 지터 딜레이 (차단 방지)."""
    await asyncio.sleep(random.uniform(_MIN_DELAY, _MAX_DELAY))


async def _handle_response(resp: httpx.Response, context: str) -> bool:
    """응답 코드 체크. True=정상, False=차단/에러."""
    if resp.status_code == 429:
        retry_after = int(resp.headers.get("Retry-After", str(_RATE_LIMIT_BASE_WAIT)))
        logger.warning("네이버 429 (rate limit) — %s, %d초 대기", context, retry_after)
        await asyncio.sleep(retry_after)
        return False

    if resp.status_code == 403:
        logger.error("네이버 403 (차단) — %s, 수집 중단 권고", context)
        return False

    if resp.status_code >= 400:
        logger.warning("네이버 HTTP %d — %s", resp.status_code, context)
        return False

    # 소프트 차단 감지 (200이지만 빈 본문)
    if len(resp.text) < 100:
        logger.warning("네이버 소프트 차단 의심 (본문 %d bytes) — %s", len(resp.text), context)
        await asyncio.sleep(_RATE_LIMIT_BASE_WAIT)
        return False

    return True


async def collect_stock_news(
    symbol: str,
    page: int = 1,
    *,
    timeout: float = 10.0,
) -> list[RawArticle]:
    """종목별 네이버 금융 뉴스를 수집한다.

    Args:
        symbol: 종목 코드 (예: "005930")
        page: 페이지 번호
        timeout: HTTP 요청 타임아웃 (초)

    Returns:
        RawArticle 리스트
    """
    articles: list[RawArticle] = []

    params = {
        "code": symbol,
        "page": str(page),
        "sm": "title_entity_id.basic",
        "clusterId": "",
    }

    client = _get_client(timeout)

    try:
        resp = await client.get(NAVER_STOCK_NEWS_URL, params=params)

        if not await _handle_response(resp, f"종목뉴스 {symbol}"):
            return articles

    except httpx.HTTPError as e:
        logger.error("네이버 뉴스 수집 실패 (%s): %s: %s", symbol, type(e).__name__, e)
        return articles

    soup = BeautifulSoup(resp.text, "lxml")
    rows = soup.select("tr .title a")

    for a_tag in rows:
        title = a_tag.get_text(strip=True)
        href = a_tag.get("href", "")
        if not title or not href:
            continue

        # 절대 URL 생성
        if href.startswith("/"):
            href = f"https://finance.naver.com{href}"

        articles.append(
            RawArticle(
                source="naver",
                title=title,
                url=href,
                published_at=now_kst(),  # 상세 페이지에서 정확한 날짜 추출 필요
                symbols=[symbol],
            )
        )

    # 날짜 추출: td.date 셀에서 시도
    date_cells = soup.select("tr .date")
    for i, date_cell in enumerate(date_cells):
        date_text = date_cell.get_text(strip=True)
        if i < len(articles) and date_text:
            try:
                articles[i].published_at = datetime.strptime(date_text, "%Y.%m.%d %H:%M")
            except ValueError:
                try:
                    articles[i].published_at = datetime.strptime(date_text, "%Y.%m.%d")
                except ValueError:
                    pass

    # 차단 방지 딜레이
    await _safe_delay()

    logger.info("네이버 뉴스 수집 완료: %s — %d건", symbol, len(articles))
    return articles


async def fetch_article_content(url: str, *, timeout: float = 10.0) -> str | None:
    """개별 기사 본문을 가져온다."""
    client = _get_client(timeout)

    try:
        resp = await client.get(url)
        if not await _handle_response(resp, f"기사본문 {url[:60]}"):
            return None
    except httpx.HTTPError as e:
        logger.warning("기사 본문 수집 실패: %s", e)
        return None

    soup = BeautifulSoup(resp.text, "lxml")

    # 네이버 뉴스 본문 영역
    body = soup.select_one("#news_read") or soup.select_one(".article_body") or soup.select_one("#content")
    if body:
        return body.get_text(strip=True)[:2000]  # 2000자 제한

    # 차단 방지 딜레이
    await _safe_delay()

    return None
