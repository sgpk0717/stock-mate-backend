"""네이버 금융 뉴스 수집기.

네이버 금융 종목 뉴스 페이지에서 기사 제목/링크/날짜를 크롤링한다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

NAVER_FINANCE_NEWS_URL = "https://finance.naver.com/news/news_list.naver"
NAVER_STOCK_NEWS_URL = "https://finance.naver.com/item/news_news.naver"


@dataclass
class RawArticle:
    """수집된 기사 원시 데이터."""

    source: str
    title: str
    url: str
    published_at: datetime
    content: str | None = None
    symbols: list[str] | None = None


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

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(
                NAVER_STOCK_NEWS_URL,
                params=params,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Referer": "https://finance.naver.com",
                },
            )
            resp.raise_for_status()
    except httpx.HTTPError as e:
        logger.error("네이버 뉴스 수집 실패 (%s): %s", symbol, e)
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
                published_at=datetime.now(),  # 상세 페이지에서 정확한 날짜 추출 필요
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

    logger.info("네이버 뉴스 수집 완료: %s — %d건", symbol, len(articles))
    return articles


async def fetch_article_content(url: str, *, timeout: float = 10.0) -> str | None:
    """개별 기사 본문을 가져온다."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                },
            )
            resp.raise_for_status()
    except httpx.HTTPError as e:
        logger.warning("기사 본문 수집 실패: %s", e)
        return None

    soup = BeautifulSoup(resp.text, "lxml")

    # 네이버 뉴스 본문 영역
    body = soup.select_one("#news_read") or soup.select_one(".article_body") or soup.select_one("#content")
    if body:
        return body.get_text(strip=True)[:2000]  # 2000자 제한

    return None
