"""BigKinds API 뉴스 수집기.

한국언론진흥재단 BigKinds API를 통해 뉴스를 수집한다.
API 키: https://www.bigkinds.or.kr/ 에서 발급.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import httpx

from app.core.config import settings

from .naver import RawArticle

logger = logging.getLogger(__name__)

BIGKINDS_SEARCH_URL = "https://tools.kinds.or.kr:8443/search/news"


async def collect_news(
    query: str,
    *,
    days: int = 7,
    max_results: int = 20,
    timeout: float = 15.0,
) -> list[RawArticle]:
    """BigKinds에서 키워드 기반 뉴스를 검색한다.

    Args:
        query: 검색 키워드 (예: "삼성전자")
        days: 최근 N일간
        max_results: 최대 결과 수
        timeout: 타임아웃 (초)

    Returns:
        RawArticle 리스트
    """
    api_key = settings.BIGKINDS_API_KEY
    if not api_key:
        logger.warning("BIGKINDS_API_KEY 미설정. BigKinds 수집 건너뜀.")
        return []

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    payload = {
        "access_key": api_key,
        "argument": {
            "query": query,
            "published_at": {
                "from": start_date.strftime("%Y-%m-%d"),
                "until": end_date.strftime("%Y-%m-%d"),
            },
            "sort": {"date": "desc"},
            "hilight": 200,
            "return_from": 0,
            "return_size": max_results,
            "fields": [
                "title",
                "content",
                "published_at",
                "provider",
                "category",
                "byline",
                "provider_link_page",
            ],
        },
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                BIGKINDS_SEARCH_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
    except (httpx.HTTPError, ValueError) as e:
        logger.error("BigKinds 수집 실패: %s", e)
        return []

    result_code = data.get("result", 0)
    if result_code != 0:
        logger.warning("BigKinds API 오류: %s", data.get("reason"))
        return []

    documents = data.get("return_object", {}).get("documents", [])

    articles: list[RawArticle] = []
    for doc in documents:
        title = doc.get("title", "")
        content = doc.get("content", "")
        published = doc.get("published_at", "")
        link = doc.get("provider_link_page", "")

        if not title:
            continue

        # 날짜 파싱
        try:
            published_at = datetime.strptime(published, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            try:
                published_at = datetime.strptime(published, "%Y-%m-%d")
            except ValueError:
                published_at = datetime.now()

        # URL: BigKinds 원문 링크 또는 고유 ID
        url = link or f"bigkinds://{doc.get('news_id', title[:50])}"

        articles.append(
            RawArticle(
                source="bigkinds",
                title=title,
                url=url,
                published_at=published_at,
                content=content[:2000] if content else None,
                symbols=None,  # 후처리에서 NER로 매핑
            )
        )

    logger.info("BigKinds 수집 완료: '%s' — %d건", query, len(articles))
    return articles


async def collect_stock_news(
    stock_name: str,
    symbol: str,
    *,
    days: int = 7,
    max_results: int = 20,
) -> list[RawArticle]:
    """종목명으로 BigKinds 뉴스를 수집한다."""
    articles = await collect_news(stock_name, days=days, max_results=max_results)
    for art in articles:
        art.symbols = [symbol]
    return articles
