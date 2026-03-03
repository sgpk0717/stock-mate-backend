"""DART OpenAPI 공시 수집기.

금융감독원 DART 전자공시시스템 API를 사용해 공시를 수집한다.
API 키: https://opendart.fss.or.kr/ 에서 발급.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import httpx

from app.core.config import settings

from .naver import RawArticle

logger = logging.getLogger(__name__)

DART_LIST_URL = "https://opendart.fss.or.kr/api/list.json"


async def collect_disclosures(
    corp_code: str | None = None,
    *,
    days: int = 7,
    page_count: int = 20,
    timeout: float = 15.0,
) -> list[RawArticle]:
    """DART 공시를 수집한다.

    Args:
        corp_code: DART 고유번호 (None이면 전체)
        days: 최근 N일간 공시
        page_count: 조회 건수
        timeout: 타임아웃 (초)

    Returns:
        RawArticle 리스트
    """
    api_key = settings.DART_API_KEY
    if not api_key:
        logger.warning("DART_API_KEY 미설정. 공시 수집 건너뜀.")
        return []

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    params: dict = {
        "crtfc_key": api_key,
        "bgn_de": start_date.strftime("%Y%m%d"),
        "end_de": end_date.strftime("%Y%m%d"),
        "page_count": str(page_count),
        "sort": "date",
        "sort_mth": "desc",
    }
    if corp_code:
        params["corp_code"] = corp_code

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(DART_LIST_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
    except (httpx.HTTPError, ValueError) as e:
        logger.error("DART 공시 수집 실패: %s", e)
        return []

    if data.get("status") != "000":
        logger.warning("DART API 응답 오류: %s", data.get("message"))
        return []

    articles: list[RawArticle] = []
    for item in data.get("list", []):
        rcept_no = item.get("rcept_no", "")
        title = item.get("report_nm", "")
        corp_name = item.get("corp_name", "")
        rcept_dt = item.get("rcept_dt", "")

        if not title or not rcept_no:
            continue

        # 공시 URL
        url = f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcept_no}"

        # 날짜 파싱
        try:
            published_at = datetime.strptime(rcept_dt, "%Y%m%d")
        except ValueError:
            published_at = datetime.now()

        articles.append(
            RawArticle(
                source="dart",
                title=f"[{corp_name}] {title}",
                url=url,
                published_at=published_at,
                content=None,  # DART 본문은 PDF → 별도 처리 필요
                symbols=None,  # corp_code → symbol 매핑 필요
            )
        )

    logger.info("DART 공시 수집 완료: %d건", len(articles))
    return articles


async def collect_stock_disclosures(
    symbol: str,
    corp_code: str,
    *,
    days: int = 30,
) -> list[RawArticle]:
    """종목별 DART 공시를 수집한다.

    symbol은 종목코드, corp_code는 DART 고유번호.
    """
    articles = await collect_disclosures(corp_code=corp_code, days=days)
    for art in articles:
        art.symbols = [symbol]
    return articles
