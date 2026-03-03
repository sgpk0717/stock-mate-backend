"""Claude API 기반 배치 감성 분석기.

뉴스 기사를 배치로 묶어 Claude에 감성 분석을 요청한다.
장마감 후 일괄 처리를 전제로 설계됨.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    """단일 기사의 감성 분석 결과."""

    article_index: int
    sentiment_score: float  # -1.0 ~ +1.0
    sentiment_magnitude: float  # 0.0 ~ 1.0 (확신도)
    market_impact: float  # 0.0 ~ 1.0
    entities: list[dict]  # [{name, symbol, relevance}]


ANALYSIS_PROMPT = """\
당신은 한국 주식시장 전문 뉴스 감성 분석가입니다.
아래 뉴스 기사들을 분석하여 각 기사에 대해 다음 정보를 JSON 배열로 반환해주세요.

## 분석 항목
1. sentiment_score: -1.0(매우 부정) ~ +1.0(매우 긍정). 주가에 미치는 영향 기준.
2. sentiment_magnitude: 0.0(불확실) ~ 1.0(매우 확실). 분석 확신도.
3. market_impact: 0.0(무관) ~ 1.0(매우 큰 영향). 주가에 대한 영향력.
4. entities: 관련 기업 목록 [{name: "기업명", symbol: "종목코드(알면)", relevance: 0.0~1.0}]

## 판단 기준
- 실적 호조, 수주, 신사업 진출 → 긍정
- 실적 부진, 소송, 규제, 사고 → 부정
- 단순 사실 전달, 중립적 보도 → 0에 가깝게
- DART 공시는 사실 기반이므로 magnitude 높게
- 영향력은 해당 뉴스가 실제 주가를 움직일 가능성 기준

## 출력 형식
반드시 아래 JSON 배열만 출력. 설명 없이 JSON만:
[
  {
    "article_index": 0,
    "sentiment_score": 0.7,
    "sentiment_magnitude": 0.8,
    "market_impact": 0.6,
    "entities": [{"name": "삼성전자", "symbol": "005930", "relevance": 0.95}]
  },
  ...
]
"""


async def analyze_batch(
    articles: list[dict],
    *,
    max_content_len: int = 500,
) -> list[SentimentResult]:
    """기사 배치에 대해 감성 분석을 수행한다.

    Args:
        articles: [{"title": str, "content": str | None, "source": str}]
        max_content_len: 본문 최대 길이 (truncate)

    Returns:
        SentimentResult 리스트
    """
    if not settings.ANTHROPIC_API_KEY:
        logger.warning("ANTHROPIC_API_KEY 미설정. 감성 분석 건너뜀.")
        return []

    if not articles:
        return []

    # 기사 텍스트 구성
    article_texts = []
    for i, art in enumerate(articles):
        title = art.get("title", "")
        content = art.get("content", "") or ""
        source = art.get("source", "unknown")

        text = f"[기사 {i}] (출처: {source})\n제목: {title}"
        if content:
            text += f"\n본문: {content[:max_content_len]}"
        article_texts.append(text)

    user_message = "\n\n".join(article_texts)

    from app.core.llm import chat

    try:
        response = await chat(
            max_tokens=2000,
            system=ANALYSIS_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        text = response.content[0].text
        results_data = _parse_results(text)

        results: list[SentimentResult] = []
        for item in results_data:
            results.append(
                SentimentResult(
                    article_index=item.get("article_index", 0),
                    sentiment_score=_clamp(item.get("sentiment_score", 0), -1, 1),
                    sentiment_magnitude=_clamp(item.get("sentiment_magnitude", 0.5), 0, 1),
                    market_impact=_clamp(item.get("market_impact", 0.5), 0, 1),
                    entities=item.get("entities", []),
                )
            )

        logger.info("감성 분석 완료: %d건", len(results))
        return results

    except Exception as e:
        logger.error("감성 분석 실패: %s", e)
        return []


def _parse_results(text: str) -> list[dict]:
    """응답에서 JSON 배열을 추출한다."""
    import re

    # ```json ... ``` 패턴
    match = re.search(r"```json\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))

    # [ 로 시작하는 JSON 배열 찾기
    start = text.find("[")
    if start >= 0:
        end = text.rfind("]")
        if end > start:
            return json.loads(text[start : end + 1])

    # 단일 객체 시도
    start = text.find("{")
    if start >= 0:
        end = text.rfind("}")
        if end > start:
            obj = json.loads(text[start : end + 1])
            return [obj] if isinstance(obj, dict) else obj

    logger.warning("감성 분석 JSON 파싱 실패: %s...", text[:200])
    return []


def _clamp(value: float, min_v: float, max_v: float) -> float:
    """값을 범위 내로 제한."""
    return max(min_v, min(max_v, float(value)))
