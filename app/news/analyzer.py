"""뉴스 배치 감성 분석기.

Gemini 기본, Anthropic 폴백.
장마감 후 일괄 처리를 전제로 설계됨.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

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

# Gemini 네이티브 JSON 모드용 스키마
SENTIMENT_JSON_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "article_index": {"type": "integer"},
            "sentiment_score": {"type": "number"},
            "sentiment_magnitude": {"type": "number"},
            "market_impact": {"type": "number"},
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "symbol": {"type": "string"},
                        "relevance": {"type": "number"},
                    },
                    "required": ["name"],
                },
            },
        },
        "required": [
            "article_index",
            "sentiment_score",
            "sentiment_magnitude",
            "market_impact",
            "entities",
        ],
    },
}


async def analyze_batch(
    articles: list[dict],
    *,
    max_content_len: int = 500,
) -> list[SentimentResult]:
    """기사 배치에 대해 감성 분석을 수행한다.

    Gemini 우선, 실패 시 Anthropic 폴백.

    Args:
        articles: [{"title": str, "content": str | None, "source": str}]
        max_content_len: 본문 최대 길이 (truncate)

    Returns:
        SentimentResult 리스트
    """
    if not articles:
        return []

    # API 키 확인
    has_gemini = bool(settings.GEMINI_API_KEY)
    has_anthropic = bool(settings.ANTHROPIC_API_KEY)

    if not has_gemini and not has_anthropic:
        logger.warning("GEMINI_API_KEY, ANTHROPIC_API_KEY 모두 미설정. 감성 분석 건너뜀.")
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
    messages = [{"role": "user", "content": user_message}]

    # 1차: Gemini 시도 (5건씩 서브배치 — 큰 배치에서 JSON 잘림 방지)
    if has_gemini:
        gemini_batch_size = 5
        all_results: list[SentimentResult] = []
        gemini_failed = False

        for i in range(0, len(articles), gemini_batch_size):
            sub_articles = article_texts[i : i + gemini_batch_size]
            sub_message = "\n\n".join(sub_articles)
            sub_messages = [{"role": "user", "content": sub_message}]

            try:
                results = await _analyze_with_gemini(sub_messages)
                # article_index 보정 (서브배치 오프셋)
                for r in results:
                    r.article_index += i
                all_results.extend(results)
            except Exception as e:
                logger.warning("Gemini 감성 분석 실패 (batch %d~%d): %s", i, i + len(sub_articles), e)
                gemini_failed = True
                break

        if not gemini_failed and all_results:
            logger.info("Gemini 감성 분석 완료: %d건", len(all_results))
            return all_results

    # 2차: Anthropic 폴백
    if has_anthropic:
        try:
            results = await _analyze_with_anthropic(messages)
            if results:
                logger.info("Anthropic 감성 분석 완료 (폴백): %d건", len(results))
                return results
        except Exception as e:
            logger.error("Anthropic 감성 분석 실패: %s", e)

    return []


async def _analyze_with_gemini(messages: list[dict]) -> list[SentimentResult]:
    """Gemini로 감성 분석."""
    from app.core.llm import chat_gemini

    response = await chat_gemini(
        system=ANALYSIS_PROMPT,
        messages=messages,
        max_tokens=8000,
        temperature=0.1,
        json_mode=True,
        caller="news.analyzer",
    )

    # Gemini JSON 수리 후 파싱
    text = _repair_json(response.text)
    results_data = _parse_results(text)
    if not results_data:
        raise ValueError("Gemini 응답에서 JSON 추출 실패")
    return _build_results(results_data)


async def _analyze_with_anthropic(messages: list[dict]) -> list[SentimentResult]:
    """Anthropic Claude로 감성 분석 (폴백)."""
    from app.core.llm import chat_simple

    response = await chat_simple(
        system=ANALYSIS_PROMPT,
        messages=messages,
        max_tokens=2000,
        caller="news.analyzer",
    )

    results_data = _parse_results(response.text)
    return _build_results(results_data)


def _build_results(results_data: list[dict]) -> list[SentimentResult]:
    """파싱된 JSON → SentimentResult 리스트."""
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
    return results


def _repair_json(text: str) -> str:
    """Gemini가 출력한 불완전 JSON을 수리한다."""
    import re

    # 마크다운 코드블록 제거
    text = re.sub(r"```json\s*\n?", "", text)
    text = re.sub(r"\n?```", "", text)
    text = text.strip()

    # 마지막 완전한 객체까지 자르기 (잘린 JSON 복구)
    # 패턴: }] 또는 } ] 로 끝나야 함
    last_bracket = text.rfind("]")
    if last_bracket > 0:
        text = text[: last_bracket + 1]
    else:
        # ] 가 없으면 마지막 } 뒤에 ] 추가
        last_brace = text.rfind("}")
        if last_brace > 0:
            text = text[: last_brace + 1] + "]"

    # trailing comma 제거: ,] → ]  또는 ,} → }
    text = re.sub(r",\s*]", "]", text)
    text = re.sub(r",\s*}", "}", text)

    return text


def _parse_results(text: str) -> list[dict]:
    """응답에서 JSON 배열을 추출한다 (Anthropic 폴백용)."""
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
