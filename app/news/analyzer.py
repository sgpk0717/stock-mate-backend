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

    # 1차: Gemini 시도 (5건씩 서브배치, asyncio.gather 병렬 — Semaphore로 동시성 제한)
    if has_gemini:
        import asyncio

        gemini_batch_size = 5
        gemini_sem = asyncio.Semaphore(3)  # 최대 3개 동시 호출

        async def _run_sub_batch(offset: int, sub_texts: list[str]) -> list[SentimentResult]:
            async with gemini_sem:
                sub_message = "\n\n".join(sub_texts)
                sub_messages = [{"role": "user", "content": sub_message}]
                results = await _analyze_with_gemini(sub_messages)
                for r in results:
                    r.article_index += offset
                return results

        tasks = []
        for i in range(0, len(articles), gemini_batch_size):
            sub_articles = article_texts[i : i + gemini_batch_size]
            tasks.append(_run_sub_batch(i, sub_articles))

        all_results: list[SentimentResult] = []
        gemini_failed = False
        try:
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            for br in batch_results:
                if isinstance(br, Exception):
                    logger.warning("Gemini 서브배치 실패: %s", br)
                    gemini_failed = True
                else:
                    all_results.extend(br)
        except Exception as e:
            logger.warning("Gemini gather 실패: %s", e)
            gemini_failed = True

        if not gemini_failed and all_results:
            logger.info("Gemini 감성 분석 완료: %d건 (병렬 %d배치)", len(all_results), len(tasks))
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


# ── 종토방 전용 경량 감성분석 ──

_DISC_PROMPT = """\
한국 주식 종목토론방 게시글의 감성을 분석하세요.
각 게시글에 대해 [감성점수, 확신도] 쌍을 반환하세요.
- 감성점수: -1.0(매우 부정/하락) ~ +1.0(매우 긍정/상승). 중립은 0.0.
- 확신도: 0.0(풍자/반어/애매) ~ 1.0(명확한 의견).

응답은 JSON 2차원 배열만 출력. 입력 순서 동일.
예: [[-0.3, 0.8], [0.5, 0.6], [0.0, 0.3]]

설명 절대 금지. 배열만 출력."""


async def analyze_discussion_batch(titles: list[str]) -> list[tuple[float, float]]:
    """종토방 게시글 경량 감성분석 — 50건씩, [score, magnitude] 쌍 반환.

    Returns:
        [(score, magnitude), ...] 리스트 (입력과 동일 순서)
    """
    if not titles:
        return []

    lines = [f"{i}. {t[:80]}" for i, t in enumerate(titles)]
    user_text = "\n".join(lines)

    try:
        from app.core.llm import chat_gemini
        resp = await chat_gemini(
            system=_DISC_PROMPT,
            messages=[{"role": "user", "content": user_text}],
            max_tokens=32000,
            temperature=0.0,
            json_mode=True,
            caller="discussion.analyzer",
        )
        text = _repair_json(resp.text)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # 잘린 JSON 복구: 마지막 유효한 ] 위치까지 파싱
            last_bracket = text.rfind("]")
            if last_bracket > 0:
                truncated = text[:last_bracket + 1]
                # 열린 [ 짝 맞추기
                if truncated.count("[") > truncated.count("]"):
                    truncated += "]"
                data = json.loads(truncated)
                logger.info("종토방 감성분석: 잘린 JSON 복구 (%d chars)", last_bracket)
            else:
                raise
        if isinstance(data, list):
            result: list[tuple[float, float]] = []
            for item in data[:len(titles)]:
                if isinstance(item, list) and len(item) >= 2:
                    result.append((_clamp(float(item[0]), -1, 1), _clamp(float(item[1]), 0, 1)))
                elif isinstance(item, (int, float)):
                    result.append((_clamp(float(item), -1, 1), 0.5))
                else:
                    result.append((0.0, 0.5))
            return result
    except Exception as e:
        logger.warning("종토방 경량 감성분석 실패: %s", e)

    return [(0.0, 0.5)] * len(titles)


# ── 공시 전용 감성분석 ──

_NOTICE_PROMPT = """\
한국 주식시장 공시(공시, 시장조치, 수시공시 등)를 분석하세요.
각 공시에 대해 해당 종목의 호재/악재 여부와 중요도를 판단하세요.

각 공시에 대해 [감성점수, 중요도] 쌍을 반환하세요.
- 감성점수: -1.0(강한 악재: 상장폐지, 횡령) ~ +1.0(강한 호재: 대규모 수주, 흑자전환). 중립은 0.0.
- 중요도: 0.0(일상적 공시: 주총소집, 정기보고) ~ 1.0(중대 공시: 최대주주변경, 합병, 유상증자).

응답은 JSON 2차원 배열만 출력. 입력 순서 동일.
예: [[0.5, 0.8], [-0.7, 0.9], [0.0, 0.1]]
설명 절대 금지. 배열만 출력."""


async def analyze_notices_batch(
    items: list[dict],
) -> list[tuple[float, float]]:
    """공시 배치 감성분석 — [score, impact] 쌍 반환.

    Args:
        items: [{"title": str, "content": str, "notice_type": str}] (최대 100건)
    """
    if not items:
        return []

    lines = []
    for i, item in enumerate(items):
        nt = item.get("notice_type", "")
        title = item.get("title", "")[:100]
        content = item.get("content", "")[:200]
        lines.append(f"{i}. [{nt}] {title} | {content}")
    user_text = "\n".join(lines)

    try:
        from app.core.llm import chat_gemini
        resp = await chat_gemini(
            system=_NOTICE_PROMPT,
            messages=[{"role": "user", "content": user_text}],
            max_tokens=16000,
            temperature=0.0,
            json_mode=True,
            caller="notice.analyzer",
        )
        text = _repair_json(resp.text)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            last = text.rfind("]")
            if last > 0:
                truncated = text[:last + 1]
                if truncated.count("[") > truncated.count("]"):
                    truncated += "]"
                data = json.loads(truncated)
            else:
                raise
        if isinstance(data, list):
            result: list[tuple[float, float]] = []
            for item_data in data[:len(items)]:
                if isinstance(item_data, list) and len(item_data) >= 2:
                    result.append((_clamp(float(item_data[0]), -1, 1), _clamp(float(item_data[1]), 0, 1)))
                else:
                    result.append((0.0, 0.1))
            return result
    except Exception as e:
        logger.warning("공시 감성분석 실패: %s", e)

    return [(0.0, 0.1)] * len(items)


# ── 실시간 뉴스 전용 감성분석 (종목 추출 포함) ──

_NEWS_PROMPT = """\
한국 주식시장 뉴스 기사를 분석하세요.
각 기사에 대해 [감성점수, 시장영향도, 관련종목코드] 를 반환하세요.

- 감성점수: -1.0(매우 부정/하락) ~ +1.0(매우 긍정/상승). 중립 0.0.
- 시장영향도: 0.0(관심 낮음) ~ 1.0(시장 전체 흔드는 뉴스).
- 관련종목코드: 기사에서 언급된 한국 상장사의 6자리 종목코드를 콤마 구분. 없으면 빈 문자열.
  예: "005930,000660" (삼성전자, SK하이닉스)

응답은 JSON 2차원 배열만 출력. 입력 순서 동일.
예: [[0.5, 0.7, "005930,000660"], [-0.3, 0.4, "035720"], [0.0, 0.2, ""]]
설명 절대 금지. 배열만 출력."""


async def analyze_news_batch(
    items: list[dict],
) -> list[tuple[float, float, str]]:
    """뉴스 배치 감성분석 — [score, impact, symbols_csv] 반환.

    Args:
        items: [{"title": str, "subcontent": str, "office": str}] (최대 200건)
    """
    if not items:
        return []

    lines = []
    for i, item in enumerate(items):
        office = item.get("office", "")
        title = item.get("title", "")[:100]
        sub = item.get("subcontent", "")[:150]
        lines.append(f"{i}. [{office}] {title} | {sub}")
    user_text = "\n".join(lines)

    try:
        from app.core.llm import chat_gemini
        resp = await chat_gemini(
            system=_NEWS_PROMPT,
            messages=[{"role": "user", "content": user_text}],
            max_tokens=32000,
            temperature=0.0,
            json_mode=True,
            caller="news_flash.analyzer",
        )
        text = _repair_json(resp.text)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            last = text.rfind("]")
            if last > 0:
                truncated = text[:last + 1]
                if truncated.count("[") > truncated.count("]"):
                    truncated += "]"
                data = json.loads(truncated)
            else:
                raise
        if isinstance(data, list):
            result: list[tuple[float, float, str]] = []
            for item_data in data[:len(items)]:
                if isinstance(item_data, list) and len(item_data) >= 3:
                    result.append((
                        _clamp(float(item_data[0]), -1, 1),
                        _clamp(float(item_data[1]), 0, 1),
                        str(item_data[2]) if item_data[2] else "",
                    ))
                elif isinstance(item_data, list) and len(item_data) >= 2:
                    result.append((_clamp(float(item_data[0]), -1, 1), _clamp(float(item_data[1]), 0, 1), ""))
                else:
                    result.append((0.0, 0.3, ""))
            return result
    except Exception as e:
        logger.warning("뉴스 감성분석 실패: %s", e)

    return [(0.0, 0.3, "")] * len(items)
