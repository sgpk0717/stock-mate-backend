"""Claude API 를 통한 자연어 → 구조화된 전략 JSON 변환."""

from __future__ import annotations

import json
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
당신은 주식 트레이딩 전략 설계 전문가입니다.
사용자가 자연어로 매수/매도 조건을 설명하면,
아래 JSON 스키마에 맞는 구조화된 전략을 생성해주세요.

## 지원 지표 목록
- rsi: RSI (params: period, 기본값 14)
- sma: 단순이동평균 (params: period)
- ema: 지수이동평균 (params: period)
- macd_hist: MACD 히스토그램 (params: fast=12, slow=26, signal=9)
- macd_cross: MACD-시그널 크로스 (bool, params: fast, slow, signal)
- bb_upper: 볼린저밴드 상단 (params: period=20, std=2.0)
- bb_lower: 볼린저밴드 하단 (params: period=20, std=2.0)
- volume_ratio: 거래량비율 = 당일거래량/N일평균 (params: period=20)
- price_change_pct: 가격변동률% (params: period=1)
- golden_cross: 골든크로스 (bool, params: fast_period, slow_period)
- dead_cross: 데드크로스 (bool, params: fast_period, slow_period)
- sentiment_score: 뉴스 감성 스코어 (-1.0~+1.0, 양수=긍정, 음수=부정)
- article_count: 뉴스 기사 수 (해당 종목의 최근 보도량)
- event_score: 이벤트 스코어 (감성×영향력×보도량 종합)

## 비교 연산자
>, >=, <, <=, ==, !=

## 출력 JSON 스키마
{
  "name": "전략 이름 (한글)",
  "description": "전략 설명 (한글)",
  "timeframe": "1d",  // 지원: "1m", "3m", "5m", "15m", "30m", "1h", "1d" (기본 "1d", 분봉은 데이터 있는 종목만 대상)
  "buy_conditions": [
    {"indicator": "rsi", "params": {"period": 14}, "op": "<=", "value": 30}
  ],
  "buy_logic": "AND" 또는 "OR",
  "sell_conditions": [...],
  "sell_logic": "AND" 또는 "OR"
}

## 추가 옵션 (사용자가 "분할매수", "손절", "트레일링", "확신도", "비중조절" 등을 언급할 때만 포함)

### position_sizing (포지션 사이징)
- mode: "fixed"(기본) | "conviction" | "atr_target" | "kelly"
- conviction일 때 weights: 조건별 가중치 dict (합계 1.0)
- atr_target일 때 atr_period(14), target_risk_pct(0.02)

### scaling (분할매매)
- enabled: true/false
- initial_pct: 1차 진입 비율 (0.5 = 50%)
- scale_in_drop_pct: 추가매수 트리거 하락률 (3.0 = -3%)
- max_scale_in: 최대 추가매수 횟수 (1)
- partial_exit_pct: 익절 시 매도 비율 (0.5 = 50%)
- partial_exit_gain_pct: 부분익절 트리거 수익률 (5.0 = +5%)

### risk_management (리스크 관리)
- stop_loss_pct: 고정 손절 % (예: 5.0 = -5%에서 손절)
- trailing_stop_pct: 고점 대비 트레일링 스탑 % (예: 2.0)
- atr_stop_multiplier: ATR × N 동적 손절 (예: 2.0)

## 규칙
- bool 지표(golden_cross, dead_cross, macd_cross)는 op="==", value=1 로 설정
- 매수 조건과 매도 조건을 반드시 모두 포함
- 사용자가 명시하지 않은 파라미터는 기본값 사용
- position_sizing, scaling, risk_management는 사용자가 언급하지 않으면 생략
- 사용자가 분봉(1m~1h) 타임프레임을 요청하면 해당 timeframe 사용 (예: "5분봉" → "5m")
- 분봉 데이터는 일부 종목만 보유하고 있으므로 자동으로 해당 종목만 대상이 됨
- 반드시 유효한 JSON만 출력
"""


async def generate_strategy_from_prompt(user_prompt: str) -> dict:
    """자연어 프롬프트 → 전략 JSON + 설명.

    Returns {"strategy": {...}, "explanation": "..."}
    """
    from app.core.llm import chat

    response = await chat(
        system=SYSTEM_PROMPT,
        max_tokens=2000,
        messages=[
            {
                "role": "user",
                "content": (
                    f"다음 조건으로 매매 전략을 JSON으로 만들어줘:\n\n{user_prompt}\n\n"
                    "반드시 아래 형식으로 응답해줘:\n"
                    "1. ```json 블록으로 전략 JSON\n"
                    "2. 그 아래에 전략 설명"
                ),
            }
        ],
    )

    text = response.content[0].text

    # JSON 블록 추출
    strategy_json = _extract_json(text)
    explanation = _extract_explanation(text)

    return {"strategy": strategy_json, "explanation": explanation}


def _extract_json(text: str) -> dict:
    """응답에서 JSON 블록을 추출한다."""
    # ```json ... ``` 패턴
    import re
    match = re.search(r"```json\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))

    # { 로 시작하는 JSON 찾기
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start : i + 1])

    raise ValueError("AI 응답에서 유효한 JSON을 찾을 수 없습니다.")


def _extract_explanation(text: str) -> str:
    """JSON 블록 이후의 설명 텍스트를 추출한다."""
    import re
    match = re.search(r"```\s*\n(.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # JSON 뒤의 텍스트
    last_brace = text.rfind("}")
    if last_brace >= 0 and last_brace < len(text) - 1:
        return text[last_brace + 1 :].strip()
    return ""
