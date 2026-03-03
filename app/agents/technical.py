"""Technical Analyst Agent — 기술적 분석 전문 에이전트.

특정 종목의 기술적 지표를 분석하고, 패턴을 감지하며,
매매 진입/이탈 지점을 제안한다.
Manager Agent에서 ask_technical_analyst 도구로 호출된다.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

from app.core.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
당신은 주식 기술적 분석(Technical Analysis) 전문가입니다.

## 역할
- 종목의 기술적 지표 현황을 분석합니다.
- 차트 패턴을 감지합니다 (추세, 지지/저항, 크로스 등).
- 매매 진입점과 이탈점을 제안합니다.

## 분석 시 고려 사항
1. RSI: 과매수(70+) / 과매도(30-) 영역 확인
2. MACD: 시그널 크로스 방향, 히스토그램 추세
3. 볼린저밴드: 밴드폭, 가격 위치 (상/중/하단)
4. 이동평균: 골든크로스/데드크로스, 정배열/역배열
5. 거래량: 평균 대비 거래량 비율, 거래량 동반 여부
6. ATR: 변동성 수준

## 응답 형식
분석 결과를 간결하게 요약하세요:
- 현재 기술적 상태 (상승/하락/횡보/전환)
- 주요 지표 수치
- 매매 시그널 (강력 매수 / 매수 / 중립 / 매도 / 강력 매도)
- 제안 진입가/이탈가 (있을 경우)
"""


async def analyze_stock(
    symbol: str,
    stock_name: str,
    strategy_draft: dict | None = None,
) -> dict:
    """종목의 기술적 분석을 수행한다.

    Args:
        symbol: 종목코드
        stock_name: 종목명
        strategy_draft: 현재 전략 초안 (있으면 전략 맥락에서 분석)

    Returns:
        {"analysis": str, "signal": str, "indicators": dict}
    """
    # 최근 캔들 데이터 로딩
    try:
        from app.backtest.data_loader import load_candles
        from app.backtest.indicators import (
            add_atr,
            add_bb,
            add_macd,
            add_rsi,
            add_volume_ratio,
        )

        end = date.today()
        start = end - timedelta(days=120)

        df = await load_candles([symbol], start, end, "1d")
        if df.is_empty() or df.height < 30:
            return {
                "analysis": f"{stock_name}({symbol})의 충분한 데이터가 없어 분석을 수행할 수 없습니다.",
                "signal": "분석 불가",
                "indicators": {},
            }

        sym_df = df.filter(df["symbol"] == symbol).sort("dt")

        # 지표 추가
        sym_df = add_rsi(sym_df)
        sym_df = add_macd(sym_df)
        sym_df = add_bb(sym_df)
        sym_df = add_volume_ratio(sym_df)
        sym_df = add_atr(sym_df)

        # 최근 값 추출
        last = sym_df.tail(1).to_dicts()[0]

        indicators = {
            "close": last.get("close"),
            "rsi": round(last.get("rsi", 0) or 0, 2),
            "macd_hist": round(last.get("macd_hist", 0) or 0, 2),
            "macd_line": round(last.get("macd_line", 0) or 0, 2),
            "macd_signal": round(last.get("macd_signal", 0) or 0, 2),
            "bb_upper": round(last.get("bb_upper", 0) or 0, 0),
            "bb_lower": round(last.get("bb_lower", 0) or 0, 0),
            "bb_middle": round(last.get("bb_middle", 0) or 0, 0),
            "volume_ratio": round(last.get("volume_ratio", 0) or 0, 2),
            "atr_14": round(last.get("atr_14", 0) or 0, 0),
        }

    except Exception as e:
        logger.warning("기술적 분석 데이터 로딩 실패 (%s): %s", symbol, e)
        indicators = {}

    # Claude API로 분석 요청
    api_key = settings.ANTHROPIC_API_KEY
    if not api_key or not indicators:
        # API 키 없거나 데이터 없으면 규칙 기반 분석
        return _rule_based_analysis(symbol, stock_name, indicators)

    try:
        from app.core.llm import chat

        user_content = f"종목: {stock_name} ({symbol})\n\n"
        user_content += "현재 기술적 지표:\n"
        for k, v in indicators.items():
            user_content += f"- {k}: {v}\n"

        if strategy_draft:
            user_content += f"\n현재 전략 초안:\n매수: {strategy_draft.get('buy_conditions', [])}\n"
            user_content += f"매도: {strategy_draft.get('sell_conditions', [])}\n"
            user_content += "\n이 전략의 관점에서 현재 기술적 상태를 분석해주세요."
        else:
            user_content += "\n이 종목의 기술적 분석을 수행해주세요."

        response = await chat(
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )

        analysis_text = response.content[0].text
        signal = _extract_signal(analysis_text, indicators)

        return {
            "analysis": analysis_text,
            "signal": signal,
            "indicators": indicators,
        }

    except Exception as e:
        logger.warning("Claude 기술적 분석 실패: %s", e)
        return _rule_based_analysis(symbol, stock_name, indicators)


def _rule_based_analysis(
    symbol: str,
    stock_name: str,
    indicators: dict,
) -> dict:
    """규칙 기반 간이 기술적 분석."""
    if not indicators:
        return {
            "analysis": f"{stock_name}({symbol})의 분석 데이터가 없습니다.",
            "signal": "분석 불가",
            "indicators": {},
        }

    rsi = indicators.get("rsi", 50)
    macd_hist = indicators.get("macd_hist", 0)
    vol_ratio = indicators.get("volume_ratio", 1)
    close = indicators.get("close", 0)
    bb_lower = indicators.get("bb_lower", 0)
    bb_upper = indicators.get("bb_upper", 0)

    parts = [f"**{stock_name} ({symbol}) 기술적 분석**\n"]

    # RSI 분석
    if rsi <= 30:
        parts.append(f"- RSI {rsi}: **과매도** 영역. 반등 가능성.")
    elif rsi >= 70:
        parts.append(f"- RSI {rsi}: **과매수** 영역. 조정 가능성.")
    else:
        parts.append(f"- RSI {rsi}: 중립 영역.")

    # MACD 분석
    if macd_hist > 0:
        parts.append(f"- MACD 히스토그램 {macd_hist}: 상승 모멘텀.")
    else:
        parts.append(f"- MACD 히스토그램 {macd_hist}: 하락 모멘텀.")

    # 볼린저밴드
    if close and bb_lower and close <= bb_lower:
        parts.append(f"- 가격({close:,.0f})이 볼린저밴드 하단({bb_lower:,.0f}) 부근.")
    elif close and bb_upper and close >= bb_upper:
        parts.append(f"- 가격({close:,.0f})이 볼린저밴드 상단({bb_upper:,.0f}) 부근.")

    # 거래량
    if vol_ratio >= 2.0:
        parts.append(f"- 거래량 비율 {vol_ratio:.1f}x: **거래량 급증**.")

    # 종합 시그널
    signal = _extract_signal("", indicators)
    parts.append(f"\n종합 시그널: **{signal}**")

    return {
        "analysis": "\n".join(parts),
        "signal": signal,
        "indicators": indicators,
    }


def _extract_signal(text: str, indicators: dict) -> str:
    """지표로부터 종합 시그널을 추출한다."""
    score = 0
    rsi = indicators.get("rsi", 50)
    macd_hist = indicators.get("macd_hist", 0)
    vol_ratio = indicators.get("volume_ratio", 1)

    if rsi <= 30:
        score += 2
    elif rsi <= 40:
        score += 1
    elif rsi >= 70:
        score -= 2
    elif rsi >= 60:
        score -= 1

    if macd_hist > 0:
        score += 1
    else:
        score -= 1

    if vol_ratio >= 2.0 and rsi <= 40:
        score += 1

    if score >= 3:
        return "강력 매수"
    elif score >= 1:
        return "매수"
    elif score <= -3:
        return "강력 매도"
    elif score <= -1:
        return "매도"
    return "중립"
