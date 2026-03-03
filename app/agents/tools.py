"""Claude Tool Use 용 도구 정의 및 실행.

Manager Agent가 대화 중 호출할 수 있는 도구들을 정의한다.
Anthropic Tool Use 스키마 형식을 따른다.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.backtest.indicators import _INDICATOR_FN, _CROSS_INDICATORS
from app.backtest.schemas import StrategySchema

logger = logging.getLogger(__name__)

# ── 지표 메타데이터 ──────────────────────────────────────

INDICATOR_DESCRIPTIONS: dict[str, dict[str, Any]] = {
    "rsi": {
        "name": "RSI (Relative Strength Index)",
        "description": "과매수/과매도 판단. 70 이상 과매수, 30 이하 과매도.",
        "params": {"period": {"default": 14, "range": "5~50"}},
        "output_column": "rsi",
    },
    "sma": {
        "name": "SMA (단순이동평균)",
        "description": "N일 종가 평균. 추세 확인 및 지지/저항 판단.",
        "params": {"period": {"default": 20, "range": "5~200"}},
        "output_column": "sma_{period}",
    },
    "ema": {
        "name": "EMA (지수이동평균)",
        "description": "최근 가격에 가중치를 둔 이동평균. SMA보다 빠르게 반응.",
        "params": {"period": {"default": 20, "range": "5~200"}},
        "output_column": "ema_{period}",
    },
    "macd_hist": {
        "name": "MACD 히스토그램",
        "description": "MACD 라인과 시그널 라인의 차이. 양수 → 상승 모멘텀.",
        "params": {
            "fast": {"default": 12},
            "slow": {"default": 26},
            "signal": {"default": 9},
        },
        "output_column": "macd_hist",
    },
    "macd_cross": {
        "name": "MACD 크로스",
        "description": "MACD 라인이 시그널 라인을 교차. bool 값 (1=크로스 발생).",
        "params": {
            "fast": {"default": 12},
            "slow": {"default": 26},
            "signal": {"default": 9},
        },
        "output_column": "macd_cross",
        "is_bool": True,
    },
    "bb_upper": {
        "name": "볼린저밴드 상단",
        "description": "이동평균 + N표준편차. 가격이 상단 돌파 시 과매수 가능성.",
        "params": {"period": {"default": 20}, "std": {"default": 2.0}},
        "output_column": "bb_upper",
    },
    "bb_lower": {
        "name": "볼린저밴드 하단",
        "description": "이동평균 - N표준편차. 가격이 하단 터치 시 반등 가능성.",
        "params": {"period": {"default": 20}, "std": {"default": 2.0}},
        "output_column": "bb_lower",
    },
    "volume_ratio": {
        "name": "거래량 비율",
        "description": "당일 거래량 / N일 평균 거래량. 2.0 이상이면 거래량 급증.",
        "params": {"period": {"default": 20, "range": "5~60"}},
        "output_column": "volume_ratio",
    },
    "price_change_pct": {
        "name": "가격 변동률 (%)",
        "description": "N일 전 대비 가격 변동률. 양수=상승, 음수=하락.",
        "params": {"period": {"default": 1, "range": "1~20"}},
        "output_column": "price_change_pct",
    },
    "golden_cross": {
        "name": "골든크로스",
        "description": "단기 SMA가 장기 SMA를 상향 돌파. bool (1=크로스 발생).",
        "params": {
            "fast_period": {"default": 5},
            "slow_period": {"default": 20},
        },
        "output_column": "golden_cross",
        "is_bool": True,
    },
    "dead_cross": {
        "name": "데드크로스",
        "description": "단기 SMA가 장기 SMA를 하향 돌파. bool (1=크로스 발생).",
        "params": {
            "fast_period": {"default": 5},
            "slow_period": {"default": 20},
        },
        "output_column": "dead_cross",
        "is_bool": True,
    },
    "atr": {
        "name": "ATR (Average True Range)",
        "description": "변동성 측정. 값이 클수록 변동성이 큰 종목.",
        "params": {"period": {"default": 14, "range": "5~50"}},
        "output_column": "atr_{period}",
    },
    "open_gap_pct": {
        "name": "시가 갭 (%)",
        "description": "전일 종가 대비 당일 시가 갭. 양수=갭업, 음수=갭다운.",
        "params": {},
        "output_column": "open_gap_pct",
    },
    "sentiment_score": {
        "name": "뉴스 감성 스코어",
        "description": "뉴스 기사의 감성 점수. -1.0(매우 부정) ~ +1.0(매우 긍정). 주가 영향 기준.",
        "params": {},
        "output_column": "sentiment_score",
    },
    "article_count": {
        "name": "뉴스 기사 수",
        "description": "해당 종목의 최근 뉴스 보도량. 보도량 급증은 시장 관심 증가 신호.",
        "params": {},
        "output_column": "article_count",
    },
    "event_score": {
        "name": "이벤트 스코어",
        "description": "감성×영향력×보도량을 종합한 이벤트 점수. 높을수록 긍정적 이벤트.",
        "params": {},
        "output_column": "event_score",
    },
}


# ── Anthropic Tool Use 스키마 ────────────────────────────

TOOLS: list[dict[str, Any]] = [
    {
        "name": "list_indicators",
        "description": (
            "사용 가능한 모든 기술적 지표 목록과 설명을 반환합니다. "
            "사용자가 어떤 지표를 사용할 수 있는지 물어볼 때 호출하세요."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "suggest_parameters",
        "description": (
            "특정 지표의 권장 파라미터 범위와 일반적인 사용법을 반환합니다. "
            "사용자가 파라미터 값을 모르거나 추천을 요청할 때 호출하세요."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "indicator": {
                    "type": "string",
                    "description": "지표 이름 (예: rsi, sma, macd_hist)",
                },
            },
            "required": ["indicator"],
        },
    },
    {
        "name": "draft_strategy",
        "description": (
            "대화에서 파악된 매수/매도 조건을 구조화된 전략 JSON으로 변환합니다. "
            "사용자가 조건을 충분히 설명했을 때 호출하여 전략 초안을 생성하세요. "
            "사용자에게 전략 초안을 보여주고 확인을 받으세요."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "전략 이름 (한글)",
                },
                "description": {
                    "type": "string",
                    "description": "전략 설명 (한글)",
                },
                "timeframe": {
                    "type": "string",
                    "description": "타임프레임 (기본: 1d)",
                    "enum": ["1m", "3m", "5m", "15m", "30m", "1h", "1d", "1w", "1M"],
                },
                "buy_conditions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "indicator": {"type": "string"},
                            "params": {"type": "object"},
                            "op": {
                                "type": "string",
                                "enum": [">", ">=", "<", "<=", "==", "!="],
                            },
                            "value": {"type": "number"},
                        },
                        "required": ["indicator", "op", "value"],
                    },
                    "description": "매수 조건 목록",
                },
                "buy_logic": {
                    "type": "string",
                    "enum": ["AND", "OR"],
                    "description": "매수 조건 결합 방식",
                },
                "sell_conditions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "indicator": {"type": "string"},
                            "params": {"type": "object"},
                            "op": {"type": "string"},
                            "value": {"type": "number"},
                        },
                        "required": ["indicator", "op", "value"],
                    },
                    "description": "매도 조건 목록",
                },
                "sell_logic": {
                    "type": "string",
                    "enum": ["AND", "OR"],
                    "description": "매도 조건 결합 방식",
                },
                "position_sizing": {
                    "type": "object",
                    "description": "포지션 사이징 설정 (사용자가 언급한 경우만)",
                },
                "scaling": {
                    "type": "object",
                    "description": "분할매매 설정 (사용자가 언급한 경우만)",
                },
                "risk_management": {
                    "type": "object",
                    "description": "리스크 관리 설정 (사용자가 언급한 경우만)",
                },
            },
            "required": [
                "name",
                "buy_conditions",
                "buy_logic",
                "sell_conditions",
                "sell_logic",
            ],
        },
    },
    {
        "name": "validate_strategy",
        "description": (
            "전략 JSON의 유효성을 검증합니다. "
            "매수/매도 조건의 지표가 실제로 존재하는지, 파라미터가 올바른지 확인합니다."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "strategy": {
                    "type": "object",
                    "description": "검증할 전략 JSON",
                },
            },
            "required": ["strategy"],
        },
    },
    {
        "name": "ask_technical_analyst",
        "description": (
            "기술적 분석 전문 에이전트에게 특정 종목의 기술적 분석을 요청합니다. "
            "현재 RSI, MACD, 볼린저밴드, 거래량 등을 분석하고 매매 시그널을 제공합니다. "
            "사용자가 특정 종목의 차트 분석이나 현재 상태를 물어볼 때 호출하세요."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "종목코드 (예: 005930)",
                },
                "stock_name": {
                    "type": "string",
                    "description": "종목명 (예: 삼성전자)",
                },
                "strategy_draft": {
                    "type": "object",
                    "description": "현재 전략 초안 (있으면 전략 맥락에서 분석)",
                },
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "ask_risk_manager",
        "description": (
            "리스크 관리 전문 에이전트에게 전략의 리스크를 평가받습니다. "
            "손절/익절 설정, 포지션 사이징, 분할매매 등을 검토하고 개선 사항을 제안합니다. "
            "사용자가 전략을 확정하기 전에 리스크 검토를 요청하거나, "
            "손절이 없는 전략을 만들었을 때 자동으로 호출하세요."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "strategy_draft": {
                    "type": "object",
                    "description": "리스크를 평가할 전략 JSON",
                },
            },
            "required": ["strategy_draft"],
        },
    },
    {
        "name": "search_sector_stocks",
        "description": (
            "자연어 쿼리로 관련 종목을 의미론적으로 검색합니다. "
            "예: '반도체 관련주', '2차전지 배터리', 'AI 인공지능'. "
            "사용자가 특정 테마/섹터의 종목을 찾거나, 전략의 대상 종목을 설정할 때 호출하세요."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "검색 쿼리 (예: '반도체 관련주')",
                },
                "top_k": {
                    "type": "integer",
                    "description": "반환할 최대 종목 수 (기본: 10)",
                },
            },
            "required": ["query"],
        },
    },
]

# 비동기 도구 이름 집합 (orchestrator에서 처리)
ASYNC_TOOLS = {"ask_technical_analyst", "ask_risk_manager", "search_sector_stocks"}


# ── 도구 실행 함수 ───────────────────────────────────────

def execute_tool(name: str, input_data: dict[str, Any]) -> str:
    """도구를 실행하고 결과를 JSON 문자열로 반환한다."""
    handler = _TOOL_HANDLERS.get(name)
    if handler is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        result = handler(input_data)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        logger.exception("Tool execution error: %s", name)
        return json.dumps({"error": str(e)})


def _handle_list_indicators(_input: dict[str, Any]) -> dict:
    indicators = []
    for key, info in INDICATOR_DESCRIPTIONS.items():
        indicators.append({
            "name": key,
            "display_name": info["name"],
            "description": info["description"],
            "params": {
                k: v.get("default") for k, v in info.get("params", {}).items()
            },
            "is_bool": info.get("is_bool", False),
        })
    return {"indicators": indicators, "total": len(indicators)}


def _handle_suggest_parameters(input_data: dict[str, Any]) -> dict:
    indicator = input_data.get("indicator", "")
    info = INDICATOR_DESCRIPTIONS.get(indicator)
    if info is None:
        return {"error": f"Unknown indicator: {indicator}"}

    suggestions: dict[str, Any] = {
        "indicator": indicator,
        "display_name": info["name"],
        "description": info["description"],
        "params": {},
    }

    for param_name, param_info in info.get("params", {}).items():
        param_detail: dict[str, Any] = {"default": param_info.get("default")}
        if "range" in param_info:
            param_detail["recommended_range"] = param_info["range"]
        suggestions["params"][param_name] = param_detail

    # 지표별 추가 사용 팁
    tips = _INDICATOR_TIPS.get(indicator, "")
    if tips:
        suggestions["usage_tip"] = tips

    return suggestions


def _handle_draft_strategy(input_data: dict[str, Any]) -> dict:
    """전략 초안을 생성하고 유효성을 검증한다."""
    # timeframe 검증: 지원하지 않는 값은 1d로 보정
    from app.backtest.data_loader import SUPPORTED_INTERVALS
    if input_data.get("timeframe", "1d") not in SUPPORTED_INTERVALS:
        input_data["timeframe"] = "1d"

    try:
        strategy = StrategySchema(**input_data)
        return {
            "status": "success",
            "strategy": strategy.model_dump(),
            "message": "전략 초안이 생성되었습니다. 사용자에게 확인을 받으세요.",
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"전략 생성 실패: {e}",
        }


def _handle_validate_strategy(input_data: dict[str, Any]) -> dict:
    """전략 JSON의 유효성을 검증한다."""
    strategy_data = input_data.get("strategy", {})
    errors: list[str] = []

    # 매수/매도 조건 존재 확인
    buy_conds = strategy_data.get("buy_conditions", [])
    sell_conds = strategy_data.get("sell_conditions", [])

    if not buy_conds:
        errors.append("매수 조건이 없습니다.")
    if not sell_conds:
        errors.append("매도 조건이 없습니다.")

    # 지표 존재 확인
    all_indicators = set(_INDICATOR_FN.keys()) | _CROSS_INDICATORS
    for i, cond in enumerate(buy_conds):
        ind = cond.get("indicator", "")
        if ind not in all_indicators:
            errors.append(f"매수 조건 #{i + 1}: 알 수 없는 지표 '{ind}'")
        if cond.get("op") not in (">", ">=", "<", "<=", "==", "!="):
            errors.append(f"매수 조건 #{i + 1}: 유효하지 않은 연산자 '{cond.get('op')}'")

    for i, cond in enumerate(sell_conds):
        ind = cond.get("indicator", "")
        if ind not in all_indicators:
            errors.append(f"매도 조건 #{i + 1}: 알 수 없는 지표 '{ind}'")
        if cond.get("op") not in (">", ">=", "<", "<=", "==", "!="):
            errors.append(f"매도 조건 #{i + 1}: 유효하지 않은 연산자 '{cond.get('op')}'")

    # Pydantic 검증
    try:
        StrategySchema(**strategy_data)
    except Exception as e:
        errors.append(f"스키마 검증 실패: {e}")

    if errors:
        return {"valid": False, "errors": errors}
    return {"valid": True, "message": "전략이 유효합니다."}


# ── 지표별 사용 팁 ───────────────────────────────────────

_INDICATOR_TIPS: dict[str, str] = {
    "rsi": (
        "일반적으로 RSI 14일 기준, 30 이하에서 매수(과매도 반등), "
        "70 이상에서 매도(과매수 조정)가 기본 전략입니다. "
        "단기 트레이딩에서는 period를 7~9로 줄이기도 합니다."
    ),
    "sma": (
        "SMA 5일(단기)과 20일(중기) 조합이 골든크로스/데드크로스에 많이 쓰입니다. "
        "SMA 60일, 120일은 중장기 추세 판단에 활용됩니다."
    ),
    "macd_hist": (
        "MACD 히스토그램이 0 이상이면 상승 모멘텀, 0 이하이면 하락 모멘텀입니다. "
        "히스토그램이 음에서 양으로 전환되는 시점이 매수 타이밍으로 활용됩니다."
    ),
    "bb_lower": (
        "볼린저밴드 하단 터치 후 반등하는 패턴에서 매수 진입합니다. "
        "std를 1.5로 줄이면 민감도가 높아지고, 2.5로 높이면 극단적 조건만 포착합니다."
    ),
    "volume_ratio": (
        "거래량 비율 2.0 이상은 평소 대비 2배 이상의 거래량 급증을 의미합니다. "
        "RSI 과매도와 함께 쓰면 바닥 확인 신호로 활용할 수 있습니다."
    ),
    "atr": (
        "ATR은 절대적인 변동성 크기를 측정합니다. "
        "포지션 사이징의 atr_target 모드와 함께 쓰면 변동성 기반 비중 조절이 가능합니다."
    ),
    "sentiment_score": (
        "뉴스 감성 스코어는 -1.0(매우 부정)~+1.0(매우 긍정) 범위입니다. "
        "0.3 이상이면 긍정적, -0.3 이하이면 부정적 뉴스가 많다는 의미입니다. "
        "RSI 과매도와 감성 스코어 양수를 조합하면 바닥에서 호재에 의한 반등을 포착할 수 있습니다."
    ),
    "article_count": (
        "보도량이 급증하면 시장의 관심이 높아졌다는 신호입니다. "
        "5건 이상이면 주목할 만한 수준이며, 감성과 함께 봐야 방향성을 판단할 수 있습니다."
    ),
    "event_score": (
        "이벤트 스코어는 감성, 영향력, 보도량을 종합한 점수입니다. "
        "0.1 이상이면 긍정적 이벤트, -0.1 이하이면 부정적 이벤트로 판단합니다."
    ),
}


_TOOL_HANDLERS = {
    "list_indicators": _handle_list_indicators,
    "suggest_parameters": _handle_suggest_parameters,
    "draft_strategy": _handle_draft_strategy,
    "validate_strategy": _handle_validate_strategy,
}
