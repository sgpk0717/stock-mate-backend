"""팩터 AI 채팅 — 기존 팩터를 선택하여 대화로 수정/평가/저장.

agents/manager.py 패턴 동일: Claude Tool Use 루프, 메모리 세션.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import polars as pl

from app.alpha.ast_converter import (
    NAMED_VARIABLE_MAP,
    ensure_alpha_features,
    parse_expression,
    sympy_to_code_string,
    sympy_to_polars,
)
from app.alpha.evaluator import (
    FactorMetrics,
    compute_factor_metrics,
    compute_forward_returns,
    compute_ic_series,
    compute_long_only_returns,
    compute_position_turnover,
    compute_quantile_returns,
)
from app.alpha.interval import bars_per_year, default_round_trip_cost
from app.core.config import settings

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 5

# ── 사용 가능한 피처 설명 (Claude 프롬프트용) ──

_FEATURES_DESC = """사용 가능한 피처 (변수명):
- close, open, high, low, volume: OHLCV 원본
- sma_20: 20일 단순이동평균
- ema_20: 20일 지수이동평균
- rsi: 14일 RSI (0~100)
- volume_ratio: 20일 평균 대비 거래량 비율
- atr_14: 14일 Average True Range
- macd_hist: MACD 히스토그램 (12/26/9)
- bb_upper, bb_lower: 볼린저 밴드 상/하한 (20일, 2σ)
- bb_width: bb_upper - bb_lower
- price_change_pct: 전일 대비 종가 변화율 (%)
- 횡단면 연산자 (같은 날 전 종목 대비 상대 위치):
  Cs_Rank_close, Cs_Rank_volume, Cs_ZScore_close, Cs_ZScore_volume
- 시계열 연산자 (개별 종목의 과거 60일 대비 위치):
  Ts_Rank_close, Ts_Rank_volume, Ts_ZScore_close, Ts_ZScore_volume

사용 가능한 수학 연산: +, -, *, /, log(), exp(), sqrt(), abs(), pow()"""


def _build_system_prompt(
    source_expression: str,
    source_metrics: dict[str, Any] | None,
    interval: str,
) -> str:
    """팩터 채팅 전용 시스템 프롬프트."""
    metrics_str = ""
    if source_metrics:
        metrics_str = (
            f"\n원본 메트릭:"
            f"\n  IC={source_metrics.get('ic_mean', 'N/A'):.4f}"
            f"  ICIR={source_metrics.get('icir', 'N/A'):.4f}"
            f"  Sharpe={source_metrics.get('sharpe', 'N/A'):.2f}"
            f"  Turnover={source_metrics.get('turnover', 'N/A'):.3f}"
            f"  MDD={source_metrics.get('max_drawdown', 'N/A'):.2%}"
        )

    return f"""\
당신은 알파 팩터 엔지니어링 전문가입니다. 사용자가 선택한 팩터를 함께 분석하고 개선합니다.

## 원본 팩터
수식: {source_expression}{metrics_str}
데이터 인터벌: {interval}

{_FEATURES_DESC}

## 역할
1. 사용자가 팩터에 대해 질문하면 수식의 경제적 의미와 메트릭을 설명하세요.
2. 사용자가 수정을 요청하면 수식을 수정하고, **반드시 evaluate_expression 도구를 호출**하여 결과를 보여주세요.
3. 수정 전후의 메트릭 변화를 비교하여 설명하세요.
4. 사용자가 만족하면 save_factor 도구로 저장하세요.

## 수식 규칙
- SymPy 파싱 가능한 수학 표현식만 작성 (위 변수명만 사용).
- sign(), Piecewise, if/else 사용 금지. 부호 반전은 -1* 로, 절댓값은 abs()로 대체.
- 수식에 마크다운(```, **, $) 사용 금지.

## 응답 스타일
- 간결하고 전문적으로 응답하세요.
- 메트릭 변화를 표로 보여주면 가독성이 좋습니다."""


# ── Claude 도구 정의 ──

TOOLS: list[dict[str, Any]] = [
    {
        "name": "evaluate_expression",
        "description": (
            "알파 팩터 수식을 평가하여 IC, ICIR, Sharpe, Turnover, MDD 메트릭을 반환합니다. "
            "수식을 수정할 때마다 이 도구를 호출하여 성능 변화를 확인하세요."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "SymPy 호환 알파 팩터 수식 (예: rsi * volume_ratio / (atr_14 + 1))",
                },
            },
            "required": ["expression"],
        },
    },
    {
        "name": "save_factor",
        "description": (
            "현재 수식을 최종 팩터로 확정합니다. "
            "사용자가 만족하고 저장을 원할 때만 호출하세요."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "저장할 최종 수식",
                },
                "name": {
                    "type": "string",
                    "description": "팩터 이름 (간결한 설명)",
                },
                "hypothesis": {
                    "type": "string",
                    "description": "팩터의 경제적 근거/가설",
                },
            },
            "required": ["expression", "name", "hypothesis"],
        },
    },
]


# ── 세션 ──


@dataclass
class FactorChatMessage:
    """채팅 메시지."""

    role: str  # "user" | "assistant"
    content: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    factor_draft: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
        }
        if self.factor_draft:
            d["factor_draft"] = self.factor_draft
        return d


@dataclass
class FactorChatSession:
    """팩터 채팅 세션."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: list[FactorChatMessage] = field(default_factory=list)
    # 원본 팩터 정보
    source_factor_id: str = ""
    source_expression: str = ""
    source_hypothesis: str = ""
    source_metrics: dict[str, Any] | None = None
    # 현재 수정 중인 팩터
    current_expression: str | None = None
    current_metrics: dict[str, Any] | None = None
    # 데이터 컨텍스트
    universe: str = ""
    start_date: str = ""
    end_date: str = ""
    interval: str = "1d"
    # 캐시 데이터 (Polars DF, API 응답에는 미포함)
    data: pl.DataFrame | None = None
    data_loaded: bool = False
    # 상태
    status: str = "active"  # "active" | "saved"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.id,
            "messages": [m.to_dict() for m in self.messages],
            "source_factor_id": self.source_factor_id,
            "source_expression": self.source_expression,
            "current_expression": self.current_expression,
            "current_metrics": self.current_metrics,
            "universe": self.universe,
            "interval": self.interval,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class FactorChatSessionStore:
    """메모리 기반 세션 저장소 (TTL 30분)."""

    def __init__(self, ttl_minutes: int = 30) -> None:
        self._sessions: dict[str, FactorChatSession] = {}
        self._ttl_minutes = ttl_minutes

    def create(self, **kwargs: Any) -> FactorChatSession:
        session = FactorChatSession(**kwargs)
        self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> FactorChatSession | None:
        session = self._sessions.get(session_id)
        if session is None:
            return None
        if self._is_expired(session):
            del self._sessions[session_id]
            return None
        return session

    def delete(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    def _is_expired(self, session: FactorChatSession) -> bool:
        updated = datetime.fromisoformat(session.updated_at)
        now = datetime.now(timezone.utc)
        return (now - updated).total_seconds() > self._ttl_minutes * 60


# 싱글톤
factor_chat_store = FactorChatSessionStore()


# ── 도구 실행 ──


def _execute_evaluate(
    session: FactorChatSession, expression: str
) -> str:
    """수식 평가 — 세션 캐시 데이터 사용."""
    if session.data is None:
        return json.dumps(
            {"error": "데이터가 아직 로드되지 않았습니다."},
            ensure_ascii=False,
        )

    try:
        parsed = parse_expression(expression)
        polars_expr = sympy_to_polars(parsed)
    except Exception as e:
        return json.dumps(
            {"error": f"수식 파싱 실패: {e}"},
            ensure_ascii=False,
        )

    try:
        col_name = "_eval_factor"
        df_eval = session.data.with_columns(polars_expr.alias(col_name))
        df_eval = df_eval.drop_nulls(subset=[col_name, "fwd_return"])

        if df_eval.height < 30:
            return json.dumps(
                {"error": f"유효 데이터 부족 ({df_eval.height}행). 수식이 대부분 null을 생성합니다."},
                ensure_ascii=False,
            )

        ann = bars_per_year(session.interval)
        rtc = default_round_trip_cost(session.interval)

        ic_series = compute_ic_series(df_eval, factor_col=col_name)
        ls_returns = compute_quantile_returns(df_eval, factor_col=col_name)
        lo_returns = compute_long_only_returns(df_eval, factor_col=col_name)
        turnover, turnover_series = compute_position_turnover(
            df_eval, factor_col=col_name
        )

        metrics = compute_factor_metrics(
            ic_series,
            ls_returns=ls_returns,
            long_only_returns=lo_returns,
            position_turnover=turnover,
            turnover_series=turnover_series,
            round_trip_cost=rtc,
            annualize=ann,
        )

        # 세션에 현재 수식/메트릭 저장
        session.current_expression = expression
        session.current_metrics = {
            "ic_mean": metrics.ic_mean,
            "ic_std": metrics.ic_std,
            "icir": metrics.icir,
            "turnover": metrics.turnover,
            "sharpe": metrics.sharpe,
            "max_drawdown": metrics.max_drawdown,
        }

        return json.dumps(
            {
                "status": "success",
                "expression": expression,
                "metrics": session.current_metrics,
            },
            ensure_ascii=False,
        )

    except Exception as e:
        return json.dumps(
            {"error": f"평가 실패: {e}"},
            ensure_ascii=False,
        )


def _execute_save(
    session: FactorChatSession,
    expression: str,
    name: str,
    hypothesis: str,
) -> str:
    """수식을 factor_draft에 저장."""
    session.current_expression = expression
    session.touch()

    draft = {
        "expression": expression,
        "name": name,
        "hypothesis": hypothesis,
        "metrics": session.current_metrics,
    }

    return json.dumps(
        {
            "status": "success",
            "message": f"팩터 '{name}'이 저장 준비 완료되었습니다.",
            "draft": draft,
        },
        ensure_ascii=False,
    )


# ── 메인 프로세서 ──


async def load_session_data(session: FactorChatSession) -> None:
    """세션에 평가용 데이터를 로드 (최초 1회)."""
    if session.data_loaded:
        return

    from app.alpha.universe import Universe, resolve_universe
    from app.backtest.data_loader import load_candles
    from datetime import date

    symbols = await resolve_universe(Universe(session.universe))
    data = await load_candles(
        symbols=symbols,
        start_date=date.fromisoformat(session.start_date),
        end_date=date.fromisoformat(session.end_date),
        interval=session.interval,
    )

    if data.height == 0:
        raise ValueError("해당 기간/유니버스에 데이터가 없습니다.")

    data = ensure_alpha_features(data)
    data = compute_forward_returns(data, periods=1)

    session.data = data
    session.data_loaded = True
    logger.info(
        "Factor chat session %s: data loaded (%d rows, %d symbols)",
        session.id,
        data.height,
        data["symbol"].n_unique() if "symbol" in data.columns else 0,
    )


async def process_message(
    session: FactorChatSession,
    user_message: str,
) -> FactorChatMessage:
    """사용자 메시지 처리 — Claude Tool Use 루프."""
    from app.core.llm import chat

    # 데이터 로드 (최초 1회)
    if not session.data_loaded:
        await load_session_data(session)

    # 사용자 메시지 추가
    user_msg = FactorChatMessage(role="user", content=user_message)
    session.messages.append(user_msg)
    session.touch()

    # API 메시지 빌드
    api_messages = _build_api_messages(session)
    system_prompt = _build_system_prompt(
        session.source_expression,
        session.source_metrics,
        session.interval,
    )

    max_tokens = getattr(settings, "AGENT_MAX_TOKENS", 4000)

    for _round in range(MAX_TOOL_ROUNDS):
        response = await chat(
            max_tokens=max_tokens,
            system=system_prompt,
            tools=TOOLS,
            messages=api_messages,
        )

        assistant_text = ""
        tool_uses: list[dict[str, Any]] = []

        for block in response.content:
            if block.type == "text":
                assistant_text += block.text
            elif block.type == "tool_use":
                tool_uses.append({
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

        if not tool_uses:
            # 텍스트만 → 대화 응답 완료
            factor_draft = None
            if session.current_expression and session.current_metrics:
                factor_draft = {
                    "expression": session.current_expression,
                    "metrics": session.current_metrics,
                }

            assistant_msg = FactorChatMessage(
                role="assistant",
                content=assistant_text,
                factor_draft=factor_draft,
            )
            session.messages.append(assistant_msg)
            session.touch()
            return assistant_msg

        # Tool 호출 처리
        api_messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for tu in tool_uses:
            if tu["name"] == "evaluate_expression":
                result_str = _execute_evaluate(
                    session, tu["input"].get("expression", "")
                )
            elif tu["name"] == "save_factor":
                result_str = _execute_save(
                    session,
                    tu["input"].get("expression", ""),
                    tu["input"].get("name", "Custom Factor"),
                    tu["input"].get("hypothesis", ""),
                )
            else:
                result_str = json.dumps({"error": f"Unknown tool: {tu['name']}"})

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu["id"],
                "content": result_str,
            })

        api_messages.append({"role": "user", "content": tool_results})

    # MAX_TOOL_ROUNDS 초과
    fallback = assistant_text or "처리 중 문제가 발생했습니다. 다시 시도해주세요."
    assistant_msg = FactorChatMessage(role="assistant", content=fallback)
    session.messages.append(assistant_msg)
    session.touch()
    return assistant_msg


def _build_api_messages(
    session: FactorChatSession,
) -> list[dict[str, Any]]:
    """세션 히스토리를 Anthropic API 형식으로 변환."""
    messages: list[dict[str, Any]] = []
    for msg in session.messages:
        if msg.role in ("user", "assistant"):
            messages.append({"role": msg.role, "content": msg.content})
    return messages
