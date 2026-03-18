"""진화형 알파 팩터 마이너.

Claude + (옵션: PySR) 기반 알파 팩터 발굴 + 궤적 변이 루프.
"""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass, field
from typing import Callable, Awaitable

import httpx
import numpy as np
import polars as pl
import sympy

from app.core.llm import chat as llm_chat, get_client as get_llm_client

from app.alpha.ast_converter import (
    ASTConversionError,
    ensure_alpha_features,
    parse_expression,
    sympy_to_code_string,
    sympy_to_polars,
    NAMED_VARIABLE_MAP,
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
from app.core.config import settings

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], Awaitable[None]]
IterationCallback = Callable[[dict], Awaitable[None]]


@dataclass
class DiscoveredFactor:
    """마이닝에서 발견된 팩터."""

    name: str
    expression_str: str
    expression_sympy: str  # srepr
    polars_code: str
    hypothesis: str
    generation: int
    metrics: FactorMetrics
    parent_ids: list[str] | None = None


@dataclass
class ExperienceMemory:
    """인메모리 경험 저장소.

    성공/실패 팩터의 수식, IC, 세대를 추적.
    Claude 프롬프트에 포함하여 중복 탐색 방지.
    """

    _successes: list[dict] = field(default_factory=list)
    _failures: list[dict] = field(default_factory=list)

    def add(
        self, expr_str: str, ic: float, generation: int, success: bool
    ) -> None:
        entry = {
            "expression": expr_str,
            "ic_mean": ic,
            "generation": generation,
        }
        if success:
            self._successes.append(entry)
            self._successes.sort(key=lambda x: x["ic_mean"], reverse=True)
        else:
            self._failures.append(entry)

    def top_k(self, k: int = 5) -> list[dict]:
        return self._successes[:k]

    def failures(self, k: int = 5) -> list[dict]:
        return self._failures[-k:]

    def format_for_prompt(self) -> str:
        lines = []
        top = self.top_k(5)
        if top:
            lines.append("=== 성공한 팩터 (IC 높은 순) ===")
            for i, f in enumerate(top, 1):
                lines.append(
                    f"{i}. {f['expression']} (IC={f['ic_mean']:.4f}, gen={f['generation']})"
                )

        fails = self.failures(5)
        if fails:
            lines.append("\n=== 최근 실패한 팩터 (IC 부족) ===")
            for i, f in enumerate(fails, 1):
                lines.append(
                    f"{i}. {f['expression']} (IC={f['ic_mean']:.4f})"
                )

        if not lines:
            return "아직 탐색 이력이 없습니다."

        return "\n".join(lines)


# ── Claude 프롬프트 ────────────────────────────

_BASE_FEATURES = """- close, open, high, low, volume: OHLCV 원본
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
  Ts_Rank_close, Ts_Rank_volume, Ts_ZScore_close, Ts_ZScore_volume"""

# 보조 데이터별 피처 설명 블록 — DB에 데이터가 있을 때만 프롬프트에 포함
_OPTIONAL_FEATURE_BLOCKS: dict[str, tuple[str, list[str]]] = {
    "investor": (
        "- 투자자 수급 (분봉: 전일 기준 T-1, 일봉: 당일, 거래량 대비 정규화):\n"
        "  foreign_net_norm: 외국인 순매수 / 거래량 (-1~+1)\n"
        "  inst_net_norm: 기관 순매수 / 거래량\n"
        "  retail_net_norm: 개인 순매수 / 거래량\n"
        "  foreign_buy_ratio: 외국인 매수비율 buy/(buy+sell) (0~1, 1=매수 일방)\n"
        "  inst_buy_ratio: 기관 매수비율 buy/(buy+sell)\n"
        "  retail_buy_ratio: 개인 매수비율 buy/(buy+sell)",
        ["foreign_net_norm", "inst_net_norm", "foreign_buy_ratio"],
    ),
    "sentiment": (
        "- 뉴스 감성 (T+1 shift 적용, 룩어헤드 방지):\n"
        "  sentiment_score: 뉴스 감성 점수 (-1~+1)\n"
        "  event_score: 이벤트 스코어 (감성 × 영향도 × log(기사수))",
        ["sentiment_score", "event_score"],
    ),
    "sector": (
        "- 섹터 횡단면 (동일 섹터 내 상대 비교):\n"
        "  sector_return: 동일 섹터 평균 수익률 (%)\n"
        "  sector_rel_strength: 종목 수익률 - 섹터 평균 (상대 강도)\n"
        "  sector_rank: 섹터 내 수익률 순위 (0~1, 1=최상위)",
        ["sector_return", "sector_rel_strength", "sector_rank"],
    ),
    "margin_short": (
        "- 신용/공매도 (분봉: 전일 기준 T-1, 일봉: 당일):\n"
        "  margin_rate: 융자잔고율 (%)\n"
        "  short_balance_rate: 대차잔고비율 (%)\n"
        "  short_volume_ratio: 공매도수량 / 거래량",
        ["margin_rate", "short_balance_rate", "short_volume_ratio"],
    ),
    "program_trading": (
        "- 프로그램 매매 (분봉: 전일 기준 T-1, 일봉: 당일):\n"
        "  pgm_net_norm: 프로그램 순매수 / 거래량 (기관 프로그램 강도)\n"
        "  pgm_buy_ratio: 프로그램 매수비율 (0~1)",
        ["pgm_net_norm", "pgm_buy_ratio"],
    ),
    "dart": (
        "- DART 재무 (최근 공시 기준, join_asof):\n"
        "  eps: 주당순이익\n"
        "  bps: 주당순자산\n"
        "  debt_to_equity: 부채비율 (%)\n"
        "  operating_margin: 영업이익률 (%)",
        ["eps", "bps", "debt_to_equity", "operating_margin"],
    ),
    # ── 이벤트 감지 피처 (OHLCV에서 자동 계산, 항상 표시) ──
    "volume_events": (
        "- 거래량 이벤트 (OHLCV에서 자동 계산):\n"
        "  vol_spike_5d: 당일 거래량 / 5일 평균 (비율, 1=평균)\n"
        "  vol_spike_20d: 당일 거래량 / 20일 평균\n"
        "  consec_low_vol_5: 최근 5일 중 저거래량 일수 (0~5)\n"
        "  vol_dry_then_spike: 거래량 고갈(3일+) 후 폭발(3배+) (0/1)",
        ["close"],  # OHLCV 항상 존재 → 항상 표시
    ),
    "price_events": (
        "- 가격 이벤트 (OHLCV에서 자동 계산):\n"
        "  consec_up: 최근 5일 중 상승일 수 (0~5)\n"
        "  consec_down: 최근 5일 중 하락일 수 (0~5)\n"
        "  gap_up_pct: 오버나이트 갭상승률 (%)\n"
        "  gap_down_pct: 오버나이트 갭하락률 (%)\n"
        "  range_breakout: 20일 최고가 돌파 (0/1)\n"
        "  range_breakdown: 20일 최저가 이탈 (0/1)",
        ["close"],
    ),
    "momentum_events": (
        "- 모멘텀 이벤트 (기술적 지표에서 자동 감지):\n"
        "  rsi_oversold_bounce: RSI 30 아래→위로 복귀 (0/1)\n"
        "  macd_cross_up: MACD 골든크로스 발생 (0/1)\n"
        "  bb_squeeze: 볼린저밴드 폭 20일 최저 — 변동성 수축 (0/1)\n"
        "  bb_breakout_upper: 종가가 BB 상단 돌파 (0/1)",
        ["close"],
    ),
    "supply_events": (
        "- 수급 이벤트 (투자자 데이터 필요):\n"
        "  foreign_accumulate_5d: 외국인 5일 연속 순매수 (0/1)\n"
        "  inst_accumulate_5d: 기관 5일 연속 순매수 (0/1)",
        ["foreign_net_norm"],
    ),
}


# ── 카테고리별 예시 수식 (LLM 편향 해소) ──
_CATEGORY_EXAMPLES: dict[str, list[dict]] = {
    "momentum": [
        {
            "hypothesis": "단기 RSI 과매도 + 거래량 폭발 = 기술적 반등 신호",
            "formula": "vol_spike_5d * step(30 - rsi) * abs(return_5d)",
        },
        {
            "hypothesis": "MACD 골든크로스 발생 시 거래량 동반 모멘텀 포착",
            "formula": "macd_cross_up * volume_ratio * (close - sma_20) / atr_14",
        },
    ],
    "event_driven": [
        {
            "hypothesis": "거래량 고갈 후 폭발은 세력 매집의 전형적 패턴",
            "formula": "vol_dry_then_spike * (close - bb_lower) / bb_width",
        },
        {
            "hypothesis": "20일 신고가 돌파 + 거래량 급증은 추세 전환의 강력한 신호",
            "formula": "range_breakout * vol_spike_20d * sign(return_5d)",
        },
    ],
    "value": [
        {
            "hypothesis": "저PER + 기관 순매수 집중 = 가치주 재평가 시작",
            "formula": "earnings_yield * inst_net_norm * (1 / (1 + debt_to_equity / 100))",
        },
        {
            "hypothesis": "장부가치 대비 저평가 + 외국인 매집 = 저평가 해소 기대",
            "formula": "book_yield * log(1 + abs(foreign_net_norm)) * sqrt(volume_ratio)",
        },
    ],
    "supply_demand": [
        {
            "hypothesis": "외국인 5일 연속 매수 + 공매도 감소 = 상승 압력 축적",
            "formula": "foreign_accumulate_5d * (1 - short_volume_ratio) * volume_ratio",
        },
        {
            "hypothesis": "기관 매수 강도와 프로그램 매수 동시 증가 = 기관 주도 상승",
            "formula": "inst_buy_ratio * pgm_buy_ratio * vol_spike_5d",
        },
    ],
    "volatility": [
        {
            "hypothesis": "볼린저 밴드 수축은 큰 가격 변동 임박 신호 (스퀴즈 전략)",
            "formula": "bb_squeeze * sign(macd_hist) * vol_spike_5d",
        },
        {
            "hypothesis": "ATR 감소 후 갭상승 + 레인지 돌파 = 확장적 돌파 매매 기회",
            "formula": "gap_up_pct * step(atr_7 - atr_21) * range_breakout",
        },
    ],
    "mean_reversion": [
        {
            "hypothesis": "RSI 과매도 반등 + BB 하단 이탈 복귀 = 평균 회귀 기회",
            "formula": "rsi_oversold_bounce * (bb_position - 0.5) * sqrt(volume_ratio)",
        },
        {
            "hypothesis": "연속 하락 후 거래량 폭증 반등 = 과매도 평균 회귀 시그널",
            "formula": "consec_down * step(price_change_pct) * vol_spike_5d",
        },
    ],
}

_CATEGORIES = list(_CATEGORY_EXAMPLES.keys())


def _build_available_features(data_columns: set[str]) -> str:
    """실제 데이터에 존재하는 컬럼 기반으로 Claude에 전달할 피처 목록 생성."""
    blocks = [f"사용 가능한 피처 (변수명):\n{_BASE_FEATURES}"]
    for _key, (desc, check_cols) in _OPTIONAL_FEATURE_BLOCKS.items():
        if any(c in data_columns for c in check_cols):
            blocks.append(desc)
    return "\n".join(blocks)

_HYPOTHESIS_SYSTEM_TEMPLATE = """당신은 퀀트 리서처입니다. 한국 주식시장의 알파 팩터를 발견해야 합니다.

{available_features}

사용 가능한 수학 연산: +, -, *, /, log(), exp(), sqrt(), abs(), pow()
조건부 함수:
  sign(x): x>0→1, x<0→-1, x=0→0 (방향/부호 추출)
  step(x): x>0→1, 아니면 0 (임계값 초과 감지)
  Max(x, y): 둘 중 큰 값
  Min(x, y): 둘 중 작은 값
  clip(x, lo, hi): x를 [lo, hi] 범위로 제한

규칙:
1. 반드시 위 변수명만 사용하세요. 목록에 없는 변수는 절대 사용 금지.
2. 경제적/기술적 근거를 가진 가설을 제시하세요.
3. 기존 성공 팩터와 다른 구조의 수식을 제안하세요 (직교성 유지).
4. 수식은 SymPy 파싱 가능한 형태로 작성하세요.
5. 단순한 단일 변수(예: close, rsi)가 아니라 비선형 조합을 만드세요.
6. 공매도 불가: '상승 종목'을 식별하는 팩터를 만드세요 (Long-only Sharpe로 평가).
7. 턴오버가 낮은(포트폴리오 변경이 적은) 안정적인 팩터가 높은 적합도를 받습니다.
8. 이벤트 피처(vol_dry_then_spike, range_breakout 등)를 적극 활용하세요. 단순 비율뿐 아니라 이벤트 감지를 조합하면 차별화된 알파를 발견할 수 있습니다.

출력 형식 (반드시 이 형식을 따르세요):
가설: [경제적 근거 한 문장]
수식: [SymPy 호환 수식]

중요:
- 수식에 마크다운(```, **, ---, $) 절대 사용 금지. 순수 수학 표현식만 작성.
- Piecewise, if/else 사용 금지. 조건부 연산은 sign(), step(), Max(), Min(), clip()을 사용하세요.
- 수식 줄에는 오직 수학 기호와 변수명만 포함. 설명/주석/LaTeX 금지."""

_MUTATION_SYSTEM_TEMPLATE = """당신은 퀀트 리서처입니다. 아래 알파 팩터 수식이 IC(Information Coefficient) 기준을 통과하지 못했습니다.
수식의 일부를 국소적으로 수정하여 IC를 개선해야 합니다.

{available_features}

사용 가능한 함수: +, -, *, /, log(), exp(), sqrt(), abs(), pow(), sign(), step(), Max(), Min(), clip()

수정 전략:
1. 노이즈가 큰 하위 표현식을 식별하여 대체
2. 연산자 교체 (log→sqrt, /→*, 등)
3. 피처 치환 (atr_14→bb_width, rsi→volume_ratio 등)
4. 상수 조정
5. 이벤트 피처 도입 (range_breakout, vol_dry_then_spike 등)
6. 조건부 함수 활용 (sign(), step()으로 방향/임계값 감지)

출력 형식:
수정사유: [한 문장]
수식: [수정된 SymPy 호환 수식]

중요:
- 수식에 마크다운(```, **, ---, $) 절대 사용 금지. 순수 수학 표현식만 작성.
- Piecewise, if/else 사용 금지. 조건부 연산은 sign(), step(), Max(), Min(), clip()을 사용하세요.
- 수식 줄에는 오직 수학 기호와 변수명만 포함. 설명/주석/LaTeX 금지."""


def _clean_expression(raw: str) -> str:
    """수식 문자열에서 마크다운 아티팩트/주석/한글 설명을 제거."""
    cleaned = raw

    # 1) 코드블록 백틱 제거 (```python, ```sympy, ``` 등)
    cleaned = re.sub(r"```(?:python|sympy)?\s*", "", cleaned)
    cleaned = cleaned.replace("`", "")

    # 2) LaTeX 구분자 제거 ($...$) — 수식 내용은 보존
    cleaned = re.sub(r"\$+", "", cleaned)

    # 3) 마크다운 볼드/이탤릭/구분선 제거
    cleaned = re.sub(r"\*{2,}", "", cleaned)   # ** bold
    cleaned = re.sub(r"_{2,}", "", cleaned)    # __ bold (변수명 _ 는 1개이므로 안전)
    cleaned = re.sub(r"-{3,}", "", cleaned)    # --- separator

    # 4) LaTeX 명령어 블록 제거 (\alpha, \frac{...}{...} 등)
    cleaned = re.sub(r"\\[a-zA-Z]+(?:\{[^}]*\})*", "", cleaned)

    # 5) # 주석 제거
    cleaned = re.split(r"\s*#", cleaned)[0]

    # 6) 후행 한글 설명 제거 (수식 뒤 한글 시작 지점)
    cleaned = re.split(r"\s+[가-힣]", cleaned)[0]

    return cleaned.strip()


def _extract_expression(text: str) -> str | None:
    """Claude 응답에서 수식 부분 추출."""
    # "수식: ..." 패턴 (멀티라인 지원, 다음 필드 경계까지)
    match = re.search(
        r"수식:\s*(.+?)(?:\n\s*(?:수정사유|가설|설명|근거)|$)",
        text,
        re.DOTALL,
    )
    if match:
        raw = match.group(1).strip()
        # 내부에 코드블록이 있으면 그 안의 내용만 추출
        inner = re.search(r"```(?:python|sympy)?\s*(.+?)\s*```", raw, re.DOTALL)
        if inner:
            raw = inner.group(1).strip()
        # 줄바꿈을 제거하여 한 줄로 합침
        raw = re.sub(r"\s*\n\s*", " ", raw)
        return _clean_expression(raw)

    # 코드 블록 내 수식
    match = re.search(r"```(?:python|sympy)?\s*(.+?)\s*```", text, re.DOTALL)
    if match:
        raw = match.group(1).strip()
        raw = re.sub(r"\s*\n\s*", " ", raw)
        return _clean_expression(raw)

    return None


def _extract_hypothesis(text: str) -> str:
    """Claude 응답에서 가설 추출."""
    match = re.search(r"가설:\s*(.+)", text)
    if match:
        return match.group(1).strip()
    return text[:200]


class EvolutionaryAlphaMiner:
    """진화 루프 오케스트레이터."""

    def __init__(
        self,
        data: pl.DataFrame,
        context: str = "",
        max_iterations: int = 5,
        ic_threshold: float = 0.03,
        orthogonality_threshold: float = 0.7,
        use_pysr: bool = False,
        vector_memory: object | None = None,
        db_session: object | None = None,
        enable_crossover: bool = False,
        interval: str = "1d",
    ):
        self.data = data
        self.context = context
        self.max_iterations = max_iterations
        self.ic_threshold = ic_threshold
        self.orthogonality_threshold = orthogonality_threshold
        self.use_pysr = use_pysr
        self.enable_crossover = enable_crossover
        self.interval = interval
        # Phase 3: 벡터 메모리 사용 시 기존 메모리 비활성화
        self.vector_memory = vector_memory
        self.db_session = db_session
        self.memory = ExperienceMemory() if vector_memory is None else None
        self.client = get_llm_client()
        # 실제 데이터 컬럼 기반 동적 피처 목록
        available_features = _build_available_features(set(data.columns))
        self._hypothesis_system_prompt = _HYPOTHESIS_SYSTEM_TEMPLATE.format(
            available_features=available_features,
        )
        self._mutation_system_prompt = _MUTATION_SYSTEM_TEMPLATE.format(
            available_features=available_features,
        )
        logger.info(
            "Miner initialized: %d columns, interval=%s, features prompt length=%d",
            len(data.columns), interval, len(available_features),
        )
        self.discovered: list[DiscoveredFactor] = []
        # 직교성 필터: 발견된 팩터의 실제 값 캐시
        self._discovered_values: dict[str, np.ndarray] = {}
        # Iteration 로그 수집
        self._iteration_logs: list[dict] = []
        self._current_iteration_attempts: list[dict] = []

    def inject_seeds(self, seeds: list[DiscoveredFactor]) -> int:
        """시드 팩터를 모집단에 주입하고 직교성 캐시를 초기화한다.

        Returns
        -------
        int
            성공적으로 주입된 시드 수.
        """
        injected = 0
        for seed in seeds:
            self.discovered.append(seed)

            # 팩터 값 벡터 계산 → 직교성 캐시에 등록
            try:
                parsed = parse_expression(seed.expression_str)
                polars_expr = sympy_to_polars(parsed)
                col = f"_seed_{injected}"
                df_eval = self.data.with_columns(polars_expr.alias(col))
                vals = df_eval[col].drop_nulls().to_numpy()
                if len(vals) > 0:
                    self._discovered_values[col] = vals
            except Exception as e:
                logger.warning(
                    "Seed factor value calc failed: %s — %s",
                    seed.expression_str[:60], e,
                )

            # 경험 메모리에도 추가
            if self.memory:
                self.memory.add(
                    seed.expression_str, seed.metrics.ic_mean, -1, success=True
                )

            injected += 1

        if injected > 0:
            logger.info("Injected %d seed factors into population", injected)

        return injected

    @property
    def iteration_logs(self) -> list[dict]:
        """누적된 iteration 로그 반환."""
        return list(self._iteration_logs)

    def build_summary(self) -> dict:
        """iteration 로그 기반 요약 통계 생성."""
        total_attempts = 0
        total_discovered = 0
        ic_values: list[float] = []
        failed_ic_values: list[float] = []
        failure_counts: dict[str, int] = {}

        for it_log in self._iteration_logs:
            for attempt in it_log.get("attempts", []):
                total_attempts += 1
                outcome = attempt.get("outcome", "")
                if outcome == "discovered":
                    total_discovered += 1
                else:
                    failure_counts[outcome] = failure_counts.get(outcome, 0) + 1

                ic = attempt.get("ic_mean")
                if ic is not None:
                    ic_values.append(ic)
                    if outcome != "discovered":
                        failed_ic_values.append(ic)

        return {
            "total_iterations": len(self._iteration_logs),
            "total_attempts": total_attempts,
            "total_discovered": total_discovered,
            "total_ic_failures": failure_counts.get("ic_below_threshold", 0),
            "total_parse_errors": failure_counts.get("parse_error", 0) + failure_counts.get("eval_error", 0),
            "total_orthogonality_rejections": failure_counts.get("orthogonality_rejected", 0),
            "avg_ic_all_attempts": float(np.mean(ic_values)) if ic_values else None,
            "max_ic_failed": float(max(failed_ic_values)) if failed_ic_values else None,
            "failure_breakdown": failure_counts,
        }

    async def run(
        self,
        progress_cb: ProgressCallback | None = None,
        iteration_cb: IterationCallback | None = None,
    ) -> list[DiscoveredFactor]:
        """메인 진화 루프."""
        total_steps = self.max_iterations
        evaluated = 0

        # Phase A: 데이터 준비
        if progress_cb:
            await progress_cb(0, 100, "데이터 준비 중...")

        self.data = ensure_alpha_features(self.data)
        self.data = compute_forward_returns(self.data, periods=1)

        for iteration in range(self.max_iterations):
            pct = int((iteration / total_steps) * 80) + 10
            if progress_cb:
                await progress_cb(
                    pct, 100,
                    f"반복 {iteration + 1}/{self.max_iterations}: 가설 생성 중..."
                )

            # Phase B: Claude 가설 생성
            try:
                hypothesis, expr_str = await self._generate_hypothesis(iteration)
            except Exception as e:
                logger.warning("Hypothesis generation failed: %s", e)
                # 로그: 가설 생성 실패
                it_log = {
                    "iteration": iteration + 1,
                    "hypothesis": f"[생성 실패] {e}",
                    "attempts": [],
                    "discovered_factor_name": None,
                }
                self._iteration_logs.append(it_log)
                if iteration_cb:
                    await iteration_cb({"type": "iteration_complete", **it_log})
                continue

            if not expr_str:
                logger.warning("No expression extracted from Claude response")
                it_log = {
                    "iteration": iteration + 1,
                    "hypothesis": hypothesis if hypothesis else "[수식 추출 실패]",
                    "attempts": [],
                    "discovered_factor_name": None,
                }
                self._iteration_logs.append(it_log)
                if iteration_cb:
                    await iteration_cb({"type": "iteration_complete", **it_log})
                continue

            # 현재 iteration의 attempt 기록 초기화
            self._current_iteration_attempts = []

            # Phase C: 평가 + 변이 루프
            factor = await self._evaluate_and_mutate(
                expr_str, hypothesis, generation=0,
                progress_cb=progress_cb, iteration_cb=iteration_cb,
            )
            evaluated += 1

            if factor:
                self.discovered.append(factor)
                await self._record_experience(
                    factor.expression_str,
                    factor.hypothesis,
                    factor.metrics.ic_mean,
                    factor.generation,
                    success=True,
                )

            # Iteration 로그 저장
            it_log = {
                "iteration": iteration + 1,
                "hypothesis": hypothesis,
                "attempts": list(self._current_iteration_attempts),
                "discovered_factor_name": factor.name if factor else None,
            }
            self._iteration_logs.append(it_log)
            if iteration_cb:
                await iteration_cb({"type": "iteration_complete", **it_log})

        # Phase 3: 유전 교차
        if self.enable_crossover and len(self.discovered) >= 2:
            crossover_children = await self._run_crossover_phase(progress_cb)
            self.discovered.extend(crossover_children)

        if progress_cb:
            await progress_cb(100, 100, f"완료: {len(self.discovered)}개 팩터 발견")

        # 최종 summary 이벤트
        if iteration_cb:
            await iteration_cb({
                "type": "mining_summary",
                **self.build_summary(),
            })

        return self.discovered

    async def _generate_hypothesis(
        self, iteration: int
    ) -> tuple[str, str | None]:
        """Claude에게 알파 팩터 가설 생성 요청."""
        if self.vector_memory is not None:
            experience_context = self.vector_memory.format_rag_context(
                self.context if self.context else "한국 주식시장 알파 팩터"
            )
        else:
            experience_context = self.memory.format_for_prompt()

        interval_desc = self._interval_description()
        # 구조화된 피드백이 context에 포함된 경우 경험 메모리 뒤에 배치
        # → RAG가 주 영향력, 피드백은 보조적 역할
        feedback_section = f"\n{self.context}" if self.context else ""

        # 카테고리 지시문: 매 반복마다 랜덤 카테고리 → 탐색 다양성 확보
        category = random.choice(_CATEGORIES)
        examples = _CATEGORY_EXAMPLES[category]
        example_text = "\n".join(
            f"  예시: 가설={e['hypothesis']}, 수식={e['formula']}"
            for e in examples
        )
        category_directive = (
            f"\n이번 반복에서는 '{category}' 유형의 팩터를 만들어주세요.\n"
            f"참고 예시:\n{example_text}\n"
            f"위 예시와 동일한 수식은 피하고, 같은 유형의 새로운 조합을 만들어주세요."
        )

        user_msg = f"""반복 {iteration + 1}번째 알파 팩터 가설을 생성해주세요.

데이터 인터벌: {interval_desc}

{experience_context}
{feedback_section}
{category_directive}

위 경험을 참고하여 새로운 알파 팩터를 제안하세요. 기존 팩터와 직교성을 유지하세요."""

        response = await self.client.messages.create(
            model=settings.AGENT_MODEL,
            max_tokens=1000,
            system=self._hypothesis_system_prompt,
            messages=[{"role": "user", "content": user_msg}],
        )

        text = response.content[0].text
        hypothesis = _extract_hypothesis(text)
        expr_str = _extract_expression(text)

        logger.info(
            "Hypothesis generated: %s | Expression: %s",
            hypothesis[:80],
            expr_str,
        )
        return hypothesis, expr_str

    async def _evaluate_and_mutate(
        self,
        expr_str: str,
        hypothesis: str,
        generation: int,
        progress_cb: ProgressCallback | None = None,
        iteration_cb: IterationCallback | None = None,
    ) -> DiscoveredFactor | None:
        """수식 평가 → 실패 시 변이 재시도."""
        max_depth = settings.ALPHA_MAX_MUTATION_DEPTH
        current_expr = expr_str
        current_hypothesis = hypothesis

        for depth in range(max_depth + 1):
            try:
                # 파싱
                parsed = parse_expression(current_expr)
                polars_expr = sympy_to_polars(parsed)

                # 팩터 적용
                col_name = f"alpha_gen{generation}_d{depth}"
                df_eval = self.data.with_columns(polars_expr.alias(col_name))
                df_eval = df_eval.drop_nulls(subset=[col_name, "fwd_return"])

                if df_eval.height < 30:
                    logger.warning("Too few rows after factor calc: %d", df_eval.height)
                    attempt = {
                        "depth": depth,
                        "expression_str": current_expr,
                        "hypothesis": current_hypothesis,
                        "ic_mean": None,
                        "passed_ic": False,
                        "orthogonality_max_corr": None,
                        "passed_orthogonality": None,
                        "outcome": "eval_error",
                        "error_message": f"유효 행 부족 ({df_eval.height}건)",
                    }
                    self._current_iteration_attempts.append(attempt)
                    if iteration_cb:
                        await iteration_cb({"type": "iteration_attempt", **attempt})
                    break

                # IC 계산
                from app.alpha.interval import bars_per_year, default_round_trip_cost
                ann = bars_per_year(self.interval)
                rtc = default_round_trip_cost(self.interval)

                ic_series = compute_ic_series(df_eval, factor_col=col_name)
                metrics = compute_factor_metrics(ic_series, annualize=ann, round_trip_cost=rtc)

                logger.info(
                    "Factor IC=%.4f (threshold=%.4f): %s",
                    metrics.ic_mean, self.ic_threshold, current_expr[:60]
                )

                if metrics.ic_mean >= self.ic_threshold:
                    # 직교성 검증: 기존 발견 팩터와 실제 값 상관계수 체크
                    factor_values = df_eval[col_name].to_numpy()
                    orth_passed, orth_max_corr = self._check_orthogonality_with_corr(
                        factor_values, current_expr
                    )

                    if orth_passed:
                        # 직교성 통과 → 캐시에 저장
                        cache_key = f"gen{generation}_d{depth}"
                        self._discovered_values[cache_key] = factor_values

                        # 전체 메트릭 계산 (Long-only returns + Turnover + Net Sharpe)
                        ls_returns = compute_quantile_returns(df_eval, factor_col=col_name)
                        long_only_returns = compute_long_only_returns(df_eval, factor_col=col_name)
                        pos_turnover, turnover_series = compute_position_turnover(df_eval, factor_col=col_name)
                        full_metrics = compute_factor_metrics(
                            ic_series,
                            ls_returns=ls_returns,
                            long_only_returns=long_only_returns,
                            position_turnover=pos_turnover,
                            turnover_series=turnover_series,
                            round_trip_cost=rtc,
                            annualize=ann,
                        )

                        logger.info(
                            "Factor DISCOVERED: IC=%.4f, Net Sharpe=%.2f, Turnover=%.3f: %s",
                            full_metrics.ic_mean, full_metrics.sharpe,
                            full_metrics.turnover, current_expr[:80],
                        )

                        # 로그: 발견
                        attempt = {
                            "depth": depth,
                            "expression_str": current_expr,
                            "hypothesis": current_hypothesis,
                            "ic_mean": full_metrics.ic_mean,
                            "passed_ic": True,
                            "orthogonality_max_corr": orth_max_corr,
                            "passed_orthogonality": True,
                            "outcome": "discovered",
                            "error_message": None,
                        }
                        self._current_iteration_attempts.append(attempt)
                        if iteration_cb:
                            await iteration_cb({"type": "iteration_attempt", **attempt})

                        return DiscoveredFactor(
                            name=f"alpha_{generation}_{depth}",
                            expression_str=current_expr,
                            expression_sympy=sympy.srepr(parsed),
                            polars_code=sympy_to_code_string(parsed),
                            hypothesis=current_hypothesis,
                            generation=generation + depth,
                            metrics=full_metrics,
                        )
                    else:
                        logger.info(
                            "Factor rejected by orthogonality filter: %s",
                            current_expr[:60],
                        )
                        # 로그: 직교성 거부
                        attempt = {
                            "depth": depth,
                            "expression_str": current_expr,
                            "hypothesis": current_hypothesis,
                            "ic_mean": metrics.ic_mean,
                            "passed_ic": True,
                            "orthogonality_max_corr": orth_max_corr,
                            "passed_orthogonality": False,
                            "outcome": "orthogonality_rejected",
                            "error_message": None,
                        }
                        self._current_iteration_attempts.append(attempt)
                        if iteration_cb:
                            await iteration_cb({"type": "iteration_attempt", **attempt})

                        # 직교성 실패 → 변이 시도
                        await self._record_experience(
                            current_expr, hypothesis, metrics.ic_mean,
                            generation + depth, success=False,
                        )
                        if depth < max_depth:
                            mutated = await self._mutate_expression(
                                current_expr, metrics.ic_mean
                            )
                            if mutated:
                                current_expr = mutated
                            else:
                                break
                        continue

                # IC 미달 → 로그 기록
                attempt = {
                    "depth": depth,
                    "expression_str": current_expr,
                    "hypothesis": current_hypothesis,
                    "ic_mean": metrics.ic_mean,
                    "passed_ic": False,
                    "orthogonality_max_corr": None,
                    "passed_orthogonality": None,
                    "outcome": "ic_below_threshold",
                    "error_message": None,
                }
                self._current_iteration_attempts.append(attempt)
                if iteration_cb:
                    await iteration_cb({"type": "iteration_attempt", **attempt})

                # 실패 → 경험 기록
                await self._record_experience(
                    current_expr, hypothesis, metrics.ic_mean,
                    generation + depth, success=False,
                )

                # 변이 시도
                if depth < max_depth:
                    mutated = await self._mutate_expression(
                        current_expr, metrics.ic_mean
                    )
                    if mutated:
                        current_expr = mutated
                    else:
                        break

            except (ASTConversionError, Exception) as e:
                logger.warning("Factor evaluation failed: %s — %s", current_expr[:60], e)
                # 로그: 파싱/평가 에러
                attempt = {
                    "depth": depth,
                    "expression_str": current_expr,
                    "hypothesis": current_hypothesis,
                    "ic_mean": None,
                    "passed_ic": False,
                    "orthogonality_max_corr": None,
                    "passed_orthogonality": None,
                    "outcome": "parse_error" if isinstance(e, ASTConversionError) else "eval_error",
                    "error_message": str(e)[:200],
                }
                self._current_iteration_attempts.append(attempt)
                if iteration_cb:
                    await iteration_cb({"type": "iteration_attempt", **attempt})

                if depth < max_depth:
                    mutated = await self._mutate_expression(current_expr, 0.0)
                    if mutated:
                        current_expr = mutated
                    else:
                        break
                else:
                    break

        return None

    async def _mutate_expression(
        self, expr_str: str, current_ic: float
    ) -> str | None:
        """Claude에게 수식 변이 요청. 파싱 사전 검증 후 반환."""
        user_msg = f"""다음 알파 팩터 수식이 IC = {current_ic:.4f}로 기준(>= {self.ic_threshold})을 통과하지 못했습니다.

수식: {expr_str}

이 수식을 국소적으로 수정하여 IC를 개선해주세요."""

        try:
            response = await self.client.messages.create(
                model=settings.AGENT_MODEL,
                max_tokens=500,
                system=self._mutation_system_prompt,
                messages=[{"role": "user", "content": user_msg}],
            )
            text = response.content[0].text
            mutated = _extract_expression(text)
            if not mutated:
                return None

            # 파싱 사전 검증: SymPy 파싱 가능한지 확인
            try:
                parse_expression(mutated)
            except ASTConversionError:
                logger.warning("Mutated expression failed to parse: %s", mutated[:60])
                return None

            logger.info("Mutation: %s → %s", expr_str[:40], mutated[:40])
            return mutated
        except Exception as e:
            logger.warning("Mutation request failed: %s", e)
            return None

    def _check_orthogonality(
        self, factor_values: np.ndarray, expr_str: str
    ) -> bool:
        """팩터값 기반 직교성 검증.

        기존 발견 팩터들과의 Pearson 상관계수 최댓값이
        orthogonality_threshold 미만이면 True (직교적).
        """
        if not self._discovered_values:
            return True

        max_corr = 0.0
        for prev_key, prev_values in self._discovered_values.items():
            min_len = min(len(factor_values), len(prev_values))
            if min_len < 10:
                continue
            try:
                corr = abs(
                    np.corrcoef(
                        factor_values[:min_len],
                        prev_values[:min_len],
                    )[0, 1]
                )
                if not np.isnan(corr):
                    max_corr = max(max_corr, corr)
            except Exception:
                continue

        return max_corr < self.orthogonality_threshold

    def _check_orthogonality_with_corr(
        self, factor_values: np.ndarray, expr_str: str
    ) -> tuple[bool, float]:
        """직교성 검증 + max_corr 값 반환 (로그용).

        기존 _check_orthogonality()와 동일 로직이나 (passed, max_corr) 튜플 반환.
        """
        if not self._discovered_values:
            return True, 0.0

        max_corr = 0.0
        for prev_key, prev_values in self._discovered_values.items():
            min_len = min(len(factor_values), len(prev_values))
            if min_len < 10:
                continue
            try:
                corr = abs(
                    np.corrcoef(
                        factor_values[:min_len],
                        prev_values[:min_len],
                    )[0, 1]
                )
                if not np.isnan(corr):
                    max_corr = max(max_corr, corr)
            except Exception:
                continue

        return max_corr < self.orthogonality_threshold, max_corr

    async def _record_experience(
        self,
        expr_str: str,
        hypothesis: str,
        ic_mean: float,
        generation: int,
        success: bool,
    ) -> None:
        """경험을 벡터 메모리 또는 인메모리 메모리에 기록."""
        if self.vector_memory is not None and self.db_session is not None:
            try:
                await self.vector_memory.add(
                    db=self.db_session,
                    expression_str=expr_str,
                    hypothesis=hypothesis,
                    ic_mean=ic_mean,
                    generation=generation,
                    success=success,
                )
            except Exception as e:
                logger.warning("Vector memory add failed: %s", e)
        elif self.memory is not None:
            self.memory.add(expr_str, ic_mean, generation, success=success)

    def _interval_description(self) -> str:
        """Claude 프롬프트용 인터벌 설명."""
        if self.interval == "1d":
            return "일봉 (1일 1봉). 지표 윈도우(sma_20, rsi 등)는 '일' 단위."
        return (
            f"{self.interval} 분봉. 지표 윈도우(sma_20=20봉, rsi=14봉 등)는 '봉' 단위. "
            "장중 패턴(시가 갭, 오전/오후 반전, 거래량 프로파일 등)을 활용하세요."
        )

    async def _run_crossover_phase(
        self,
        progress_cb: ProgressCallback | None = None,
    ) -> list[DiscoveredFactor]:
        """성공 팩터 간 유전 교차로 자식 팩터 생성."""
        from app.alpha.evolution import (
            ScoredFactor,
            crossover,
            tournament_select,
        )

        if progress_cb:
            await progress_cb(90, 100, "유전 교차 진행 중...")

        # 성공 팩터로 population 구성
        population: list[ScoredFactor] = []
        for factor in self.discovered:
            try:
                parsed = parse_expression(factor.expression_str)
                population.append(
                    ScoredFactor(
                        expression=parsed,
                        expression_str=factor.expression_str,
                        hypothesis=factor.hypothesis,
                        ic_mean=factor.metrics.ic_mean,
                        generation=factor.generation,
                        parent_ids=[factor.expression_str[:60]],
                    )
                )
            except (ASTConversionError, Exception):
                continue

        if len(population) < 2:
            return []

        children: list[DiscoveredFactor] = []
        tournament_k = settings.ALPHA_FACTORY_TOURNAMENT_K

        # 토너먼트 선택 → 교차 → 평가 (최대 3라운드)
        max_rounds = min(3, len(population) // 2)
        for round_idx in range(max_rounds):
            parents = tournament_select(population, k=tournament_k, n_select=2)
            if len(parents) < 2:
                break

            offspring_exprs = crossover(parents[0].expression, parents[1].expression)
            parent_ids = [
                p.factor_id or p.expression_str[:30]
                for p in parents
            ]

            for child_expr in offspring_exprs:
                try:
                    polars_expr = sympy_to_polars(child_expr)
                    col_name = f"alpha_crossover_{round_idx}"
                    df_eval = self.data.with_columns(polars_expr.alias(col_name))
                    df_eval = df_eval.drop_nulls(subset=[col_name, "fwd_return"])

                    if df_eval.height < 30:
                        continue

                    from app.alpha.interval import bars_per_year, default_round_trip_cost
                    ic_series = compute_ic_series(df_eval, factor_col=col_name)
                    metrics = compute_factor_metrics(
                        ic_series,
                        annualize=bars_per_year(self.interval),
                        round_trip_cost=default_round_trip_cost(self.interval),
                    )

                    child_expr_str = str(child_expr)
                    hypothesis = f"교차: {parents[0].expression_str[:30]} × {parents[1].expression_str[:30]}"

                    if metrics.ic_mean >= self.ic_threshold:
                        children.append(
                            DiscoveredFactor(
                                name=f"alpha_crossover_{round_idx}",
                                expression_str=child_expr_str,
                                expression_sympy=sympy.srepr(child_expr),
                                polars_code=sympy_to_code_string(child_expr),
                                hypothesis=hypothesis,
                                generation=max(p.generation for p in parents) + 1,
                                metrics=metrics,
                                parent_ids=parent_ids,
                            )
                        )
                        await self._record_experience(
                            child_expr_str, hypothesis,
                            metrics.ic_mean, parents[0].generation + 1,
                            success=True,
                        )
                    else:
                        await self._record_experience(
                            child_expr_str, hypothesis,
                            metrics.ic_mean, parents[0].generation + 1,
                            success=False,
                        )

                except Exception as e:
                    logger.debug("Crossover child evaluation failed: %s", e)

        logger.info("Crossover phase: %d children discovered", len(children))
        return children
