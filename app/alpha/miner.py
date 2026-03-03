"""진화형 알파 팩터 마이너.

Claude + (옵션: PySR) 기반 알파 팩터 발굴 + 궤적 변이 루프.
"""

from __future__ import annotations

import logging
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

_AVAILABLE_FEATURES = """사용 가능한 피처 (변수명):
- close, open, high, low, volume: OHLCV 원본
- sma_20: 20일 단순이동평균
- ema_20: 20일 지수이동평균
- rsi: 14일 RSI (0~100)
- volume_ratio: 20일 평균 대비 거래량 비율
- atr_14: 14일 Average True Range
- macd_hist: MACD 히스토그램 (12/26/9)
- bb_upper, bb_lower: 볼린저 밴드 상/하한 (20일, 2σ)
- bb_width: bb_upper - bb_lower
- price_change_pct: 전일 대비 종가 변화율 (%)"""

_HYPOTHESIS_SYSTEM_PROMPT = f"""당신은 퀀트 리서처입니다. 한국 주식시장의 알파 팩터를 발견해야 합니다.

{_AVAILABLE_FEATURES}

사용 가능한 수학 연산: +, -, *, /, log(), exp(), sqrt(), abs(), pow()

규칙:
1. 반드시 위 변수명만 사용하세요.
2. 경제적/기술적 근거를 가진 가설을 제시하세요.
3. 기존 성공 팩터와 다른 구조의 수식을 제안하세요 (직교성 유지).
4. 수식은 SymPy 파싱 가능한 형태로 작성하세요.
5. 단순한 단일 변수(예: close, rsi)가 아니라 비선형 조합을 만드세요.

출력 형식 (반드시 이 형식을 따르세요):
가설: [경제적 근거 한 문장]
수식: [SymPy 호환 수식]

중요:
- 수식에 마크다운(```, **, ---, $) 절대 사용 금지. 순수 수학 표현식만 작성.
- sign(), Piecewise, if/else 사용 금지. 부호 반전은 -1 * 로, 절댓값은 abs()로 대체.
- 수식 줄에는 오직 수학 기호와 변수명만 포함. 설명/주석/LaTeX 금지."""

_MUTATION_SYSTEM_PROMPT = f"""당신은 퀀트 리서처입니다. 아래 알파 팩터 수식이 IC(Information Coefficient) 기준을 통과하지 못했습니다.
수식의 일부를 국소적으로 수정하여 IC를 개선해야 합니다.

{_AVAILABLE_FEATURES}

수정 전략:
1. 노이즈가 큰 하위 표현식을 식별하여 대체
2. 연산자 교체 (log→sqrt, /→*, 등)
3. 피처 치환 (atr_14→bb_width, rsi→volume_ratio 등)
4. 상수 조정

출력 형식:
수정사유: [한 문장]
수식: [수정된 SymPy 호환 수식]

중요:
- 수식에 마크다운(```, **, ---, $) 절대 사용 금지. 순수 수학 표현식만 작성.
- sign(), Piecewise, if/else 사용 금지. 부호 반전은 -1 * 로, 절댓값은 abs()로 대체.
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
    ):
        self.data = data
        self.context = context
        self.max_iterations = max_iterations
        self.ic_threshold = ic_threshold
        self.orthogonality_threshold = orthogonality_threshold
        self.use_pysr = use_pysr
        self.enable_crossover = enable_crossover
        # Phase 3: 벡터 메모리 사용 시 기존 메모리 비활성화
        self.vector_memory = vector_memory
        self.db_session = db_session
        self.memory = ExperienceMemory() if vector_memory is None else None
        self.client = get_llm_client()
        self.discovered: list[DiscoveredFactor] = []
        # 직교성 필터: 발견된 팩터의 실제 값 캐시
        self._discovered_values: dict[str, np.ndarray] = {}
        # Iteration 로그 수집
        self._iteration_logs: list[dict] = []
        self._current_iteration_attempts: list[dict] = []

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

        user_msg = f"""반복 {iteration + 1}번째 알파 팩터 가설을 생성해주세요.

시장 맥락: {self.context if self.context else '한국 주식시장 일봉 데이터'}

{experience_context}

위 경험을 참고하여 새로운 알파 팩터를 제안하세요. 기존 팩터와 직교성을 유지하세요."""

        response = await self.client.messages.create(
            model=settings.AGENT_MODEL,
            max_tokens=1000,
            system=_HYPOTHESIS_SYSTEM_PROMPT,
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
                ic_series = compute_ic_series(df_eval, factor_col=col_name)
                metrics = compute_factor_metrics(ic_series)

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

                        # 로그: 발견
                        attempt = {
                            "depth": depth,
                            "expression_str": current_expr,
                            "hypothesis": current_hypothesis,
                            "ic_mean": metrics.ic_mean,
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
                            metrics=metrics,
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
                system=_MUTATION_SYSTEM_PROMPT,
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

                    ic_series = compute_ic_series(df_eval, factor_col=col_name)
                    metrics = compute_factor_metrics(ic_series)

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
