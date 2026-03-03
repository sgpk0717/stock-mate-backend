"""마이너 로직 단위 테스트 (T15-T20)."""

from __future__ import annotations

from app.alpha.miner import (
    ExperienceMemory,
    _clean_expression,
    _extract_expression,
    _extract_hypothesis,
)


class TestExtractExpression:
    """수식 추출 테스트."""

    def test_t15_basic_extraction(self):
        """T15: 기본 '수식: ...' 패턴 추출."""
        text = "가설: RSI 과매도 반등\n수식: log(rsi) * volume_ratio"
        result = _extract_expression(text)
        assert result == "log(rsi) * volume_ratio"

    def test_t16_code_block_extraction(self):
        """T16: 코드 블록 내 수식 추출."""
        text = "다음은 제안하는 수식입니다:\n```python\nlog(rsi)\n```"
        result = _extract_expression(text)
        assert result == "log(rsi)"

    def test_t17_no_expression(self):
        """T17: 수식 없는 텍스트 → None."""
        text = "이 텍스트에는 수식이 포함되어 있지 않습니다."
        result = _extract_expression(text)
        assert result is None

    def test_t20_trailing_korean_stripped(self):
        """T20: 수식 뒤 한글 설명 트리밍."""
        text = "수식: log(rsi) * volume_ratio 이것은 좋은 팩터입니다"
        result = _extract_expression(text)
        assert result == "log(rsi) * volume_ratio"

    def test_trailing_comment_stripped(self):
        """수식 뒤 # 주석 트리밍."""
        text = "수식: log(rsi) * volume_ratio  # 과매도 신호"
        result = _extract_expression(text)
        assert result == "log(rsi) * volume_ratio"

    def test_multiline_expression(self):
        """수식이 줄바꿈 포함 시 한 줄로 합침."""
        text = "수식: log(volume_ratio) *\n(30 - rsi) / atr_14\n가설: 다음 가설"
        result = _extract_expression(text)
        assert result is not None
        assert "log(volume_ratio)" in result
        assert "atr_14" in result
        # 줄바꿈이 제거되어야
        assert "\n" not in result


    def test_extract_with_inner_codeblock(self):
        """수식: 내부에 코드블록이 포함된 경우."""
        text = "수식: ```python\nlog(rsi) * volume_ratio\n``` --- **SymPy"
        result = _extract_expression(text)
        assert result == "log(rsi) * volume_ratio"

    def test_extract_inline_backticks_and_separator(self):
        """실제 Claude 출력: 수식 뒤 백틱 + 구분선."""
        text = (
            "가설: RSI 과열 반전\n"
            "수식: ((bb_upper - close) / (bb_width + 1)) * (100 - rsi) ``` ---"
        )
        result = _extract_expression(text)
        assert result == "((bb_upper - close) / (bb_width + 1)) * (100 - rsi)"

    def test_extract_bold_prefix_and_latex(self):
        """실제 Claude 출력: 볼드 + LaTeX 후행."""
        text = (
            "수식: ** ``` macd_hist / (atr_14 * rsi) ``` "
            "$\\alpha = \\frac{x}{y}$ ---"
        )
        result = _extract_expression(text)
        assert result == "macd_hist / (atr_14 * rsi)"


class TestCleanExpression:
    """수식 클리닝 테스트."""

    def test_clean_comment(self):
        result = _clean_expression("log(rsi) * vol  # 좋은 팩터")
        assert result == "log(rsi) * vol"

    def test_clean_korean_suffix(self):
        result = _clean_expression("log(rsi) * vol 이것은")
        assert result == "log(rsi) * vol"

    def test_clean_no_suffix(self):
        result = _clean_expression("log(rsi) * vol")
        assert result == "log(rsi) * vol"

    def test_clean_trailing_backticks(self):
        """후행 백틱 + 구분선 제거."""
        result = _clean_expression("log(rsi) * vol ``` ---")
        assert result == "log(rsi) * vol"

    def test_clean_bold_markers(self):
        """마크다운 볼드 마커 제거."""
        result = _clean_expression("** log(rsi) * vol **")
        assert result == "log(rsi) * vol"

    def test_clean_latex_suffix(self):
        """LaTeX 구분자 + 명령어 제거."""
        result = _clean_expression(
            "log(rsi) $\\alpha = \\frac{x}{y}$"
        )
        # LaTeX 명령어와 $ 제거 → 남는 건 log(rsi) + 잔여 공백/문자
        assert "log(rsi)" in result
        assert "$" not in result
        assert "\\" not in result

    def test_clean_mixed_artifacts(self):
        """백틱 + 볼드 + 구분선 혼합."""
        result = _clean_expression(
            "``` -1 * (close - bb_lower) / bb_width ``` --- **SymPy"
        )
        assert "-1 * (close - bb_lower) / bb_width" in result
        assert "```" not in result
        assert "---" not in result


class TestExtractHypothesis:
    """가설 추출 테스트."""

    def test_basic(self):
        text = "가설: RSI 과매도 반등 신호\n수식: log(rsi)"
        result = _extract_hypothesis(text)
        assert result == "RSI 과매도 반등 신호"

    def test_no_hypothesis_prefix(self):
        text = "이것은 가설 형식이 없는 텍스트입니다."
        result = _extract_hypothesis(text)
        # 200자까지 반환
        assert len(result) <= 200


class TestExperienceMemory:
    """경험 메모리 테스트."""

    def test_t18_top_k_sorted(self):
        """T18: top_k는 IC 높은 순 정렬."""
        mem = ExperienceMemory()
        mem.add("expr_a", 0.05, 0, success=True)
        mem.add("expr_b", 0.10, 1, success=True)
        mem.add("expr_c", 0.03, 2, success=True)

        top = mem.top_k(3)
        assert len(top) == 3
        assert top[0]["ic_mean"] == 0.10
        assert top[1]["ic_mean"] == 0.05
        assert top[2]["ic_mean"] == 0.03

    def test_t19_empty_memory(self):
        """T19: 빈 메모리 → 안내 메시지."""
        mem = ExperienceMemory()
        prompt = mem.format_for_prompt()
        assert "아직 탐색 이력이 없습니다" in prompt

    def test_failures_recent_order(self):
        """실패 팩터는 최신순 (마지막 k개)."""
        mem = ExperienceMemory()
        mem.add("fail_1", 0.01, 0, success=False)
        mem.add("fail_2", 0.02, 1, success=False)
        mem.add("fail_3", 0.005, 2, success=False)

        fails = mem.failures(2)
        assert len(fails) == 2
        # 마지막 2개: fail_2, fail_3
        assert fails[0]["expression"] == "fail_2"
        assert fails[1]["expression"] == "fail_3"

    def test_format_with_data(self):
        """데이터 있을 때 프롬프트 형식 확인."""
        mem = ExperienceMemory()
        mem.add("log(rsi)", 0.05, 0, success=True)
        mem.add("sqrt(vol)", 0.01, 1, success=False)

        prompt = mem.format_for_prompt()
        assert "성공한 팩터" in prompt
        assert "실패한 팩터" in prompt
        assert "log(rsi)" in prompt
