"""AST 변환기 단위 테스트 (T01-T06)."""

from __future__ import annotations

import polars as pl
import pytest
import sympy

from app.alpha.ast_converter import (
    ASTConversionError,
    NAMED_VARIABLE_MAP,
    ensure_alpha_features,
    parse_expression,
    sympy_to_code_string,
    sympy_to_polars,
)


class TestParseExpression:
    """수식 파싱 테스트."""

    def test_t01_complex_expression(self):
        """T01: 복합 수식 파싱 성공."""
        expr = parse_expression("log(volume_ratio) * (30 - rsi) / atr_14")
        assert isinstance(expr, sympy.Basic)
        # Polars Expr 변환도 성공해야
        polars_expr = sympy_to_polars(expr)
        assert polars_expr is not None

    def test_t02_all_named_variables(self):
        """T02: 모든 NAMED_VARIABLE_MAP 키 파싱 에러 없음."""
        for var_name in NAMED_VARIABLE_MAP:
            expr = parse_expression(var_name)
            assert isinstance(expr, sympy.Basic), f"Failed for: {var_name}"

    def test_t03_unknown_variable(self):
        """T03: 미지 변수 → ASTConversionError."""
        expr = parse_expression("unknown_var + close")
        # parse_expression은 성공하지만 (sympy가 Symbol로 만듦)
        # sympy_to_polars에서 _resolve_column이 실패해야
        with pytest.raises(ASTConversionError, match="Unknown variable"):
            sympy_to_polars(expr)

    def test_t04_nested_expression(self):
        """T04: 중첩 수식 (sqrt, abs, 나눗셈) 변환 성공."""
        expr = parse_expression("sqrt(abs(close - sma_20) / atr_14)")
        polars_expr = sympy_to_polars(expr)
        assert polars_expr is not None


class TestSympyToCodeString:
    """코드 문자열 변환 테스트."""

    def test_t05_code_string_contains_polars(self):
        """T05: 출력이 pl.col/pl.lit 포함."""
        expr = parse_expression("log(volume_ratio) * rsi")
        code = sympy_to_code_string(expr)
        assert "pl.col" in code
        # log() 변환 확인
        assert ".log()" in code


class TestPolarsApplication:
    """실제 Polars DataFrame 적용 테스트."""

    def test_t06_apply_to_dataframe(self, sample_ohlcv_with_indicators):
        """T06: 실제 DF에 팩터 적용 → 비null 값 존재."""
        df = sample_ohlcv_with_indicators

        expr = parse_expression("log(volume_ratio) * (30 - rsi) / atr_14")
        polars_expr = sympy_to_polars(expr)

        result = df.with_columns(polars_expr.alias("alpha_test"))

        assert "alpha_test" in result.columns
        # 초기 행은 지표 워밍업으로 null일 수 있지만, 일부 행은 값이 있어야
        non_null = result.filter(pl.col("alpha_test").is_not_null())
        assert non_null.height > 0, "All alpha values are null"

        # NaN이 아닌 값도 있어야
        non_nan = non_null.filter(pl.col("alpha_test").is_not_nan())
        assert non_nan.height > 0, "All non-null alpha values are NaN"
