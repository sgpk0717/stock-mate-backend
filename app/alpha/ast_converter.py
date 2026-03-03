"""SymPy AST → Polars Expression 재귀 변환기.

PySR 또는 Claude가 생성한 SymPy 수식을
Polars DataFrame에 적용 가능한 Expression으로 변환한다.
"""

from __future__ import annotations

import hashlib

import sympy
import polars as pl

from app.backtest.indicators import (
    add_atr,
    add_bb,
    add_ema,
    add_macd,
    add_rsi,
    add_sma,
    add_volume_ratio,
    add_price_change_pct,
)

# PySR 변수 → Polars 컬럼 매핑
VARIABLE_MAP: dict[str, str] = {
    "x0": "close",
    "x1": "open",
    "x2": "high",
    "x3": "low",
    "x4": "volume",
    "x5": "sma_20",
    "x6": "rsi",
    "x7": "volume_ratio",
    "x8": "atr_14",
    "x9": "macd_hist",
    "x10": "bb_upper",
    "x11": "bb_lower",
    "x12": "price_change_pct",
}

# 사람 읽기용 이름 → Polars 컬럼 (Claude-only 모드에서 사용)
NAMED_VARIABLE_MAP: dict[str, str] = {
    "close": "close",
    "open": "open",
    "high": "high",
    "low": "low",
    "volume": "volume",
    "sma_20": "sma_20",
    "sma": "sma_20",
    "rsi": "rsi",
    "volume_ratio": "volume_ratio",
    "vol_ratio": "volume_ratio",
    "atr": "atr_14",
    "atr_14": "atr_14",
    "macd_hist": "macd_hist",
    "macd": "macd_hist",
    "bb_upper": "bb_upper",
    "bb_lower": "bb_lower",
    "bb_width": "bb_width",
    "price_change_pct": "price_change_pct",
    "pct_change": "price_change_pct",
    "ema_20": "ema_20",
}

# 모든 유효한 변수명 합집합
_ALL_VARIABLES = {**VARIABLE_MAP, **NAMED_VARIABLE_MAP}


class ASTConversionError(Exception):
    """SymPy → Polars 변환 실패."""


def _resolve_column(name: str) -> str:
    """변수명을 Polars 컬럼명으로 해석."""
    col = _ALL_VARIABLES.get(name)
    if col is None:
        raise ASTConversionError(f"Unknown variable: {name}")
    return col


def sympy_to_polars(expr: sympy.Basic) -> pl.Expr:
    """SymPy 표현식을 Polars Expression으로 재귀 변환."""

    # Symbol → pl.col
    if isinstance(expr, sympy.Symbol):
        return pl.col(_resolve_column(str(expr)))

    # 정수/실수 → pl.lit
    if isinstance(expr, (sympy.Integer, sympy.Float, sympy.Rational)):
        return pl.lit(float(expr))

    # python int/float (sympify 결과에서 발생 가능)
    if isinstance(expr, (int, float)):
        return pl.lit(float(expr))

    # 덧셈: Add(a, b, c, ...)
    if isinstance(expr, sympy.Add):
        args = [sympy_to_polars(a) for a in expr.args]
        result = args[0]
        for a in args[1:]:
            result = result + a
        return result

    # 곱셈: Mul(a, b, c, ...)
    if isinstance(expr, sympy.Mul):
        args = [sympy_to_polars(a) for a in expr.args]
        result = args[0]
        for a in args[1:]:
            result = result * a
        return result

    # 거듭제곱: Pow(base, exp)
    if isinstance(expr, sympy.Pow):
        base = sympy_to_polars(expr.args[0])
        exp_val = expr.args[1]

        # sqrt: Pow(x, 1/2)
        if exp_val == sympy.Rational(1, 2) or exp_val == sympy.Float(0.5):
            return base.sqrt()

        # 정수 지수
        if isinstance(exp_val, (sympy.Integer, sympy.Float, sympy.Rational)):
            return base.pow(float(exp_val))

        # 일반 지수 (변수)
        exp_expr = sympy_to_polars(exp_val)
        return base.pow(exp_expr)

    # log
    if isinstance(expr, sympy.log):
        return sympy_to_polars(expr.args[0]).log()

    # exp
    if isinstance(expr, sympy.exp):
        return sympy_to_polars(expr.args[0]).exp()

    # abs
    if isinstance(expr, sympy.Abs):
        return sympy_to_polars(expr.args[0]).abs()

    # NegativeOne: -1 (Mul(-1, x)로 처리됨)
    if expr == sympy.S.NegativeOne:
        return pl.lit(-1.0)

    # One, Zero 등 상수
    if expr.is_number:
        return pl.lit(float(expr))

    raise ASTConversionError(
        f"Unsupported SymPy node: {type(expr).__name__} ({expr})"
    )


def sympy_to_code_string(expr: sympy.Basic) -> str:
    """SymPy 표현식을 Polars 코드 문자열로 변환 (DB 저장용)."""

    if isinstance(expr, sympy.Symbol):
        return f'pl.col("{_resolve_column(str(expr))}")'

    if isinstance(expr, (sympy.Integer, sympy.Float, sympy.Rational)):
        return f"pl.lit({float(expr)})"

    if isinstance(expr, (int, float)):
        return f"pl.lit({float(expr)})"

    if isinstance(expr, sympy.Add):
        parts = [sympy_to_code_string(a) for a in expr.args]
        return " + ".join(f"({p})" for p in parts)

    if isinstance(expr, sympy.Mul):
        parts = [sympy_to_code_string(a) for a in expr.args]
        return " * ".join(f"({p})" for p in parts)

    if isinstance(expr, sympy.Pow):
        base = sympy_to_code_string(expr.args[0])
        exp_val = expr.args[1]
        if exp_val == sympy.Rational(1, 2) or exp_val == sympy.Float(0.5):
            return f"({base}).sqrt()"
        return f"({base}).pow({float(exp_val)})"

    if isinstance(expr, sympy.log):
        return f"({sympy_to_code_string(expr.args[0])}).log()"

    if isinstance(expr, sympy.exp):
        return f"({sympy_to_code_string(expr.args[0])}).exp()"

    if isinstance(expr, sympy.Abs):
        return f"({sympy_to_code_string(expr.args[0])}).abs()"

    if expr.is_number:
        return f"pl.lit({float(expr)})"

    raise ASTConversionError(
        f"Cannot convert to code string: {type(expr).__name__} ({expr})"
    )


# 기저 지표가 이미 추가되었는지 확인하기 위한 컬럼명 집합
_ALPHA_FEATURE_COLUMNS = {
    "sma_20", "rsi", "volume_ratio", "atr_14",
    "macd_hist", "macd_line", "macd_signal",
    "bb_upper", "bb_lower", "bb_middle",
    "price_change_pct", "ema_20",
}


def ensure_alpha_features(df: pl.DataFrame) -> pl.DataFrame:
    """PySR 변수에 필요한 기저 지표를 DataFrame에 추가.

    이미 존재하는 컬럼은 건너뛴다.
    """
    existing = set(df.columns)

    if "sma_20" not in existing:
        df = add_sma(df, period=20)

    if "ema_20" not in existing:
        df = add_ema(df, period=20)

    if "rsi" not in existing:
        df = add_rsi(df, period=14)

    if "volume_ratio" not in existing:
        df = add_volume_ratio(df, period=20)

    if "atr_14" not in existing:
        df = add_atr(df, period=14)

    if "macd_hist" not in existing:
        df = add_macd(df, fast=12, slow=26, signal=9)

    if "bb_upper" not in existing or "bb_lower" not in existing:
        df = add_bb(df, period=20, std=2.0)

    if "price_change_pct" not in existing:
        df = add_price_change_pct(df, period=1)

    # bb_width: bb_upper - bb_lower (편의용)
    if "bb_width" not in existing and "bb_upper" in df.columns:
        df = df.with_columns(
            (pl.col("bb_upper") - pl.col("bb_lower")).alias("bb_width")
        )

    return df


def expression_hash(expr: sympy.Basic) -> str:
    """Merkle-style 바텀업 구조 해시. 상수값 무시, 구조만 해싱.

    동일 구조(상수만 다른) 수식은 같은 해시를 반환한다.
    예: close * 2.0 + rsi == close * 3.0 + rsi
    """

    def _hash_node(node: sympy.Basic) -> str:
        # 리프: 숫자 → "N" (값 무시)
        if isinstance(node, (sympy.Integer, sympy.Float, sympy.Rational)):
            return "N"
        if isinstance(node, (int, float)):
            return "N"
        if node.is_number:
            return "N"

        # 리프: 심볼 → "S:name"
        if isinstance(node, sympy.Symbol):
            return f"S:{node.name}"

        # 내부 노드: "TypeName(child1,child2,...)"
        type_name = type(node).__name__
        if hasattr(node, "args") and node.args:
            child_hashes = sorted(_hash_node(a) for a in node.args)
            content = f"{type_name}({','.join(child_hashes)})"
        else:
            content = type_name

        return hashlib.sha256(content.encode()).hexdigest()[:16]

    return _hash_node(expr)


def tree_depth(expr: sympy.Basic) -> int:
    """AST 깊이 계산. 리프=0."""
    if not hasattr(expr, "args") or not expr.args:
        return 0
    return 1 + max(tree_depth(a) for a in expr.args)


def tree_size(expr: sympy.Basic) -> int:
    """AST 노드 수 계산. 모든 노드 카운트."""
    if not hasattr(expr, "args") or not expr.args:
        return 1
    return 1 + sum(tree_size(a) for a in expr.args)


def parse_expression(expr_str: str) -> sympy.Basic:
    """문자열을 SymPy 표현식으로 파싱.

    Claude가 생성한 수식 문자열을 sympify로 변환한다.
    변수명을 SymPy Symbol로 인식시키기 위해 local_dict를 제공.
    """
    local_dict = {name: sympy.Symbol(name) for name in _ALL_VARIABLES}
    # 자주 쓰이는 함수 추가
    local_dict["log"] = sympy.log
    local_dict["exp"] = sympy.exp
    local_dict["sqrt"] = sympy.sqrt
    local_dict["abs"] = sympy.Abs

    try:
        return sympy.sympify(expr_str, locals=local_dict)
    except (sympy.SympifyError, SyntaxError, TypeError) as e:
        raise ASTConversionError(f"Failed to parse expression: {expr_str}") from e
