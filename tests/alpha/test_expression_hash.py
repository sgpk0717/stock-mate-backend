"""구조적 해싱 + tree_depth + tree_size 단위 테스트.

Phase 1: ast_converter.py에 추가될 expression_hash(), tree_depth(), tree_size() 함수.
"""

from __future__ import annotations

import sympy

from app.alpha.ast_converter import parse_expression


class TestExpressionHash:
    """Merkle-style 구조적 해싱 테스트."""

    def test_same_structure_different_constants_same_hash(self):
        """상수값만 다른 수식은 같은 해시를 가져야 한다."""
        from app.alpha.ast_converter import expression_hash

        expr_a = parse_expression("close * 2.0 + rsi")
        expr_b = parse_expression("close * 3.0 + rsi")
        assert expression_hash(expr_a) == expression_hash(expr_b)

    def test_different_structure_different_hash(self):
        """구조가 다른 수식은 다른 해시를 가져야 한다."""
        from app.alpha.ast_converter import expression_hash

        expr_a = parse_expression("close * rsi + volume_ratio")
        expr_b = parse_expression("close + rsi * volume_ratio")
        assert expression_hash(expr_a) != expression_hash(expr_b)

    def test_same_expression_same_hash(self):
        """완전히 동일한 수식은 같은 해시를 가져야 한다."""
        from app.alpha.ast_converter import expression_hash

        expr_a = parse_expression("log(close / sma_20)")
        expr_b = parse_expression("log(close / sma_20)")
        assert expression_hash(expr_a) == expression_hash(expr_b)

    def test_hash_is_hex_string(self):
        """해시 결과는 hex 문자열이어야 한다."""
        from app.alpha.ast_converter import expression_hash

        expr = parse_expression("close * rsi")
        h = expression_hash(expr)
        assert isinstance(h, str)
        assert len(h) > 0
        # hex 문자만 포함
        assert all(c in "0123456789abcdef" for c in h)

    def test_different_functions_different_hash(self):
        """다른 함수 적용은 다른 해시를 생성해야 한다."""
        from app.alpha.ast_converter import expression_hash

        expr_a = parse_expression("log(close)")
        expr_b = parse_expression("sqrt(close)")
        assert expression_hash(expr_a) != expression_hash(expr_b)

    def test_symbol_only_hash(self):
        """단일 심볼도 해시 가능해야 한다."""
        from app.alpha.ast_converter import expression_hash

        expr = parse_expression("close")
        h = expression_hash(expr)
        assert isinstance(h, str)
        assert len(h) > 0

    def test_commutative_operations_same_hash(self):
        """교환법칙: close + rsi == rsi + close 같은 해시."""
        from app.alpha.ast_converter import expression_hash

        expr_a = parse_expression("close + rsi")
        expr_b = parse_expression("rsi + close")
        # SymPy가 자동 정렬하므로 같아야 함
        assert expression_hash(expr_a) == expression_hash(expr_b)


class TestTreeDepth:
    """AST 깊이 계산 테스트."""

    def test_single_symbol_depth_zero(self):
        """단일 심볼의 깊이는 0."""
        from app.alpha.ast_converter import tree_depth

        expr = parse_expression("close")
        assert tree_depth(expr) == 0

    def test_single_operation_depth_one(self):
        """단일 연산(close + rsi)의 깊이는 1."""
        from app.alpha.ast_converter import tree_depth

        expr = parse_expression("close + rsi")
        assert tree_depth(expr) == 1

    def test_nested_depth(self):
        """중첩 연산의 깊이가 올바르게 계산되는지."""
        from app.alpha.ast_converter import tree_depth

        # log(close + rsi) → depth 2: log > Add > (close, rsi)
        expr = parse_expression("log(close + rsi)")
        depth = tree_depth(expr)
        assert depth >= 2

    def test_constant_depth_zero(self):
        """상수의 깊이는 0."""
        from app.alpha.ast_converter import tree_depth

        expr = parse_expression("3.14")
        assert tree_depth(expr) == 0


class TestTreeSize:
    """AST 노드 수 계산 테스트."""

    def test_single_symbol_size_one(self):
        """단일 심볼의 노드 수는 1."""
        from app.alpha.ast_converter import tree_size

        expr = parse_expression("close")
        assert tree_size(expr) == 1

    def test_addition_size(self):
        """close + rsi의 노드 수는 3 (Add, close, rsi)."""
        from app.alpha.ast_converter import tree_size

        expr = parse_expression("close + rsi")
        size = tree_size(expr)
        assert size == 3

    def test_complex_expression_size(self):
        """복합 수식의 노드 수가 올바르게 카운트되는지."""
        from app.alpha.ast_converter import tree_size

        # log(close / sma_20) * rsi
        # Mul(log(Mul(close, Pow(sma_20, -1))), rsi) 구조
        expr = parse_expression("log(close / sma_20) * rsi")
        size = tree_size(expr)
        # 최소 5개 이상의 노드
        assert size >= 5

    def test_size_greater_than_depth(self):
        """항상 size >= depth + 1."""
        from app.alpha.ast_converter import tree_depth, tree_size

        expr = parse_expression("log(volume_ratio) * (30 - rsi) / atr_14")
        assert tree_size(expr) >= tree_depth(expr) + 1
