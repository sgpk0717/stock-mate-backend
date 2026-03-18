"""유전 연산 — SymPy AST 기반 교차/변이/선택.

DEAP 의존성 없이 SymPy의 트리 구조를 직접 조작한다.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field

import numpy as np
import sympy

from app.alpha.ast_converter import (
    ASTConversionError,
    NAMED_VARIABLE_MAP,
    parse_expression,
    sympy_to_code_string,
    sympy_to_polars,
)

logger = logging.getLogger(__name__)

# 피처 치환용 변수 목록
_FEATURES = list(NAMED_VARIABLE_MAP.keys())
_MAX_DEPTH = 6


def _get_subtrees(expr: sympy.Basic) -> list[sympy.Basic]:
    """재귀적으로 모든 하위 트리를 수집."""
    result = [expr]
    if hasattr(expr, "args"):
        for arg in expr.args:
            result.extend(_get_subtrees(arg))
    return result


def _get_depth(expr: sympy.Basic) -> int:
    """AST 깊이 계산."""
    if not hasattr(expr, "args") or not expr.args:
        return 0
    return 1 + max(_get_depth(a) for a in expr.args)


def crossover(
    expr_a: sympy.Basic,
    expr_b: sympy.Basic,
) -> list[sympy.Basic]:
    """AST 하위 트리 교환으로 자식 생성.

    각 부모에서 랜덤 하위 트리를 선택 → 서브트리 교환.
    깊이 제한 초과 또는 Polars 변환 불가 시 제외.
    """
    try:
        subtrees_a = _get_subtrees(expr_a)
        subtrees_b = _get_subtrees(expr_b)

        # 루트 자체는 제외 — 완전 교체 방지
        candidates_a = [s for s in subtrees_a if s != expr_a]
        candidates_b = [s for s in subtrees_b if s != expr_b]

        if not candidates_a or not candidates_b:
            return []

        swap_a = random.choice(candidates_a)
        swap_b = random.choice(candidates_b)

        child_a = expr_a.subs(swap_a, swap_b)
        child_b = expr_b.subs(swap_b, swap_a)

        results = []
        for child in [child_a, child_b]:
            if _get_depth(child) <= _MAX_DEPTH:
                try:
                    sympy_to_polars(child)
                    results.append(child)
                except (ASTConversionError, Exception):
                    pass

        return results

    except Exception as e:
        logger.debug("Crossover failed: %s", e)
        return []


def mutate(expr: sympy.Basic) -> sympy.Basic | None:
    """SymPy AST를 국소적으로 변이.

    변이 전략 (랜덤 선택):
    1. 연산자 교체 (Add↔Mul)
    2. 상수 섭동 (±20%)
    3. 피처 치환
    4. 함수 래핑/제거
    """
    strategies = [
        _mutate_operator,
        _mutate_constant,
        _mutate_feature,
        _mutate_function,
    ]
    random.shuffle(strategies)

    for strategy in strategies:
        try:
            result = strategy(expr)
            if result is not None and result != expr:
                if _get_depth(result) <= _MAX_DEPTH:
                    sympy_to_polars(result)
                    return result
        except (ASTConversionError, Exception):
            continue

    return None


def _mutate_operator(expr: sympy.Basic) -> sympy.Basic | None:
    """Add↔Mul 연산자 교체."""
    subtrees = _get_subtrees(expr)
    targets = [s for s in subtrees if isinstance(s, (sympy.Add, sympy.Mul))]
    if not targets:
        return None

    target = random.choice(targets)
    if isinstance(target, sympy.Add):
        replacement = sympy.Mul(*target.args)
    else:
        replacement = sympy.Add(*target.args)

    return expr.subs(target, replacement)


def _mutate_constant(expr: sympy.Basic) -> sympy.Basic | None:
    """상수를 ±20% 섭동."""
    subtrees = _get_subtrees(expr)
    constants = [
        s
        for s in subtrees
        if isinstance(s, (sympy.Integer, sympy.Float, sympy.Rational))
        and float(s) != 0
    ]
    if not constants:
        return None

    target = random.choice(constants)
    factor = random.uniform(0.8, 1.2)
    new_val = sympy.Float(round(float(target) * factor, 6))
    return expr.subs(target, new_val)


def _mutate_feature(expr: sympy.Basic) -> sympy.Basic | None:
    """피처 심볼을 다른 피처로 교체."""
    subtrees = _get_subtrees(expr)
    symbols = [s for s in subtrees if isinstance(s, sympy.Symbol)]
    if not symbols:
        return None

    target = random.choice(symbols)
    current_name = str(target)
    candidates = [f for f in _FEATURES if f != current_name]
    if not candidates:
        return None

    new_feature = sympy.Symbol(random.choice(candidates))
    return expr.subs(target, new_feature)


def _mutate_function(expr: sympy.Basic) -> sympy.Basic | None:
    """함수 래핑(x→log(x)) 또는 제거(log(x)→x)."""
    subtrees = _get_subtrees(expr)

    # 기존 함수 제거 시도
    wrapped = [s for s in subtrees if isinstance(s, (sympy.log, sympy.Abs, sympy.sign, sympy.Heaviside))]
    if wrapped and random.random() < 0.5:
        target = random.choice(wrapped)
        return expr.subs(target, target.args[0])

    # 심볼에 함수 래핑
    symbols = [s for s in subtrees if isinstance(s, sympy.Symbol)]
    if symbols:
        target = random.choice(symbols)
        func = random.choice([sympy.log, sympy.sqrt, sympy.Abs, sympy.sign, sympy.Heaviside])
        return expr.subs(target, func(target))

    return None


@dataclass
class ScoredFactor:
    """토너먼트 선택용 팩터."""

    expression: sympy.Basic
    expression_str: str
    hypothesis: str
    ic_mean: float
    generation: int
    parent_ids: list[str] = field(default_factory=list)
    factor_id: str | None = None
    fitness_composite: float = 0.0
    tree_depth: int = 0
    tree_size: int = 0
    expression_hash: str = ""
    operator_origin: str = ""
    ic_std: float = 0.0
    icir: float = 0.0
    turnover: float = 0.0          # 포지션 턴오버 (일별 포트폴리오 변경 비율)
    sharpe: float = 0.0            # Long-only Sharpe (상위 분위 포트폴리오)
    max_drawdown: float = 0.0


def hoist_mutation(expr: sympy.Basic) -> sympy.Basic | None:
    """랜덤 서브트리를 루트로 승격. 트리 크기 감소 효과.

    복잡한 수식에서 의미 있는 서브트리를 추출하여 bloat을 억제한다.
    리프 노드만 존재하면 None 반환.
    """
    subtrees = _get_subtrees(expr)
    # 루트와 리프 제외 — 중간 노드만 후보
    candidates = [
        s for s in subtrees
        if s != expr
        and hasattr(s, "args") and s.args  # 리프가 아닌 서브트리
    ]

    if not candidates:
        return None

    hoisted = random.choice(candidates)

    try:
        # Polars 변환 가능 확인
        sympy_to_polars(hoisted)
        return hoisted
    except (ASTConversionError, Exception):
        return None


def ephemeral_constant_mutation(expr: sympy.Basic) -> sympy.Basic | None:
    """리프 심볼을 랜덤 상수(log-uniform 0.01~100)로 교체.

    GP의 Ephemeral Random Constant (ERC) 패턴.
    """
    import math as _math

    subtrees = _get_subtrees(expr)
    symbols = [s for s in subtrees if isinstance(s, sympy.Symbol)]
    if not symbols:
        return None

    target = random.choice(symbols)
    # log-uniform: exp(uniform(log(0.01), log(100)))
    log_val = random.uniform(_math.log(0.01), _math.log(100))
    constant = round(_math.exp(log_val), 4)

    result = expr.subs(target, sympy.Float(constant))

    if result == expr:
        return None

    try:
        if _get_depth(result) <= _MAX_DEPTH:
            sympy_to_polars(result)
            return result
    except (ASTConversionError, Exception):
        pass

    return None


def tournament_select(
    population: list[ScoredFactor],
    k: int = 3,
    n_select: int = 2,
    parsimony: bool = True,
    multi_objective: bool = True,
) -> list[ScoredFactor]:
    """토너먼트 선택 (Multi-Objective Mini-Lexicase + Parsimony Pressure).

    multi_objective=True: IC/Sharpe/Turnover 3축 mini-lexicase로 다양성 유지.
    parsimony=True: tree_size 1차 필터 적용.
    """
    if len(population) < n_select:
        return list(population)

    def _select_one() -> ScoredFactor:
        tournament = random.sample(population, min(k, len(population)))
        if parsimony and len(tournament) > 1:
            # 1차: tree_size로 상위 70% 필터
            size_sorted = sorted(tournament, key=lambda f: f.tree_size)
            keep_n = max(2, int(len(tournament) * 0.7))
            tournament = size_sorted[:keep_n]

        if multi_objective and len(tournament) > 1:
            # 3축 mini-lexicase: [ic_mean, long_only_sharpe, -position_turnover]
            objectives = [
                ("ic_mean", lambda f: f.ic_mean if f.ic_mean is not None else 0.0),
                ("long_only_sharpe", lambda f: f.sharpe if f.sharpe is not None else 0.0),
                ("low_turnover", lambda f: -(f.turnover if f.turnover is not None else 0.0)),
            ]
            random.shuffle(objectives)
            candidates = list(tournament)
            for _, key_fn in objectives:
                if len(candidates) <= 1:
                    break
                best_val = max(key_fn(c) for c in candidates)
                # ε = MAD (Median Absolute Deviation)
                vals = [key_fn(c) for c in candidates]
                median_val = sorted(vals)[len(vals) // 2]
                mad = max(1e-10, float(np.median([abs(v - median_val) for v in vals])))
                candidates = [c for c in candidates if key_fn(c) >= best_val - mad]
            return random.choice(candidates)
        else:
            return max(tournament, key=lambda f: f.fitness_composite)

    selected: list[ScoredFactor] = []
    attempts = 0
    while len(selected) < n_select and attempts < n_select * 3:
        winner = _select_one()
        if winner not in selected:
            selected.append(winner)
        attempts += 1

    if len(selected) < n_select:
        return list(population[:n_select])

    return selected
