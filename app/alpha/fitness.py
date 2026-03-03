"""다목적 복합 적합도 함수.

IC, ICIR, turnover, 복잡도를 가중 합산하여 단일 fitness 스코어를 산출한다.
"""

from __future__ import annotations


def compute_composite_fitness(
    ic_mean: float,
    icir: float,
    turnover: float,
    tree_depth: int,
    tree_size: int,
    w_ic: float = 0.40,
    w_icir: float = 0.30,
    w_turnover: float = 0.15,
    w_complexity: float = 0.15,
    max_depth: int = 10,
    max_size: int = 30,
) -> float:
    """다목적 복합 적합도. 높을수록 좋음.

    Parameters
    ----------
    ic_mean : IC 평균 (높을수록 좋음)
    icir : IC Information Ratio (높을수록 좋음)
    turnover : 시그널 턴오버 (낮을수록 좋음, 패널티)
    tree_depth : AST 깊이 (낮을수록 좋음, 패널티)
    tree_size : AST 노드 수 (낮을수록 좋음, 패널티)
    w_ic, w_icir, w_turnover, w_complexity : 가중치
    max_depth, max_size : 정규화 기준
    """
    depth_norm = tree_depth / max(max_depth, 1)
    size_norm = tree_size / max(max_size, 1)
    complexity_penalty = (depth_norm + size_norm) / 2.0

    return (
        ic_mean * w_ic
        + icir * w_icir
        - turnover * w_turnover
        - complexity_penalty * w_complexity
    )
