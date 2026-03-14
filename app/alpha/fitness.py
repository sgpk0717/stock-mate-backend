"""다목적 복합 적합도 함수.

IC, ICIR, Long-only Sharpe, MDD, 포지션 턴오버, 복잡도를 가중 합산하여
단일 fitness 스코어를 산출한다.
"""

from __future__ import annotations


def compute_composite_fitness(
    ic_mean: float,
    icir: float,
    turnover: float,
    tree_depth: int,
    tree_size: int,
    sharpe: float = 0.0,
    max_drawdown: float = 0.0,
    w_ic: float = 0.25,
    w_icir: float = 0.15,
    w_sharpe: float = 0.25,
    w_mdd: float = 0.05,
    w_turnover: float = 0.20,
    w_complexity: float = 0.10,
    max_depth: int = 10,
    max_size: int = 30,
) -> float:
    """다목적 복합 적합도. 높을수록 좋음.

    Parameters
    ----------
    ic_mean : IC 평균 (높을수록 좋음)
    icir : IC Information Ratio (높을수록 좋음)
    turnover : 포지션 턴오버 (0~1, 낮을수록 좋음, 패널티)
    tree_depth : AST 깊이 (낮을수록 좋음, 패널티)
    tree_size : AST 노드 수 (낮을수록 좋음, 패널티)
    sharpe : Long-only Sharpe Ratio (높을수록 좋음)
    max_drawdown : 최대 낙폭 (음수, 낮을수록 나쁨, 패널티)
    w_ic, w_icir, w_sharpe, w_mdd, w_turnover, w_complexity : 가중치
    max_depth, max_size : 정규화 기준
    """
    # Long-only Sharpe 정규화: [-0.5, 2.0] → [0, 1]
    # (Long-only Sharpe는 L/S보다 낮은 경향이므로 범위 축소)
    sharpe_norm = max(0.0, min(1.0, (sharpe + 0.5) / 2.5))

    # MDD 정규화: [0%, 50%] → [0, 1] (패널티)
    mdd_norm = min(1.0, abs(max_drawdown) / 0.50)

    # 복잡도 정규화
    depth_norm = tree_depth / max(max_depth, 1)
    size_norm = tree_size / max(max_size, 1)
    complexity_penalty = (depth_norm + size_norm) / 2.0

    return (
        ic_mean * w_ic
        + icir * w_icir
        + sharpe_norm * w_sharpe
        - mdd_norm * w_mdd
        - turnover * w_turnover
        - complexity_penalty * w_complexity
    )
