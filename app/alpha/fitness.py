"""다목적 복합 적합도 함수.

IC, ICIR, Long-only Sharpe, MDD, 포지션 턴오버, 복잡도를 가중 합산하여
단일 fitness 스코어를 산출한다.

Round 3 (2026-03-21): IC 정규화 추가 + ICIR 가중치 강화.
- 기존: IC 원시값(0~0.26) × 가중치 → Sharpe(0~1)에 비해 스케일 1/4 → IC 무력화
- 변경: IC를 [0, 1]로 정규화하여 다른 요소와 동등한 스케일 경쟁
"""

from __future__ import annotations

import math

# Turnover 하한: 이보다 낮으면 Buy-and-Hold (스캘핑 무용) → 즉시 탈락
_TURNOVER_FLOOR = 0.005  # 0.5% (일봉 가치팩터 턴오버 1~2% 허용, 딥리서치 권고)

# IC 정규화 기준: 이 값 이상이면 ic_norm=1.0 (포화)
# 5분봉 KOSPI200 현실적 상한 ~0.15 (0.12 이상이면 매우 우수)
_IC_NORM_CEIL = 0.15


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
    # ── Hard filter: Buy-and-Hold 퇴화 팩터 즉시 제거 ──
    # 스캘핑 목적상 최소 턴오버 미달 팩터는 무조건 탈락
    if turnover < _TURNOVER_FLOOR:
        return -1e6

    # ── IC 정규화: [0, _IC_NORM_CEIL] → [0, 1] ──
    # 원시 IC(0~0.26)는 Sharpe_norm(0~1) 대비 스케일 1/4이라
    # 같은 가중치를 줘도 fitness 기여가 미미했음. 정규화로 해소.
    ic_norm = min(1.0, max(0.0, ic_mean / _IC_NORM_CEIL))

    # Long-only Sharpe 정규화: [-0.5, 2.0] → [0, 1]
    # (Long-only Sharpe는 L/S보다 낮은 경향이므로 범위 축소)
    sharpe_norm = max(0.0, min(1.0, (sharpe + 0.5) / 2.5))

    # MDD 정규화: [0%, 50%] → [0, 1] (패널티)
    mdd_norm = min(1.0, abs(max_drawdown) / 0.50)

    # 복잡도 정규화
    depth_norm = tree_depth / max(max_depth, 1)
    size_norm = tree_size / max(max_size, 1)
    complexity_penalty = (depth_norm + size_norm) / 2.0

    # Turnover 로그 스케일: 선형 패널티 → 로그 패널티
    # 저turnover(0.05)와 고turnover(0.9) 차이를 완화하여 IC 높은 팩터가 불이익 안 받도록
    turnover_penalty = -math.log(max(turnover, 0.05))  # log(0.05)≈-3, log(1.0)=0

    return (
        ic_norm * w_ic
        + icir * w_icir
        + sharpe_norm * w_sharpe
        - mdd_norm * w_mdd
        + turnover_penalty * w_turnover  # 로그 스케일 (음수값 → 패널티)
        - complexity_penalty * w_complexity
    )
