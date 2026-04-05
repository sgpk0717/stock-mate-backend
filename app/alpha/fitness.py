"""다목적 복합 적합도 함수.

IC, ICIR, Long-only Sharpe, MDD, 포지션 턴오버, 복잡도를 가중 합산하여
단일 fitness 스코어를 산출한다.

Round 3 (2026-03-21): IC 정규화 추가 + ICIR 가중치 강화.
- 기존: IC 원시값(0~0.26) × 가중치 → Sharpe(0~1)에 비해 스케일 1/4 → IC 무력화
- 변경: IC를 [0, 1]로 정규화하여 다른 요소와 동등한 스케일 경쟁
"""

from __future__ import annotations

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
    coverage_pct: float = 1.0,
    coverage_exp: float = 0.4,
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
    coverage_pct : IC 유효 비율 (0~1, 1.0=전체 기간 데이터 존재)
    coverage_exp : 커버리지 패널티 지수 (0.4=딥리서치 권고)
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

    # [2026-03-31] 딥리서치 R1+R2 공통 권장 — 턴오버 패널티 선형화
    # 프로세스: /deep-research → 2건 보고서 교차 분석 → 공통 Tier 1 권장
    # 변경: -log(max(T, 0.05)) → -max(0, T-0.005)*15 | 판단: log 패널티는 저턴오버(buy-and-hold) 팩터를 과도하게 선호. 15bp = 한국 시장 round-trip 비용
    turnover_penalty = -max(0, turnover - 0.005) * 15

    raw_fitness = (
        ic_norm * w_ic
        + icir * w_icir
        + sharpe_norm * w_sharpe
        - mdd_norm * w_mdd
        + turnover_penalty * w_turnover  # 선형 스케일 (음수값 → 패널티)
        - complexity_penalty * w_complexity
    )

    # 커버리지 패널티: 데이터가 부분적으로만 존재하는 팩터의 적합도 하향
    # coverage=1.0 → 1.0 (무패널티), 0.5 → 0.76, 0.01 → 0.16
    if coverage_pct < 1.0 and coverage_exp > 0:
        # [2026-03-31] 딥리서치 R2 권장 — 커버리지 floor 0.7 적용
        # 프로세스: /deep-research → 2건 보고서 교차 분석 → 공통 Tier 1 권장
        # 변경: max(0.01, pct) → max(0.7, pct) | 판단: 데이터 50% 미만이어도 floor 0.7로 패널티 상한 제한. 뉴스/프로그램 팩터 차단 해소
        coverage_factor = max(0.7, coverage_pct) ** coverage_exp
        return raw_fitness * coverage_factor

    return raw_fitness
