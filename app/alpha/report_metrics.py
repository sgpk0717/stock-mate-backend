"""알파 마이닝 리포트 메트릭 수집.

_build_report_data()에서 호출되어 피처 다양성, 커버리지, IC 트렌드 등을 계산한다.
"""

from __future__ import annotations

import logging
from collections import Counter

from app.alpha.ast_converter import FEATURE_FAMILY_MAP, classify_niche

logger = logging.getLogger(__name__)

# Phase 3에서 추가한 15개 파생 피처
_DERIVED_FEATURES = [
    "foreign_net_ema5", "foreign_net_ema20", "inst_net_ema5", "inst_net_ema20",
    "foreign_flow_accel", "smart_dumb_gap",
    "days_since_disclosure", "eps_fresh",
    "sentiment_ema3", "event_score_ema5",
    "short_asi_60", "margin_rate_roc10",
    "foreign_accum_10d", "foreign_accum_20d", "inst_accum_10d",
]


def compute_family_distribution(
    population: list,
) -> dict[str, float]:
    """모집단의 8개 피처 패밀리 비율 계산.

    Parameters
    ----------
    population : list[ScoredFactor]

    Returns
    -------
    dict[str, float] — 패밀리명 → 비율 (0~1)
    """
    if not population:
        return {}
    counts: Counter[str] = Counter()
    for f in population:
        niche = classify_niche(f.expression)
        counts[niche] += 1
    total = len(population)
    all_families = [
        "price", "volume", "momentum", "volatility",
        "supply", "fundamental", "sentiment", "market_micro",
    ]
    return {fam: round(counts.get(fam, 0) / total, 3) for fam in all_families}


def compute_family_delta(
    current: dict[str, float],
    previous: dict[str, float],
) -> dict[str, float]:
    """이전 세대 대비 패밀리 비율 변화."""
    if not previous:
        return {k: 0.0 for k in current}
    return {k: round(current.get(k, 0) - previous.get(k, 0), 3) for k in current}


def compute_derived_feature_usage(
    offspring: list,
) -> dict[str, int]:
    """신규 파생 피처 15개의 사용 횟수 집계.

    Parameters
    ----------
    offspring : list[ScoredFactor] — 이번 세대에서 생성된 자식 팩터
    """
    usage: Counter[str] = Counter()
    for f in offspring:
        try:
            symbols = {str(s) for s in f.expression.free_symbols}
        except Exception:
            continue
        for feat in _DERIVED_FEATURES:
            col_name = FEATURE_FAMILY_MAP.get(feat)
            if feat in symbols or (col_name and col_name in symbols):
                usage[feat] += 1
    # 미사용 피처도 0으로 포함
    return {feat: usage.get(feat, 0) for feat in _DERIVED_FEATURES}


def compute_coverage_health(
    population: list,
) -> dict:
    """커버리지 건강 지표 — Tier A/B/C 분류.

    Parameters
    ----------
    population : list[ScoredFactor] — coverage_pct 필드 필요
    """
    tier_a = [f for f in population if getattr(f, "coverage_pct", 1.0) >= 0.8]
    tier_b = [f for f in population if 0.5 <= getattr(f, "coverage_pct", 1.0) < 0.8]
    tier_c = [f for f in population if getattr(f, "coverage_pct", 1.0) < 0.5]

    def _avg(lst: list) -> float:
        if not lst:
            return 0.0
        return round(sum(getattr(f, "coverage_pct", 1.0) for f in lst) / len(lst), 3)

    return {
        "tier_a": {"count": len(tier_a), "avg_pct": _avg(tier_a)},
        "tier_b": {"count": len(tier_b), "avg_pct": _avg(tier_b)},
        "tier_c": {"count": len(tier_c), "avg_pct": _avg(tier_c)},
    }


async def compute_ic_trend(
    interval: str = "1d",
    limit: int = 10,
) -> list[dict]:
    """DB에서 최근 N세대의 평균IC/최고IC/ICIR 추이 조회."""
    from sqlalchemy import text

    from app.core.database import async_session

    query = text("""
        SELECT generation,
               ROUND(AVG(ic_mean)::numeric, 4) as avg_ic,
               ROUND(MAX(ic_mean)::numeric, 4) as best_ic,
               ROUND(AVG(CASE WHEN icir != 0 THEN icir END)::numeric, 3) as avg_icir,
               COUNT(*) as factor_count
        FROM alpha_factors
        WHERE interval = :interval AND ic_mean > 0
        GROUP BY generation
        ORDER BY generation DESC
        LIMIT :limit
    """)

    async with async_session() as session:
        result = await session.execute(query, {"interval": interval, "limit": limit})
        rows = result.fetchall()

    return [
        {
            "gen": row.generation,
            "avg_ic": float(row.avg_ic or 0),
            "best_ic": float(row.best_ic or 0),
            "avg_icir": float(row.avg_icir or 0),
            "factor_count": row.factor_count,
        }
        for row in reversed(rows)
    ]
