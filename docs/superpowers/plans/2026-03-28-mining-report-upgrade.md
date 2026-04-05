# Mining Report Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 마이닝 리포트를 피처 다양성, 커버리지 건강, IC 트렌드, 파생 피처 활용률을 포함하는 전문 대시보드로 업그레이드한다.

**Architecture:** 백엔드에 메트릭 수집 모듈(`report_metrics.py`)을 추가하고 scheduler의 `_build_report_data()`를 확장한다. Redis 캐시를 통해 `/alpha/mining-report` API를 제공하고, 프론트엔드에 Recharts(v3.8.0, 이미 설치됨) 기반 Mining Dashboard 탭(7개 차트)을 추가한다.

**Tech Stack:** Python/FastAPI/Polars/Redis (백엔드), React/TypeScript/Recharts/TanStack Query/shadcn (프론트엔드)

---

### Task 1: 메트릭 수집 모듈 생성

**Files:**
- Create: `app/alpha/report_metrics.py`

- [ ] **Step 1: report_metrics.py 생성**

```python
"""알파 마이닝 리포트 메트릭 수집.

_build_report_data()에서 호출되어 피처 다양성, 커버리지, IC 트렌드 등을 계산한다.
"""

from __future__ import annotations

import logging
from collections import Counter

import sympy

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
```

- [ ] **Step 2: 빌드 확인**

Run: `cd stock-mate-backend && docker-compose up -d --build --no-deps app worker`
Expected: 빌드 성공, import 에러 없음

- [ ] **Step 3: 커밋**

```bash
git add app/alpha/report_metrics.py
git commit -m "feat: add mining report metrics module"
```

---

### Task 2: scheduler.py — _build_report_data() 확장 + 폴백 리포트 개편

**Files:**
- Modify: `app/alpha/scheduler.py` — `_build_report_data()` (lines 698-764), `_build_fallback_report()` (lines 829-919), `_generate_llm_report()` (lines 766-827)

- [ ] **Step 1: _build_report_data()에 신규 메트릭 추가**

`_build_report_data()` 내부에서 `report_metrics` 함수들을 호출하여 반환 dict에 4개 블록을 추가한다.
evolution engine의 population과 offspring에 접근하기 위해 `self._engine` 참조를 사용한다.

scheduler.py 파일 상단 import에 추가:
```python
from app.alpha.report_metrics import (
    compute_family_distribution,
    compute_family_delta,
    compute_derived_feature_usage,
    compute_coverage_health,
    compute_ic_trend,
)
```

`_build_report_data()` 반환 dict의 마지막 항목(`generation_ic_trend`) 뒤에 추가:
```python
    # ── 신규 메트릭 (마이닝 리포트 고도화) ──
    population = self._engine._population if self._engine else []
    offspring = candidates_data.get("offspring", []) if candidates_data else []

    family_dist = compute_family_distribution(population)
    report["family_distribution"] = family_dist
    report["family_delta"] = compute_family_delta(
        family_dist,
        self._state.prev_family_distribution,
    )
    self._state.prev_family_distribution = family_dist

    report["derived_feature_usage"] = compute_derived_feature_usage(offspring)
    report["coverage_health"] = compute_coverage_health(population)
```

`_build_report_data()`를 호출하는 곳 바로 뒤(Redis 캐싱 + ic_trend 비동기 호출):
```python
    # IC 트렌드는 DB 조회이므로 report_data 구성 직후 비동기로
    report_data["ic_trend"] = await compute_ic_trend(
        interval=config.get("data_interval", "1d"), limit=10,
    )

    # Redis 캐싱 (프론트엔드 API용)
    try:
        interval = config.get("data_interval", "1d")
        import json
        await self._redis.set(
            f"alpha:mining_report:{interval}",
            json.dumps(report_data, default=str),
            ex=600,
        )
    except Exception:
        pass
```

`_state`에 `prev_family_distribution` 필드를 추가 (scheduler state dataclass):
```python
prev_family_distribution: dict[str, float] = field(default_factory=dict)
```

- [ ] **Step 2: _build_fallback_report() 개편**

기존 폴백 리포트를 제안한 구조로 전면 교체:

```python
def _build_fallback_report(self, data: dict) -> str:
    """텔레그램 폴백 리포트 (LLM 실패 시)."""
    gen = data.get("generation", "?")
    elapsed = data.get("elapsed", "?")
    discovered = data.get("discovered_factors", [])
    funnel = data.get("funnel", {})
    total = data.get("total_discovered", 0)

    # ── Executive Summary ──
    best_ic = max((f.get("ic_mean", 0) for f in discovered), default=0)
    n_found = len(discovered)
    if n_found > 0:
        emoji = "🔥" if best_ic >= 0.05 else "✅"
        summary = f"{emoji} <b>Gen {gen}</b>: {n_found}개 발견 (최고 IC {best_ic:.4f}) [{elapsed}]"
    else:
        emoji = "🔬"
        summary = f"{emoji} <b>Gen {gen}</b>: 탐색 중 [{elapsed}]"

    lines = [summary, ""]

    # ── 핵심 수치 ──
    attempted = funnel.get("attempted", 0)
    rate = (n_found / attempted * 100) if attempted > 0 else 0
    best_sharpe = max((f.get("sharpe", 0) for f in discovered), default=0)
    lines.append(
        f"📊 발견 {n_found}개 / 평가 {attempted}개 / "
        f"통과율 {rate:.1f}% / 총 {total}개"
    )
    lines.append("")

    # ── 상위 팩터 ──
    if discovered:
        lines.append("🏆 <b>상위 전략</b>")
        for i, f in enumerate(discovered[:3], 1):
            ic = f.get("ic_mean", 0)
            sh = f.get("sharpe", 0)
            expr = f.get("expression", "?")[:70]
            lines.append(f"  {i}. IC {ic:.4f} | Sharpe {sh:.2f}")
            lines.append(f"     <code>{expr}</code>")
        lines.append("")

    # ── 퍼널 ──
    eval_ok = funnel.get("eval_ok", 0)
    ic_pass = funnel.get("ic_pass", 0)
    cpcv = funnel.get("cpcv_candidates", 0)
    lines.append("🔬 <b>파이프라인</b>")
    lines.append(f"  {attempted} → {eval_ok} → {ic_pass} → {cpcv} → {n_found}")
    lines.append("")

    # ── 패밀리 분포 ──
    family_dist = data.get("family_distribution", {})
    family_delta = data.get("family_delta", {})
    if family_dist:
        lines.append("📊 <b>패밀리 분포</b>")
        for fam in sorted(family_dist, key=family_dist.get, reverse=True):
            pct = family_dist[fam] * 100
            bar_len = int(pct / 5)
            bar = "█" * bar_len + "░" * (10 - bar_len)
            delta = family_delta.get(fam, 0) * 100
            delta_str = f" ({delta:+.0f}pp)" if abs(delta) >= 1 else ""
            lines.append(f"  {fam:10s} {bar} {pct:4.0f}%{delta_str}")
        lines.append("")

    # ── 커버리지 ──
    cov = data.get("coverage_health", {})
    if cov:
        ta = cov.get("tier_a", {})
        tb = cov.get("tier_b", {})
        lines.append(
            f"📈 커버리지  A(>80%): {ta.get('count', 0)}개 | "
            f"B(50-80%): {tb.get('count', 0)}개"
        )
        lines.append("")

    # ── IC 트렌드 ──
    ic_trend = data.get("ic_trend", [])
    if ic_trend and len(ic_trend) >= 2:
        lines.append("📉 <b>IC 추이</b>")
        max_ic = max(t["avg_ic"] for t in ic_trend) or 0.01
        for t in ic_trend[-5:]:
            bar_len = int(t["avg_ic"] / max_ic * 8) if max_ic > 0 else 0
            bar = "■" * bar_len
            mark = " ★" if t["avg_ic"] == max_ic else ""
            lines.append(f"  Gen{t['gen']:>3d} ┤{bar:<8s} {t['avg_ic']:.4f}{mark}")
        lines.append("")

    # ── 연산자 Top 3 ──
    op_stats = data.get("operator_stats", {})
    if op_stats:
        sorted_ops = sorted(
            op_stats.items(),
            key=lambda x: x[1].get("avg_fitness_delta", 0),
            reverse=True,
        )[:3]
        lines.append("⚙️ <b>연산자 Top 3</b>")
        for op, stats in sorted_ops:
            calls = stats.get("calls", 0)
            delta = stats.get("avg_fitness_delta", 0)
            lines.append(f"  {op}: {calls}회 (avg {delta:+.4f})")

    return "\n".join(lines)
```

- [ ] **Step 3: Gemini 프롬프트 강화**

`_generate_llm_report()`의 system prompt에 신규 데이터 해석 지시를 추가한다.
기존 프롬프트 끝부분(섹션 7 이후)에 추가:

```
8. 피처 다양성 분석: family_distribution 비율과 delta를 해석. 특정 패밀리가 과도하거나 과소한 경우 원인 분석.
9. 커버리지 건강: coverage_health의 Tier 분포를 해석. 데이터 부족으로 탈락한 팩터가 많으면 데이터 수집 강화 권고.
10. IC 트렌드: ic_trend 시계열 추세를 해석. 정체/하락/상승 패턴 식별.
11. 신규 피처: derived_feature_usage 중 활발히 사용되는 피처와 미사용 피처를 언급.
```

- [ ] **Step 4: 빌드 + 검증**

Run: `cd stock-mate-backend && docker-compose up -d --build --no-deps app worker`
Expected: 빌드 성공

- [ ] **Step 5: 커밋**

```bash
git add app/alpha/scheduler.py
git commit -m "feat: upgrade mining report with family diversity, coverage health, IC trend"
```

---

### Task 3: API 엔드포인트 + 스키마

**Files:**
- Modify: `app/schemas/alpha.py` (append after line 326)
- Modify: `app/routers/alpha.py` (append after line 1367)

- [ ] **Step 1: MiningReportResponse 스키마 추가**

`app/schemas/alpha.py` 끝에 추가:

```python
# ── 마이닝 리포트 대시보드 ──

class FunnelData(BaseModel):
    attempted: int = 0
    eval_ok: int = 0
    ic_pass: int = 0
    wf_overfit: int = 0
    sharpe_fail: int = 0
    cpcv_candidates: int = 0

class CoverageTier(BaseModel):
    count: int = 0
    avg_pct: float = 0.0

class CoverageHealth(BaseModel):
    tier_a: CoverageTier = CoverageTier()
    tier_b: CoverageTier = CoverageTier()
    tier_c: CoverageTier = CoverageTier()

class IcTrendPoint(BaseModel):
    gen: int
    avg_ic: float
    best_ic: float
    avg_icir: float = 0.0
    factor_count: int = 0

class DiscoveredFactorSummary(BaseModel):
    expression: str
    ic_mean: float = 0.0
    icir: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    turnover: float = 0.0
    hypothesis: str | None = None

class MiningReportResponse(BaseModel):
    generation: int = 0
    cycle_num: int = 0
    elapsed: str = ""
    universe: str = "KOSPI200"
    data_interval: str = "1d"
    ic_threshold: float = 0.03
    total_discovered: int = 0
    funnel: FunnelData = FunnelData()
    operator_stats: dict[str, dict] = {}
    discovered_factors: list[DiscoveredFactorSummary] = []
    family_distribution: dict[str, float] = {}
    family_delta: dict[str, float] = {}
    derived_feature_usage: dict[str, int] = {}
    coverage_health: CoverageHealth = CoverageHealth()
    ic_trend: list[IcTrendPoint] = []
```

- [ ] **Step 2: GET /alpha/mining-report 엔드포인트 추가**

`app/routers/alpha.py` 끝에 추가:

```python
@router.get("/mining-report", response_model=MiningReportResponse)
async def get_mining_report(
    interval: str = "1d",
):
    """최신 마이닝 리포트 데이터 반환 (Redis 캐시)."""
    import json
    from app.core.redis import get_redis

    redis = await get_redis()
    cached = await redis.get(f"alpha:mining_report:{interval}")
    if cached:
        return json.loads(cached)
    return MiningReportResponse()
```

schemas import에 `MiningReportResponse` 추가.

- [ ] **Step 3: 빌드 + 검증**

Run: `cd stock-mate-backend && docker-compose up -d --build --no-deps app worker`
Run: `curl -s http://localhost:8007/alpha/mining-report | python -m json.tool | head -20`
Expected: 빈 기본값 JSON 또는 캐시된 리포트 데이터

- [ ] **Step 4: 커밋**

```bash
git add app/schemas/alpha.py app/routers/alpha.py
git commit -m "feat: add GET /alpha/mining-report API endpoint"
```

---

### Task 4: 프론트엔드 타입 + API + 훅

**Files:**
- Modify: `src/types/alpha.ts` (append)
- Modify: `src/api/alpha.ts` (append)
- Modify: `src/hooks/queries/use-alpha.ts` (append)

- [ ] **Step 1: MiningReport 타입 추가**

`src/types/alpha.ts` 끝에 추가:

```typescript
// ── Mining Report Dashboard ──

export interface FunnelData {
  attempted: number
  eval_ok: number
  ic_pass: number
  wf_overfit: number
  sharpe_fail: number
  cpcv_candidates: number
}

export interface CoverageTier {
  count: number
  avg_pct: number
}

export interface CoverageHealth {
  tier_a: CoverageTier
  tier_b: CoverageTier
  tier_c: CoverageTier
}

export interface IcTrendPoint {
  gen: number
  avg_ic: number
  best_ic: number
  avg_icir: number
  factor_count: number
}

export interface DiscoveredFactorSummary {
  expression: string
  ic_mean: number
  icir: number
  sharpe: number
  max_drawdown: number
  turnover: number
  hypothesis: string | null
}

export interface MiningReport {
  generation: number
  cycle_num: number
  elapsed: string
  universe: string
  data_interval: string
  ic_threshold: number
  total_discovered: number
  funnel: FunnelData
  operator_stats: Record<string, { calls: number; avg_fitness_delta: number }>
  discovered_factors: DiscoveredFactorSummary[]
  family_distribution: Record<string, number>
  family_delta: Record<string, number>
  derived_feature_usage: Record<string, number>
  coverage_health: CoverageHealth
  ic_trend: IcTrendPoint[]
}
```

- [ ] **Step 2: API 함수 추가**

`src/api/alpha.ts` 끝에 추가:

```typescript
export async function fetchMiningReport(interval = "1d"): Promise<MiningReport> {
  return apiFetch(`/alpha/mining-report?interval=${interval}`)
}
```

import에 `MiningReport` 타입 추가.

- [ ] **Step 3: Query 훅 추가**

`src/hooks/queries/use-alpha.ts` 끝에 추가:

```typescript
export function useMiningReport(interval = "1d") {
  return useQuery({
    queryKey: ["alpha", "mining-report", interval],
    queryFn: () => fetchMiningReport(interval),
    refetchInterval: 30_000,
    placeholderData: keepPreviousData,
  })
}
```

import에 `fetchMiningReport` 추가.

- [ ] **Step 4: 커밋**

```bash
cd stock-mate-frontend
git add src/types/alpha.ts src/api/alpha.ts src/hooks/queries/use-alpha.ts
git commit -m "feat: add mining report types, API, query hook"
```

---

### Task 5: 프론트엔드 차트 컴포넌트 (7개)

**Files:**
- Create: `src/components/alpha/mining/ExecutiveSummary.tsx`
- Create: `src/components/alpha/mining/NicheDonut.tsx`
- Create: `src/components/alpha/mining/IcTrendChart.tsx`
- Create: `src/components/alpha/mining/FunnelChart.tsx`
- Create: `src/components/alpha/mining/CoverageHistogram.tsx`
- Create: `src/components/alpha/mining/FeatureUsageGrid.tsx`
- Create: `src/components/alpha/mining/OperatorChart.tsx`

- [ ] **Step 1: ExecutiveSummary 컴포넌트**

```tsx
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import type { MiningReport } from "@/types/alpha"

export function ExecutiveSummary({ data }: { data: MiningReport }) {
  const { generation, elapsed, total_discovered, funnel, discovered_factors } = data
  const bestIc = Math.max(...discovered_factors.map((f) => f.ic_mean), 0)
  const bestSharpe = Math.max(...discovered_factors.map((f) => f.sharpe), 0)
  const rate = funnel.attempted > 0
    ? ((discovered_factors.length / funnel.attempted) * 100).toFixed(1)
    : "0"

  return (
    <Card>
      <CardContent className="p-4">
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-lg font-semibold">
            Gen {generation}
            <span className="ml-2 text-sm font-normal text-gray-500">{elapsed}</span>
          </h3>
          <Badge variant={discovered_factors.length > 0 ? "default" : "secondary"}>
            {discovered_factors.length > 0 ? `${discovered_factors.length}개 발견` : "탐색 중"}
          </Badge>
        </div>
        <div className="grid grid-cols-4 gap-3">
          <div className="text-center">
            <div className="text-2xl font-bold">{total_discovered}</div>
            <div className="text-xs text-gray-500">총 발견</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold">{bestIc.toFixed(4)}</div>
            <div className="text-xs text-gray-500">최고 IC</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold">{bestSharpe.toFixed(2)}</div>
            <div className="text-xs text-gray-500">최고 Sharpe</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold">{rate}%</div>
            <div className="text-xs text-gray-500">통과율</div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
```

- [ ] **Step 2: NicheDonut 컴포넌트**

```tsx
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from "recharts"

const COLORS = [
  "#4056F4", "#E3B23C", "#6B7280", "#3B82F6",
  "#10B981", "#F59E0B", "#EF4444", "#8B5CF6",
]

const FAMILY_LABELS: Record<string, string> = {
  price: "가격", volume: "거래량", momentum: "모멘텀",
  volatility: "변동성", supply: "수급", fundamental: "재무",
  sentiment: "감성", market_micro: "미시구조",
}

export function NicheDonut({ distribution, delta }: {
  distribution: Record<string, number>
  delta: Record<string, number>
}) {
  const chartData = Object.entries(distribution)
    .filter(([, v]) => v > 0)
    .map(([name, value]) => ({
      name: FAMILY_LABELS[name] || name,
      value: Math.round(value * 100),
      delta: Math.round((delta[name] || 0) * 100),
    }))
    .sort((a, b) => b.value - a.value)

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm">패밀리 분포</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={220}>
          <PieChart>
            <Pie
              data={chartData}
              cx="50%"
              cy="50%"
              innerRadius={50}
              outerRadius={80}
              dataKey="value"
              label={({ name, value }) => `${name} ${value}%`}
              labelLine={false}
            >
              {chartData.map((_, i) => (
                <Cell key={i} fill={COLORS[i % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip
              formatter={(value: number, name: string, props: any) => {
                const d = props.payload.delta
                const sign = d > 0 ? "+" : ""
                return [`${value}% (${sign}${d}pp)`, name]
              }}
            />
          </PieChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )
}
```

- [ ] **Step 3: IcTrendChart 컴포넌트**

```tsx
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts"
import type { IcTrendPoint } from "@/types/alpha"

export function IcTrendChart({ data }: { data: IcTrendPoint[] }) {
  if (!data.length) return null

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm">IC 추이</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis dataKey="gen" tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} domain={["auto", "auto"]} />
            <Tooltip
              formatter={(v: number, name: string) => [v.toFixed(4), name]}
            />
            <Line
              type="monotone"
              dataKey="avg_ic"
              name="평균 IC"
              stroke="#4056F4"
              strokeWidth={2}
              dot={{ r: 3 }}
            />
            <Line
              type="monotone"
              dataKey="best_ic"
              name="최고 IC"
              stroke="#E3B23C"
              strokeWidth={2}
              dot={{ r: 3 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )
}
```

- [ ] **Step 4: FunnelChart 컴포넌트**

```tsx
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts"
import type { FunnelData } from "@/types/alpha"

const FUNNEL_COLORS = ["#6B7280", "#3B82F6", "#4056F4", "#E3B23C", "#10B981"]

export function FunnelChart({ data, discovered }: { data: FunnelData; discovered: number }) {
  const chartData = [
    { name: "후보 생성", value: data.attempted },
    { name: "평가 완료", value: data.eval_ok },
    { name: "IC 통과", value: data.ic_pass },
    { name: "CPCV", value: data.cpcv_candidates },
    { name: "최종 합격", value: discovered },
  ]

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm">탐색 파이프라인</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={chartData} layout="vertical">
            <XAxis type="number" tick={{ fontSize: 11 }} />
            <YAxis type="category" dataKey="name" width={70} tick={{ fontSize: 11 }} />
            <Tooltip />
            <Bar dataKey="value" radius={[0, 4, 4, 0]}>
              {chartData.map((_, i) => (
                <Cell key={i} fill={FUNNEL_COLORS[i]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )
}
```

- [ ] **Step 5: CoverageHistogram 컴포넌트**

```tsx
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { CoverageHealth } from "@/types/alpha"

export function CoverageHistogram({ data }: { data: CoverageHealth }) {
  const tiers = [
    { label: "A (>80%)", ...data.tier_a, color: "#10B981" },
    { label: "B (50-80%)", ...data.tier_b, color: "#F59E0B" },
    { label: "C (<50%)", ...data.tier_c, color: "#EF4444" },
  ]
  const total = tiers.reduce((s, t) => s + t.count, 0) || 1

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm">커버리지 건강</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {tiers.map((tier) => (
          <div key={tier.label}>
            <div className="flex justify-between text-sm mb-1">
              <span>{tier.label}</span>
              <span className="font-medium">{tier.count}개 ({(tier.avg_pct * 100).toFixed(0)}%)</span>
            </div>
            <div className="w-full bg-gray-100 rounded-full h-2">
              <div
                className="h-2 rounded-full transition-all"
                style={{
                  width: `${(tier.count / total) * 100}%`,
                  backgroundColor: tier.color,
                }}
              />
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  )
}
```

- [ ] **Step 6: FeatureUsageGrid 컴포넌트**

```tsx
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

export function FeatureUsageGrid({ usage }: { usage: Record<string, number> }) {
  const sorted = Object.entries(usage).sort((a, b) => b[1] - a[1])
  const maxCount = Math.max(...sorted.map(([, v]) => v), 1)

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm">파생 피처 활용률</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-3 gap-2">
          {sorted.map(([name, count]) => {
            const intensity = count / maxCount
            const bg = count > 0
              ? `rgba(64, 86, 244, ${0.1 + intensity * 0.5})`
              : "#f9fafb"
            return (
              <div
                key={name}
                className="rounded px-2 py-1.5 text-xs"
                style={{ backgroundColor: bg }}
              >
                <div className="font-medium truncate">{name}</div>
                <div className="text-gray-600">{count}회</div>
              </div>
            )
          })}
        </div>
      </CardContent>
    </Card>
  )
}
```

- [ ] **Step 7: OperatorChart 컴포넌트**

```tsx
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts"

export function OperatorChart({ stats }: {
  stats: Record<string, { calls: number; avg_fitness_delta: number }>
}) {
  const chartData = Object.entries(stats)
    .map(([name, s]) => ({
      name: name.replace("ast_", "").replace("llm_", "LLM:"),
      calls: s.calls,
      delta: Number((s.avg_fitness_delta * 1000).toFixed(1)),
    }))
    .sort((a, b) => b.delta - a.delta)
    .slice(0, 7)

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm">연산자 효율</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={chartData} layout="vertical">
            <XAxis type="number" tick={{ fontSize: 11 }} />
            <YAxis type="category" dataKey="name" width={100} tick={{ fontSize: 10 }} />
            <Tooltip
              formatter={(v: number, name: string) => [
                name === "calls" ? `${v}회` : `${v}‰`,
                name === "calls" ? "호출" : "Δfitness",
              ]}
            />
            <Bar dataKey="delta" fill="#4056F4" name="Δfitness (‰)" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )
}
```

- [ ] **Step 8: 커밋**

```bash
cd stock-mate-frontend
git add src/components/alpha/mining/
git commit -m "feat: add 7 mining dashboard chart components"
```

---

### Task 6: MiningDashboard 레이아웃 + AlphaLabPage 탭 추가

**Files:**
- Create: `src/components/alpha/mining/MiningDashboard.tsx`
- Modify: `src/pages/AlphaLabPage.tsx`

- [ ] **Step 1: MiningDashboard 레이아웃**

```tsx
import { useMiningReport } from "@/hooks/queries/use-alpha"
import { ExecutiveSummary } from "./ExecutiveSummary"
import { NicheDonut } from "./NicheDonut"
import { IcTrendChart } from "./IcTrendChart"
import { FunnelChart } from "./FunnelChart"
import { CoverageHistogram } from "./CoverageHistogram"
import { FeatureUsageGrid } from "./FeatureUsageGrid"
import { OperatorChart } from "./OperatorChart"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

export function MiningDashboard() {
  const { data, isLoading } = useMiningReport()

  if (isLoading || !data) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-400">
        마이닝 리포트 로딩 중...
      </div>
    )
  }

  if (!data.generation) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-400">
        마이닝 데이터가 없습니다. Factory를 실행하면 여기에 대시보드가 표시됩니다.
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <ExecutiveSummary data={data} />

      <div className="grid grid-cols-2 gap-4">
        <NicheDonut
          distribution={data.family_distribution}
          delta={data.family_delta}
        />
        <IcTrendChart data={data.ic_trend} />
      </div>

      <div className="grid grid-cols-2 gap-4">
        <FunnelChart
          data={data.funnel}
          discovered={data.discovered_factors.length}
        />
        <CoverageHistogram data={data.coverage_health} />
      </div>

      <FeatureUsageGrid usage={data.derived_feature_usage} />

      <div className="grid grid-cols-2 gap-4">
        <OperatorChart stats={data.operator_stats} />
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">상위 팩터</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {data.discovered_factors.slice(0, 5).map((f, i) => (
              <div key={i} className="border rounded p-2">
                <div className="flex justify-between text-sm">
                  <span className="font-medium">IC {f.ic_mean.toFixed(4)}</span>
                  <span>Sharpe {f.sharpe.toFixed(2)}</span>
                </div>
                <code className="text-xs text-gray-600 block mt-1 truncate">
                  {f.expression}
                </code>
              </div>
            ))}
            {data.discovered_factors.length === 0 && (
              <div className="text-sm text-gray-400 text-center py-4">
                이번 세대에서 발견된 팩터 없음
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
```

- [ ] **Step 2: AlphaLabPage에 Dashboard 탭 추가**

AlphaLabPage.tsx의 TabsList에 4번째 탭 추가:

기존 3개 TabsTrigger 뒤에:
```tsx
<TabsTrigger value="dashboard">Dashboard</TabsTrigger>
```

기존 마지막 TabsContent(improvement) 뒤에:
```tsx
<TabsContent value="dashboard" className="p-4">
  <MiningDashboard />
</TabsContent>
```

import 추가:
```tsx
import { MiningDashboard } from "@/components/alpha/mining/MiningDashboard"
```

- [ ] **Step 3: 빌드 확인**

Run: `cd stock-mate-frontend && npm run build`
Expected: 에러 없이 빌드 성공

- [ ] **Step 4: 커밋**

```bash
cd stock-mate-frontend
git add src/components/alpha/mining/MiningDashboard.tsx src/pages/AlphaLabPage.tsx
git commit -m "feat: add Mining Dashboard tab to AlphaLab"
```

---

### Task 7: 통합 빌드 + 검증

**Files:** (전체)

- [ ] **Step 1: 백엔드 전체 재빌드**

Run: `cd stock-mate-backend && docker-compose up -d --build --no-deps app worker mcp`

- [ ] **Step 2: API 엔드포인트 확인**

Run: `curl -s http://localhost:8007/alpha/mining-report | python -m json.tool | head -30`
Expected: JSON 응답 (빈 기본값 또는 캐시된 데이터)

- [ ] **Step 3: 프론트엔드 빌드**

Run: `cd stock-mate-frontend && npm run build`
Expected: 에러 없이 빌드 성공

- [ ] **Step 4: 브라우저 확인**

AlphaLab → Dashboard 탭 확인. 마이닝 데이터가 없으면 빈 상태 메시지가 보여야 한다.
마이닝 1사이클 실행 후 차트가 채워지는지 확인.

- [ ] **Step 5: 최종 커밋**

```bash
cd stock-mate-backend && git add -A && git commit -m "feat: mining report upgrade - metrics, telegram, dashboard"
cd stock-mate-frontend && git add -A && git commit -m "feat: mining report dashboard with 7 charts"
```
