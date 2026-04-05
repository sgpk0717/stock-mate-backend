# Mining Report Upgrade Design

## Context

알파 마이닝 데이터 활용 최적화(Phase 1-4) 완료 후, 리포트 시스템도 이에 맞게 고도화한다.
현재 텔레그램 리포트는 퍼널 통계와 상위 팩터 3개만 보여주며, 프론트엔드 AlphaLab은 진행률 바와 팩터 테이블만 제공한다.

새로 추가된 coverage_pct, stratified sampling, 15개 파생 피처의 효과를 추적하고, 마이닝 품질을 다각도로 시각화하는 전문 리포트로 업그레이드한다.

## 변경 범위

1. **백엔드**: 메트릭 수집 + API 엔드포인트
2. **텔레그램**: 리포트 구조 확장 (폴백 + Gemini 프롬프트)
3. **프론트엔드**: Mining Dashboard 탭 + 7개 차트 컴포넌트

---

## 1. 백엔드 메트릭 수집

### 1.1 `_build_report_data()` 확장 (scheduler.py)

기존 반환 dict에 4개 블록 추가:

```python
report_data = {
    # ... 기존 필드 유지 ...

    # [신규] 피처 패밀리 다양성
    "family_distribution": {
        "price": 0.22, "volume": 0.12, "momentum": 0.15,
        "volatility": 0.11, "supply": 0.18, "fundamental": 0.12,
        "sentiment": 0.05, "market_micro": 0.05,
    },
    "family_delta": {  # 이전 세대 대비 변화
        "price": -0.08, "supply": +0.05, ...
    },

    # [신규] 신규 파생 피처 활용률
    "derived_feature_usage": {
        "foreign_flow_accel": 12,
        "eps_fresh": 8,
        "smart_dumb_gap": 6,
        ...
    },

    # [신규] 커버리지 건강
    "coverage_health": {
        "tier_a": {"count": 42, "avg_pct": 0.95},  # >80%
        "tier_b": {"count": 8, "avg_pct": 0.62},   # 50-80%
        "tier_c": {"count": 0, "avg_pct": 0.0},    # <50%
        "rejected_low_coverage": 5,
    },

    # [신규] IC 트렌드 (최근 10세대)
    "ic_trend": [
        {"gen": 11, "avg_ic": 0.042, "best_ic": 0.065, "icir": 0.35},
        {"gen": 12, "avg_ic": 0.048, "best_ic": 0.071, "icir": 0.38},
        ...
    ],
}
```

### 1.2 메트릭 수집 헬퍼

`app/alpha/report_metrics.py` (신규 파일):

| 함수 | 입력 | 출력 |
|------|------|------|
| `compute_family_distribution(population)` | list[ScoredFactor] | dict[str, float] |
| `compute_family_delta(current, previous)` | 2x dict | dict[str, float] |
| `compute_derived_feature_usage(offspring)` | list[ScoredFactor] | dict[str, int] |
| `compute_coverage_health(population)` | list[ScoredFactor] | dict (tier_a/b/c) |
| `compute_ic_trend(db, interval, limit=10)` | async Session | list[dict] |

`classify_niche()`와 `FEATURE_FAMILY_MAP` 재사용.

### 1.3 API 엔드포인트

`GET /alpha/mining-report` (routers/alpha.py):
- 최신 세대의 report_data를 Redis 캐시에서 반환
- 캐시 키: `alpha:mining_report:{interval}`
- TTL: 600초 (10분)
- 스키마: `MiningReportResponse` (Pydantic)

`_build_report_data()` 완료 시 Redis에 캐싱:
```python
await redis.set("alpha:mining_report:1d", json.dumps(report_data), ex=600)
```

---

## 2. 텔레그램 리포트 구조

### 2.1 폴백 리포트 구조 (`_build_fallback_report()`)

```
🧬 Gen {N}: {1줄 요약}

📊 핵심 수치
  발견 {n}개 / 평가 {m}개 / 통과율 {r}% / 최고 Sharpe {s}

🏆 상위 전략
  1. IC {ic} | Sharpe {sh} | cov {cov}%
     {expression_str[:80]}
  2. ...
  3. ...

🔬 탐색 파이프라인
  {attempted} → {eval_ok} → {ic_pass} → {cpcv} → {discovered}
  통과율: {rate}% (이전 {prev_rate}% → {delta})

📊 패밀리 분포
  price     {bar} {pct}% ({delta})
  supply    {bar} {pct}% ({delta})
  momentum  {bar} {pct}%
  fundament {bar} {pct}% ({delta})
  sentiment {bar} {pct}% ({delta})
  volatil   {bar} {pct}%
  volume    {bar} {pct}%
  mkt_micro {bar} {pct}% ({delta})

📈 커버리지
  A(>80%): {n}개 | B(50-80%): {n}개 | 탈락: {n}건

📉 IC 추이
  {gen} ┤{bar} {ic}
  {gen} ┤{bar} {ic} ★
  ...

⚙️ 연산자 Top 3
  {op}: {calls}회 (avg +{delta_fit})
  ...
```

### 2.2 Gemini 프롬프트 강화

`_generate_llm_report()`의 system prompt에 신규 데이터 설명 추가:
- family_distribution, family_delta 해석 지시
- coverage_health 분석 지시
- ic_trend 추세 분석 지시
- derived_feature_usage 중 미사용 피처 언급 지시

### 2.3 Executive Summary 생성

1줄 요약 로직 (폴백):
- 신기록 IC → "IC {ic} 신기록 달성"
- 신규 패밀리 등장 → "{family}에서 첫 발견"
- 발견 0개 → "탐색 중 (IC 최고 {ic})"

---

## 3. 프론트엔드 Mining Dashboard

### 3.1 페이지 구조

AlphaLabPage에 탭 추가: `Factors` | `Mining` | `Dashboard`

`Dashboard` 탭 = `MiningDashboard` 컴포넌트:
```
┌─────────────────────────────────────────────┐
│  Executive Summary Card (1줄 요약 + 핵심 수치) │
├──────────────────┬──────────────────────────┤
│  NicheDonut      │  IcTrendChart            │
│  (패밀리 분포)    │  (세대별 IC 추이)         │
├──────────────────┼──────────────────────────┤
│  FunnelChart     │  CoverageHistogram       │
│  (탐색 파이프라인) │  (커버리지 분포)          │
├──────────────────┴──────────────────────────┤
│  FeatureUsageHeatmap (피처 × 세대 사용빈도)   │
├──────────────────┬──────────────────────────┤
│  OperatorChart   │  TopFactorsCard          │
│  (연산자 효율)    │  (상위 5개 팩터)          │
└──────────────────┴──────────────────────────┘
```

### 3.2 컴포넌트 목록

| 컴포넌트 | 차트 | 라이브러리 | 데이터 |
|---------|------|----------|--------|
| `MiningDashboard` | 레이아웃 | — | `useMiningReport()` |
| `ExecutiveSummary` | 텍스트 카드 | shadcn Card | 요약 + 4개 수치 배지 |
| `NicheDonut` | 도넛 | Recharts PieChart | family_distribution |
| `IcTrendChart` | 라인 | Recharts LineChart | ic_trend[] |
| `FunnelChart` | 가로 바 | Recharts BarChart | funnel |
| `CoverageHistogram` | 히스토그램 | Recharts BarChart | coverage_health |
| `FeatureUsageHeatmap` | 격자 | div + Tailwind | derived_feature_usage |
| `OperatorChart` | 가로 바 | Recharts BarChart | operator_stats |
| `TopFactorsCard` | 카드 리스트 | shadcn Card | discovered_factors |

### 3.3 의존성

- `recharts` 추가 (`npm install recharts`)
- TanStack Query 훅: `useMiningReport()` (30초 폴링)
- API: `GET /alpha/mining-report`
- 타입: `types/mining-report.ts`

### 3.4 색상

프로젝트 디자인 시스템 준수:
- Primary `#4056F4` — 주요 차트 색상
- Secondary `#E3B23C` — 강조/하이라이트
- Grayscale — 배경, 보조 요소
- 패밀리별 색상: 8가지 grayscale 톤 + primary/secondary 강조

---

## 수정 파일 목록

### 백엔드 (4개)
| 파일 | 변경 |
|------|------|
| `app/alpha/report_metrics.py` | 신규: 5개 메트릭 수집 함수 |
| `app/alpha/scheduler.py` | `_build_report_data()` 확장 + Redis 캐싱 + 폴백 리포트 개편 + Gemini 프롬프트 강화 |
| `app/routers/alpha.py` | `GET /alpha/mining-report` 엔드포인트 + 스키마 |
| `app/schemas/alpha.py` | `MiningReportResponse` Pydantic 모델 |

### 프론트엔드 (11개)
| 파일 | 변경 |
|------|------|
| `package.json` | recharts 의존성 추가 |
| `src/types/mining-report.ts` | 신규: MiningReport 타입 |
| `src/api/alpha.ts` | fetchMiningReport() 추가 |
| `src/hooks/queries/use-alpha.ts` | useMiningReport() 훅 |
| `src/pages/AlphaLabPage.tsx` | Dashboard 탭 추가 |
| `src/components/alpha/MiningDashboard.tsx` | 신규: 대시보드 레이아웃 |
| `src/components/alpha/ExecutiveSummary.tsx` | 신규 |
| `src/components/alpha/NicheDonut.tsx` | 신규 |
| `src/components/alpha/IcTrendChart.tsx` | 신규 |
| `src/components/alpha/FunnelChart.tsx` | 신규 |
| `src/components/alpha/CoverageHistogram.tsx` | 신규 |
