# Phase 2: 인과 추론 파이프라인 — 보완 사항

> Phase 2 전문가 검증 후 도출된 개선 사항.
> CRITICAL 이슈는 즉시 수정 완료. 아래는 향후 개선할 HIGH/MEDIUM/LOW 항목.

## 수정 완료 (CRITICAL)

| # | 항목 | 수정 내용 |
|---|------|----------|
| B-C1 | p-value 추출 불안정 | `_extract_p_value()` 다중 경로 시도 + fallback |
| B-C2 | asyncpg 직접 연결 | SQLAlchemy 커넥션 풀 재사용으로 변경 |
| B-C3 | `proceed_when_unidentifiable` 무경고 | estimand 존재 여부 확인 + 경고 로그 |
| B-C4 | 교란변수 캐시 키 미검증 | 날짜+종목 복합 키로 캐시 유효성 검증 |
| B-C5 | sector_id 미로딩 | `load_sector_mapping()` 호출 + confounders에 매핑 |
| B-C6 | 최소 샘플 30 → 100 | 6변수 회귀에 적합한 100으로 상향 |
| B-C7 | DoWhy API 호환성 | `IdentifiedEstimand` 속성 존재 여부 `getattr` 체크 |
| F-C1 | 에러 표시 없음 | 에러 메시지 UI 표시 영역 추가 |
| F-C2 | 로딩 스피너 없음 | 스피너 + 예상 소요시간 안내 추가 |
| F-C3 | `.toFixed()` null crash | optional chaining (`?.toFixed()`) 적용 |
| F-C4 | 쿼리 invalidation 누락 | `["alpha-factor", factorId]` 키도 invalidate |
| F-C5 | 화살표 마커 색상 고정 | treatment/default 2개 마커 분리 |

---

## HIGH — 향후 구현 권장

### B-H1: 교란변수 확장 — 거래량/환율/VIX

현재 교란변수: market_return, market_volatility, base_rate, sector_id (4개).
추가 고려:
- 거래량 비율 (market_volume_ratio)
- 원/달러 환율 변동률
- VIX (글로벌 변동성)
- 외국인/기관 순매수 비율

**영향 범위:** `confounders.py`, `causal.py` (DAG 확장), 프론트엔드 DAG 노드 추가

### B-H2: 다중 추정 방법 교차 검증

현재 `backdoor.linear_regression`만 사용.
Propensity Score Matching, IV Regression 등 2개 이상 추정 방법으로 교차 검증하면 robust 판정의 신뢰성 향상.

```python
methods = ["backdoor.linear_regression", "backdoor.propensity_score_matching"]
estimates = [model.estimate_effect(identified, method_name=m) for m in methods]
# 다수결 또는 일관성 기준으로 최종 판정
```

### B-H3: 시계열 교차 검증 (Rolling Window)

현재 전체 기간을 한 번에 분석. 시계열 데이터에서는 Rolling Window로 인과 관계의 시간 안정성 검증 필요.

- 1년 윈도우, 1개월 롤링
- 각 윈도우별 ATE 일관성 확인
- 구조 변화(structural break) 감지

### B-H4: ATE 정규화

현재 ATE 값이 절대값으로 해석하기 어려움.
Cohen's d 또는 표준화 효과 크기로 변환하면 팩터 간 비교가 용이.

### B-H5: 반증 테스트 확장

현재 2가지 반증 (Placebo + Random Common Cause).
추가 반증:
- **Data Subset**: 데이터의 일부만으로 검증 → 안정성
- **Add Unobserved Common Cause**: 관측되지 않은 교란 변수 시뮬레이션
- **Bootstrapping**: 부트스트랩 신뢰구간

### F-H1: 인과 검증 결과 캐싱

검증 결과를 서버에서 DB에 저장하지만, 프론트에서 상세 결과(placebo_passed 등)는 별도 API 호출 없이 AlphaFactor 응답에 포함되면 UX 개선.

### F-H2: DAG 인터랙티브 시각화

현재 정적 SVG. 노드 클릭 시 해당 교란변수의 상세 통계, 엣지 hover 시 경로 효과 크기 표시 등.

### F-H3: 검증 히스토리

동일 팩터의 검증 히스토리 (날짜별 ATE 추이) 차트. 시간에 따른 인과 효과 안정성 시각화.

---

## MEDIUM — 코드 품질 개선

### B-M1: DoWhy 버전 핀닝

DoWhy API가 버전 간 차이가 큼 (0.8 vs 0.11+). `requirements.txt`에 정확한 버전 범위 지정 권장.

```
dowhy>=0.11.0,<0.13.0
```

### B-M2: 교란변수 데이터 품질 메트릭

`load_confounders()` 반환 시 데이터 품질 보고:
- 결측률 (%)
- 날짜 커버리지
- 이상치 비율

### B-M3: 로깅 구조화

현재 `logger.info/warning`에 문자열 포맷.
structured logging (JSON) + 검증 ID 추적을 도입하면 운영 시 디버깅 용이.

### B-M4: confounders_df 재사용 최적화

배치 검증 시 confounders_df를 팩터 데이터와 매번 merge. 팩터별 날짜/종목이 동일하면 한 번만 merge하고 재사용.

### F-M1: StatusBadge 한국어화

현재 `discovered`, `validated`, `mirage` 영어 표시. 한국어로:
- discovered → "발견"
- validated → "검증됨"
- mirage → "미라지"
- deployed → "배포됨"

### F-M2: DAG 노드 반응형 폰트

작은 화면에서 SVG 텍스트가 잘 안 보일 수 있음. `fontSize`를 viewBox 비례로 조정.

### F-M3: 인과 검증 버튼 비활성 조건

`status === "mirage"`인 팩터는 재검증이 무의미할 수 있음. 상태에 따른 버튼 텍스트/비활성 처리.

---

## LOW — 향후 고려

### B-L1: BOK 기준금리 자동 업데이트

현재 `data/bok_base_rate.json` 수동 관리. 한국은행 API 연동으로 자동 업데이트 고려.

### B-L2: 벤치마크 수익률 다양화

현재 KOSPI 200 ETF(069500)만 사용. KOSDAQ, 업종별 ETF 등 다양한 벤치마크 지원.

### B-L3: 인과 검증 타임아웃

DoWhy 실행이 비정상적으로 오래 걸릴 경우 타임아웃 (기본 5분) 설정.

### F-L1: DAG 엣지 애니메이션

검증 진행 중 DAG 엣지에 흐르는 점선 애니메이션 → 시각적 피드백.

### F-L2: 검증 결과 PDF 내보내기

검증 결과(DAG + 통계)를 PDF로 내보내기. 리포트 용도.

### F-L3: 키보드 접근성

CausalBadge, DAG 노드에 `tabIndex`, `onKeyDown` 추가. 스크린 리더 지원.

---

## 수정 파일 목록

| 파일 | 수정 사항 |
|------|----------|
| `app/alpha/causal.py` | `_extract_p_value()` 추가, `_MIN_SAMPLES=100`, estimand 경고 |
| `app/alpha/confounders.py` | SQLAlchemy 세션 풀 사용, numpy import 제거 |
| `app/alpha/causal_runner.py` | sector_id 로딩, 캐시 키 검증, confounders 정렬 |
| `tests/alpha/test_causal.py` | C09 테스트 샘플 300으로 상향 |
| `src/components/alpha/AlphaFactorDetail.tsx` | 에러 표시, 로딩 스피너, null safety |
| `src/components/alpha/CausalDAGView.tsx` | 반응형 SVG, 화살표 마커 분리 |
| `src/hooks/queries/use-alpha.ts` | 개별 팩터 쿼리 invalidation 추가 |
