# Alpha Mining System — 개선 히스토리

이 문서는 알파 팩터 마이닝 프로세스의 개선 이력을 추적합니다.
`/improve-mining` 스킬 실행 시 자동으로 참조 및 업데이트됩니다.

---

## 현재 상태 요약 (2026-03-19 기준)

| 지표 | 값 | 비고 |
|------|-----|------|
| 총 팩터 | 3,073개 | 251세대 걸쳐 발견 |
| avg IC | 0.082 | 주가 예측력 평균 |
| max IC | 0.262 | |
| avg Sharpe | 1.36 | |
| max Sharpe | 4.38 | |
| avg ICIR | 0.53 | 예측 안정성 (1.0 이상이 실전 유의미) |
| max ICIR | 0.67 | |
| 인과 검증 통과 | 874/3,073 (28.4%) | |
| 수렴도 | 극심 | 전체 팩터의 83%가 volume 패밀리, Top 30 100% volume 포함 |
| 중복 수식 | 상위 1개 수식 28회 중복 | |

### 핵심 문제

1. **모집단 수렴**: 거의 모든 팩터가 `(high + log(zscore_volume)) / atr` 변형
2. **탐색 공간 포화**: gen 200 이후 max IC 정체 (~0.115)
3. **ICIR 전반적 저조**: avg 0.53 — 시간대별 불안정
4. **외부 데이터 미활용**: 수급/감성/펀더멘털 피처 사용률 0.3%
5. **인과 검증도 단일 가족**: 다양성 부재로 검증 통과도 같은 패턴에서만

---

## 개선 라운드

### Round 1: Niche-Based Population Diversity Cap
- **날짜**: 2026-03-19
- **문제 진단**: 모집단 크기 제한(500) 시 fitness 순 정렬 → volume 가족만 생존(83%) → 다른 피처 조합 탐색 불가 (Population Takeover)
- **개선 내용**:
  - 8개 피처 패밀리 분류 체계 도입 (price, volume, momentum, volatility, supply, fundamental, sentiment, market_micro)
  - 모집단 트리밍에 니치 캡 적용: 패밀리당 최대 25% 점유율 (ALPHA_NICHE_MAX_PCT=0.25)
  - 2-Pass 알고리즘: Pass 1에서 니치별 캡 적용, Pass 2에서 미달분 보충
- **수정 파일**:
  - `app/alpha/ast_converter.py`: `FEATURE_FAMILY_MAP`, `classify_niche()` 추가
  - `app/alpha/evolution_engine.py`: `_niche_cap_trim()` 메서드 추가 + 기존 트리밍 교체 (line 538-540)
  - `app/core/config.py`: `ALPHA_NICHE_MAX_PCT=0.25` 추가
- **기대 효과**:
  - volume 패밀리 비중 83% → ≤25%
  - 다양한 패밀리 팩터 생존 → 진화 기회 확보
  - 외부 데이터 피처 활용 팩터 등장
- **검증 기준** (5사이클 후):
  - [필수] volume 패밀리 비중 ≤30%
  - [필수] 모집단 내 5개+ 멤버 패밀리 ≥4개
  - [권장] 비volume discovered 팩터 ≥1개
  - [안전] 최고 IC 하락 ≤20%
  - [관찰] 최고 ICIR 유지 또는 개선
- **검증 결과**: PENDING
  - 데이터: (다음 `/improve-mining` 실행 시 측정)
- **교훈**: (검증 후 기록)
