# Alpha Mining System — 개선 히스토리

이 문서는 알파 팩터 마이닝 프로세스의 개선 이력을 추적합니다.
`/improve-mining` 스킬 실행 시 자동으로 참조 및 업데이트됩니다.

---

## 현재 상태 요약 (2026-03-21 00:40 KST 기준)

| 지표 | 값 | 비고 |
|------|-----|------|
| 총 팩터 | 7,430개 | 348세대 걸쳐 생성 |
| avg IC | 0.091 | 주가 예측력 평균 |
| max IC | 0.262 | 명목. 실질 IC 천장 ~0.116 (ICIR>0 기준) |
| avg Sharpe | 1.35 | |
| max Sharpe | 4.38 | |
| avg ICIR | 0.55 | 예측 안정성 (1.0 이상이 실전 유의미) |
| max ICIR | 0.69 | ICIR ≥ 0.7 팩터 0개 |
| 활성 모집단 volume 비중 | ~90% | gen 340+ 기준. Round 1~2 다양성 개선 한계 |
| 활성 모집단 크기 | 149개 | |

### 핵심 상황

1. **ICIR 천장 0.69**: 7,430개 팩터 중 ICIR ≥ 0.7 = 0개. 실전 유의미(1.0+) 도달 불가 → **Round 3에서 fitness ICIR 중심 재편 적용**
2. **fitness-Sharpe 지배 구조**: Sharpe가 fitness의 70% 결정. IC 기여 5%. 스케일 불일치 원인 → **Round 3에서 IC 정규화로 해소**
3. **다양성 한계 확인 완료**: Round 1~2로 메커니즘은 동작하나, 5분봉 KOSPI200에서 volume IC 우위가 구조적. 추가 다양성 개선보다 환경 변경이 필요
4. **IC 천장 유지**: gen 297~348에서도 max IC ~0.117 수준 유지. 돌파 없음

---

## 개선 라운드

### Round 1: Niche-Based Population Diversity Cap
- **시각**: 2026-03-19 03:50 KST
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
- **검증 결과**: 부분 성공 (30사이클 경과, gen 253~284)
  - volume 패밀리 비중: 83% → 64.5% (활성 모집단) — 목표 30% 미달
  - 패밀리 수 (5개+ 멤버): 4개 (volume 40, price 9, momentum 6, fundamental 3) — 목표 달성 (경계)
  - 비volume discovered: 다수 발견 (fundamental Sharpe 3.07, supply Sharpe 1.88) — 달성
  - 최고 IC: 유지 (0.217) — 달성
  - 최고 ICIR: 0.67 → 0.69 개선 — 달성
- **교훈**: 니치캡은 "다양한 팩터가 살 공간"을 마련하는 데 성공했지만, 비volume 팩터의 "공급" 자체가 부족하여 빈 슬롯을 volume overflow로 채우게 됨. 다음 라운드에서 비volume 팩터 생성량을 늘려야 함.

### Round 2: 수렴 감지 시 비volume 다양성 시드 강제 주입
- **시각**: 2026-03-19 09:30 KST
- **문제 진단**: Round 1의 니치캡이 공간을 만들었지만, AST 변이 1개 심볼 교체로는 volume 니치 탈출 불가 (핵심 zscore_volume 잔존). LLM 시드도 RAG 영향으로 volume 경향.
- **개선 내용**:
  - 수렴 감지 조건 확장: 기존(피처 top10 중 8개+) + 니치 60% 이상
  - 수렴 감지 시 비volume 하드코딩 시드 15개를 offspring에 강제 주입
  - OHLCV + 기술적 지표 기반만 사용 (외부 데이터 피처 제외 — 미존재 위험 방지)
  - 4개 패밀리 (price, momentum, volatility, mixed_no_volume) × 3~5개 시드 템플릿
- **수정 파일**:
  - `app/alpha/evolution_engine.py`: 수렴 감지 확장 + `_make_diversity_seeds()` 추가
- **기대 효과**:
  - volume 패밀리 비중 64.5% → ≤40%
  - 비volume discovered 팩터 증가
- **검증 기준** (5사이클 후):
  - [필수] volume 패밀리 비중 ≤40%
  - [필수] 비volume discovered 팩터 ≥3개
  - [안전] 최고 IC 하락 ≤20%
- **검증 결과**: 부분 실패 (12세대 경과, gen 285~296)
  - volume 패밀리 비중: 64.5% → 62.5% (활성 모집단) — 목표 40% **미달**
  - 전체 생성 팩터 중 volume: 94.6% (313/331) — 진화 과정에서 volume 압도적 재생산
  - 비volume IC ≥ 0.03 팩터: 3개 (price 니치, max IC 0.051) — **경계 달성**
  - 최고 IC: 0.116 (하락 없음) — **달성**
  - 니치캡 트리밍 동작 확인: 트리밍 직후 volume 32~39%, 이후 비volume 도태로 복원
  - 다양성 시드 16개 생존 (IC > 0이나 volume 대비 2~30배 낮은 IC)
- **교훈**: 니치캡(Round 1)과 다양성 시드(Round 2)는 **메커니즘으로서 정상 동작**한다. 근본 한계는 코드가 아닌 데이터 특성: 5분봉 KOSPI200에서 volume 기반 피처의 IC가 다른 피처 대비 3~30배 높아 어떤 보호/주입 메커니즘으로도 fitness 경쟁에서 비volume이 자연도태됨. 외부 데이터(수급/재무)는 일별 빈도라서 5분봉 내 변별력이 구조적으로 제한됨. 같은 방향(다양성)의 추가 개선보다 환경 변경(인터벌/유니버스)이 더 효과적일 가능성 높음.
- **교훈 요약**: 코드 수준의 다양성 개선(Round 1~2)은 메커니즘으로서 정상이나, 5분봉 KOSPI200에서 volume의 IC 우위가 압도적이라 fitness 경쟁에서 비volume이 자연도태됨.

### Round 3: Fitness 함수 ICIR 중심 재편 — IC 정규화 + 가중치 재조정
- **시각**: 2026-03-21 00:40 KST
- **문제 진단**: Fitness-Sharpe 상관 0.70 (지배적), Fitness-IC 상관 0.053 (무력화). IC 원시값(0~0.26)이 Sharpe_norm(0~1)의 1/4 스케일이라, IC 가중치 30%에도 불구하고 실질 기여가 7%에 불과. ICIR은 전체 7,430개 팩터 중 0.7을 초과하는 것이 0개로, 실전 유의미(1.0+) 수준 도달 불가. 시스템이 "예측 안정성"이 아닌 "절대 수익률"을 최적화 중.
- **개선 내용**:
  - IC 정규화 추가: `ic_norm = min(1.0, ic_mean / 0.15)` → IC를 [0, 1] 범위로 변환하여 Sharpe/ICIR과 동등한 스케일 경쟁
  - ICIR 가중치 증가: 0.20 → 0.30 (ICIR이 가장 큰 예측 품질 요소가 됨)
  - Sharpe 가중치 감소: 0.20 → 0.10 (Sharpe 지배 구조 해소)
- **수정 파일**:
  - `app/alpha/fitness.py`: `_IC_NORM_CEIL=0.15` 상수 추가, `ic_norm` 정규화 적용
  - `app/core/config.py`: `ALPHA_FITNESS_W_ICIR=0.30`, `ALPHA_FITNESS_W_SHARPE=0.10`
- **기대 효과** (변경 전→후 기여도 비교, IC=0.10/ICIR=0.60/Sharpe=1.4 기준):
  - IC 기여: 7.1% → 34.7% (+4.9배) — IC가 실질적으로 fitness에 영향
  - ICIR 기여: 28.4% → 31.3% (+0.10) — ICIR이 최대 예측 품질 요소
  - Sharpe 기여: 36.0% → 13.2% (-0.63) — Sharpe 지배 해소
  - 진화 방향 전환: "수익률 높은 팩터" → "예측이 안정적인 팩터"
- **검증 기준** (10사이클 후):
  - [필수] fitness-ICIR 상관 ≥ 0.40 (현재 0.30)
  - [필수] fitness-IC 상관 ≥ 0.15 (현재 0.053)
  - [권장] max ICIR ≥ 0.70 (현재 0.6924)
  - [안전] avg IC 하락 ≤15% (현재 0.091)
  - [안전] max IC 유지 (현재 0.116 실질, 0.262 명목)
  - [관찰] discovered 팩터의 ICIR 분포 변화
- **검증 결과**: PENDING
