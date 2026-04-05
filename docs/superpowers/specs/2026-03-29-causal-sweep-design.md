# Causal Sweep — 인과검증 전수 처리 기능

## Context

마이닝이 계속 돌면서 팩터가 쌓이지만, 인과검증은 1시간마다 상위 30%만 처리하여 미검증 팩터가 2,600+건 적체됨. 마이닝을 일시중단하고 미검증 전수를 한번에 처리한 뒤 자동 재시작하는 기능이 필요.

## 설계

### 백엔드 API

**`POST /alpha/causal-sweep`**

파라미터: `interval` (기본 "1d")

동작:
1. Factory 중지 (이미 중지 상태면 스킵)
2. 해당 interval의 미검증(causal_robust IS NULL) + discovered 상태 + ic_mean > 0 팩터 전체 ID 수집
3. 기존 `validate-batch` 인프라 재사용하여 배치 검증 시작
4. 백그라운드 태스크에서 검증 완료 감지 → Factory 자동 재시작 (이전 config 유지)

반환: `{ job_id, total, interval, auto_restart: true }`

상태 조회: 기존 `GET /alpha/validate/{job_id}/status` 그대로 사용

**`POST /alpha/causal-sweep/cancel`**

파라미터: `job_id`, `interval` (기본 "1d")

동작:
1. 진행 중인 배치 검증 취소
2. Factory 즉시 재시작

### 프론트엔드

**위치**: `AlphaFactoryControl.tsx` 팩토리 헤더 영역

**컴포넌트**: `CausalSweepPanel` (신규)

상태별 UI:
- **대기**: `⚡ 인과검증 전수 (N건)` 버튼
- **진행 중**: 진행률 바 + 통과/탈락 카운터 + [취소하고 마이닝 복귀] 버튼
- **미검증 0건**: 버튼 비활성 "전수 검증 완료"

데이터: 미검증 건수는 `GET /alpha/factory/status` 응답에 `causal_pending_count` 필드 추가.

## 수정 파일

### 백엔드 (3파일)
| 파일 | 변경 |
|------|------|
| `app/routers/alpha.py` | `POST /alpha/causal-sweep`, `POST /alpha/causal-sweep/cancel` 엔드포인트 |
| `app/alpha/scheduler.py` | `get_last_config()` 메서드 (재시작 시 이전 config 복원용) |
| `app/routers/alpha.py` | `GET /alpha/factory/status` 응답에 `causal_pending_count` 추가 |

### 프론트엔드 (5파일)
| 파일 | 변경 |
|------|------|
| `src/types/alpha.ts` | CausalSweepResponse 타입 + AlphaFactoryStatus에 causal_pending_count 추가 |
| `src/api/alpha.ts` | startCausalSweep(), cancelCausalSweep() 함수 |
| `src/hooks/queries/use-alpha.ts` | useStartCausalSweep(), useCancelCausalSweep() 훅 |
| `src/components/alpha/CausalSweepPanel.tsx` | 신규: 전수검증 버튼 + 진행률 패널 |
| `src/components/alpha/AlphaFactoryControl.tsx` | CausalSweepPanel 삽입 |
