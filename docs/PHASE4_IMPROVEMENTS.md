# Phase 4 유저 테스트 결과 — 보완점 문서

**테스트 일시:** 2026-03-02
**테스트 범위:** ABM 시뮬레이션 + MCP 대시보드 (E2E)
**테스트 시나리오:** flash_crash, liquidity_crisis, rate_shock, supply_chain (4종)

---

## CRITICAL (즉시 수정 필요)

### C-1. PriceChart `setMarkers` 크래시 → 화면 완전 백지
- **상태:** 수정 완료
- **원인:** Lightweight Charts v5에서 `lineSeries.setMarkers()` API 제거. v5는 `createSeriesMarkers()` 사용
- **수정:** `PriceChart.tsx` — `createSeriesMarkers(lineSeries, markers)` 패턴으로 변경
- **영향:** 결과 화면 전체가 React error boundary 없이 백지 (치명적 UX 결함)

### C-2. WebSocket `/ws/simulation/{run_id}` 404
- **상태:** 수정 완료
- **원인:** `app/routers/ws.py`에 simulation WebSocket 라우트 미등록
- **수정:** `ws.py`에 `@router.websocket("/simulation/{run_id}")` 추가
- **영향:** 진행률 실시간 표시 불가

### C-3. `/mcp/tools` 엔드포인트 500 Internal Server Error
- **상태:** 미수정
- **원인:** Docker 이미지에 `fastmcp` 패키지 미설치. `requirements.txt`에 추가는 했으나 이미지 리빌드 안 됨
- **수정:** `docker compose up -d --build` 실행하여 이미지 리빌드 필요
- **영향:** MCP 대시보드에서 도구 목록 로딩 실패 (CORS 에러로 표시)

---

## HIGH (시뮬레이션 품질 — 핵심 가치에 직결)

### H-1. 시나리오 충격이 현실적이지 않음 — crash_depth < 0.5%
- **현상:** 모든 4종 시나리오에서 crash_depth가 0.2~0.4%에 불과
  - flash_crash: 0.24%
  - liquidity_crisis: 0.33%
  - rate_shock: 0.33%
  - supply_chain: 0.39%
- **원인 분석:**
  1. **flash_crash의 시장가 매도 물량이 너무 적음**: `sell_volume_multiple=20`이면 20회 × randint(10,50)건 = 약 600주. 호가창의 매수 잔량이 수천 주이므로 충격이 흡수됨
  2. **FundamentalAgent가 50스텝 워밍업 필요**: `len(price_history) < 50`이면 아무 행동 안 함 → 시장 초반에 유동성 제공자가 부족
  3. **에이전트 주문 수량이 너무 작음**: 전체 에이전트가 1~5주씩 거래 → LOB에 미미한 영향
- **권장 수정:**
  - flash_crash: 매도 물량을 현재의 10~50배 증가 (LOB 전체 매수 잔량의 50%+ 소화)
  - 에이전트 주문 크기를 cash/shares 비율로 더 공격적으로 설정
  - FundamentalAgent 워밍업을 20~30스텝으로 단축

### H-2. 가격이 장시간 정체(flatline) → 거래 없음
- **현상:**
  - price_series의 처음 50스텝이 전부 50000.0 (변화 없음)
  - 스텝 130~250 구간에서 50145.0 고정
  - 스텝 250(시나리오 주입) 이후도 50095.0 고정
- **원인:**
  1. FundamentalAgent의 50스텝 워밍업 동안 Noise/Chartist만 활동
  2. NoiseTrader의 `intensity=0.3` + `randint(1,5)` → 주문 간격이 너무 길고 수량이 너무 적음
  3. 에이전트 간 가격 수렴 후 아무도 주문하지 않는 **교착 상태** 발생
- **권장 수정:**
  - NoiseTrader intensity를 0.6~0.8로 상향
  - ChartistAgent lookback을 10스텝으로 축소 (더 빠른 반응)
  - 매 스텝 랜덤 주문을 추가하는 "market maker" 역할 에이전트 도입
  - 초기 호가 시딩 후 워밍업 페이즈(50스텝)에서 인위적 랜덤 거래 생성

### H-3. recovery_steps 항상 0
- **현상:** 모든 시나리오에서 `recovery_steps=0`
- **원인:** recovery_steps 계산 조건이 `p < peak_price * 0.95` (5% 하락)인데, 실제 crash_depth가 0.3% 수준이므로 조건 자체가 충족되지 않음
- **권장 수정:**
  - crash_depth 기반으로 threshold를 동적 설정 (실제 낙폭의 50%를 크래시 기준으로)
  - 또는 시나리오 주입 시점(inject_at_step) 전후의 가격차를 기준으로 계산

### H-4. Volume이 0으로 떨어지면 복구 불가
- **현상:** volume_series에서 스텝 141 이후 0 유지 (500스텝까지)
- **원인:** 모든 에이전트가 주문을 중단하면 거래량이 0으로 유지. "유동성 소멸" 현상
- **권장 수정:**
  - NoiseTrader에 최소 활동 보장 (intensity=0이어도 가끔 주문)
  - MarketMaker 에이전트 추가 (항상 BBO에 양방향 주문 유지)

---

## MEDIUM (UI/UX 개선)

### M-1. 차트 X축이 "1970년" 날짜로 표시
- **현상:** PriceChart의 X축이 `1970년 1월 1일` 부터의 날짜로 표시
- **원인:** Lightweight Charts는 `time` 필드를 Unix timestamp(초)로 해석. 스텝 인덱스(0,1,2...)를 넘기면 1970년 1~6초로 렌더링
- **권장 수정:**
  - X축을 인덱스 모드로 전환 (`timeScale: { rightOffset: 0, barSpacing: 6 }` + custom time formatter)
  - 또는 각 스텝을 가상의 timestamp로 변환 (예: `Date.now() + step * 60000`)

### M-2. 실행 기록 목록의 status가 즉시 갱신되지 않음
- **현상:** 새 테스트 시작 후 히스토리에 "대기"로 표시되다가, 다른 동작 후에야 "완료"로 갱신
- **원인:** `useStartStressTest()`의 `onSuccess`에서 `invalidateQueries(["stress-tests"])`는 호출되지만, 시뮬레이션이 완료된 후 목록 refetch가 트리거되지 않음
- **권장 수정:**
  - `useStressTest(id)`의 adaptive refetchInterval이 COMPLETED 감지 시 `queryClient.invalidateQueries(["stress-tests"])` 호출
  - 또는 WebSocket `completed` 이벤트 수신 시 `stress-tests` 목록 invalidate

### M-3. 시나리오 미선택 시 기본 시나리오가 flash_crash로 하드코딩
- **현상:** 프리셋을 선택하지 않고 바로 "시작" 누르면 `type: "flash_crash"` 기본값
- **권장 수정:**
  - 시나리오 미선택 시 "시작" 버튼 비활성화
  - 또는 사용자에게 시나리오 선택 필수 안내 (validation)

### M-4. 고급설정 "LLM" 에이전트 수 기본값 0
- **현상:** LLM 에이전트가 기본 0이라서 사용자가 활성화하려면 직접 변경 필요
- **권장 수정:**
  - LLM 에이전트 사용 시 비용 안내 툴팁 추가
  - ANTHROPIC_API_KEY 미설정 시 LLM 입력 필드 비활성화 + 안내 메시지

### M-5. LOB 깊이 차트 — ask/bid 바 방향이 동일
- **현상:** 매도(ask) 바가 오른쪽 정렬, 매수(bid) 바가 왼쪽 정렬인데, 시각적으로 대칭성이 떨어짐
- **권장 수정:**
  - 호가 깊이를 좌우 대칭 레이아웃으로 변경 (bid←→ask)
  - 또는 가운데 가격을 기준으로 양방향 수평 바 차트

---

## LOW (개선 사항)

### L-1. StrategyAgent가 비어있으면 strategy_pnl 항상 0
- **현상:** `strategy_json: {}` 전달 시 StrategyAgent가 어떤 매매도 하지 않음
- **권장 수정:**
  - 비어있는 전략일 때 "Buy & Hold" 기본 행동 적용
  - 또는 백테스트 전략 선택 UI를 시뮬레이션에도 통합

### L-2. 커스텀 시나리오 생성 시 결과 시나리오 미리보기 없음
- **현상:** AI가 생성한 시나리오의 explanation만 텍스트로 보이고, params 확인 불가
- **권장 수정:**
  - 생성된 ScenarioConfig의 params를 JSON 뷰어로 표시
  - "적용" 버튼으로 사용자 확인 후 시나리오 설정

### L-3. 진행률 바가 COMPLETED 후에도 잠시 보임
- **현상:** 시뮬레이션이 매우 빨리 완료되면 (< 1초) 진행률 바가 100%로 잠깐 표시된 후 사라짐
- **권장 수정:**
  - COMPLETED 상태 감지 시 즉시 숨기기 (현재 동작과 동일하나, 2초 폴링 간격으로 인한 지연)

### L-4. 에이전트별 PnL 시각화 없음
- **현상:** `results.agent_metrics`에 에이전트 타입별 avg_pnl/total_pnl이 있지만 UI에서 미표시
- **권장 수정:**
  - StressMetrics 하단에 에이전트별 PnL 바 차트 또는 테이블 추가
  - 각 에이전트 유형의 수, 평균 PnL, 총 PnL 표시

### L-5. depth_series 스냅샷 간격이 50스텝으로 고정
- **현상:** 500스텝 시뮬레이션에서 depth 스냅샷이 10개뿐 (50스텝마다)
- **권장 수정:**
  - 시나리오 주입 전후 구간에서 스냅샷 빈도 증가 (예: 주입 ±20스텝은 매 스텝)
  - 총 스텝 수에 비례하여 스냅샷 수 조정 (최소 20개)

---

## 수정 완료 항목 요약

| # | 항목 | 파일 | 상태 |
|---|------|------|------|
| C-1 | setMarkers v5 API | `src/components/simulation/PriceChart.tsx` | 수정 완료 |
| C-2 | WS simulation 라우트 | `app/routers/ws.py` | 수정 완료 |

## 미수정 항목 우선순위

| 순위 | # | 항목 | 예상 난이도 |
|------|---|------|-----------|
| 1 | C-3 | fastmcp Docker 빌드 | 낮음 (docker build) |
| 2 | H-1 | 시나리오 충격 강화 | 중간 (agents.py, exchange.py 튜닝) |
| 3 | H-2 | 가격 정체/교착 해소 | 중간 (NoiseTrader 강화, MarketMaker 추가) |
| 4 | H-3 | recovery_steps 계산 | 낮음 (exchange.py 수정) |
| 5 | M-1 | 차트 X축 포맷 | 낮음 (PriceChart.tsx) |
| 6 | M-2 | 히스토리 status 갱신 | 낮음 (use-simulation.ts) |
| 7 | H-4 | Volume 0 복구 | 중간 (MarketMaker 에이전트) |
| 8 | M-3~5 | UI/UX 사소한 개선 | 낮음 |
| 9 | L-1~5 | 추가 기능 | 중간~낮음 |
