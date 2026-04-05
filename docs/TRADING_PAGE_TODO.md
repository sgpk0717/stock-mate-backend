# 자동매매 페이지 미해결 이슈 (2026-03-24)

## 즉시 수정 완료 (프론트엔드 + 백엔드)

1. [x] `[object Object]/10` 렌더링 → positions JSON 파싱 (DecisionMonitor.tsx)
2. [x] 상단 요약 카드 ₩0 → Paper 모드에서 세션 데이터 기반 (TradingPage.tsx)
3. [x] 매매 시간 오후 6:35 → UTC→KST 시간 보정 (TradingPage.tsx)
4. [x] 사유 "지정가 매수 체결" → 조건 기반 자연어 사유 (live_runner.py)
5. [x] 보유 포지션 0개 → Paper 모드 세션 포지션 표시 (TradingPage.tsx)
6. [ ] 매도 없음 — 아래 #13 참조 (분봉 수집 실패가 근본 원인)
7. [x] 중복 세션 방지 → start_session에 중복 체크 (live_runner.py)
8. [x] Redis 카운터 불일치 → _sync_session_to_redis에서 직접 카운터 계산 (live_runner.py)
9. [x] 알파 랭킹 스코어 0.500 → 미구현 상태 명시 + 경고 표시 (AlphaRanking.tsx)
10. [x] 매수/매도 목록 동일 → sell은 보유종목 필터 (AlphaRanking.tsx)
11. [x] 알파 랭킹 UX → 팩터명/스코어/컬럼 설명 추가 (AlphaRanking.tsx)
12. [x] 매매 이력 "매매 없음" → 실시간 세션 trade_count 보정 (TradeDailyHistory.tsx)
13. [ ] 데이터부족 대량 → 아래 상세
14. [x] 매수 판단 로그 안 보임 → "실행된 매매 판단" 고정 섹션 (DecisionMonitor.tsx)
15. [x] 판단 사유 자연어 → 클릭 시 조건 펼침 (DecisionMonitor.tsx)
16. [x] 가격 소수점 → Math.round() 프론트 표시 (TradingPage.tsx)
17. [x] B1 용어 → "1차 매수" 등 STEP_LABELS 매핑 (TradingPage.tsx)
18. [ ] 최대 포지션 15/10 → 아래 상세
19. [x] raw 디버그 텍스트 → 자연어 변환 (DecisionMonitor.tsx)
20. [ ] 누적 P&L -410.95% → 아래 상세

---

## TODO: 미해결 이슈 상세

### #6 + #13: 매도 없음 + 데이터부족 대량 (근본 원인: 분봉 수집 실패)

**현상:**
- 928종목이 "데이터 부족: 0봉 (최소 30봉 필요)"
- 매수 15건 후 매도 시그널 체크 불가 (새 봉이 없으므로)

**근본 원인 분석:**
- `_intraday_collect_loop()`이 KIS API를 사용하여 분봉 수집
- 수집 대상이 유니버스 전체(~950종목)인데 KIS API rate limit(15req/s)으로 전체 수집에 시간 소요
- 샘플 종목 하나의 마지막 시각으로 전체 수집 시작점을 판단 → 일부 종목 누락
- 비거래일/장 마감 후에는 새 봉이 없어 "데이터 부족" 정상

**수정 방향:**
1. 장중 분봉 수집기의 종목별 수집 시각 추적 (글로벌 샘플 종목 1개 → 종목별 last_dt)
2. 수집 실패 종목 리스트를 Redis에 저장 → 프론트에서 수집 진행률 표시
3. 유니버스 크기에 따른 KIS API 배치 크기 자동 조정
4. 비거래일/장외 시간에는 "데이터부족" 대신 "장외 시간" 메시지 표시

**즉시 대응 (완료):**
- 프론트에서 "데이터부족" 메시지의 raw 텍스트를 자연어로 변환

### #18: 최대 포지션 15/10 초과

**현상:**
- 최대 포지션 제한이 10인데 15로 표시
- 세션 카드에는 "포지션 0개"

**원인 분석:**
- 3개 세션이 독립적으로 포지션을 관리 (세션당 max_positions=10)
- 각 세션이 다른 봉 데이터를 보고 각각 매수 → 총합 15개
- 모니터링은 첫 번째 세션의 데이터만 표시
- positions 필드가 `[object Object]`로 표시되어 실제 개수 파악 불가했음

**수정 방향:**
1. 워크플로우 orchestrator에서 세션 간 포지션 공유 또는 전체 합산 제한
2. `evaluate_buy()`의 `current_positions`에 모든 세션의 포지션 합산 전달
3. 프론트: 모니터링에서 전체 세션 포지션 합산 표시 (완료 — #1 수정에서 해결)

### #20: 누적 P&L -410.95% 비현실적

**현상:**
- 현물 매매에서 -410% 손실 불가
- 3/18일 하루 -284.54% 기록

**원인 분석:**
- `pnl_pct`이 자본금 기준이 아닌 개별 포지션 손익의 단순 합산일 가능성
- 또는 세션 간 중복 집계 (같은 날 3개 세션의 pnl_pct를 합산)
- workflow_runs.pnl_pct 계산 로직 확인 필요 (trade_reviewer.py)

**수정 방향:**
1. `pnl_pct` = `total_pnl_amount / initial_capital * 100`으로 통일
2. 세션별 pnl_pct를 자본금 비중으로 가중 평균
3. 프론트: -100% 이하 값에 경고 표시
