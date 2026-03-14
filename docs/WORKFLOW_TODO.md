# Stock Mate 자동화 워크플로우 — 구현 TODO

> **최종 업데이트**: 2026-03-13
> **참조 문서**: `C:\Users\Rex\Documents\StockMate_자동화워크플로우_설계서_v2.md`
> **현재 상태**: 전체 TODO 완료 (CRITICAL 4 + HIGH 8 + MEDIUM 7). 워크플로우 FSM + APScheduler + OpenClaw + MCP 13도구 + Telegram 봇 가동.

---

## 시스템 전체 맥락 (이 문서를 처음 보는 경우)

Stock Mate는 매일 이 사이클을 자동으로 반복한다:

```
[야간 마이닝] → [08:30 팩터 선택] → [09:00 매매 시작] → [15:30 청산]
→ [16:30 리뷰+피드백] → [18:00 마이닝 시작] → 반복
```

**실행 주체 2개:**
- **APScheduler (백엔드)**: 시간-정확 실행 (매매 시작/청산/마이닝). `app/workflow/orchestrator.py`
- **OpenClaw (AI 에이전트)**: 지능적 판단 (분석/보고/코드수정). 크론잡 8개, MCP로 백엔드 호출

**핵심 원칙**: 매매 실행은 코드가, 판단은 LLM이, 중요 결정은 사람이 승인.

---

## CRITICAL — 이것 없으면 자동매매 자체가 불가

### ~~C1. TradingContext DB 영속화~~ ✅ 완료 (2026-03-13)

**현재 문제**: `app/trading/context.py`의 `_contexts: dict`가 메모리 전용.
서버 재시작(Docker rebuild, 크래시 등) 시 활성 매매 컨텍스트가 전부 소실된다.
09:00에 만든 TradingContext가 장중에 서버 재시작되면 사라져서 매매가 중단됨.

**해야 할 것**:
1. `app/workflow/models.py`에 `TradingContextModel` SQLAlchemy 모델 추가
   - 설계서 §13.1의 `trading_contexts` DDL 참조
   - 컬럼: id(UUID PK), mode(paper/real), status(active/archived/error), strategy(JSONB),
     risk_management(JSONB), source_factor_id(UUID), auto_created(bool), created_at, updated_at
2. `app/trading/context.py` 수정
   - `_contexts: dict` → DB 테이블 + 메모리 LRU 캐시 (이중 레이어)
   - `create_context()`: DB INSERT + 메모리 캐시
   - `get_context()`: 메모리 먼저 → miss시 DB 조회
   - `archive_context()`: status='archived' 업데이트
3. `app/main.py` lifespan에서 서버 시작 시 `status='active'` 컨텍스트 자동 로드
4. Alembic 마이그레이션 생성

**수용 기준**: 서버 재시작 후 `GET /trading/contexts`에서 active 컨텍스트가 복원되어 있어야 함.

---

### ~~C2. LiveSession 상태 복구~~ ✅ 완료 (2026-03-13)

**현재 문제**: `app/trading/live_runner.py`의 `_sessions: dict`가 메모리 전용.
장중(09:00~15:30) 서버 재시작 시 활성 매매 세션이 사라진다.
KIS에 보낸 주문은 증권사에 남아있지만, 봇은 포지션을 모르는 상태가 됨.

**해야 할 것**:
1. LiveSession 상태를 DB `trading_contexts` 테이블의 `session_state` JSONB 컬럼에 저장
   - 저장 대상: 현재 포지션 목록, 마지막 시그널 시각, trade_log(최근 N건)
   - 주기: 매 시그널 체크(30초)마다 또는 포지션 변경 시 저장
2. 서버 시작 시 복구 로직:
   - `status='active'` 컨텍스트 로드 → LiveSession 재생성
   - KIS API로 실제 잔고 동기화 (`kis_client.get_balance()`)
   - 동기화 결과와 DB 상태 비교 → 불일치 시 KIS 잔고 기준으로 보정
3. 복구 후 LiveRunner 시그널 루프 자동 재개

**의존**: C1 (TradingContext DB) 먼저 완료해야 함.

**수용 기준**: 장중 `docker-compose restart app` 후 매매가 자동 재개되고, KIS 잔고와 일치해야 함.

---

### ~~C3. 매매 로그 DB 영속화~~ ✅ 완료 (2026-03-13)

**현재 문제**: `LiveSession.trade_log`가 `list[dict]` 메모리 전용.
재시작 시 당일 매매 기록이 사라진다.
16:30 TradeReviewer가 리뷰할 데이터가 없어서 피드백 루프가 끊김.

**해야 할 것**:
1. `app/workflow/models.py`에 `LiveTrade` 모델 추가
   - 컬럼: id(UUID), context_id(FK→trading_contexts), symbol, side(BUY/SELL),
     price, qty, pnl_pct, scale_step(B1/B2/S-STOP/S-TRAIL/S-HALF),
     signal_source(팩터 시그널 값), executed_at(TIMESTAMPTZ)
2. `app/trading/live_runner.py` 수정
   - 주문 체결 시 `LiveTrade` INSERT
   - `trade_log` 리스트도 유지 (호환성), DB가 source of truth
3. `app/workflow/trade_reviewer.py` 수정
   - 메모리 trade_log 대신 DB에서 당일 `LiveTrade` 조회하여 리뷰 생성

**의존**: C1 (TradingContext DB).

**수용 기준**: 서버 재시작 후에도 `GET /trading/session/{id}/trades`에서 당일 매매 내역이 조회됨.

---

### ~~C4. trading_contexts Alembic 마이그레이션~~ ✅ 완료 (이전 세션)

**현재 문제**: 설계서 §13.1에 DDL이 정의되어 있지만 실제 마이그레이션이 없다.
C1~C3 구현 전에 테이블이 DB에 존재해야 함.

**해야 할 것**:
1. Alembic 마이그레이션 생성: `trading_contexts`, `live_trades` 테이블
2. `alpha_factors` 테이블에 staleness 컬럼 4개 추가 (H3 선행 작업):
   - `last_evaluated_at TIMESTAMPTZ`
   - `live_ic_7d FLOAT`
   - `live_sharpe_7d FLOAT`
   - `staleness_warning BOOLEAN DEFAULT false`
3. `docker-compose run --rm app alembic upgrade head` 실행

**수용 기준**: `\dt` 에서 `trading_contexts`, `live_trades` 테이블 존재. `alpha_factors`에 새 컬럼 존재.

---

## HIGH — 자동매매는 되지만 안정성/정확성 저하

### ~~H1. 주문 상태 추적~~ ✅ 완료 (2026-03-13)

**현재 문제**: KIS에 주문을 보내면 fire-and-forget. `pending_orders`가 항상 빈 배열.
주문이 미체결/부분체결되어도 봇이 모른다. 같은 종목에 중복 주문 가능.

**해야 할 것**:
1. `app/trading/kis_order.py`에 주문 후 상태 폴링 로직 추가
   - KIS 주문 체결 조회 API: `TTTC8001R` (실전) / `VTTC8001R` (모의)
   - 주문 후 5초 간격으로 최대 3회 체결 확인
2. `pending_orders` 딕셔너리 관리 (주문번호 → 주문 정보)
3. LiveRunner 시그널 루프에서 미체결 주문 존재 시 해당 종목 신규 주문 스킵

**수용 기준**: 주문 후 체결 확인까지 추적. 미체결 주문이 있는 종목에 중복 주문 안 나감.

---

### ~~H2. 포트폴리오 수준 MDD 서킷브레이커~~ ✅ 완료 (2026-03-13)

**현재 문제**: 개별 포지션 손절(`stop_loss_pct`)만 있음.
10개 포지션이 동시에 -4%씩 떨어지면 개별 손절(-5%)에 안 걸리지만 포트폴리오는 -4%.
`WORKFLOW_MAX_DRAWDOWN_PCT=10.0` 설정이 있지만 실제 감시 코드가 없음.

**해야 할 것**:
1. `app/trading/live_runner.py`의 시그널 체크 루프에 포트폴리오 MDD 계산 추가
   - 당일 시작 자산 vs 현재 평가금액
   - `(현재 - 고점) / 고점 * 100`
2. MDD > `WORKFLOW_MAX_DRAWDOWN_PCT` 시:
   - 모든 포지션 시장가 매도 (전량 청산)
   - LiveSession 중지
   - workflow_events에 "portfolio_circuit_breaker" 이벤트 기록
   - (OpenClaw이 다음 미드데이 체크에서 감지하여 텔레그램 알림)

**수용 기준**: 포트폴리오 전체 MDD가 임계치 초과 시 전량 청산 + 이벤트 로그.

---

### ~~H3. 팩터 스탈니스 추적 + 자동 은퇴~~ ✅ 완료 (2026-03-13)

**현재 문제**: 한번 발견된 팩터가 영원히 "매매 가능" 상태.
시장이 바뀌어서 IC가 0에 수렴해도 AutoSelector가 계속 선택할 수 있음.
설계서 §12.3의 7일/14일/30일 노화 관리가 없음.

**해야 할 것**:
1. C4에서 추가한 `alpha_factors` staleness 컬럼 활용
2. `app/workflow/orchestrator.py`의 `handle_review()` 또는 별도 스케줄러 잡에서:
   - 활성 팩터들의 최근 7일 실매매 IC 재계산 → `live_ic_7d` 업데이트
   - `live_ic_7d < ic_mean * 0.5` → `staleness_warning = True`
   - 14일 연속 IC 하락 → `status='stale'` (AutoSelector에서 제외)
   - 30일 이상 `stale` → `status='retired'`
3. `app/workflow/auto_selector.py`에서 `status IN ('stale', 'retired')` 팩터 필터링

**의존**: C4 (staleness 컬럼), C3 (매매 로그 DB — IC 재계산에 실적 데이터 필요).

**수용 기준**: 2주간 IC가 하락한 팩터가 자동으로 `stale` 마킹되어 매매에서 제외됨.

---

### ~~H4. 인과 검증 실패 분류 → 마이닝 피드백~~ ✅ 완료 (2026-03-13)

**현재 문제**: `app/alpha/causal.py`에서 인과 검증 실패 시 단순 `status='mirage'`만 설정.
어떤 단계에서 왜 실패했는지 분류가 안 됨.
다음 마이닝에서 같은 유형의 실패를 반복할 수 있음.

**해야 할 것**:
1. `app/alpha/causal.py`의 `validate()` 반환값에 `failure_type` 필드 추가:
   - `LOW_IC`: 기본 예측력 부족
   - `CONFOUNDED`: 플라시보 미통과 (허위 상관)
   - `FRAGILE`: 랜덤 원인 미통과 (교란 변수에 민감)
   - `REGIME_SHIFT`: 체제 변화 미통과 (전/후반 불일치)
2. `alpha_factors` 테이블에 `causal_failure_type VARCHAR(20)` 컬럼 추가
3. `app/workflow/feedback_engine.py`의 `generate_context()`에서:
   - 최근 실패 팩터의 failure_type 집계
   - 예: `CONFOUNDED` 3건 → "최근 허위 상관 팩터 다수. 독립적 피처 조합 탐색 필요"
   - 이 텍스트가 mining_context에 포함 → Claude가 다음 가설에서 회피

**수용 기준**: 인과 검증 실패 팩터에 failure_type이 기록되고, mining_context에 실패 유형 요약이 포함됨.

---

### ~~H5. 백테스트 타임아웃~~ ✅ 완료 (2026-03-13)

**현재 문제**: `app/backtest/runner.py`에서 백테스트가 무한 실행 가능.
대규모 데이터(수년치 틱)로 백테스트하면 OOM이나 무한 루프 가능.

**해야 할 것**:
1. `app/backtest/runner.py`의 `run_backtest_task()` 내부를 `asyncio.wait_for()` 래핑
2. 기본 타임아웃: 30분 (`BACKTEST_TIMEOUT_SECONDS=1800` 환경변수)
3. 타임아웃 시 `status='FAILED'`, `error_message='타임아웃 (30분 초과)'`

**수용 기준**: 30분 초과 백테스트가 자동 중단되고 FAILED 상태로 DB 기록.

---

### ~~H6. AutoSelector 점수에 MDD 반영~~ ✅ 이미 구현됨

> `app/core/config.py`에 `WORKFLOW_SCORE_W_MDD = 0.15` 존재.
> `app/workflow/auto_selector.py`의 `_compute_score()`에 MDD 정규화 구현됨.
> 탐색 결과 이미 동작 중이므로 별도 작업 불필요.

---

### ~~H7. OpenClaw 헬스체크 + 독립 모드 (장애 대비)~~ ✅ 이미 구현됨

**현재 문제**: `app/workflow/orchestrator.py`의 `check_openclaw_health()`가 5분마다 실행되지만,
실제로 OpenClaw 게이트웨이 포트(18789)에 핑하는지 미검증.
핑 실패 시 재시작/독립모드 전환 로직이 스텁(stub)일 수 있음.

**해야 할 것**:
1. `check_openclaw_health()` 확인/수정:
   - `httpx.AsyncClient`로 `http://127.0.0.1:18789/health` GET
   - 3회 연속 실패 → `openclaw gateway restart` 시도 (subprocess)
   - 6회 연속 실패 (30분) → 독립 모드 플래그 설정
2. 독립 모드 동작:
   - `_independent_mode: bool = False` 플래그
   - True일 때: 매매 실행(APScheduler)은 정상, OpenClaw 의존 작업(분석/보고)만 스킵
   - 텔레그램 직접 알림: `httpx`로 Telegram Bot API 직접 호출 (OpenClaw 우회)
   - 봇 토큰/chatId는 환경변수에서 로드
3. `app/core/config.py`에 추가:
   - `OPENCLAW_HEALTH_URL: str = "http://127.0.0.1:18789/health"`
   - `TELEGRAM_BOT_TOKEN: str = ""` (독립 모드 직접 알림용)
   - `TELEGRAM_CHAT_ID: str = ""` (독립 모드 직접 알림용)

**수용 기준**: OpenClaw 프로세스 종료 후 30분 내에 텔레그램으로 장애 알림이 오고, 매매는 계속됨.

---

### ~~H8. OpenClaw 03:00 데몬 자동 재시작~~ ✅ 완료 (2026-03-13)

**현재 문제**: 설계서 §9.0에서 OpenClaw 13시간+ 메모리 누수 이슈 언급.
03:00 KST 자동 재시작이 설정되어 있지 않음.

**해야 할 것**:
1. Windows Task Scheduler 작업 등록:
   ```
   schtasks /create /tn "OpenClaw Daily Restart" /tr "openclaw gateway restart" /sc daily /st 03:00 /f
   ```
2. 또는 APScheduler에 03:00 잡 추가:
   ```python
   # orchestrator.py setup_scheduler() 내
   scheduler.add_schedule(restart_openclaw, CronTrigger(hour=3, minute=0))
   ```
   - `subprocess.run(["openclaw", "gateway", "restart"])` 실행

**수용 기준**: 매일 03:00에 OpenClaw 게이트웨이가 재시작되고, 07:00 모닝 브리프가 정상 실행됨.

---

## MEDIUM — 기능 완성도

### ~~M1. 텔레그램 Inline Keyboard 양방향 컨펌~~ ✅ 완료 (2026-03-13)

**현재 문제**: OpenClaw이 보내는 텔레그램 메시지가 전부 단방향 보고(L1).
설계서 §11에서 정의한 L3/L4 승인 메커니즘(인라인 버튼)이 없음.
실거래 전환, 대규모 코드 수정 같은 중요 결정을 사용자가 승인/거부할 수 없음.

**해야 할 것**:
1. OpenClaw 크론잡 프롬프트에 승인 필요 로직 추가
   - 현재: "텔레그램으로 보고하세요"
   - 변경: "텔레그램으로 보고하고, [승인 필요] 항목은 사용자 응답을 기다리세요"
2. OpenClaw의 텔레그램 채널이 양방향 지원하는지 확인
   - `openclaw.json`의 `channels.telegram.dmPolicy: "pairing"` — DM 가능
   - OpenClaw이 "질문 → 응답 대기" 할 수 있는지 확인 필요
3. 대안: 백엔드에 텔레그램 봇 직접 연동
   - `python-telegram-bot` 패키지로 Inline Keyboard 구현
   - `/approve {task_id}`, `/reject {task_id}` 콜백 핸들러
   - approval_queue 테이블: task_id, description, status(pending/approved/rejected), created_at

**참고**: OpenClaw 자체의 텔레그램 인터랙션 능력에 따라 구현 방식이 달라짐. 먼저 OpenClaw docs 확인.

---

### ~~M2. 텔레그램 봇 명령어 핸들러~~ ✅ 완료 (2026-03-13)

**현재 문제**: 설계서 §11.4의 6개 명령어(`/status`, `/stop`, `/resume`, `/skip`, `/report`, `/factors`)가 없음.
사용자가 텔레그램에서 봇에게 명령을 보내도 아무 반응 없음.

**해야 할 것**:
1. 방법 A (OpenClaw 네이티브): OpenClaw에 텔레그램 메시지 수신 시 MCP 도구 호출 설정
   - OpenClaw의 `channels.telegram` 설정에서 DM 수신 + 명령어 파싱
   - "/status" 수신 → MCP get_workflow_status() 호출 → 결과 응답
2. 방법 B (백엔드 직접): `python-telegram-bot`으로 명령어 핸들러 구현
   - `app/telegram/bot.py` 신규 생성
   - 각 명령어 → 내부 API 호출 → 텔레그램 응답
   - main.py lifespan에서 봇 시작

**수용 기준**: 텔레그램에서 `/status` 보내면 현재 워크플로우 상태가 응답으로 옴.

---

### ~~M3. Claude Code Agent SDK 연동 (프로젝트 자동 개선)~~ ✅ 이미 구현됨

**현재 문제**: 22:00 `project_improvement` 크론잡이 버그를 발견해도 실제 코드 수정이 안 됨.
설계서 §10.1의 `scripts/claude_code_agent.py`가 없음.
OpenClaw → Claude Code 위임 경로가 없음.

**해야 할 것**:
1. `scripts/claude_code_agent.py` 생성:
   ```python
   from claude_agent_sdk import query, ClaudeAgentOptions

   async def fix_code(task: str, files: list[str]) -> dict:
       result = await query(
           prompt=f"[Stock Mate 버그 수정] {task}",
           options=ClaudeAgentOptions(
               allowed_tools=["Read", "Edit", "Write", "Bash", "Glob", "Grep"],
               cwd="C:/Users/Rex/stock-mate",
               max_turns=20,
           )
       )
       return parse_result(result)
   ```
2. OpenClaw project_improvement 크론잡 프롬프트에 호출 방법 추가:
   - "경미한 버그 발견 시: `python C:/Users/Rex/stock-mate/scripts/claude_code_agent.py --task '{설명}'` 실행"
3. 안전장치 (설계서 §10.2):
   - 변경 가능 파일: stock-mate 리포 내로 제한
   - 50줄 초과 변경: 텔레그램 승인 필요
   - git push 금지 (commit만)
   - 환경변수/리스크 파라미터 변경 금지

**선행**: `pip install claude-agent-sdk` (PyPI v0.1.48+)

**수용 기준**: OpenClaw 크론잡에서 Claude Code Agent를 호출하여 실제 코드 수정 + git commit이 이루어짐.

---

### ~~M4. 워크플로우 프론트엔드 대시보드~~ ✅ 이미 구현됨

**현재 문제**: 자동화 워크플로우 상태를 볼 수 있는 UI가 없음.
현재 phase, 오늘 팩터, 매매 실적, 마이닝 진행률을 웹에서 확인할 수 없음.

**해야 할 것**:
1. `stock-mate-frontend/src/pages/WorkflowPage.tsx` 신규 생성
   - 상단: 현재 FSM 상태 (IDLE/PRE_MARKET/TRADING/...) + 시각적 진행 바
   - 중앙: 오늘 선택된 팩터 정보 + 실시간 PnL
   - 하단: 최근 7일 workflow_runs 히스토리 (date, phase, pnl_pct, trade_count)
2. `stock-mate-frontend/src/hooks/queries/use-workflow.ts` 신규 생성
   - `useWorkflowStatus()`: `GET /workflow/status` 5초 폴링
   - `useWorkflowHistory()`: `GET /workflow/history`
   - `useWorkflowTrigger()`: `POST /workflow/trigger`
3. `App.tsx`에 `/workflow` 라우트 추가, Sidebar에 메뉴 추가

**수용 기준**: `http://localhost:5173/workflow`에서 현재 워크플로우 상태와 히스토리가 표시됨.

---

### ~~M5. max_cycles 마이닝 야간 예산~~ ✅ 완료 (2026-03-13)

**현재 문제**: 설계서 §3.1에서 `max_cycles=10` (야간 예산) 명시.
현재는 시간 간격(6시간)만 있고 최대 사이클 수 제한 없음.
API 비용 폭주 방지 장치가 없음.

**해야 할 것**:
1. `app/core/config.py`에 `ALPHA_FACTORY_MAX_CYCLES: int = 10` 추가
2. `app/alpha/scheduler.py`의 `_run_cycle()` 또는 스케줄러 루프에서:
   - `_cycle_count` 카운터 관리
   - `_cycle_count >= max_cycles` → 팩토리 자동 중지
3. `app/workflow/orchestrator.py`의 `handle_mining()`에서 `max_cycles` 전달

**수용 기준**: 10사이클 완료 후 팩토리가 자동 중지됨. `get_workflow_status`에서 사이클 수 확인 가능.

---

### ~~M6. get_factor_performance MCP 도구~~ ✅ 완료 (2026-03-13)

**현재 문제**: 설계서 §4.1.1 #13에서 팩터별 실매매 성적 추적 도구 정의.
OpenClaw의 post_market_analysis가 `get_factor_performance`를 호출하려 하지만 MCP에 없음.

**해야 할 것**:
1. `app/mcp/server.py`에 `get_factor_performance` 도구 추가:
   ```python
   @mcp.tool()
   async def get_factor_performance(factor_id: str = "", days: int = 7) -> str:
       """팩터별 실매매 성적 조회. factor_id 미지정 시 최근 사용 팩터 전체."""
       # live_feedback 테이블에서 조회
       # 반환: factor_name, 일별 pnl_pct, 승률, 평균 IC, 백테스트 대비 갭
   ```

**수용 기준**: `mcporter call stockmate.get_factor_performance` 정상 응답.

---

### ~~M7. post_market_analysis 시간 복원 (16:00)~~ ✅ 완료 (2026-03-13)

**현재 문제**: 내가 FeedbackEngine 덮어쓰기 문제 때문에 17:00으로 옮겼지만,
설계서 원래 의도는 16:00이다.
FeedbackEngine이 기존 mining_context를 보존하도록 이미 수정했으므로,
원래 시간(16:00)으로 복원해도 안전함.

**해야 할 것**:
1. `openclaw cron edit 91382f28-6c48-43e5-a4cc-c686e95844d5 --cron "0 16 * * 1-5"`
2. 흐름 확인:
   - 16:00 OpenClaw submit_trading_feedback → mining_context에 append
   - 16:30 FeedbackEngine → 새 텍스트 생성 + 기존(OpenClaw 피드백) 보존
   - 18:00 mining → 합쳐진 context 사용

**수용 기준**: 크론잡 시간이 16:00이고, 16:30 리뷰 후에도 OpenClaw 피드백이 mining_context에 남아있음.

---

## 참고: 이미 완료된 항목 (중복 작업 방지)

- [x] WorkflowOrchestrator FSM + APScheduler 6잡 (`app/workflow/orchestrator.py`)
- [x] AutoSelector (`app/workflow/auto_selector.py`)
- [x] TradeReviewer (`app/workflow/trade_reviewer.py`)
- [x] FeedbackEngine (`app/workflow/feedback_engine.py`) — 덮어쓰기 버그 수정 완료
- [x] MCP 도구 12개 (`app/mcp/server.py`)
- [x] MCP 거버넌스 (`app/mcp/governance.py`)
- [x] Workflow REST API (`app/routers/workflow.py`) — status/trigger/history/events/best-factors
- [x] workflow_runs, workflow_events, live_feedback DB 테이블 + 마이그레이션
- [x] OpenClaw 크론잡 8개 등록 (morning_brief, pre_market_check, midday_check, post_market_analysis, mining_start_check, mining_review, project_improvement, overnight_check)
- [x] OpenClaw 텔레그램 delivery 설정 (chatId: 7852817441)
- [x] mcporter 글로벌 config + OpenClaw 워크스페이스 config
- [x] MCP SSE 서버 Docker 외부 접근 수정 (host="0.0.0.0")

## 구현 순서 권장

```
✅ C4 (마이그레이션) → ✅ C1 (Context DB) → ✅ C3 (매매로그 DB) → ✅ C2 (세션 복구)
→ ✅ H1 (주문 추적) → ✅ H2 (포트폴리오 MDD) → ✅ H6 (AutoSelector MDD)
→ ✅ H3 (스탈니스) → ✅ H4 (인과 실패 분류) → ✅ H5 (백테스트 타임아웃)
→ ✅ H7 (OpenClaw 헬스) → ✅ H8 (03:00 재시작)
→ M 항목들 (순서 무관)
```
