# Stock Mate - 상세 시스템 설계 문서

스캘핑(초단타)의 핵심인 '초저지연(Low Latency)'과 '데이터 병목 방지'에 초점을 맞춘 아키텍처.

---

## 1. 프로세스 아키텍처 및 통신 설계 (IPC)

키움증권의 32비트 제약 때문에 백엔드는 반드시 2개의 파이썬 프로세스로 분리되며,
두 프로세스 간 통신 속도는 1ms(밀리초) 이하로 유지되어야 한다.

### 수집 프로세스 (Data Pump - Python 32-bit)
- **역할:** 키움증권 Open API+에 접속하여 실시간 틱 데이터(호가, 체결가)만 수신하는 깡통 프로세스.
- **실행 환경:** Windows 전용, Docker 불가. 호스트에서 네이티브 실행.
- **통신:** ZeroMQ (ZMQ) PUB/SUB 패턴.
  - 초당 약 40만 건 이상, 1ms 이하 지연.
  - Redis 대비 압도적 성능.

### 메인 프로세스 (Main Engine - Python 64-bit)
- **역할:** ZMQ를 통해 틱 데이터를 수신, 매매 전략(스캘핑/스윙) 판단, KIS API로 주문 전송.
- **프레임워크:** FastAPI (REST API + WebSocket 서버).
- **실행 환경:** Docker 컨테이너.

### 종목 필터링 파이프라인 (2-Tier)

전체 상장 종목 ~2,000개에서 실시간 매매 대상까지 2단계로 좁힌다.

**Tier 1 — 서버사이드 조건검색 (키움):**
- `SendCondition()` API로 키움 서버에서 사전 필터링.
- 거래량, 등락률, 시가총액 등 기본 조건으로 ~2,000개 → 약 50개 압축.
- 조건검색 결과는 실시간으로 편입/이탈 알림 수신 가능.

**Tier 2 — 로컬 AI 커스텀 조건식:**
- Tier 1 통과 종목 50개에 대해 고급 지표를 로컬에서 실시간 연산.
- **Numba JIT 컴파일** + **NumPy 벡터 연산**으로 마이크로초 단위 계산 달성.
- 연산 대상: 이동평균, 볼린저밴드, RSI, VWAP 등 기술적 지표.
- 패턴 인식: 골든크로스, VWAP 돌파, 호가 스프레드 급변 등.
- 50개 → 최종 매매 대상 (조건 충족 종목만 주문 진입).

### 주문 제어 (Rate Limiter)
- KIS API 초당 15회 호출 제한 방어.
- `Token Bucket` 알고리즘: `capacity=15`, `refill_rate=15/sec`.
- `asyncio` 기반 커스텀 클래스로 구현.
- 큐 대기 시 최대 timeout 3초 — 초과 시 `OrderRejected(reason="rate_limit")` 반환.
- 드랍된 주문은 로그에 기록하여 추후 분석 가능.

---

## 2. 데이터베이스 스키마 (PostgreSQL)

### 테이블 구조

#### accounts (계좌 및 자산 상태)
| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | SERIAL PK | |
| mode | VARCHAR | 'REAL' 또는 'PAPER' |
| total_capital | NUMERIC(18,2) | 초기 투자금 |
| current_balance | NUMERIC(18,2) | 현재 예수금 |

#### positions (현재 보유 종목)
| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | SERIAL PK | |
| symbol | VARCHAR(20) | 종목코드 |
| mode | VARCHAR | 'REAL' 또는 'PAPER' |
| qty | INTEGER | 보유 수량 |
| avg_price | NUMERIC(18,2) | 매입 평균 단가 |

#### orders (주문 및 체결 이력)
| 컬럼 | 타입 | 설명 |
|------|------|------|
| order_id | UUID PK | |
| symbol | VARCHAR(20) | 종목코드 |
| side | VARCHAR | 'BUY', 'SELL' |
| type | VARCHAR | 'MARKET', 'LIMIT' |
| price | NUMERIC(18,2) | 주문/체결 가격 |
| qty | INTEGER | 수량 |
| status | VARCHAR | 'PENDING', 'FILLED', 'PARTIAL', 'CANCELLED', 'REJECTED' |
| mode | VARCHAR | 'REAL' 또는 'PAPER' |
| trade_log | JSONB | 체결 상세 (슬리피지, AI 파라미터 등) |
| created_at | TIMESTAMPTZ | |

`trade_log` JSONB 예시:
```json
{
  "slippage_ms": 75,
  "intended_price": 72800,
  "filled_price": 72900,
  "risk_score": 0.3,
  "strategy": "scalping_vwap"
}
```

#### stock_ticks (초당 체결/호가 데이터)
| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | BIGSERIAL | |
| ts | TIMESTAMPTZ | 체결 시각 |
| symbol | VARCHAR(20) | 종목코드 |
| price | NUMERIC(18,2) | 체결가 |
| volume | BIGINT | 체결량 |

### DB 최적화 (TimescaleDB)

PostgreSQL에 **TimescaleDB 확장**을 적용하여 시계열 데이터를 효율적으로 처리한다.

**Hypertable:**
- `stock_ticks` 테이블을 hypertable로 변환 — 자동 청크(chunk) 분할.
  ```sql
  SELECT create_hypertable('stock_ticks', 'ts');
  ```
- `symbol` + `ts` 복합 인덱스.

**Continuous Aggregate (자동 캔들 생성):**
```sql
CREATE MATERIALIZED VIEW candles_1m
WITH (timescaledb.continuous) AS
SELECT time_bucket('1 minute', ts) AS bucket,
       symbol,
       first(price, ts) AS open,
       max(price) AS high,
       min(price) AS low,
       last(price, ts) AS close,
       sum(volume) AS volume
FROM stock_ticks
GROUP BY bucket, symbol;
```
- 동일 패턴으로 `candles_5m`, `candles_1h` 뷰 추가.
- DB 레벨에서 자동 갱신 — 애플리케이션 코드 불필요.

**데이터 수명 관리:**
- 압축 정책: 7일 이상 원본 틱 자동 압축 (90%+ 저장 공간 절감).
- 보존 정책: 90일 이후 원본 틱 자동 삭제 (캔들 Aggregate는 영구 보존).

---

## 2.5. 캐싱 레이어 (Redis)

최근 틱 데이터를 메모리에 캐싱하여, 실시간 지표 계산 시 DB 조회 없이 즉시 처리한다.

### 자료구조
- **Redis List** (`LPUSH` + `LTRIM`): 종목별 고정 길이 리스트 (FIFO).
  - 키: `tick:{symbol}` → 최근 500개 틱.
  - 새 틱 수신 시: `LPUSH tick:005930 <json>` → `LTRIM tick:005930 0 499`.

### 활용
- Tier 2 지표 계산: 이동평균, VWAP, 호가 스프레드 등.
- DB 부하 없이 최근 100~500개 틱에 즉시 접근.
- 캐시 미스 시 TimescaleDB 폴백.

### Docker 구성
```yaml
redis:
  image: redis:7-alpine
  ports:
    - "6379:6379"
  volumes:
    - redisdata:/data
```

---

## 3. 모의투자 (Paper Trading) 엔진

증권사 서버를 거치지 않고 백엔드에서 100% 동작하는 모의투자 엔진.

### 체결 로직 (Matching Logic)
- FastAPI 메모리 상에 가상 주문 큐 생성.
- **시장가 매수:** 최우선 매도 호가로 강제 체결 (현재가 아님).
- **시장가 매도:** 최우선 매수 호가로 강제 체결.
- **지정가:** 해당 가격에 틱이 도달했을 때만 체결.

### 부분 체결 (Partial Fill)
- 주문 수량이 해당 호가 잔량을 초과할 경우, 잔량만큼만 체결하고 나머지는 대기.
  - 예: 1,000주 매수 주문 → 매도1호가 잔량 300주 → 300주 체결 + 700주 잔여 (status: `PARTIAL`).
- 잔여 수량은 다음 틱에서 재매칭 시도.

### 슬리피지(Slippage) 시뮬레이션
- `asyncio.sleep(random.uniform(0.05, 0.10))` — 50~100ms 인위적 지연.
- 딜레이 동안 가격이 변동했을 경우, **최신 호가로 재매칭** (실전과 동일한 체결가 보장).
- 모든 체결에 JSONB `trade_log` 기록:
  - 슬리피지 소요 시간 (ms), 의도 가격 vs 실제 체결 가격, 전략명.

### 세금 및 수수료 (2026년 기준 하드코딩)
| 항목 | 매수 | 매도 |
|------|------|------|
| 증권거래세 | - | 0.20% (코스피/코스닥 동일, 농특세 포함) |
| KIS API 수수료 | 0.015% | 0.015% |
| **합계** | **0.015%** | **0.215%** |

잔고 계산: `current_balance -= 금액 * 수수료율`

---

## 4. 프론트엔드 차트 아키텍처

### Lightweight Charts (TradingView)
- HTML5 Canvas 기반, 35KB.
- 초당 수십 번 틱 업데이트에도 렌더링 지연 없음.

### 차트 데이터 로드 플로우

**1단계 — 초기 로드 (REST):**
```
GET /api/candles?symbol=005930&interval=1m&limit=100
```
- TimescaleDB `candles_1m` Continuous Aggregate에서 최근 100개 캔들 조회.
- `series.setData(candles)` 로 차트 초기 렌더링.

**2단계 — 실시간 구독 (WebSocket):**
```
ws://host/ws/tick/{symbol}
```
- 새 틱 수신 시 `series.update({ time, open, high, low, close })` 로 최신 캔들 갱신.

### OHLC 변환 로직 (프론트엔드)
- WebSocket raw tick → 현재 1분봉에 merge.
  - `open`: 해당 분의 첫 번째 틱.
  - `high`: `Math.max(current.high, tick.price)`.
  - `low`: `Math.min(current.low, tick.price)`.
  - `close`: 최신 틱 가격.
- 분이 바뀌면 새 캔들 생성, 이전 캔들 확정.

### 메모리 팽창 방지
- 차트 데이터 배열: 항상 `slice(-1000)` 처리.
- 최근 1,000개 캔들/틱만 유지, 나머지 폐기.

---

## 5. 24시간 무인 구동

### 키움 API 자동 로그인
- `pyautogui` 사용 불가 (화면보호기 상태에서 실패).
- **`pywinauto`** 사용: 윈도우 핸들 직접 제어.
  ```python
  app = Application().connect(path="khministarter.exe")
  dlg = app.window(title_re="Open API Login")
  dlg.Edit2.set_text(password)
  dlg.Button5.click()
  ```
  - 화면보호기, 잠금화면 상태에서도 동작 (백그라운드 핸들 제어).
  - 로그인 성공/실패 이벤트로 상태 확인 후 재시도 로직 포함.

### KIS API 토큰 자동 갱신
- 토큰 만료: 24시간.
- `APScheduler` 비동기 스케줄러로 매일 오전 8시(장 시작 전) 자동 갱신.
- FastAPI lifespan 내에서 스케줄러 시작/종료.

---

## 6. Docker 구성

```yaml
services:
  app:        # FastAPI (64-bit Main Engine)
    - port 8007
    - depends_on: postgres, redis (healthcheck)
    - volume mount for hot-reload

  postgres:   # PostgreSQL 16 + TimescaleDB
    - image: timescale/timescaledb:latest-pg16
    - port 5432
    - named volume: pgdata
    - healthcheck: pg_isready

  redis:      # Redis 7 (틱 캐싱)
    - image: redis:7-alpine
    - port 6379
    - named volume: redisdata
```

**Docker에 포함되지 않는 것:**
- 키움 32-bit Data Pump (Windows 네이티브 전용)
- 향후 ZMQ 포트(5555) 매핑 필요

---

## 7. 환경 변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| PORT | 8007 | FastAPI 포트 |
| DEBUG | true | 디버그 모드 |
| POSTGRES_HOST | localhost | Docker 내: postgres |
| POSTGRES_PORT | 5432 | |
| POSTGRES_USER | stockmate | |
| POSTGRES_PASSWORD | stockmate | |
| POSTGRES_DB | stockmate | |
| REDIS_HOST | localhost | Docker 내: redis |
| REDIS_PORT | 6379 | |
| ZMQ_HOST | 127.0.0.1 | Data Pump ZMQ 주소 |
| ZMQ_PORT | 5555 | Data Pump ZMQ 포트 |
| CORS_ORIGINS | * | 허용 오리진 |
