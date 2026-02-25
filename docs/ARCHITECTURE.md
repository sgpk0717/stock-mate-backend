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

### 주문 제어 (Throttling)
- KIS API 초당 15회 호출 제한 방어.
- `Token Bucket` 알고리즘 기반 속도 제한기를 클래스로 구현.
- 초과 시 큐에 대기시키거나 안전하게 드랍(Drop).

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
| created_at | TIMESTAMPTZ | |

#### stock_ticks (초당 체결/호가 데이터)
| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | BIGSERIAL | |
| ts | TIMESTAMPTZ | 체결 시각 |
| symbol | VARCHAR(20) | 종목코드 |
| price | NUMERIC(18,2) | 체결가 |
| volume | BIGINT | 체결량 |

### DB 최적화
- `stock_ticks` 테이블은 `ts` 기준 테이블 파티셔닝 (월별 또는 주별).
- `symbol` + `ts` 복합 인덱스.

---

## 3. 모의투자 (Paper Trading) 엔진

증권사 서버를 거치지 않고 백엔드에서 100% 동작하는 모의투자 엔진.

### 체결 로직 (Matching Logic)
- FastAPI 메모리 상에 가상 주문 큐 생성.
- **시장가 매수:** 최우선 매도 호가로 강제 체결 (현재가 아님).
- **시장가 매도:** 최우선 매수 호가로 강제 체결.
- **지정가:** 해당 가격에 틱이 도달했을 때만 체결.

### 슬리피지(Slippage) 시뮬레이션
- 시장가 주문 시 0.05초~0.1초 지연 후 체결 확정.
- 실전과 유사한 환경 조성.

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

### WebSocket 실시간 렌더링
1. FastAPI ↔ React WebSocket 연결.
2. 새 틱 발생 시 `{ time: timestamp, value: price }` JSON 전송.
3. `chart.update()` 메서드로 캔들스틱 실시간 갱신.

### 메모리 팽창 방지
- 차트 데이터 배열: 항상 `slice(-1000)` 처리.
- 최근 1,000개 캔들/틱만 유지, 나머지 폐기.

---

## 5. 24시간 무인 구동

### 키움 API 자동 로그인
- `pyautogui` 사용 불가 (화면보호기 상태에서 실패).
- **`pywinauto`** 사용: 윈도우 핸들 직접 제어.
  - 비밀번호를 `Edit2`에 전송, `Button5` 클릭.
  - 백그라운드에서도 동작.

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
    - depends_on: postgres (healthcheck)
    - volume mount for hot-reload

  postgres:   # PostgreSQL 16
    - port 5432
    - named volume: pgdata
    - healthcheck: pg_isready
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
| CORS_ORIGINS | * | 허용 오리진 |
