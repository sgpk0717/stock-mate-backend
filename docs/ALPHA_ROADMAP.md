# Stock Mate 2026 퀀트 트렌드 통합 로드맵

## Context

Stock Mate는 현재 수동 지표 조합 기반의 백테스트 + Claude AI 자연어 전략 생성으로 동작한다. 이 로드맵은 시스템이 **스스로 알파 팩터를 발견하고, 인과적으로 검증하며, 가상 시장에서 스트레스 테스트한 뒤, MCP 인프라를 통해 안전하게 실행**하는 자가 진화형 트레이딩 플랫폼으로 격상시킨다.

4단계 순서: **Phase 1 → 2 → 3 → 4** (각 Phase는 이전 Phase 위에 쌓임)

---

## Phase 1: 진화형 알파 탐색 (Evolutionary Alpha Discovery)

### 목표
Claude + PySR(기호회귀)로 **수식 기반의 해석 가능한 알파 팩터**를 자동 발굴하고, IC(Information Coefficient) 기반으로 평가하며, 궤적 수준 변이(Trajectory Mutation)로 지속 진화시킨다.

### 아키텍처

```
[POST /alpha/mine]
     │
     ▼ asyncio.create_task (backtest/runner.py 패턴)
[EvolutionaryAlphaMiner]
     │
     ├─① Claude: 가설 생성 (경험 메모리 참조)
     │   "RSI와 거래량비의 비선형 관계로 과매도 반등 신호 포착"
     │
     ├─② PySR: 기호 회귀 탐색
     │   → SymPy AST: log(volume_ratio) * (30 - rsi) / atr
     │
     ├─③ AST Converter: SymPy → Polars Expression
     │   → pl.col("volume_ratio").log() * (pl.lit(30) - pl.col("rsi")) / pl.col("atr_14")
     │
     ├─④ Evaluator: IC 스크리닝 (Spearman with forward returns)
     │   IC >= 0.03 → PASS, else → 변이 시도
     │
     ├─⑤ 실패 시: Claude 궤적 변이 (AST 부분 노드 수정)
     │   "log() 대신 sqrt()로, atr 대신 bb_width로 교체 제안"
     │   → 재평가 (최대 5회 반복)
     │
     └─⑥ 성공 시: DB 저장 + 경험 메모리 업데이트
```

### 신규 의존성

```
# requirements.txt 추가
pysr>=0.19.0          # 기호 회귀 (Julia 런타임 필요)
sympy>=1.13.0         # 수학적 표현식 AST
juliacall>=0.9.0      # PySR의 Julia 브릿지
scipy>=1.14.0         # Spearman 상관계수 (IC 계산)
```

> **PySR/Julia 설치**: 첫 `import pysr` 시 Julia 자동 다운로드(~400MB). Docker에서는 빌드 시 `python -c "import pysr; pysr.install()"` 사전 실행. 로컬 개발 시 `use_pysr=False` 플래그로 Claude-only 모드 fallback 지원.

### 환경변수 추가 (`app/core/config.py`)

```python
ALPHA_MAX_PYSR_ITERATIONS: int = 40     # PySR 탐색 세대 수
ALPHA_IC_THRESHOLD_PASS: float = 0.03   # IC 최소 통과 기준
ALPHA_IC_THRESHOLD_GOOD: float = 0.05   # "우수" 팩터 기준
ALPHA_MAX_MUTATION_DEPTH: int = 5       # 변이 재귀 제한
ALPHA_PYSR_TIMEOUT_SECONDS: int = 300   # PySR 타임아웃
```

### DB 스키마 (Alembic 마이그레이션 `e7f8a9b0c1d2`)

**`alpha_mining_runs`** — 마이닝 실행 기록 (BacktestRun 패턴 동일)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | UUID PK | 실행 ID |
| name | String(200) | 실행명 |
| context | JSON | Claude에 전달된 시장 맥락 |
| config | JSON | PySR 설정 + 임계값 |
| status | String | PENDING / RUNNING / COMPLETED / FAILED |
| progress | Integer | 0-100 |
| factors_found | Integer | 통과한 팩터 수 |
| total_evaluated | Integer | 평가한 수식 수 |
| error_message | Text | 실패 시 에러 |
| created_at / completed_at | DateTime | 시간 |

**`alpha_factors`** — 발견된 알파 팩터

| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | UUID PK | 팩터 ID |
| mining_run_id | UUID FK | 마이닝 실행 참조 |
| name | String(200) | 팩터명 |
| expression_str | Text | 사람 읽기용 수식 (`log(vol) * (30 - rsi) / atr`) |
| expression_sympy | Text | SymPy 직렬화 (srepr) |
| polars_code | Text | Polars Expression 코드 문자열 |
| hypothesis | Text | Claude의 원본 가설 |
| generation | Integer | 몇 세대 변이인지 |
| ic_mean | Float | 평균 IC (Spearman) |
| ic_std | Float | IC 표준편차 |
| icir | Float | IC / IC_std (Information Ratio) |
| turnover | Float | 팩터 턴오버 |
| sharpe | Float | 팩터 롱숏 Sharpe |
| max_drawdown | Float | 최대 낙폭 |
| status | String | discovered / validated / mirage / deployed |
| causal_robust | Boolean | Phase 2에서 설정 |
| causal_effect_size | Float | Phase 2에서 설정 |
| causal_p_value | Float | Phase 2에서 설정 |
| created_at / updated_at | DateTime | 시간 |

인덱스: `mining_run_id`, `status`, `ic_mean`

### 신규 백엔드 파일

#### `app/alpha/__init__.py`
빈 모듈 init.

#### `app/alpha/models.py`
SQLAlchemy 모델 2개 (AlphaMiningRun, AlphaFactor). `backtest/models.py`의 BacktestRun 패턴 따름.

#### `app/alpha/schemas.py`
Pydantic V2 스키마:
- `AlphaMineRequest` — POST /alpha/mine 요청 (name, context, symbols, dates, max_iterations, ic_threshold, use_pysr)
- `AlphaMineResponse` — 즉시 응답 (id, status, created_at)
- `AlphaFactorResponse` — 팩터 상세 (모든 메트릭 + causal 필드)
- `AlphaMiningRunResponse` — 실행 상태/진행률
- `AlphaFactorBacktestRequest` — 팩터로 백테스트 (factor_id, buy/sell threshold, dates)
- `CausalValidationResponse` — Phase 2 인과 검증 결과

#### `app/alpha/ast_converter.py` — 핵심
SymPy AST → Polars Expression 재귀 변환기.

**변수 매핑** (PySR 변수 → Polars 컬럼):
```python
VARIABLE_MAP = {
    "x0": "close", "x1": "open", "x2": "high", "x3": "low", "x4": "volume",
    "x5": "sma_20", "x6": "rsi", "x7": "volume_ratio", "x8": "atr_14",
    "x9": "macd_hist", "x10": "bb_upper", "x11": "bb_lower", "x12": "price_change_pct",
}
```

**지원 SymPy 노드**:
| SymPy 노드 | Polars 변환 |
|-----------|-----------|
| `Symbol("x0")` | `pl.col("close")` |
| `Integer(5)` / `Float(0.5)` | `pl.lit(5.0)` |
| `Add(a, b)` | `a + b` |
| `Mul(a, b)` | `a * b` |
| `Pow(x, 0.5)` | `x.sqrt()` |
| `Pow(x, n)` | `x.pow(n)` |
| `log(x)` | `x.log()` |
| `exp(x)` | `x.exp()` |
| `Abs(x)` | `x.abs()` |

함수:
- `sympy_to_polars(expr) → pl.Expr` — 재귀 변환
- `sympy_to_code_string(expr) → str` — DB 저장용 코드 문자열
- `ensure_alpha_features(df) → pl.DataFrame` — PySR 변수용 기저 지표 추가 (기존 `indicators.py`의 `add_indicator()` 재사용)

#### `app/alpha/evaluator.py`
IC 및 팩터 메트릭 평가:
- `compute_forward_returns(df, periods=1)` — T+1 수익률 컬럼 추가
- `compute_ic_series(df, factor_col)` — 일별 cross-sectional Spearman IC
- `compute_factor_metrics(ic_series)` — ic_mean, ic_std, icir, turnover, sharpe, mdd 집계
- `evaluate_factor(df, factor_expr, name)` — 팩터 하나의 전체 평가 파이프라인

> IC 계산에 `scipy.stats.spearmanr` 사용. 데이터 전처리는 Polars, 상관계수만 scipy.

#### `app/alpha/miner.py` — 핵심
EvolutionaryAlphaMiner 클래스:

```
class ExperienceMemory:
    """인메모리 + DB 백업 경험 저장소.
    성공/실패 팩터의 수식, IC, 세대를 추적.
    Claude 프롬프트에 포함하여 중복 탐색 방지."""

    add(expr_str, ic, generation, success)
    top_k(k=10) → 상위 성공 팩터
    failures(k=5) → 최근 실패 팩터
    format_for_prompt() → Claude 프롬프트용 문자열

class EvolutionaryAlphaMiner:
    """진화 루프 오케스트레이터.

    async run(progress_cb):
        Phase A: 데이터 준비 (load_candles + ensure_alpha_features)
        Phase B: Claude 가설 생성 (경험 메모리 참조)
        Phase C: PySR 탐색 (가설당 Pareto front 수식 목록)
        Phase D: IC 평가 (Polars 네이티브)
        Phase E: 실패 시 Claude 변이 (AST 부분 수정)
        Phase F: C-E 반복 (max_iterations)
    """
```

**Claude 가설 생성 프롬프트 핵심**:
- 사용 가능한 OHLCV + 12개 기저 지표 설명
- 경험 메모리: 상위 5개 성공 패턴 + 최근 5개 실패 패턴
- "기존 성공 팩터와 직교성(orthogonality) 유지" 지시
- 출력: 자연어 가설 (경제적 근거 포함)

**Claude 변이 프롬프트 핵심**:
- 실패한 수식의 AST 트리 구조 + IC 점수 제공
- "노이즈를 유발하는 하위 노드를 국소적으로 식별하여 수정" 지시
- 출력: 수정된 SymPy 호환 수식 문자열

**PySR fallback**: `use_pysr=False` 시 Claude가 직접 SymPy 수식 문자열 생성 → `sympy.sympify()` 파싱. Julia 없이도 동작하지만 탐색 폭이 좁아짐.

#### `app/alpha/runner.py`
비동기 실행기 (backtest/runner.py 패턴 동일):
1. DB에 RUNNING 상태 기록
2. EvolutionaryAlphaMiner.run() 실행 + progress_cb → WebSocket
3. 결과 DB 저장, COMPLETED/FAILED 상태 전환

#### `app/alpha/backtest_bridge.py`
발견된 팩터 → 기존 백테스트 통합:
- `register_alpha_factor(factor)` — `indicators._INDICATOR_FN`에 동적 등록
- `add_alpha_indicator(df, factor_id)` — Polars DF에 팩터 컬럼 추가
- 등록 후 `ConditionSchema(indicator="alpha_<uuid_short>", op=">", value=0.0)` 가능

#### `app/routers/alpha.py`
REST API:

| Method | Path | 설명 |
|--------|------|------|
| POST | `/alpha/mine` | 마이닝 시작 (202 반환) |
| GET | `/alpha/mine/{run_id}` | 실행 상태/진행률 |
| GET | `/alpha/mines` | 실행 목록 |
| GET | `/alpha/factors` | 팩터 목록 (status/min_ic 필터) |
| GET | `/alpha/factor/{id}` | 팩터 상세 |
| POST | `/alpha/factor/{id}/backtest` | 팩터로 백테스트 실행 |
| POST | `/alpha/factor/{id}/validate` | 인과 검증 트리거 (Phase 2) |
| DELETE | `/alpha/factor/{id}` | 팩터 삭제 |
| DELETE | `/alpha/mine/{run_id}` | 실행 삭제 |

### 기존 파일 수정

| 파일 | 변경 내용 |
|------|---------|
| `app/core/config.py` | ALPHA_* 환경변수 5개 추가 |
| `app/main.py` | `app.include_router(alpha.router)` 추가 |
| `app/backtest/indicators.py` | `register_custom_indicator(name, fn)` 함수 추가 — 동적 지표 등록 훅 |
| `app/backtest/engine.py` | `_build_condition_expr()`에서 `alpha_*` 접두사 지표 인식 추가 |
| `requirements.txt` | pysr, sympy, juliacall, scipy 추가 |

### 신규 프론트엔드 파일

| 파일 | 설명 |
|------|------|
| `types/alpha.ts` | AlphaFactor, AlphaMiningRun, AlphaMineRequest 등 인터페이스 |
| `api/alpha.ts` | API 함수 8개 (startMining, fetchFactors, backtestWithFactor 등) |
| `hooks/queries/use-alpha.ts` | TanStack Query 훅 8개 (useMutation/useQuery 패턴) |
| `pages/AlphaLabPage.tsx` | 메인 페이지 (2컬럼 그리드, BacktestPage 패턴) |
| `components/alpha/AlphaMineConfig.tsx` | 마이닝 설정 폼 (날짜/종목/PySR 토글) |
| `components/alpha/AlphaMineProgress.tsx` | 진행률 바 (WebSocket) |
| `components/alpha/AlphaFactorTable.tsx` | 팩터 목록 테이블 (IC/ICIR/Sharpe 정렬) |
| `components/alpha/AlphaFactorDetail.tsx` | 수식 + 메트릭 + 인과 배지 |
| `components/alpha/AlphaICChart.tsx` | IC 시계열 차트 (Lightweight Charts area) |
| `components/alpha/AlphaMineHistory.tsx` | 마이닝 실행 이력 |

기존 프론트 수정:
- `App.tsx` — `/alpha` 라우트 추가
- `Sidebar.tsx` — "알파 탐색" 메뉴 추가
- `api/index.ts`, `hooks/queries/index.ts` — re-export 추가

### 구현 순서 (8단계)

1. DB 모델 + Alembic 마이그레이션 (alpha_mining_runs, alpha_factors)
2. Pydantic 스키마 + 환경변수 (config.py)
3. SymPy→Polars AST 변환기 (ast_converter.py)
4. IC 평가기 (evaluator.py)
5. 코어 마이너 + 러너 (miner.py, runner.py)
6. 백테스트 브릿지 + 기존 파일 수정 (backtest_bridge.py, indicators.py, engine.py)
7. API 라우터 + main.py 등록
8. 프론트엔드 (types, api, hooks, 페이지, 컴포넌트)

### 설계 vs 구현 차이점

> 아래는 설계 시점의 의도와 실제 구현 결과가 달라진 항목이다.
> 모든 변경은 의도적이며, 설계보다 실용적인 방향으로 조정되었다.

#### 1. PySR / Julia 의존성 미포함 (Claude-only 모드 채택)

**설계**: `pysr>=0.19.0`, `juliacall>=0.9.0`을 `requirements.txt`에 추가하여, PySR 기호 회귀 엔진이 Julia 런타임 위에서 Pareto-optimal 수식을 자동 탐색한다. Claude는 가설 생성과 변이만 담당.

**실제 구현**: PySR과 juliacall을 포함하지 않았다. `miner.py`의 `EvolutionaryAlphaMiner`는 `use_pysr=False`가 기본값이며, Claude가 가설 생성부터 SymPy 호환 수식 생성까지 직접 수행한다. PySR 호출 코드 자체가 존재하지 않는다.

**변경 사유**:
- Julia 런타임 설치가 ~400MB로 무겁고, Docker 이미지 빌드 시간이 크게 증가한다
- PySR 첫 실행 시 Julia 패키지 프리컴파일이 추가로 5~10분 소요된다
- Claude-only 모드로도 `log(volume_ratio) * (30 - rsi) / atr` 같은 해석 가능한 수식을 충분히 생성할 수 있음을 확인하였다
- 향후 PySR이 필요해지면 `use_pysr=True` 플래그와 함께 활성화할 수 있도록 아키텍처는 유지하였다

**영향 범위**: `requirements.txt`, `app/alpha/miner.py` (Phase C 단계 생략)

#### 2. 변수 매핑 확장 (`NAMED_VARIABLE_MAP` 추가)

**설계**: PySR은 `x0`, `x1`, ... `x12` 형태의 인덱스 변수명을 사용하므로, `VARIABLE_MAP`에서 `{"x0": "close", "x1": "open", ...}` 13개 매핑만 정의한다.

**실제 구현**: PySR 없이 Claude가 직접 수식을 생성하므로, Claude는 `"rsi"`, `"volume_ratio"`, `"atr"` 같은 사람 읽기 이름을 사용한다. 이를 처리하기 위해 `ast_converter.py`에 `NAMED_VARIABLE_MAP`을 추가하였다.

```python
# ast_converter.py
NAMED_VARIABLE_MAP = {
    "close", "open", "high", "low", "volume",
    "sma", "sma_20", "rsi", "vol_ratio", "volume_ratio",
    "atr", "atr_14", "macd", "macd_hist",
    "bb_upper", "bb_lower", "bb_width",
    "price_change_pct", "pct_change", "ema_20",
}
```

`parse_expression()`이 SymPy 파싱 시 이 이름들을 유효한 심볼로 인식하고, `sympy_to_polars()` 변환 시 해당 Polars 컬럼명으로 매핑한다. 기존 `VARIABLE_MAP`(x0-x12)도 그대로 유지하여 향후 PySR 활성화 시 호환된다.

**영향 범위**: `app/alpha/ast_converter.py` (L30-45)

#### 3. EMA 지표 추가

**설계**: 기저 지표로 SMA(20일)만 포함. `x5 = sma_20`.

**실제 구현**: `ema_20`을 기저 지표에 추가하였다. `ensure_alpha_features()` 함수가 `indicators.add_ema(df, period=20)` 호출. NAMED_VARIABLE_MAP에도 `"ema_20"` 포함.

**변경 사유**: EMA(지수이동평균)는 SMA(단순이동평균)보다 최근 가격에 가중치를 많이 주어 모멘텀/추세 팩터 생성에 유리하다. Claude가 수식에 EMA를 사용할 수 있도록 기저 지표를 확대하였다.

**영향 범위**: `app/alpha/ast_converter.py` (NAMED_VARIABLE_MAP), `ensure_alpha_features()` 함수

#### 4. Turnover 메트릭 재정의 (포트폴리오 턴오버 → Signal Flip Rate)

**설계**: `turnover` = 포트폴리오 턴오버. 팩터 기반 포트폴리오 구성이 날짜별로 얼마나 바뀌는지 측정. 퀀트 업계 표준 정의.

**실제 구현**: `turnover` = Signal Flip Rate (IC 부호 전환 비율). 일별 IC 시리즈에서 양→음, 음→양으로 부호가 바뀐 비율을 계산한다.

```python
# evaluator.py (L161-163)
sign_changes = (ic_series[1:] * ic_series[:-1]) < 0
turnover = float(sign_changes.sum()) / max(len(sign_changes), 1)
```

**변경 사유**: Phase 1은 팩터 "발견" 단계이므로, 실제 포트폴리오를 구성하지 않는다. 포트폴리오 턴오버를 계산하려면 종목별 랭킹 → 포트폴리오 구성 → 리밸런싱 시뮬레이션이 필요한데, 이는 Phase 1의 범위를 초과한다. 대신, IC 부호 안정성(팩터 신호가 자주 뒤집히지 않는지)으로 팩터 품질을 근사하였다. 부호 전환이 적을수록 안정적인 팩터이다.

**영향 범위**: `app/alpha/evaluator.py` (compute_factor_metrics 내부)

#### 5. Sharpe 메트릭 재정의 (롱숏 Sharpe → IC Sharpe)

**설계**: `sharpe` = 팩터 롱숏 Sharpe. 팩터 상위 10% 매수 + 하위 10% 매도 포트폴리오의 연환산 Sharpe Ratio.

**실제 구현**: `sharpe` = IC Sharpe. 누적 IC 일변화량의 평균/표준편차 × √252.

```python
# evaluator.py (L147-154)
cumulative_ic = np.cumsum(ic_series)
daily_changes = np.diff(cumulative_ic)
if len(daily_changes) > 0 and np.std(daily_changes) > 1e-12:
    sharpe = float(np.mean(daily_changes) / np.std(daily_changes) * np.sqrt(252))
```

**변경 사유**: Turnover와 같은 이유. 롱숏 포트폴리오 시뮬레이션은 Phase 1 범위 초과. IC의 일간 변화가 안정적이면(양의 IC가 꾸준히 유지) Sharpe가 높게 나온다. 이는 "팩터가 일관되게 수익률을 예측하는가"를 측정하므로, 팩터 품질 평가에 충분히 적합하다.

**영향 범위**: `app/alpha/evaluator.py` (compute_factor_metrics 내부)

#### 6. IC 계산에 단일 종목 롤링 윈도우 fallback 추가

**설계**: IC = 날짜별 횡단면(cross-sectional) Spearman 상관계수. 같은 날짜에 여러 종목의 (팩터값, T+1 수익률) 쌍으로 상관계수 계산.

**실제 구현**: 다종목 마이닝 시 설계대로 cross-sectional IC를 계산한다. 추가로, **단일 종목**(`symbols=["005930"]`)으로 마이닝할 경우 같은 날짜에 종목이 1개뿐이므로 횡단면 IC를 계산할 수 없다. 이 경우 **20일 롤링 윈도우** 내에서 시계열 Spearman IC를 계산하는 fallback 경로를 추가하였다.

```python
# evaluator.py (L64-96)
if len(symbols) > 1:
    # 날짜별 cross-sectional IC
    grouped = df.group_by("dt")  # ...
else:
    # 단일 종목: 20일 롤링 윈도우 IC
    for i in range(window, len(df)):
        window_slice = df[i-window:i]
        ic, _ = spearmanr(window_slice["alpha_factor"], window_slice["fwd_return"])
```

**변경 사유**: 사용자가 특정 종목 1개에 대해서만 팩터를 탐색하는 유스케이스가 실제로 발생하였다. 횡단면 IC가 불가능한 상황에서 아무 결과도 반환하지 않는 것보다, 시계열 IC라도 제공하는 것이 실용적이다.

**영향 범위**: `app/alpha/evaluator.py` (compute_ic_series 내부 분기)

#### 7. Claude 수식 추출 정규식 강화

**설계**: Claude가 SymPy 호환 수식 문자열을 반환하면 `sympy.sympify()`로 파싱한다. 별도 추출 로직은 명시하지 않았다.

**실제 구현**: Claude의 응답 형식이 일관되지 않아, 다양한 패턴에서 수식을 추출하는 정규식 로직을 `miner.py`에 추가하였다.

```python
# miner.py (L155-185) — _extract_expression()
# 패턴 1: "수식: log(volume_ratio) * (30 - rsi)"
# 패턴 2: ```python\nlog(volume_ratio) * (30 - rsi)\n```
# 패턴 3: 줄 끝에 한글 설명이 붙은 경우 → 제거
# 패턴 4: 여러 줄에 걸친 수식 → 줄바꿈 연결
```

추가로 `_clean_expression()` 함수가 후행 주석(`# 이 수식은...`), 한글 설명(`여기서 rsi는...`), 마크다운 서식 등을 제거한다.

**변경 사유**: Claude는 자연어 AI이므로, "수식만 출력하라"고 지시해도 설명을 덧붙이거나, 코드블록으로 감싸거나, 줄바꿈을 넣는 경우가 빈번하다. 수식 추출의 robust성이 마이닝 성공률에 직접 영향을 주므로, 여러 패턴에 대응하는 파서가 필수적이었다.

**영향 범위**: `app/alpha/miner.py` (_extract_expression, _clean_expression)

#### 8. AlphaICChart.tsx 미구현

**설계**: `components/alpha/AlphaICChart.tsx` — IC 시계열을 Lightweight Charts area 차트로 시각화하는 컴포넌트.

**실제 구현**: 이 컴포넌트는 구현되지 않았다. `AlphaFactorDetail.tsx`에서 IC 관련 수치(ic_mean, ic_std, icir)는 텍스트로 표시하지만, 시계열 차트는 없다.

**변경 사유**: Phase 1의 핵심 기능은 "팩터 발견 + IC 스크리닝"이며, IC 시계열 시각화는 UX 개선 항목이다. 핵심 파이프라인 구현을 우선하였으며, 이 컴포넌트는 `docs/PHASE1_IMPROVEMENTS.md`에 향후 구현 항목으로 기록하였다.

**영향 범위**: `src/components/alpha/` (파일 미존재), `AlphaFactorDetail.tsx` (차트 연동 코드 없음)

---

## Phase 2: 인과 추론 파이프라인 (Causal Inference)

### 목표
Phase 1에서 IC를 통과한 팩터에 대해 DoWhy 4단계 인과 검증(모델링→식별→추정→반증)을 수행하여 **팩터 신기루(Factor Mirage)**를 제거한다.

### 아키텍처

```
[AlphaFactor (IC >= 0.03)]
          │
          ▼
[FactorMirageFilter.validate()]
          │
   ┌──────┼──────────────────────────┐
   │      │                          │
   ▼      ▼                          ▼
Step 1  Step 2                    교란 변수
DAG     식별(Identify)            로더
구축    Backdoor Criterion         │
   │                              ├─ KOSPI ETF(069500) 수익률/변동성
   ▼                              ├─ BOK 기준금리 (정적 JSON)
Step 3: 추정                      └─ 섹터 ID (stock_masters.sector)
Linear Regression
   │
   ▼
Step 4: 반증
   ├─ Placebo Treatment: 팩터를 랜덤 노이즈로 대체 → 효과가 0 수렴해야 통과
   └─ Random Common Cause: 랜덤 교란 변수 추가 → 효과 변동 없어야 통과
          │
          ▼
   is_causally_robust?
   ├─ True  → status = "validated"
   └─ False → status = "mirage"
```

### 신규 의존성

```
dowhy>=0.11.0         # 인과 추론 (4단계 검증)
networkx>=3.2.0       # DAG 그래프 (DoWhy 의존)
```

> **Polars↔pandas 브릿지**: DoWhy는 pandas 필수. 인과 검증 시에만 `.to_pandas()` 변환 (팩터당 ~500행 수준, 메인 파이프라인은 100% Polars 유지).

### 환경변수 추가

```python
CAUSAL_PLACEBO_THRESHOLD: float = 0.05      # 플라시보 효과 허용 상한
CAUSAL_RANDOM_CAUSE_THRESHOLD: float = 0.05 # 랜덤 교란 효과 변동 허용 상한
CAUSAL_NUM_SIMULATIONS: int = 100           # 반증 시뮬레이션 횟수
```

### 신규 백엔드 파일

#### `app/alpha/confounders.py`
교란 변수 데이터 로더:
- `load_confounders(start, end) → pd.DataFrame` — 모든 교란 변수 병합
- `_load_market_data()` — KOSPI ETF(069500) 일봉에서 수익률 + 20일 실현 변동성 계산. 기존 `data_loader.load_candles()` 재사용
- `_load_base_rate()` — `data/bok_base_rate.json` 읽어서 일별 전방 충전(forward-fill)
- `_load_sector_mapping()` — `stock_masters.sector` 에서 심볼→섹터 정수 ID 매핑

#### `app/alpha/causal.py` — 핵심
```python
class FactorMirageFilter:
    """DoWhy 4단계 인과 검증.

    validate(factor_values, forward_returns, dates) → dict:
        is_causally_robust: bool
        causal_effect_size: float
        p_value: float
        placebo_passed: bool
        placebo_effect: float
        random_cause_passed: bool
        random_cause_delta: float
        dag_edges: list  # 프론트 시각화용
    """
```

**DAG 구조** (고정):
```
market_return ──→ alpha_factor ──→ forward_return ←── market_return
market_volatility ──→ alpha_factor ──→ forward_return ←── market_volatility
base_rate ──────────────────────────→ forward_return
sector_id ──→ alpha_factor ──→ forward_return ←── sector_id
```

**추정 방법**: `backdoor.linear_regression` (v1), 향후 `backdoor.econml.dml.DML`로 업그레이드 가능.

**반증 기준**:
- `placebo_passed`: `abs(placebo_new_effect) < 0.05` — 플라시보 효과가 0 근처여야
- `random_cause_passed`: `abs(new_effect - original_effect) < 0.05` — 랜덤 교란에 흔들리지 않아야

#### `app/alpha/causal_runner.py`
비동기 배치 검증기:
1. 교란 변수 1회 로드 (전체 팩터 공유)
2. 팩터별: 팩터값 재계산 → FactorMirageFilter.validate()
3. DB 업데이트: `alpha_factors.status`, `causal_*` 필드

### 정적 데이터 파일

**`data/bok_base_rate.json`** — 한국은행 기준금리 이력:
```json
[
  {"date": "2020-05-28", "rate": 0.50},
  {"date": "2021-08-26", "rate": 0.75},
  {"date": "2022-04-14", "rate": 1.50},
  {"date": "2023-01-13", "rate": 3.50},
  {"date": "2024-10-11", "rate": 3.25},
  {"date": "2025-11-28", "rate": 2.75}
]
```

### Phase 1과의 통합 지점

1. **자동 후처리**: `alpha/runner.py`에서 마이닝 완료 후 IC 통과 팩터에 대해 자동으로 `validate_factors_causally()` 호출
2. **수동 검증 API**: `POST /alpha/factor/{id}/validate` — 개별 팩터 재검증
3. **상태 전이**: `discovered` → (IC pass) → (causal pass) → `validated` / (causal fail) → `mirage`

### 프론트엔드 추가

| 파일 | 설명 |
|------|------|
| `components/alpha/CausalBadge.tsx` | 상태 배지 — "검증됨"(green), "미라지"(red), "미검증"(gray) + 툴팁(effect_size, p_value) |
| `components/alpha/CausalDAGView.tsx` | DAG 시각화 — 순수 SVG, 6노드 8엣지 고정 레이아웃, pass/fail 컬러링 |

`AlphaFactorDetail.tsx`에 CausalBadge + CausalDAGView 통합.
`AlphaFactorTable.tsx`에 "검증 상태" 컬럼 추가.

### 구현 순서 (5단계)

1. DoWhy 의존성 + 환경변수 추가
2. 교란 변수 로더 (confounders.py) + BOK 기준금리 JSON
3. FactorMirageFilter 인과 검증기 (causal.py)
4. 비동기 배치 검증기 + runner.py 통합 (causal_runner.py)
5. 프론트엔드 (CausalBadge, CausalDAGView, 테이블 컬럼)

### 설계 vs 구현 차이점

> 아래는 설계 시점의 의도와 실제 구현 결과가 달라진 항목이다.
> 모든 변경은 구현 과정 및 전문가 검증(코드 리뷰)을 거쳐 의도적으로 조정되었다.

#### 1. `CAUSAL_AUTO_VALIDATE` 환경변수 추가

**설계**: 마이닝 완료 후 IC를 통과한 팩터에 대해 **무조건** 자동으로 `validate_factors_causally()`를 호출한다. 별도 on/off 설정 없음.

**실제 구현**: `app/core/config.py`에 `CAUSAL_AUTO_VALIDATE: bool = False` 환경변수를 추가하였다. `app/alpha/runner.py`의 마이닝 완료 후 자동 검증 훅이 이 플래그가 `True`일 때만 실행된다.

```python
# runner.py (L121-135)
if settings.CAUSAL_AUTO_VALIDATE and len(discovered) > 0:
    try:
        from app.alpha.causal_runner import validate_factors_batch
        async with async_session() as causal_db:
            validated_count = await validate_factors_batch(run_id, causal_db)
    except Exception as e:
        logger.warning("Auto causal validation failed for run %s: %s", run_id, e)
```

**변경 사유**: DoWhy 인과 검증은 팩터당 수십 초~수 분이 소요되는 CPU-bound 작업이다. 팩터가 10개 발견되면 자동 검증에 10분 이상 걸릴 수 있어, 마이닝 자체의 완료가 크게 지연된다. 운영 환경에서는 마이닝 완료를 먼저 확인하고, 필요한 팩터만 수동(`POST /alpha/factor/{id}/validate`)으로 검증하는 워크플로우가 더 실용적이다. `CAUSAL_AUTO_VALIDATE=true`로 설정하면 설계 원안대로 자동 실행된다.

**영향 범위**: `app/core/config.py` (L75), `app/alpha/runner.py` (L121-135)

#### 2. `load_sector_mapping()` public 함수로 전환

**설계**: `_load_sector_mapping()` — private 함수(`_` 접두사). `load_confounders()` 내부에서만 호출되는 헬퍼 함수.

**실제 구현**: `load_sector_mapping()` — public async 함수. `_` 접두사 없음. `confounders.py`에서 export되며, `causal_runner.py`에서도 직접 import하여 호출한다.

```python
# confounders.py (L131)
async def load_sector_mapping(symbols: list[str] | None = None) -> dict[str, int]:

# causal_runner.py (L20)
from app.alpha.confounders import load_confounders, load_sector_mapping
```

**변경 사유**: 설계 시에는 `load_confounders()` 내부에서 sector_id를 포함한 전체 교란 변수 DataFrame을 한 번에 반환하려 했다. 하지만 실제 구현 과정에서 sector_id는 **종목별(per-symbol)** 데이터인 반면, 나머지 교란 변수(market_return, market_volatility, base_rate)는 **날짜별(per-date)** 데이터라는 차원 불일치가 발생하였다. sector_id를 별도로 로드하여 `causal_runner.py`에서 팩터 데이터의 symbol 컬럼과 직접 매핑하는 것이 데이터 정합성 면에서 올바른 접근이다.

추가로, SQLAlchemy 커넥션 풀을 재사용하도록 변경하였다(원래 설계는 `asyncpg.connect()` 직접 연결).

**영향 범위**: `app/alpha/confounders.py` (L131-172), `app/alpha/causal_runner.py` (L20, L108-120)

#### 3. `validate()` 시그니처 및 반환 타입 변경

**설계**:
```python
def validate(factor_values, forward_returns, dates) -> dict:
    # dates: 날짜 배열 → 내부에서 교란 변수 로드
    # 반환: {"is_causally_robust": bool, "causal_effect_size": float, ...}
```

**실제 구현**:
```python
def validate(
    self,
    factor_values: np.ndarray,
    forward_returns: np.ndarray,
    confounders_df: pd.DataFrame,  # dates 대신 교란 변수 DataFrame 직접 전달
) -> CausalValidationResult:       # dict 대신 dataclass 반환
```

**변경 사유 (시그니처)**:
설계에서는 `dates` 파라미터를 받아 `validate()` 내부에서 교란 변수를 로드하려 했다. 하지만 이렇게 하면:
- `validate()` 안에서 비동기 DB 호출이 필요해져 `async def`가 되어야 하지만, DoWhy 자체가 동기 API이므로 `asyncio.to_thread()`와 충돌
- 배치 검증 시 동일 교란 변수를 팩터마다 중복 로드

대신 **호출자(`causal_runner.py`)가 교란 변수를 미리 로드하여 전달**하는 방식을 채택하였다. 이로써 `validate()`는 순수 동기 함수로 유지되고, 배치 검증 시 교란 변수를 1회만 로드할 수 있다.

**변경 사유 (반환 타입)**:
dict 대신 `@dataclass CausalValidationResult`를 사용하면:
- IDE 자동완성 및 타입 검사 지원
- 필드 누락 방지 (dict는 키 오타 시 런타임 에러)
- `dag_edges` 기본값을 `field(default_factory=...)`로 자동 할당

```python
# causal.py (L109-120)
@dataclass
class CausalValidationResult:
    is_causally_robust: bool
    causal_effect_size: float
    p_value: float
    placebo_passed: bool
    placebo_effect: float
    random_cause_passed: bool
    random_cause_delta: float
    dag_edges: list[dict] = field(default_factory=lambda: list(DAG_EDGES))
```

**영향 범위**: `app/alpha/causal.py` (L109-120, L136-180), `app/alpha/causal_runner.py` (L115-120)

#### 4. BOK 기준금리 JSON 확장 (6 → 14개 항목)

**설계**: 주요 금리 변경 이벤트 6개만 포함:
```json
[
  {"date": "2020-05-28", "rate": 0.50},
  {"date": "2021-08-26", "rate": 0.75},
  {"date": "2022-04-14", "rate": 1.50},
  {"date": "2023-01-13", "rate": 3.50},
  {"date": "2024-10-11", "rate": 3.25},
  {"date": "2025-11-28", "rate": 2.75}
]
```

**실제 구현**: 2020~2025 기간의 **모든** 한국은행 기준금리 변경 이벤트 14개를 포함. 예를 들어 2021-11-25(1.00%), 2022-01-14(1.25%), 2022-05-26(1.75%) 등 중간 인상 단계도 모두 기록하였다.

**변경 사유**: 교란 변수로서의 기준금리는 forward-fill(전방 충전) 방식으로 일별 시계열을 생성한다. 중간 이벤트가 누락되면 예를 들어 2021-08-26(0.75%) → 2022-04-14(1.50%) 사이 8개월 동안 실제로는 0.75 → 1.00 → 1.25 → 1.50으로 단계적 인상이 있었는데, 이를 한 번에 0.75 → 1.50으로 점프하는 것으로 잘못 반영한다. 모든 이벤트를 포함해야 교란 변수의 시계열 정확도가 보장된다.

**영향 범위**: `data/bok_base_rate.json`

#### 5. 최소 샘플 수 상향 (30 → 100)

**설계**: 최소 샘플 수를 명시하지 않았으며, 초기 구현에서는 `n < 30`일 때 검증을 건너뛰었다.

**실제 구현**: 전문가 검증(코드 리뷰) 후 `_MIN_SAMPLES = 100`으로 상향하였다.

```python
# causal.py (L61)
_MIN_SAMPLES = 100  # 6변수 선형회귀에 최소 100개 (≈ 변수당 15-17개)
```

**변경 사유**: DoWhy의 `backdoor.linear_regression` 추정에서 사용하는 변수는 6개(treatment: `alpha_factor`, outcome: `forward_return`, 교란 변수: `market_return`, `market_volatility`, `base_rate`, `sector_id`)이다. 통계학의 경험 법칙(rule of thumb)에 따르면 선형회귀에서 **변수당 최소 10~20개** 관측치가 필요하다. 6변수 × 15 ≈ 90, 여유를 두어 100으로 설정하였다. 30행에서는 회귀 계수의 표준 오차가 크고, p-value의 검정력(statistical power)이 낮아 "의미 있는" 인과 효과도 통계적으로 유의하지 않게 나올 수 있다.

추가로, 반증 테스트(Placebo, Random Common Cause)에서 `num_simulations=100`회 시뮬레이션을 돌리므로, 원본 데이터가 너무 적으면 시뮬레이션 결과의 분산이 커져 false positive/negative가 증가한다.

**영향 범위**: `app/alpha/causal.py` (L61, L196, L231), `tests/alpha/test_causal.py` (C09 테스트 샘플 200→300)

---

## Phase 3: 진화형 알파 팩터 공장 (Autonomous Alpha Factory)

### 목표
Phase 1의 수동 마이닝을 **24시간 자율 팩터 공장**으로 확장. 벡터 DB 기반 경험 메모리, 유전 교차(Crossover), 팩터 포트폴리오 합성을 추가한다.

### 아키텍처

```
[AlphaFactoryScheduler] (백그라운드 asyncio 태스크)
         │
         │ 주기적 실행 (기본 6시간 간격)
         ▼
[EvolutionaryAlphaMiner] (Phase 1)
         │
         ├─ 가설 생성 시: ExperienceMemory에서 RAG 검색
         │   → 과거 성공/실패 임베딩 벡터 유사도 조회
         │   → 직교성(cos_sim < 0.7) 보장
         │
         ├─ 평가 후: IC pass → Phase 2 인과 검증
         │
         ├─ 세대 간: 유전 교차 (Crossover)
         │   factorA: log(vol) * (30 - rsi)
         │   factorB: sqrt(atr) / price_change
         │   → child: log(vol) * sqrt(atr)  (하위 트리 교환)
         │
         └─ 결과: alpha_factors + alpha_experiences DB 저장

[FactorPortfolioBuilder]
         │
         ├─ validated 팩터 N개 선택
         ├─ 상관행렬 계산 → 다각화 보장
         ├─ IC 가중 합성 → 복합 알파 시그널
         └─ 복합 팩터를 ConditionSchema 지표로 등록
```

### 신규 의존성

```
deap>=1.4.0           # 유전 알고리즘 (교차/변이/선택 연산)
```

### DB 추가 테이블 (마이그레이션 `f8g9h0i1j2k3`)

**`alpha_experiences`** — 벡터 임베딩 기반 경험 메모리

| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | UUID PK | |
| factor_id | UUID FK | alpha_factors 참조 |
| expression_str | Text | 수식 문자열 |
| hypothesis | Text | 가설 |
| embedding | JSON | 768-dim 벡터 (ko-sroberta로 가설+수식 인코딩) |
| ic_mean | Float | IC |
| success | Boolean | 성공 여부 |
| generation | Integer | 세대 |
| parent_ids | JSON | 부모 팩터 ID 목록 (계보 추적) |
| created_at | DateTime | |

인덱스: `success`, `ic_mean`

### 신규 백엔드 파일

#### `app/alpha/memory.py`
벡터 기반 경험 메모리:
- `ExperienceVectorMemory` 클래스
- `add(factor, hypothesis, success)` — ko-sroberta 임베딩 생성 → DB 저장. 기존 `sector/embedder.py`의 `encode_text()` 재사용
- `retrieve_similar(query, k=10)` — 코사인 유사도 검색. 기존 `sector/search.py`의 `cosine_similarity_batch()` 재사용
- `check_orthogonality(new_expr, threshold=0.7)` — 기존 성공 팩터와의 유사도 < threshold 확인
- `format_rag_context(query)` — Claude 프롬프트용: 유사 성공 5개 + 유사 실패 3개

#### `app/alpha/evolution.py`
유전 연산:
- `crossover(exprA, exprB) → list[sympy.Expr]` — AST 하위 트리 교환
  - DEAP의 `gp.cxOnePoint` 패턴 차용
  - 랜덤 하위 노드 선택 → 교차점에서 서브트리 스왑
- `mutate(expr) → sympy.Expr` — 연산자 교체, 상수 섭동, 피처 치환
- `tournament_select(population, k=3) → list` — IC 기준 토너먼트 선택
- `run_generation(parents, data) → list[AlphaFactor]` — 한 세대 실행
  - 선택 → 교차 → 변이 → 평가 → IC 필터 → 인과 검증

#### `app/alpha/scheduler.py`
자율 마이닝 스케줄러:
```python
class AlphaFactoryScheduler:
    """백그라운드에서 주기적으로 마이닝 사이클 실행.

    start() — asyncio 태스크 시작
    stop()  — 중지

    사이클:
    1. 최근 시장 데이터 로드
    2. ExperienceVectorMemory에서 RAG 컨텍스트 생성
    3. EvolutionaryAlphaMiner.run() (Phase 1)
    4. FactorMirageFilter.validate() (Phase 2)
    5. 유전 교차 (상위 팩터 간)
    6. 결과 저장 + WebSocket 알림
    """
```

설정: `ALPHA_FACTORY_INTERVAL_HOURS: int = 6`, `ALPHA_FACTORY_AUTO_START: bool = False`

#### `app/alpha/portfolio.py`
팩터 포트폴리오 합성:
- `build_composite_factor(factor_ids, method="ic_weighted") → CompositeAlphaFactor`
  - `equal_weight`: 단순 평균
  - `ic_weighted`: IC 비례 가중 합성
- `compute_correlation_matrix(factor_ids) → pl.DataFrame` — 팩터 간 상관행렬
- `register_composite(composite) → str` — 복합 팩터를 지표로 등록 (`alpha_composite_<id>`)

### API 추가

| Method | Path | 설명 |
|--------|------|------|
| POST | `/alpha/factory/start` | 자율 공장 시작 |
| POST | `/alpha/factory/stop` | 자율 공장 중지 |
| GET | `/alpha/factory/status` | 공장 상태 (running/stopped, 통계) |
| POST | `/alpha/portfolio/build` | 복합 팩터 생성 |
| GET | `/alpha/portfolio/{id}` | 복합 팩터 상세 |
| GET | `/alpha/portfolio/{id}/correlation` | 팩터 간 상관행렬 |

### 기존 파일 수정

| 파일 | 변경 |
|------|------|
| `app/alpha/miner.py` | ExperienceMemory → ExperienceVectorMemory 교체, 유전 교차 통합 |
| `app/main.py` | lifespan에 AlphaFactoryScheduler start/stop 추가 (ALPHA_FACTORY_AUTO_START 시) |
| `app/core/config.py` | ALPHA_FACTORY_* 환경변수 추가 |

### 프론트엔드 추가

| 파일 | 설명 |
|------|------|
| `components/alpha/AlphaFactoryControl.tsx` | 공장 시작/중지 + 실시간 상태 (WebSocket) |
| `components/alpha/FactorLineageTree.tsx` | 팩터 계보 트리 (parent→child SVG) |
| `components/alpha/FactorCorrelationHeatmap.tsx` | 상관행렬 히트맵 (canvas 기반) |
| `components/alpha/CompositeFactorBuilder.tsx` | 팩터 선택 → 가중치 → 복합 팩터 생성 UI |

`AlphaLabPage.tsx`에 Factory 탭 추가 (탭 구조: 탐색 / 공장 / 포트폴리오).

### 구현 순서 (6단계)

1. alpha_experiences 테이블 + 마이그레이션
2. 벡터 경험 메모리 (memory.py) — sector/embedder.py 재사용
3. 유전 연산 (evolution.py) — crossover, mutate, tournament_select
4. 자율 스케줄러 (scheduler.py) + config.py/main.py 수정
5. 팩터 포트폴리오 합성 (portfolio.py) + API 엔드포인트
6. 프론트엔드 (FactoryControl, LineageTree, CorrelationHeatmap, CompositeBuilder)

### 설계 vs 구현 차이점

> 아래는 설계 시점의 의도와 실제 구현 결과가 달라진 항목이다.
> 모든 변경은 구현 과정 및 전문가 검증(코드 리뷰)을 거쳐 의도적으로 조정되었다.

#### 1. `register_composite()` 미구현

**설계**: `portfolio.py`에 `register_composite(composite_id, factor_expressions) → str` 함수를 구현하여, 복합 팩터를 `backtest/indicators.py`의 `_INDICATOR_FN`에 동적 등록한다.

**실제 구현**: 해당 함수를 구현하지 않았다. `build_composite_factor()`가 가중합 수식 문자열(`"0.4000 * (rsi) + 0.6000 * (volume_ratio)"`)을 반환하고, DB에 `expression_str`로 저장된다.

**변경 사유**: 복합 팩터의 수식은 단일 SymPy 표현식이 아닌 **여러 팩터의 가중 합산**이다. 각 구성 팩터의 `expression_str`을 파싱하여 Polars 표현식으로 변환한 뒤 가중합을 계산하는 과정은 `_INDICATOR_FN`의 단일 함수 등록 패턴과 맞지 않는다. 복합 팩터를 백테스트에 사용하려면 별도의 `register_alpha_factor()` (기존 `backtest_bridge.py`) 경로로 처리하는 것이 적절하다.

**영향 범위**: 없음 (미구현 항목)

#### 2. `check_orthogonality()` 마이너 루프 미통합

**설계**: `ExperienceVectorMemory.check_orthogonality()`를 마이너의 가설 생성 → 평가 사이에 삽입하여, 기존 성공 팩터와 유사도가 높은(cos_sim ≥ 0.7) 후보를 사전에 필터링한다.

**실제 구현**: 메서드는 `memory.py`에 구현되었으나, `miner.py`에서 호출하지 않는다. 대신 `format_rag_context()`가 "유사하지만 실패한 팩터" 섹션을 Claude 프롬프트에 포함하여 간접적으로 중복 생성을 방지한다.

**변경 사유**: 직교성 체크에는 임베딩 모델 호출(~50ms)이 필요하다. 마이너 루프가 반복당 Claude API 호출(~2-5초)을 수행하므로 50ms는 무시할 수준이지만, RAG 컨텍스트에 실패 팩터를 명시적으로 포함하는 것이 Claude의 생성 다양성 유도에 더 효과적이라 판단하였다. `check_orthogonality()`는 향후 AlphaFactoryScheduler의 사이클 간 중복 방지(inter-cycle dedup)에 활용할 수 있다.

**영향 범위**: `app/alpha/miner.py` (호출 추가 시 3줄 삽입)

#### 3. 벡터 메모리 batch embedding 미적용

**설계**: 경험 추가 시 배치 임베딩으로 처리하여 GPU/모델 호출 효율을 높인다.

**실제 구현**: `add()` 메서드에서 개별 `_encode_text_safe()` 호출. `load_cache()`는 DB에서 이미 저장된 임베딩을 로드하므로 배치 인코딩 불필요.

**변경 사유**: 경험은 마이닝 루프에서 한 번에 하나씩 추가된다(평가 성공/실패 시). 배치 임베딩은 여러 텍스트를 모아서 한 번에 인코딩할 때 효율적이지만, 현재 흐름에서는 항상 단건이다. `sentence-transformers`의 `model.encode()`는 단건 호출에도 GPU 배치 크기 1로 동작하므로 추가 오버헤드가 없다. 대량 경험 마이그레이션(DB import) 시나리오에서 배치 인코딩을 추가할 수 있다.

**영향 범위**: `app/alpha/memory.py` (배치 메서드 추가 시)

#### 4. `CompositeFactorResponse`에 `weights` dict 미포함

**설계**: 복합 팩터 생성 응답에 각 구성 팩터의 가중치를 포함한다.

**실제 구현**: `CompositeResult` dataclass에 `weights: dict[str, float]` 필드가 존재하나, `CompositeFactorResponse` Pydantic 스키마와 API 응답에는 포함되지 않았다.

**변경 사유**: 프론트엔드 CompositeFactorBuilder에서 생성 결과를 `window.alert()`로 간단히 표시하는 현재 UI에서는 가중치 상세 정보를 활용할 곳이 없다. 향후 복합 팩터 상세 페이지에서 가중치 파이 차트 등을 추가할 때 `CompositeFactorResponse`에 `weights` 필드를 추가하면 된다.

**영향 범위**: `app/alpha/schemas.py`, `app/routers/alpha.py`, `src/types/alpha.ts`

#### 5. `tournament_select` 중복 부모 방지 추가

**설계**: `tournament_select()`는 단순히 k개 토너먼트에서 n_select번 반복 선택한다. 중복 가능.

**실제 구현**: 전문가 검증 후, 같은 부모가 2번 선택되는 경우(자기 교차, crossover(A, A))를 방지하는 중복 제거 로직을 추가하였다.

```python
# evolution.py — 수정 후
while len(selected) < n_select and attempts < n_select * 3:
    tournament = random.sample(population, min(k, len(population)))
    winner = max(tournament, key=lambda f: f.ic_mean)
    if winner not in selected:
        selected.append(winner)
    attempts += 1
```

**변경 사유**: `crossover(expr_a, expr_b)`에서 `expr_a == expr_b`이면 하위 트리 교환이 사실상 항등 변환이 되어 유전 교차의 의미가 없다. population 크기가 작을 때(2-3개) 중복 선택 확률이 특히 높으므로 방지가 필요하다.

**영향 범위**: `app/alpha/evolution.py` (L211-225)

---

## Phase 4: 금융 월드 모델 (ABM) + MCP 통합

### 목표
A) **에이전트 기반 시장 시뮬레이터(ABM)**로 카운터팩추얼 스트레스 테스트 수행
B) **FastMCP 서버**로 KIS/DB/뉴스/알파를 표준화된 도구로 캡슐화하여 Claude 에이전트가 자율 호출

### Part A: Agent-Based Market Simulation

#### 아키텍처

```
[StressTestRequest]
        │
        ▼
[VirtualExchange] ─── Limit Order Book (LOB)
        │              Price-time priority matching
        │              Market impact modeling
        │
        ├─ [FundamentalAgent] x 20  ─ 내재가치 기반 매매
        ├─ [ChartistAgent] x 30    ─ 모멘텀/역추세 규칙
        ├─ [NoiseTrader] x 100     ─ 랜덤 주문
        └─ [LLMAgent] x 5          ─ Claude 추론 (익명화된 맥락)
                                      └ 손실회피, 군집행동 편향 파라미터
        │
        ▼
[CounterfactualEngine]
        │
        ├─ "금리 급등 2%p" 시나리오 주입
        ├─ "유동성 고갈 90%" 시나리오 주입
        └─ "공급망 충격" 시나리오 주입 (Claude 생성)
        │
        ▼
[전략 투입 → MDD/생존률/회복시간 측정]
```

#### 신규 의존성

```
# ABM 관련 — 외부 의존성 최소화, 자체 구현
# (PyMarketSim 대신 경량 자체 LOB 엔진)
```

> ABM은 외부 라이브러리 의존 없이 자체 구현. PyMarketSim은 유지보수가 불안정하므로 경량 LOB 엔진을 직접 작성.

#### DB 추가 (마이그레이션 `g9h0i1j2k3l4`)

**`stress_test_runs`**

| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | UUID PK | |
| strategy_json | JSON | 테스트 대상 전략 |
| scenario_type | String | rate_shock / liquidity_crisis / supply_chain / custom |
| scenario_config | JSON | 시나리오 파라미터 |
| agent_config | JSON | 에이전트 구성 (수/편향) |
| status | String | PENDING / RUNNING / COMPLETED / FAILED |
| progress | Integer | 0-100 |
| results | JSON | {survived, mdd, recovery_days, price_series, order_book_snapshots} |
| created_at / completed_at | DateTime | |

#### 신규 백엔드 파일

**`app/simulation/__init__.py`**

**`app/simulation/orderbook.py`** — 경량 LOB 엔진:
```python
class LimitOrderBook:
    """호가창 시뮬레이터.
    - bids/asks: SortedDict (price → list[Order])
    - match(): Price-time priority 매칭
    - market_order(side, qty) → list[Fill]
    - limit_order(side, price, qty) → Order
    - cancel(order_id)
    - get_mid_price(), get_spread(), get_depth()
    """
```

**`app/simulation/agents.py`** — 이질적 에이전트:
```python
class BaseAgent:
    """에이전트 베이스. 매 스텝마다 decide() → Order | None"""

class FundamentalAgent(BaseAgent):
    """내재가치 = SMA(200) +/- noise. price < intrinsic → BUY"""

class ChartistAgent(BaseAgent):
    """momentum: price > SMA(20) → BUY, else → SELL"""

class NoiseTrader(BaseAgent):
    """랜덤 매매. intensity 파라미터로 활동량 조절"""

class LLMAgent(BaseAgent):
    """Claude 추론 기반. 익명화된 시장 상태 + 행동 편향 주입.
    loss_aversion: float = 2.5 (카너먼 계수)
    herding_factor: float = 0.3 (군집 행동 강도)"""
```

> LLMAgent는 매 N스텝마다 Claude 호출 (API 비용 제어). 나머지 스텝은 이전 결정 유지.

**`app/simulation/exchange.py`** — 가상 거래소:
```python
class VirtualExchange:
    """다중 에이전트 시뮬레이션 환경.

    init(agents, initial_price, config)
    run_steps(n) → SimulationResult
    inject_event(event_type, params) — 시나리오 주입
    get_metrics() → {price_series, volume, spread, depth}
    """
```

**`app/simulation/scenarios.py`** — 카운터팩추얼 시나리오:
```python
PRESET_SCENARIOS = {
    "rate_shock": {"type": "fundamental_shift", "magnitude": -0.15},
    "liquidity_crisis": {"type": "volume_drain", "reduction": 0.9},
    "flash_crash": {"type": "cascade_sell", "trigger_pct": -0.05},
    "supply_chain": {"type": "news_shock", "prompt": "..."},
}

class CounterfactualEngine:
    """시나리오 주입 + 전략 생존력 평가.

    run_stress_test(strategy, scenario, exchange_config) → StressTestResult
    generate_counterfactual_news(scenario_type) → str  # Claude 생성
    """
```

**`app/simulation/runner.py`** — 비동기 실행기 (runner.py 패턴).

**`app/simulation/anonymizer_enhanced.py`** — 강화된 익명화:
기존 `news/anonymizer.py` 확장:
- 기존: 기업명 → `[COMPANY_A]`
- 추가: 날짜 → `[DATE]`, CEO명 → `[PERSON]`, 티커 → `[TICKER]`
- 기업명 사전: `stock_masters.name` (기존 로직 유지)
- 인물/날짜: 정규식 패턴 매칭 (spaCy 없이 경량 구현)

> spaCy (500MB+) 대신 정규식 기반 NER로 경량화. 한국어 기업명은 기존 `stock_masters` 사전으로 충분.

**`app/routers/simulation.py`** — API:

| Method | Path | 설명 |
|--------|------|------|
| POST | `/simulation/stress-test` | 스트레스 테스트 시작 |
| GET | `/simulation/stress-test/{id}` | 결과 조회 |
| GET | `/simulation/stress-tests` | 목록 |
| GET | `/simulation/scenarios` | 프리셋 시나리오 목록 |
| POST | `/simulation/scenario/generate` | Claude로 커스텀 시나리오 생성 |

### Part B: MCP Data Bus

#### 아키텍처

```
[Claude AI Agent] (MCP Client)
        │
        │ MCP Protocol (SSE transport)
        ▼
[FastMCP Server] ─── app/mcp/server.py
        │
        ├─ Tool: execute_order(ticker, action, qty)
        │   └→ kis_order.py (기존) + Governance 검증
        │
        ├─ Tool: query_stock_data(symbol, interval, dates)
        │   └→ candle_service.py (기존)
        │
        ├─ Tool: fetch_sentiment(symbol, date)
        │   └→ news DB 조회
        │
        ├─ Tool: run_alpha_scan(context)
        │   └→ alpha/miner.py (Phase 1)
        │
        ├─ Tool: get_portfolio_status()
        │   └→ positions + P&L 조회
        │
        ├─ Resource: realtime_orderbook/{symbol}
        │   └→ WebSocket 스냅샷
        │
        └─ Governance Layer
            ├─ 수량 하드 리밋 (max 1000주)
            ├─ BUY/SELL만 허용
            ├─ 감사 로깅 (모든 Tool 호출)
            └─ Real 모드 시 human-in-the-loop 승인
```

#### 신규 의존성

```
fastmcp>=2.0.0        # MCP 서버 프레임워크
```

#### DB 추가 (마이그레이션 `g9h0i1j2k3l4`에 포함)

**`mcp_audit_logs`** — MCP 도구 호출 감사 로그

| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | UUID PK | |
| tool_name | String | 호출된 도구명 |
| input_params | JSON | 입력 파라미터 |
| output | JSON | 결과 |
| status | String | success / error / blocked |
| blocked_reason | String | governance에 의해 차단 시 사유 |
| execution_ms | Integer | 실행 시간 |
| created_at | DateTime | |

#### 신규 백엔드 파일

**`app/mcp/__init__.py`**

**`app/mcp/server.py`** — FastMCP 서버:
```python
from mcp.server.fastmcp import FastMCP, Context

mcp = FastMCP(name="StockMate_Gateway")

@mcp.tool()
async def execute_order(ticker: str, action: str, quantity: int, ctx: Context) -> str:
    """증권사 주문 실행 (KIS API). Governance 검증 후 실행."""

@mcp.tool()
async def query_stock_data(symbol: str, interval: str, start: str, end: str, ctx: Context) -> str:
    """시계열 OHLCV 데이터 조회 (TimescaleDB)."""

@mcp.tool()
async def fetch_sentiment(symbol: str, date: str, ctx: Context) -> str:
    """종목 뉴스 감성 분석 데이터 조회."""

@mcp.tool()
async def run_alpha_scan(context: str, ctx: Context) -> str:
    """AI 알파 팩터 탐색 실행."""

@mcp.tool()
async def get_portfolio_status(ctx: Context) -> str:
    """현재 포트폴리오 (보유종목 + 평가손익) 조회."""

@mcp.resource("orderbook://{symbol}")
async def get_orderbook(symbol: str) -> str:
    """실시간 호가창 스냅샷."""
```

**`app/mcp/governance.py`** — 보안/거버넌스:
```python
class GovernanceLayer:
    """MCP 도구 호출 전/후 검증.

    pre_check(tool_name, params) → (allowed: bool, reason: str)
    - 수량 리밋: qty <= 1000 (하드)
    - 액션 검증: BUY/SELL만
    - Real 모드: human_approval_required=True

    audit_log(tool_name, params, result, status)
    - mcp_audit_logs 테이블에 기록
    """
```

**`app/mcp/bridge.py`** — FastAPI↔MCP 브릿지:
```python
"""FastAPI와 MCP 서버를 동일 프로세스에서 실행.
SSE transport로 외부 클라이언트(Claude Desktop 등) 연결 지원.
"""

async def start_mcp_server():
    """FastAPI lifespan에서 호출. 별도 asyncio 태스크로 MCP 서버 구동."""
```

**`app/routers/mcp.py`** — MCP 관리 API:

| Method | Path | 설명 |
|--------|------|------|
| GET | `/mcp/status` | MCP 서버 상태 |
| GET | `/mcp/tools` | 등록된 도구 목록 |
| GET | `/mcp/audit` | 감사 로그 조회 (페이지네이션) |
| PUT | `/mcp/governance` | 거버넌스 규칙 업데이트 |

### 기존 파일 수정

| 파일 | 변경 |
|------|------|
| `app/main.py` | lifespan에 MCP 서버 start/stop, simulation router 포함 |
| `app/news/anonymizer.py` | 날짜/인물/티커 마스킹 로직 추가 (기존 기업명 마스킹 유지) |
| `app/core/config.py` | MCP/Simulation 환경변수 추가 |
| `requirements.txt` | fastmcp 추가 |

### 프론트엔드 추가

| 파일 | 설명 |
|------|------|
| `pages/SimulationPage.tsx` | 스트레스 테스트 페이지 |
| `components/simulation/StressTestConfig.tsx` | 시나리오 선택 + 에이전트 구성 |
| `components/simulation/StressTestResult.tsx` | 결과 (가격 시계열 + MDD + 생존 여부) |
| `components/simulation/LOBVisualization.tsx` | 호가창 깊이 차트 (시뮬레이션 중) |
| `components/mcp/MCPDashboard.tsx` | MCP 도구 상태 + 감사 로그 |
| `components/mcp/AuditLog.tsx` | 도구 호출 이력 테이블 |
| `types/simulation.ts` | StressTestRun, Scenario, AgentConfig 인터페이스 |
| `types/mcp.ts` | MCPTool, AuditLogEntry 인터페이스 |
| `api/simulation.ts` | 스트레스 테스트 API 함수 |
| `api/mcp.ts` | MCP 관리 API 함수 |
| `hooks/queries/use-simulation.ts` | TanStack Query 훅 |
| `hooks/queries/use-mcp.ts` | TanStack Query 훅 |

`App.tsx`에 `/simulation`, `/mcp` 라우트 추가.
`Sidebar.tsx`에 "스트레스 테스트", "MCP 게이트웨이" 메뉴 추가.

### 구현 순서 (8단계)

1. stress_test_runs + mcp_audit_logs 테이블 + 마이그레이션
2. 경량 LOB 엔진 (orderbook.py)
3. 이질적 에이전트 4종 (agents.py)
4. 가상 거래소 + 카운터팩추얼 엔진 (exchange.py, scenarios.py)
5. 강화된 익명화 (anonymizer_enhanced.py)
6. ABM 비동기 러너 + API (runner.py, routers/simulation.py)
7. FastMCP 서버 + 거버넌스 + 브릿지 (mcp/server.py, governance.py, bridge.py)
8. 프론트엔드 (SimulationPage, MCPDashboard, 컴포넌트)

---

## 전체 파일 변경 요약

### Phase 1 (신규 15 + 수정 5)
**백엔드 신규 (8)**: `alpha/__init__`, `models`, `schemas`, `ast_converter`, `evaluator`, `miner`, `runner`, `backtest_bridge`, `routers/alpha`
**프론트 신규 (9)**: `types/alpha`, `api/alpha`, `hooks/queries/use-alpha`, `pages/AlphaLabPage`, `components/alpha/` x 6
**DB**: 마이그레이션 1개 (2 테이블)
**수정 (5)**: `config.py`, `main.py`, `indicators.py`, `engine.py`, `requirements.txt`

### Phase 2 (신규 5 + 수정 3)
**백엔드 신규 (3)**: `alpha/confounders`, `alpha/causal`, `alpha/causal_runner`
**프론트 신규 (2)**: `components/alpha/CausalBadge`, `CausalDAGView`
**데이터 (1)**: `data/bok_base_rate.json`
**수정 (3)**: `alpha/runner.py` (자동 검증 통합), `alpha/schemas.py` (CausalValidationResponse), `routers/alpha.py` (검증 엔드포인트)

### Phase 3 (신규 8 + 수정 4)
**백엔드 신규 (4)**: `alpha/memory`, `alpha/evolution`, `alpha/scheduler`, `alpha/portfolio`
**프론트 신규 (4)**: `components/alpha/` x 4 (FactoryControl, LineageTree, CorrelationHeatmap, CompositeBuilder)
**DB**: 마이그레이션 1개 (alpha_experiences)
**수정 (4)**: `alpha/miner.py`, `main.py`, `config.py`, `requirements.txt`

### Phase 4 (신규 18 + 수정 5)
**백엔드 신규 (10)**: `simulation/__init__`, `orderbook`, `agents`, `exchange`, `scenarios`, `runner`, `anonymizer_enhanced`, `mcp/__init__`, `mcp/server`, `mcp/governance`, `mcp/bridge`, `routers/simulation`, `routers/mcp`
**프론트 신규 (12)**: `pages/SimulationPage`, `types/simulation`, `types/mcp`, `api/simulation`, `api/mcp`, `hooks/queries/use-simulation`, `hooks/queries/use-mcp`, `components/simulation/` x 3, `components/mcp/` x 2
**DB**: 마이그레이션 1개 (stress_test_runs, mcp_audit_logs)
**수정 (5)**: `main.py`, `config.py`, `news/anonymizer.py`, `requirements.txt`, `App.tsx`

---

## 검증 방법

### Phase 1 검증
1. `POST /alpha/mine` 호출 (use_pysr=false, Claude-only 모드)
2. WebSocket으로 진행률 확인
3. `GET /alpha/factors`로 발견된 팩터 확인 (IC 값, 수식)
4. `POST /alpha/factor/{id}/backtest`로 팩터 기반 백테스트 실행
5. BacktestPage에서 결과 확인 (기존 백테스트와 동일한 UI)

### Phase 2 검증
1. Phase 1에서 발견된 팩터에 `POST /alpha/factor/{id}/validate` 호출
2. 인과 검증 결과 확인: `is_causally_robust`, `placebo_passed`, `random_cause_passed`
3. 의도적 "가짜 팩터" (random noise) 생성 → mirage로 분류되는지 확인
4. AlphaLabPage에서 CausalBadge/CausalDAGView 렌더링 확인

### Phase 3 검증
1. `POST /alpha/factory/start` → 자율 마이닝 시작
2. 6시간(또는 테스트용 1분) 간격으로 새 팩터 발견 확인
3. FactorLineageTree에서 부모→자식 계보 표시 확인
4. FactorCorrelationHeatmap에서 팩터 간 상관관계 확인
5. CompositeFactorBuilder로 복합 팩터 생성 → 백테스트

### Phase 4 검증
1. `POST /simulation/stress-test` (flash_crash 시나리오)
2. 시뮬레이션 진행 중 LOB 시각화 확인
3. 전략 생존 여부 + MDD 결과 확인
4. MCP: `GET /mcp/tools`로 등록된 도구 확인
5. Claude Desktop에서 MCP 서버 연결 → "삼성전자 10주 매수" 자연어 → execute_order 호출 확인
6. `GET /mcp/audit`에서 감사 로그 확인
