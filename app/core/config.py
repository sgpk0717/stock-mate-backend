from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Application
    APP_TITLE: str = "Stock Mate API"
    APP_VERSION: str = "0.1.0"
    PORT: int = 8007
    DEBUG: bool = False

    # Database
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "stockmate"
    POSTGRES_PASSWORD: str = "stockmate"
    POSTGRES_DB: str = "stockmate"

    @property
    def async_database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def sync_database_url(self) -> str:
        """Used by Alembic (which requires a synchronous driver)."""
        return (
            f"postgresql+psycopg2://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    # ZeroMQ (Data Pump 연결)
    ZMQ_HOST: str = "127.0.0.1"
    ZMQ_PORT: int = 5555

    # 실시간 데이터 소스 (true=개발용 시뮬레이터, false=키움 ZMQ 대기)
    USE_SIMULATOR: bool = False

    # AI — Anthropic (에이전트, 전략, 알파)
    ANTHROPIC_API_KEY: str = ""

    # AI — Gemini (뉴스 감성분석 등 비용 민감 작업)
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-3-flash-preview"

    # Agent (다중 턴 대화형 전략 수립)
    AGENT_SESSION_TTL_MINUTES: int = 30
    AGENT_MODEL: str = "claude-sonnet-4-6"
    AGENT_MAX_TOKENS: int = 4000

    # News (뉴스 수집 + 감성 분석)
    DART_API_KEY: str = ""
    BIGKINDS_API_KEY: str = ""
    NEWS_COLLECT_HOUR: int = 18  # 수집 시각 (KST)
    NEWS_BATCH_SIZE: int = 10  # Claude 배치 분석 단위

    # KIS (한국투자증권 Open API)
    KIS_APP_KEY: str = ""
    KIS_APP_SECRET: str = ""
    KIS_ACCOUNT_NO: str = ""  # XXXXXXXX-XX (종합계좌번호-계좌상품코드)
    KIS_BASE_URL: str = "https://openapi.koreainvestment.com:9443"
    KIS_MOCK_URL: str = "https://openapivts.koreainvestment.com:29443"

    # Alpha Mining (진화형 알파 탐색)
    ALPHA_MAX_PYSR_ITERATIONS: int = 40
    ALPHA_IC_THRESHOLD_PASS: float = 0.03
    ALPHA_IC_THRESHOLD_GOOD: float = 0.05
    ALPHA_MAX_MUTATION_DEPTH: int = 5
    ALPHA_PYSR_TIMEOUT_SECONDS: int = 300
    ALPHA_PYSR_MAX_SIZE: int = 15
    ALPHA_PYSR_PARSIMONY: float = 0.03

    # Phase 2: Causal Inference
    CAUSAL_PLACEBO_THRESHOLD: float = 0.05
    CAUSAL_RANDOM_CAUSE_THRESHOLD: float = 0.05
    CAUSAL_NUM_SIMULATIONS: int = 20
    CAUSAL_USE_FAST_ENGINE: bool = True  # NumPy 고속 엔진 (False → DoWhy 레거시)

    # Evolution Engine (진화형 팩토리)
    ALPHA_POPULATION_SIZE: int = 100  # 5분봉 대규모 데이터 OOM 방지 (메모리 안정화 후 점진적 증가)
    ALPHA_ELITE_PCT: float = 0.05
    ALPHA_AST_MUTATION_RATIO: float = 0.92
    ALPHA_LLM_MUTATION_RATIO: float = 0.08
    ALPHA_FITNESS_W_IC: float = 0.30
    ALPHA_FITNESS_W_ICIR: float = 0.20
    ALPHA_FITNESS_W_SHARPE: float = 0.20
    ALPHA_FITNESS_W_MDD: float = 0.05
    ALPHA_FITNESS_W_TURNOVER: float = 0.10
    ALPHA_FITNESS_W_COMPLEXITY: float = 0.15
    ALPHA_SHARPE_THRESHOLD: float = 0.3  # discovered 최소 Sharpe 기준
    ALPHA_MAX_TREE_DEPTH: int = 10
    ALPHA_MAX_TREE_SIZE: int = 30
    ALPHA_TOURNAMENT_K: int = 5

    # Evolution Engine 병렬화
    ALPHA_LLM_MAX_CONCURRENT: int = 20       # LLM 동시 호출 수 (Tier 3 기준)
    ALPHA_LLM_RETRY_MAX: int = 2             # 429/timeout 최대 재시도
    ALPHA_LLM_RETRY_BASE_DELAY: float = 2.0  # 지수 백오프 기본 대기(초)
    ALPHA_EVAL_BATCH_SIZE: int = 5            # 배치 평가 크기 (5분봉 대규모 데이터 OOM 방지)

    # Phase 3: Alpha Factory
    ALPHA_FACTORY_AUTO_START: bool = False
    ALPHA_FACTORY_INTERVAL_MINUTES: int = 360  # 6시간
    ALPHA_FACTORY_MAX_ITERATIONS: int = 5
    ALPHA_FACTORY_CROSSOVER_ENABLED: bool = True
    ALPHA_FACTORY_TOURNAMENT_K: int = 3
    ALPHA_FACTORY_ORTHOGONALITY_THRESHOLD: float = 0.7
    ALPHA_FACTORY_MAX_CYCLES: int = 10  # 야간 마이닝 최대 사이클 수 (API 비용 예산)

    # Backtest
    BACKTEST_TIMEOUT_SECONDS: int = 1800  # 30분

    # Phase 4: Simulation (ABM)
    SIMULATION_DEFAULT_STEPS: int = 1000
    SIMULATION_LLM_CALL_INTERVAL: int = 20

    # Phase 4: MCP Data Bus
    MCP_ENABLED: bool = True
    MCP_SSE_PORT: int = 8008
    MCP_MAX_ORDER_QTY: int = 1000
    MCP_HUMAN_APPROVAL_REAL: bool = True

    # Daily Scheduler (일일 배치 수집)
    DAILY_SCHEDULER_ENABLED: bool = True
    DAILY_COLLECT_HOUR: int = 16
    DAILY_COLLECT_MINUTE: int = 30
    DAILY_PYKRX_THROTTLE_SEC: float = 1.0
    DAILY_NEWS_TOP_N: int = 200
    TICK_ROTATION_SCHEDULE_FILE: str = "tick_rotation_schedule.json"
    TICK_ROTATION_BATCH_SIZE: int = 200
    TICK_ROTATION_INTERVAL_MIN: int = 10

    # Program Trading Collector (KIS 프로그램 매매 수집)
    PGM_TRADING_ENABLED: bool = True
    PGM_TRADING_COLLECT_INTERVAL_MINUTES: int = 5
    PGM_TRADING_SYMBOLS_LIMIT: int = 200  # 수집 대상 종목 수 (시총 상위)

    # Workflow Orchestrator (일일 자동매매 워크플로우)
    WORKFLOW_ENABLED: bool = True
    WORKFLOW_TRADING_MODE: str = "paper"  # "paper" | "real"
    WORKFLOW_INITIAL_CAPITAL: float = 100_000_000
    WORKFLOW_MAX_POSITIONS: int = 10
    WORKFLOW_STOP_LOSS_PCT: float = 5.0
    WORKFLOW_MAX_DRAWDOWN_PCT: float = 10.0
    WORKFLOW_UNIVERSE: str = "KOSPI200"
    WORKFLOW_MIN_FACTOR_SHARPE: float = 0.3
    WORKFLOW_MIN_FACTOR_IC: float = 0.03
    WORKFLOW_REQUIRE_CAUSAL: bool = False
    WORKFLOW_FACTOR_MAX_AGE_DAYS: int = 30
    WORKFLOW_DATA_INTERVAL: str = "5m"  # 분봉 단타 (1m/3m/5m)

    # AutoSelector 가중치 (6요소 — 설계서 §8.2)
    WORKFLOW_SCORE_W_IC: float = 0.25
    WORKFLOW_SCORE_W_SHARPE: float = 0.20
    WORKFLOW_SCORE_W_ICIR: float = 0.15
    WORKFLOW_SCORE_W_MDD: float = 0.15
    WORKFLOW_SCORE_W_CAUSAL: float = 0.15
    WORKFLOW_SCORE_W_RECENCY: float = 0.10

    # Feedback Loop (피드백 루프)
    WORKFLOW_FEEDBACK_STALE_DAYS: int = 7
    WORKFLOW_FEEDBACK_RETIRE_DAYS: int = 30
    WORKFLOW_FEEDBACK_IC_DROP_THRESHOLD: float = 0.5

    # Parameter Auto-Tuning (파라미터 자동 튜닝)
    WORKFLOW_PARAM_EVAL_ENABLED: bool = True
    WORKFLOW_PARAM_EVAL_LOOKBACK_DAYS: int = 7
    WORKFLOW_PARAM_EVAL_MIN_TRADES: int = 20
    WORKFLOW_PARAM_EVAL_MIN_CONFIDENCE: float = 0.6

    # Divergence Detector (팩터 라이브-백테스트 다이버전스 자동 감지)
    WORKFLOW_DIVERGENCE_CHECK_ENABLED: bool = True
    WORKFLOW_DIVERGENCE_HALT_THRESHOLD: float = -10.0   # 누적 pnl% 자동 정지
    WORKFLOW_DIVERGENCE_WARN_THRESHOLD: float = -5.0    # 누적 pnl% 경고
    WORKFLOW_DIVERGENCE_MIN_DAYS: int = 2               # 최소 실매매 일수

    # Telegram (OpenClaw 독립 알림용 폴백)
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""

    # OpenClaw
    OPENCLAW_HEALTH_URL: str = "http://host.docker.internal:18789/health"
    OPENCLAW_MAX_MEMORY_MB: int = 2048

    # Worker Mode ("inline" | "external" | "worker")
    # inline: 기존 동작 (팩토리+인과검증 API 내 실행)
    # external: API는 DB 경유 위임 (REST만 서빙)
    # worker: 워커 프로세스 (팩토리+인과검증 실행, 명령큐 소비)
    WORKER_MODE: str = "inline"

    # CORS
    CORS_ORIGINS: list[str] = ["*"]


settings = Settings()
