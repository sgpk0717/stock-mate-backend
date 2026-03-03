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

    # AI (Claude API for backtest strategy generation)
    ANTHROPIC_API_KEY: str = ""

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
    CAUSAL_AUTO_VALIDATE: bool = False
    CAUSAL_PLACEBO_THRESHOLD: float = 0.05
    CAUSAL_RANDOM_CAUSE_THRESHOLD: float = 0.05
    CAUSAL_NUM_SIMULATIONS: int = 100

    # Evolution Engine (진화형 팩토리)
    ALPHA_POPULATION_SIZE: int = 500
    ALPHA_ELITE_PCT: float = 0.05
    ALPHA_AST_MUTATION_RATIO: float = 0.92
    ALPHA_LLM_MUTATION_RATIO: float = 0.08
    ALPHA_FITNESS_W_IC: float = 0.40
    ALPHA_FITNESS_W_ICIR: float = 0.30
    ALPHA_FITNESS_W_TURNOVER: float = 0.15
    ALPHA_FITNESS_W_COMPLEXITY: float = 0.15
    ALPHA_MAX_TREE_DEPTH: int = 10
    ALPHA_MAX_TREE_SIZE: int = 30
    ALPHA_TOURNAMENT_K: int = 5

    # Phase 3: Alpha Factory
    ALPHA_FACTORY_AUTO_START: bool = False
    ALPHA_FACTORY_INTERVAL_MINUTES: int = 360  # 6시간
    ALPHA_FACTORY_MAX_ITERATIONS: int = 5
    ALPHA_FACTORY_CROSSOVER_ENABLED: bool = True
    ALPHA_FACTORY_TOURNAMENT_K: int = 3
    ALPHA_FACTORY_ORTHOGONALITY_THRESHOLD: float = 0.7

    # Phase 4: Simulation (ABM)
    SIMULATION_DEFAULT_STEPS: int = 1000
    SIMULATION_LLM_CALL_INTERVAL: int = 20

    # Phase 4: MCP Data Bus
    MCP_ENABLED: bool = False
    MCP_SSE_PORT: int = 8008
    MCP_MAX_ORDER_QTY: int = 1000
    MCP_HUMAN_APPROVAL_REAL: bool = True

    # CORS
    CORS_ORIGINS: list[str] = ["*"]


settings = Settings()
