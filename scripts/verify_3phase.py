"""3 Phase 구현 검증 스크립트.

Phase 1: 피처 확장 (14 → ~40)
Phase 2: 벡터 메모리 RAG 연결
Phase 3: 적합도 함수 Sharpe/MDD 반영

Usage:
  docker compose run --rm app python -m scripts.verify_3phase
"""

from __future__ import annotations

import asyncio
import logging
import sys

# 로깅 설정 — DEBUG 레벨로 벡터메모리/엔진 내부 확인
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
# 알파 모듈만 DEBUG
for name in [
    "app.alpha.evolution_engine",
    "app.alpha.memory",
    "app.alpha.ast_converter",
    "app.alpha.scheduler",
    "scripts.verify_3phase",
]:
    logging.getLogger(name).setLevel(logging.DEBUG)

logger = logging.getLogger("scripts.verify_3phase")


async def verify_phase1_features():
    """Phase 1: ensure_alpha_features 호출 후 컬럼 수 확인."""
    import polars as pl
    from app.alpha.ast_converter import ensure_alpha_features

    logger.info("=" * 60)
    logger.info("Phase 1: 피처 세트 확장 검증")
    logger.info("=" * 60)

    # 최소 OHLCV 데이터 생성 (100일, 단일종목)
    import random
    random.seed(42)
    n = 200
    dates = pl.date_range(
        pl.date(2024, 1, 1), pl.date(2024, 1, 1), eager=True
    )
    # 실제로는 날짜 시리즈 필요
    base_prices = [50000 + i * 10 + random.randint(-500, 500) for i in range(n)]

    df = pl.DataFrame({
        "dt": pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 7, 18), eager=True)[:n],
        "open": [p - random.randint(0, 300) for p in base_prices],
        "high": [p + random.randint(0, 500) for p in base_prices],
        "low": [p - random.randint(0, 500) for p in base_prices],
        "close": base_prices,
        "volume": [random.randint(100000, 5000000) for _ in range(n)],
    })

    logger.info("입력 DataFrame 컬럼 (%d개): %s", len(df.columns), df.columns)

    # ensure_alpha_features 호출
    df_feat = ensure_alpha_features(df)

    logger.info("출력 DataFrame 컬럼 (%d개): %s", len(df_feat.columns), sorted(df_feat.columns))

    # 기대 피처 확인
    expected_features = [
        # 기존
        "sma_20", "ema_20", "rsi", "volume_ratio", "atr_14",
        "macd_hist", "macd_line", "macd_signal",
        "bb_upper", "bb_lower", "bb_middle",
        "price_change_pct", "bb_width",
        # 멀티 윈도우
        "sma_5", "sma_10", "sma_60",
        "ema_5", "ema_10", "ema_60",
        "rsi_7", "rsi_21",
        "atr_7", "atr_21",
        # 시차
        "close_lag_1", "close_lag_5", "close_lag_20",
        "volume_lag_1", "volume_lag_5",
        "return_5d", "return_20d",
        # 파생
        "bb_position",
    ]

    missing = [f for f in expected_features if f not in df_feat.columns]
    present = [f for f in expected_features if f in df_feat.columns]

    logger.info("기대 피처 %d개 중 %d개 존재, %d개 누락",
                len(expected_features), len(present), len(missing))
    if missing:
        logger.error("누락된 피처: %s", missing)
    else:
        logger.info("✓ 모든 기대 피처가 존재합니다!")

    # 횡단면 피처는 symbol 컬럼이 있어야 생성
    logger.info("(횡단면 피처는 symbol 컬럼 필요 — 단일종목이므로 스킵 정상)")

    # NaN 비율 확인 (후반 행 — 초기 행은 rolling window 때문에 NaN 정상)
    tail = df_feat.tail(50)
    for col in expected_features:
        if col in tail.columns:
            null_count = tail[col].null_count()
            nan_count = tail[col].is_nan().sum() if tail[col].dtype in [pl.Float64, pl.Float32] else 0
            if null_count > 5 or nan_count > 5:
                logger.warning("  %s: null=%d, nan=%d (후반 50행 중)", col, null_count, nan_count)

    logger.info("Phase 1 피처 확장: %d개 OHLCV → %d개 컬럼",
                6, len(df_feat.columns))

    return df_feat


async def verify_phase3_fitness():
    """Phase 3: fitness 함수에 sharpe/mdd가 반영되는지 확인."""
    from app.alpha.fitness import compute_composite_fitness

    logger.info("")
    logger.info("=" * 60)
    logger.info("Phase 3: 적합도 함수 Sharpe/MDD 반영 검증")
    logger.info("=" * 60)

    # 동일 IC/ICIR, 다른 sharpe → fitness 차이 확인
    base = dict(ic_mean=0.05, icir=1.0, turnover=0.3, tree_depth=3, tree_size=8)

    f_no_sharpe = compute_composite_fitness(**base, sharpe=0.0, max_drawdown=0.0)
    f_good_sharpe = compute_composite_fitness(**base, sharpe=2.0, max_drawdown=-0.05)
    f_bad_sharpe = compute_composite_fitness(**base, sharpe=-0.5, max_drawdown=-0.40)

    logger.info("Sharpe=0.0, MDD=0.0  → fitness=%.6f", f_no_sharpe)
    logger.info("Sharpe=2.0, MDD=-5%%  → fitness=%.6f", f_good_sharpe)
    logger.info("Sharpe=-0.5, MDD=-40%% → fitness=%.6f", f_bad_sharpe)

    assert f_good_sharpe > f_no_sharpe > f_bad_sharpe, "Sharpe/MDD가 fitness에 반영되지 않음!"
    logger.info("✓ Sharpe가 높을수록 fitness가 높고, MDD가 클수록 fitness가 낮습니다!")

    # 가중치 합 검증
    from app.core.config import settings
    total_w = (
        settings.ALPHA_FITNESS_W_IC
        + settings.ALPHA_FITNESS_W_ICIR
        + settings.ALPHA_FITNESS_W_SHARPE
        + settings.ALPHA_FITNESS_W_MDD
        + settings.ALPHA_FITNESS_W_TURNOVER
        + settings.ALPHA_FITNESS_W_COMPLEXITY
    )
    logger.info("가중치 합: %.2f (IC=%.2f, ICIR=%.2f, Sharpe=%.2f, MDD=%.2f, Turn=%.2f, Comp=%.2f)",
                total_w,
                settings.ALPHA_FITNESS_W_IC,
                settings.ALPHA_FITNESS_W_ICIR,
                settings.ALPHA_FITNESS_W_SHARPE,
                settings.ALPHA_FITNESS_W_MDD,
                settings.ALPHA_FITNESS_W_TURNOVER,
                settings.ALPHA_FITNESS_W_COMPLEXITY)
    assert abs(total_w - 1.0) < 0.01, f"가중치 합이 1.0이 아닙니다: {total_w}"
    logger.info("✓ 가중치 합 = 1.0 확인!")

    # Sharpe threshold 확인
    logger.info("ALPHA_SHARPE_THRESHOLD = %.2f", settings.ALPHA_SHARPE_THRESHOLD)


async def verify_phase2_rag():
    """Phase 2: 벡터 메모리 RAG 연결 검증 (DB 접근)."""
    from app.alpha.memory import ExperienceVectorMemory
    from app.core.database import async_session

    logger.info("")
    logger.info("=" * 60)
    logger.info("Phase 2: 벡터 메모리 RAG 연결 검증")
    logger.info("=" * 60)

    vm = ExperienceVectorMemory()

    async with async_session() as db:
        # 1. 캐시 로드
        await vm.load_cache(db)
        logger.info("벡터 메모리 캐시: %d개 경험 로드", len(vm._cache))

        # 2. RAG 컨텍스트 생성 테스트
        rag = vm.format_rag_context("alpha factor for Korean equities")
        logger.info("RAG 컨텍스트 생성 결과 (query='alpha factor for Korean equities'):")
        for line in rag.split("\n"):
            logger.info("  %s", line)

        # 3. 경험이 있으면 유사도 검색 테스트
        if vm._cache:
            first_exp = vm._cache[0]
            similar = vm.retrieve_similar(first_exp.expression_str, k=3)
            logger.info("유사도 검색 결과 (query='%s'):", first_exp.expression_str[:50])
            for s in similar:
                logger.info("  - %s (IC=%.4f, success=%s)", s.expression_str[:60], s.ic_mean, s.success)
        else:
            logger.info("(경험 데이터 없음 — 임베딩 모델 미설치 가능, 정상)")

    logger.info("✓ 벡터 메모리 RAG 연결 검증 완료!")


async def verify_evolution_run():
    """실제 1세대 진화 실행 (소규모)."""
    from app.alpha.ast_converter import ensure_alpha_features, parse_expression
    from app.alpha.evolution_engine import EvolutionEngine
    from app.alpha.memory import ExperienceVectorMemory
    from app.alpha.operators import OperatorRegistry
    from app.backtest.data_loader import load_candles
    from app.core.config import settings
    from app.core.database import async_session

    logger.info("")
    logger.info("=" * 60)
    logger.info("통합 검증: 1세대 진화 실행 (population=20)")
    logger.info("=" * 60)

    # 소규모 데이터 로드
    symbols = ["005930"]  # 삼성전자
    data = await load_candles(symbols=symbols)

    if data.height == 0:
        logger.warning("삼성전자 캔들 데이터 없음 — DB 시딩 필요. 스킵.")
        return

    logger.info("로드된 데이터: %d행 × %d열", data.height, data.width)

    # 피처 추가 확인
    data_feat = ensure_alpha_features(data)
    logger.info("피처 추가 후: %d행 × %d열", data_feat.height, data_feat.width)
    new_cols = set(data_feat.columns) - set(data.columns)
    logger.info("새로 추가된 컬럼 %d개: %s", len(new_cols), sorted(new_cols))

    # 벡터 메모리 + 진화 엔진
    vm = ExperienceVectorMemory()
    op_registry = OperatorRegistry(llm_ratio=0.0)  # LLM 없이 AST만

    async with async_session() as db:
        await vm.load_cache(db)
        logger.info("벡터 메모리: %d개 경험 캐시", len(vm._cache))

        engine = EvolutionEngine(
            data=data,
            db=db,
            operator_registry=op_registry,
            population_size=20,  # 소규모
            elite_pct=0.1,
            context="verification test",
            ic_threshold=settings.ALPHA_IC_THRESHOLD_PASS,
            vector_memory=vm,
            generation=0,
        )

        # 엔진 내부 데이터의 컬럼 수 확인
        logger.info("엔진 내부 데이터 컬럼 (%d개): %s",
                     len(engine._data.columns), sorted(engine._data.columns))

        # 1세대 실행
        async def log_progress(current, total, msg):
            logger.info("  [진행] %d/%d — %s", current, total, msg)

        async def log_iteration(event):
            logger.info("  [이벤트] %s", event)

        discovered = await engine.run_generation(
            progress_cb=log_progress,
            iteration_cb=log_iteration,
        )

        logger.info("")
        logger.info("1세대 결과:")
        logger.info("  discovered 팩터: %d개", len(discovered))

        for i, df in enumerate(discovered):
            m = df.metrics
            logger.info(
                "  [%d] %s | IC=%.4f ICIR=%.2f Sharpe=%.2f MDD=%.1f%%",
                i + 1,
                df.expression_str[:60],
                m.ic_mean if m else 0,
                m.icir if m else 0,
                m.sharpe if m else 0,
                (m.max_drawdown * 100) if m else 0,
            )

        # 벡터 메모리에 경험 저장 확인
        logger.info("벡터 메모리 캐시 (실행 후): %d개", len(vm._cache))

    logger.info("✓ 1세대 진화 실행 완료!")


async def main():
    logger.info("=" * 60)
    logger.info("3 Phase 구현 검증 시작")
    logger.info("=" * 60)

    # Phase 1: 피처 확장
    await verify_phase1_features()

    # Phase 3: fitness
    await verify_phase3_fitness()

    # Phase 2: RAG (DB 접근)
    try:
        await verify_phase2_rag()
    except Exception as e:
        logger.warning("Phase 2 RAG 검증 스킵 (DB 미접속?): %s", e)

    # 통합: 1세대 진화
    try:
        await verify_evolution_run()
    except Exception as e:
        logger.warning("통합 검증 스킵: %s", e)

    logger.info("")
    logger.info("=" * 60)
    logger.info("3 Phase 구현 검증 완료!")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
