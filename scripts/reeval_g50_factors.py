"""g50 팩터 재평가 스크립트.

g50 팩터가 turnover=0, Sharpe 과대평가 상태로 저장된 버그를 수정한다.
현재 정상 평가 파이프라인으로 재평가하여 DB를 업데이트한다.

사용법:
    docker-compose run --rm app python -m scripts.reeval_g50_factors
"""

from __future__ import annotations

import asyncio
import logging
import sys

import sympy
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import async_session
from app.alpha.models import AlphaFactor
from app.alpha.ast_converter import sympy_to_polars
from app.alpha.evaluator import (
    compute_ic_series,
    compute_quantile_returns,
    compute_long_only_returns,
    compute_position_turnover,
    compute_factor_metrics,
    compute_forward_returns,
)
from app.alpha.fitness import compute_composite_fitness
from app.alpha.interval import default_round_trip_cost, is_intraday

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


async def load_data(session: AsyncSession, interval: str, symbols: list[str] | None = None):
    """평가용 캔들 데이터 로딩 (EvolutionEngine과 동일 로직)."""
    from datetime import date, timedelta

    end = date.today()
    start = end - timedelta(days=365)

    if not symbols:
        # pykrx 대신 DB의 stock_masters에서 KOSPI200 종목 가져오기
        from sqlalchemy import text
        result = await session.execute(
            text("SELECT symbol FROM stock_masters WHERE market = 'KOSPI' ORDER BY symbol LIMIT 200")
        )
        symbols = [row[0] for row in result.fetchall()]
        if not symbols:
            logger.error("DB에서 종목을 가져올 수 없습니다")
            return None
        logger.info("DB에서 종목 %d개 로드", len(symbols))

    from app.alpha.interval import max_symbols_for_mining
    max_sym = max_symbols_for_mining(interval)
    if len(symbols) > max_sym:
        symbols = symbols[:max_sym]
        logger.info("OOM 방지: %d종목으로 제한", max_sym)

    from app.backtest.data_loader import load_enriched_candles
    df = await load_enriched_candles(
        symbols=symbols,
        start_date=start,
        end_date=end,
        interval=interval,
    )
    if df is None or df.height == 0:
        logger.error("캔들 데이터 없음")
        return None

    logger.info("캔들 데이터 로드 완료: %d행, %d종목", df.height, df["symbol"].n_unique())

    # 알파 피처 추가 (atr_14, macd_hist, zscore_volume 등)
    from app.alpha.ast_converter import ensure_alpha_features
    df = ensure_alpha_features(df)
    logger.info("알파 피처 추가 완료: %d컬럼", len(df.columns))

    return df


def evaluate_factor_full(expr_str: str, data, interval: str):
    """단일 팩터 전체 메트릭 재계산."""
    import polars as pl

    # open 등 Python 내장 함수와 충돌 방지
    local_dict = {name: sympy.Symbol(name) for name in [
        "open", "close", "high", "low", "volume",
    ]}
    try:
        expr = sympy.sympify(expr_str, locals=local_dict)
    except Exception:
        # expression_sympy 필드 폴백 시도
        logger.warning("sympify 실패, 건너뜀: %s", expr_str[:60])
        return None
    polars_expr = sympy_to_polars(expr)

    if is_intraday(interval):
        from app.alpha.evaluator import _collapse_to_daily
        df = data.with_columns(polars_expr.alias("alpha_factor")).select(
            ["symbol", "dt", "close", "alpha_factor"]
        )
        df = _collapse_to_daily(df, factor_col="alpha_factor")
        df = df.drop_nulls(subset=["alpha_factor", "fwd_return"])
    else:
        df = data.with_columns(polars_expr.alias("alpha_factor"))
        if "fwd_return" not in df.columns:
            df = compute_forward_returns(df, periods=1)
        df = df.filter(
            pl.col("alpha_factor").is_not_null()
            & pl.col("alpha_factor").is_not_nan()
            & pl.col("fwd_return").is_not_null()
            & pl.col("fwd_return").is_not_nan()
        )

    if df.height < 10:
        logger.warning("데이터 부족 (%d행)", df.height)
        return None

    ic_series = compute_ic_series(df, factor_col="alpha_factor")
    ls_returns = compute_quantile_returns(df, factor_col="alpha_factor")
    lo_returns = compute_long_only_returns(df, factor_col="alpha_factor")
    avg_turnover, turnover_series = compute_position_turnover(df, factor_col="alpha_factor")

    metrics = compute_factor_metrics(
        ic_series,
        ls_returns=ls_returns,
        long_only_returns=lo_returns,
        position_turnover=avg_turnover,
        turnover_series=turnover_series,
        annualize=252.0,
        round_trip_cost=default_round_trip_cost(interval),
    )
    return metrics


async def main():
    logger.info("=== g50 팩터 재평가 시작 ===")

    async with async_session() as session:
        # g50 팩터 조회 (turnover=0인 것만 — 정상 평가된 것은 제외)
        result = await session.execute(
            select(AlphaFactor).where(
                AlphaFactor.generation == 50,
                AlphaFactor.turnover == 0.0,
            )
        )
        factors = list(result.scalars().all())
        logger.info("g50 팩터 (turnover=0) %d개 발견", len(factors))

        if not factors:
            logger.error("g50 팩터가 없습니다")
            return

        # 인터벌 확인
        interval = factors[0].interval or "5m"
        logger.info("인터벌: %s", interval)

        # 수식별 중복 제거 (같은 수식은 1번만 평가)
        expr_map: dict[str, list[AlphaFactor]] = {}
        for f in factors:
            expr_map.setdefault(f.expression_str, []).append(f)

        logger.info("고유 수식 %d개", len(expr_map))

        # 데이터 로드
        data = await load_data(session, interval)
        if data is None:
            return

        # 각 수식별 재평가
        updated = 0
        for expr_str, factor_group in expr_map.items():
            logger.info("--- 수식: %s", expr_str[:80])
            logger.info("    기존: sharpe=%.2f, turnover=%.4f, fitness=%.4f",
                        factor_group[0].sharpe or 0,
                        factor_group[0].turnover or 0,
                        factor_group[0].fitness_composite or 0)

            metrics = evaluate_factor_full(expr_str, data, interval)
            if metrics is None:
                logger.warning("    재평가 실패 — 건너뜀")
                continue

            # fitness 재계산
            fitness = compute_composite_fitness(
                ic_mean=metrics.ic_mean,
                icir=metrics.icir,
                turnover=metrics.turnover,
                tree_depth=factor_group[0].tree_depth or 3,
                tree_size=factor_group[0].tree_size or 11,
                sharpe=metrics.sharpe,
                max_drawdown=metrics.max_drawdown,
            )

            logger.info("    재평가: sharpe=%.2f, turnover=%.4f, fitness=%.4f, ic=%.4f, mdd=%.4f",
                        metrics.sharpe, metrics.turnover, fitness, metrics.ic_mean, metrics.max_drawdown)

            # DB 업데이트 (같은 수식의 모든 팩터)
            factor_ids = [f.id for f in factor_group]
            await session.execute(
                update(AlphaFactor)
                .where(AlphaFactor.id.in_(factor_ids))
                .values(
                    ic_mean=metrics.ic_mean,
                    ic_std=metrics.ic_std,
                    icir=metrics.icir,
                    sharpe=metrics.sharpe,
                    turnover=metrics.turnover,
                    max_drawdown=metrics.max_drawdown,
                    fitness_composite=fitness,
                )
            )
            updated += len(factor_ids)

        await session.commit()
        logger.info("=== 완료: %d개 팩터 업데이트 ===", updated)

        # 검증: 업데이트 후 값 확인
        result = await session.execute(
            select(AlphaFactor).where(AlphaFactor.generation == 50)
        )
        for f in result.scalars().all():
            logger.info("  검증 | %s  sharpe=%.2f  turnover=%.4f  fitness=%.4f",
                        f.name, f.sharpe or 0, f.turnover or 0, f.fitness_composite or 0)


if __name__ == "__main__":
    asyncio.run(main())
