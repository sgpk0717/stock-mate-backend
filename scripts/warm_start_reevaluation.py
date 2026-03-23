"""기존 팩터 Warm Start 재평가.

메모리 안전 + 속도 균형 설계:
- 외부 루프: 종목 10개 청크 (피처 1회 생성, 메모리 ~500MB)
- 내부 루프: 전 팩터 수식 적용 → collapse → 일별 결과(4컬럼)만 디스크 임시 저장
- 전 청크 완료 후: 팩터별 일별 결과 로딩 → IC 계산

피크 메모리: ~500MB (10종목 피처) + ~100MB (수식 적용/collapse 임시)

Usage:
    docker compose run --rm app python -m scripts.warm_start_reevaluation --dry-run
    docker compose run --rm app python -m scripts.warm_start_reevaluation
"""

import argparse
import asyncio
import gc
import logging
import math
import os
import shutil
import time
import tempfile

import numpy as np
import polars as pl
from sqlalchemy import text, update

from app.alpha.ast_converter import parse_expression, sympy_to_polars, ensure_alpha_features
from app.alpha.evaluator import (
    compute_ic_series,
    compute_long_only_returns,
    compute_position_turnover,
)
from app.alpha.fitness import compute_composite_fitness
from app.alpha.interval import default_round_trip_cost
from app.alpha.models import AlphaFactor
from app.backtest.data_loader import _load_raw_candles
from app.core.config import settings
from app.core.database import async_session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

IC_THRESHOLD = 0.03
T_STAT_THRESHOLD = 2.0
FACTOR_COL = "_ws_factor"
SYMBOL_CHUNK = 10


async def get_5m_symbols() -> list[str]:
    async with async_session() as db:
        result = await db.execute(
            text("SELECT DISTINCT symbol FROM stock_candles WHERE interval = '5m' ORDER BY symbol")
        )
        return [r[0] for r in result.fetchall()]


async def get_old_factors(cutoff_gen: int) -> list[dict]:
    async with async_session() as db:
        result = await db.execute(text("""
            SELECT DISTINCT ON (expression_str)
                id, expression_str, birth_generation, tree_depth, tree_size
            FROM alpha_factors
            WHERE birth_generation <= :cutoff AND expression_str IS NOT NULL
            ORDER BY expression_str, fitness_composite DESC NULLS LAST
        """), {"cutoff": cutoff_gen})
        return [
            {"id": r[0], "expression_str": r[1], "birth_gen": r[2],
             "tree_depth": r[3] or 3, "tree_size": r[4] or 10}
            for r in result.fetchall()
        ]


def collapse_chunk(df: pl.DataFrame, factor_col: str) -> pl.DataFrame:
    """분봉 → 일별. 일별 첫 봉 팩터값 + close-to-close fwd_return."""
    date_col = "dt"
    if df[date_col].dtype in (pl.Datetime,):
        df = df.with_columns(pl.col(date_col).dt.date().alias("_date"))
    else:
        df = df.with_columns(pl.col(date_col).alias("_date"))

    daily = (
        df.sort(["symbol", date_col])
        .group_by(["symbol", "_date"])
        .agg([
            pl.col(factor_col).first().alias(factor_col),
            pl.col("close").last().alias("close"),
        ])
        .sort(["symbol", "_date"])
    )
    daily = daily.with_columns(
        (pl.col("close").shift(-1).over("symbol") / pl.col("close") - 1.0)
        .alias("fwd_return")
    ).rename({"_date": "dt"})
    return daily.select(["dt", "symbol", factor_col, "fwd_return"]).drop_nulls()


async def main(cutoff_gen: int = 348, dry_run: bool = False) -> None:
    t0 = time.time()

    # 1. 팩터 파싱
    old_factors = await get_old_factors(cutoff_gen)
    logger.info("=== 재평가 대상: %d개 (gen <= %d) ===", len(old_factors), cutoff_gen)
    if not old_factors:
        return

    parsed = []
    parse_fail = 0
    for f in old_factors:
        try:
            expr = parse_expression(f["expression_str"])
            polars_expr = sympy_to_polars(expr)
            parsed.append({**f, "_expr": polars_expr})
        except Exception:
            parse_fail += 1
    logger.info("파싱: %d 성공, %d 실패", len(parsed), parse_fail)
    if not parsed:
        return

    # 2. 임시 디렉토리: 팩터별 일별 결과를 parquet로 저장
    tmp_dir = tempfile.mkdtemp(prefix="ws_reeval_")
    logger.info("임시 디렉토리: %s", tmp_dir)

    # 3. 종목 청크 루프 (외부) → 팩터 루프 (내부)
    symbols = await get_5m_symbols()
    symbol_chunks = [symbols[i:i + SYMBOL_CHUNK] for i in range(0, len(symbols), SYMBOL_CHUNK)]
    total_chunks = len(symbol_chunks)
    logger.info("종목: %d개, 청크: %d개 (×%d종목)", len(symbols), total_chunks, SYMBOL_CHUNK)

    for ci, chunk_syms in enumerate(symbol_chunks):
        # 5분봉 로딩 (10종목)
        df_5m = await _load_raw_candles(chunk_syms, None, None, db_interval="5m", as_datetime=True)
        if df_5m.is_empty():
            continue

        # 피처 생성 (103컬럼, 10종목분만 — ~500MB)
        df_5m = ensure_alpha_features(df_5m)

        # 각 팩터 수식 적용 → collapse → parquet append
        for fi, f in enumerate(parsed):
            try:
                df_f = df_5m.with_columns(f["_expr"].alias(FACTOR_COL))
                daily = collapse_chunk(df_f, FACTOR_COL)
                if daily.is_empty():
                    continue
                # 팩터별 parquet에 append
                pq_path = os.path.join(tmp_dir, f"{fi}.parquet")
                if os.path.exists(pq_path):
                    existing = pl.read_parquet(pq_path)
                    daily = pl.concat([existing, daily])
                    del existing
                daily.write_parquet(pq_path)
            except Exception:
                pass

        del df_5m
        gc.collect()

        if (ci + 1) % 20 == 0 or ci + 1 == total_chunks:
            logger.info("  청크 %d/%d (%.0f%%)", ci + 1, total_chunks, (ci + 1) / total_chunks * 100)

    # 4. 팩터별 IC 계산 (parquet에서 로딩)
    logger.info("=== IC 계산 시작 ===")
    survived = []
    failed = parse_fail
    low_ic = 0
    low_tstat = 0
    low_turnover = 0
    cost = default_round_trip_cost("5m")

    for fi, f in enumerate(parsed):
        pq_path = os.path.join(tmp_dir, f"{fi}.parquet")
        if not os.path.exists(pq_path):
            failed += 1
            continue

        try:
            df_eval = pl.read_parquet(pq_path)
            if df_eval.height < 30:
                failed += 1
                continue

            ic_series = compute_ic_series(df_eval, factor_col=FACTOR_COL)
            if not ic_series:
                failed += 1
                continue

            ic_mean = float(np.mean(ic_series))
            ic_std = float(np.std(ic_series, ddof=1)) if len(ic_series) > 1 else 0.0
            icir = ic_mean / ic_std if ic_std > 1e-12 else 0.0

            if abs(ic_mean) < IC_THRESHOLD:
                low_ic += 1
                continue

            n_obs = len(ic_series)
            t_stat = ic_mean / (ic_std / np.sqrt(n_obs)) if n_obs > 1 and ic_std > 1e-12 else 0.0
            if abs(t_stat) < T_STAT_THRESHOLD:
                low_tstat += 1
                continue

            pos_turnover, turnover_series = compute_position_turnover(df_eval, factor_col=FACTOR_COL)
            if pos_turnover < 0.02:
                low_turnover += 1
                continue

            long_only_returns = compute_long_only_returns(df_eval, factor_col=FACTOR_COL)
            if long_only_returns and turnover_series:
                net_rets = [r - t * cost for r, t in zip(long_only_returns, turnover_series[:len(long_only_returns)])]
                net_mean = float(np.mean(net_rets)) if net_rets else 0.0
                net_std = float(np.std(net_rets, ddof=1)) if len(net_rets) > 1 else 1e-8
                sharpe = (net_mean / max(net_std, 1e-8)) * math.sqrt(252.0)
            else:
                sharpe = 0.0

            if long_only_returns:
                cum = np.cumprod([1 + r for r in long_only_returns])
                peak = np.maximum.accumulate(cum)
                dd = (cum - peak) / np.where(peak > 1e-12, peak, 1.0)
                max_drawdown = float(np.min(dd))
            else:
                max_drawdown = 0.0

            fitness = compute_composite_fitness(
                ic_mean=ic_mean, icir=icir, turnover=pos_turnover,
                tree_depth=f["tree_depth"], tree_size=f["tree_size"],
                sharpe=sharpe, max_drawdown=max_drawdown,
                w_ic=settings.ALPHA_FITNESS_W_IC, w_icir=settings.ALPHA_FITNESS_W_ICIR,
                w_sharpe=settings.ALPHA_FITNESS_W_SHARPE, w_mdd=settings.ALPHA_FITNESS_W_MDD,
                w_turnover=settings.ALPHA_FITNESS_W_TURNOVER, w_complexity=settings.ALPHA_FITNESS_W_COMPLEXITY,
            )

            survived.append({
                "id": f["id"], "expression_str": f["expression_str"],
                "new_ic": ic_mean, "new_icir": icir, "new_sharpe": sharpe,
                "new_turnover": pos_turnover, "t_stat": t_stat, "n_obs": n_obs, "fitness": fitness,
            })
        except Exception:
            failed += 1

        if (fi + 1) % 100 == 0 or fi + 1 == len(parsed):
            logger.info(
                "  IC %d/%d (%.0f%%): 생존 %d, 실패 %d, IC미달 %d, t미달 %d, TO미달 %d",
                fi + 1, len(parsed), (fi + 1) / len(parsed) * 100,
                len(survived), failed, low_ic, low_tstat, low_turnover,
            )

    # 5. 임시 파일 정리
    shutil.rmtree(tmp_dir, ignore_errors=True)
    logger.info("임시 파일 정리 완료")

    # 6. 결과
    rate = len(survived) / len(old_factors) * 100 if old_factors else 0
    logger.info("=== 재평가 완료 ===")
    logger.info("  대상: %d → 생존: %d (%.1f%%)", len(old_factors), len(survived), rate)
    logger.info("  탈락: 실패 %d, IC미달 %d, t미달 %d, TO미달 %d", failed, low_ic, low_tstat, low_turnover)

    if survived:
        ics = [s["new_ic"] for s in survived]
        logger.info("  생존 IC: avg=%.4f, max=%.4f", np.mean(ics), np.max(ics))
        top10 = sorted(survived, key=lambda x: x["fitness"], reverse=True)[:10]
        logger.info("  === Top 10 ===")
        for rank, s in enumerate(top10, 1):
            logger.info("  %2d. IC=%.4f ICIR=%.3f Sharpe=%.2f t=%.1f | %s",
                        rank, s["new_ic"], s["new_icir"], s["new_sharpe"], s["t_stat"],
                        s["expression_str"][:60])

    if dry_run:
        logger.info("=== DRY-RUN: DB 업데이트 없음 ===")
    elif survived:
        logger.info("=== DB 업데이트 중... ===")
        async with async_session() as db:
            for s in survived:
                await db.execute(
                    update(AlphaFactor).where(AlphaFactor.id == s["id"]).values(
                        ic_mean=s["new_ic"], ic_std=0.0, icir=s["new_icir"],
                        sharpe=s["new_sharpe"], turnover=s["new_turnover"],
                        max_drawdown=0.0, fitness_composite=s["fitness"],
                    )
                )
            await db.commit()
        logger.info("=== %d개 업데이트 완료 ===", len(survived))

    logger.info("=== 총 소요: %.1f분 ===", (time.time() - t0) / 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="기존 팩터 Warm Start 재평가")
    parser.add_argument("--cutoff", type=int, default=348)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(cutoff_gen=args.cutoff, dry_run=args.dry_run))
