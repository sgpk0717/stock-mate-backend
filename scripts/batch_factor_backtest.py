"""52개 인과검증 통과 팩터 일괄 백테스트 (순차 실행, DB 메모리 안전).

사용법:
    docker exec stockmate-api python -m scripts.batch_factor_backtest

환경변수:
    BATCH_REBALANCE_FREQ: 리밸런싱 주기 (기본 weekly)
    BATCH_MAX_POSITIONS: 최대 포지션 (기본 30)
    BATCH_TOP_PCT: 상위 % (기본 0.2)
    BATCH_EOD_LIQUIDATION: 장 마감 청산 (기본 false)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from datetime import date, timedelta

logger = logging.getLogger(__name__)


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from sqlalchemy import select
    from app.alpha.models import AlphaFactor
    from app.backtest.cost_model import CostConfig, default_cost_config
    from app.backtest.models import BacktestRun
    from app.core.database import async_session
    from app.alpha.factor_backtest import execute_factor_backtest
    from app.alpha.universe import Universe, resolve_universe

    # ── 파라미터 (환경변수 또는 기본값) ──
    rebalance_freq = os.getenv("BATCH_REBALANCE_FREQ", "weekly")
    max_positions = int(os.getenv("BATCH_MAX_POSITIONS", "30"))
    top_pct = float(os.getenv("BATCH_TOP_PCT", "0.2"))
    eod_liquidation = os.getenv("BATCH_EOD_LIQUIDATION", "false").lower() == "true"
    interval = "5m"
    initial_capital = 100_000_000

    logger.info("=== 일괄 팩터 백테스트 시작 ===")
    logger.info(
        "파라미터: interval=%s, rebalance=%s, max_pos=%d, top_pct=%.1f%%, eod=%s",
        interval, rebalance_freq, max_positions, top_pct * 100, eod_liquidation,
    )

    # ── 인과검증 통과 팩터 로드 ──
    async with async_session() as db:
        result = await db.execute(
            select(AlphaFactor)
            .where(
                AlphaFactor.factor_type == "single",
                AlphaFactor.causal_robust == True,  # noqa: E712
            )
            .order_by(AlphaFactor.sharpe.desc().nullslast())
        )
        factors = list(result.scalars().all())

    logger.info("인과검증 통과 팩터: %d개", len(factors))
    if not factors:
        logger.error("팩터 없음. 종료.")
        return

    # ── 유니버스 심볼 ──
    symbols = await resolve_universe(Universe.KOSPI200)
    logger.info("유니버스: KOSPI200 %d종목", len(symbols))

    # ── 기간 (전체 5m 데이터 범위) ──
    start_date = date(2025, 2, 18)
    end_date = date.today() - timedelta(days=1)

    # ── 순차 실행 ──
    results: list[dict] = []
    t0 = time.time()

    for i, factor in enumerate(factors):
        run_id = uuid.uuid4()
        logger.info(
            "[%d/%d] %s (IC=%.4f, Sharpe=%.2f) 시작...",
            i + 1, len(factors),
            factor.name or factor.expression_str[:40],
            factor.ic_mean or 0,
            factor.sharpe or 0,
        )

        # DB에 PENDING 레코드 생성
        async with async_session() as db:
            run = BacktestRun(
                id=run_id,
                strategy_name=f"Alpha: {factor.name or factor.expression_str[:30]}",
                strategy_json={
                    "name": f"Alpha: {factor.name}",
                    "expression": factor.expression_str,
                    "mode": "cross_sectional_portfolio",
                    "interval": interval,
                    "top_pct": top_pct,
                    "max_positions": max_positions,
                    "rebalance_freq": rebalance_freq,
                    "eod_liquidation": eod_liquidation,
                    "factor_id": str(factor.id),
                },
                start_date=start_date,
                end_date=end_date,
                status="PENDING",
            )
            db.add(run)
            await db.commit()

        # 백테스트 실행 (동기적 — 1개씩 순차)
        try:
            await execute_factor_backtest(
                run_id=run_id,
                expression_str=factor.expression_str,
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                top_pct=top_pct,
                max_positions=max_positions,
                rebalance_freq=rebalance_freq,
                band_threshold=0.05,
                interval=interval,
                stop_loss_pct=0.0,
                max_drawdown_pct=0.0,
                eod_liquidation=eod_liquidation,
                skip_opening_minutes=0,
                engine="loop",
            )
        except Exception as e:
            logger.error("[%d/%d] 실패: %s", i + 1, len(factors), e)
            continue

        # 결과 조회
        async with async_session() as db:
            run_result = await db.get(BacktestRun, run_id)
            if run_result and run_result.metrics:
                m = run_result.metrics
                sharpe = m.get("sharpe_ratio", 0) or 0
                ret = m.get("total_return", 0) or 0
                trades = m.get("total_trades", 0) or 0
                results.append({
                    "name": factor.name,
                    "ic": factor.ic_mean,
                    "sharpe_mining": factor.sharpe,
                    "sharpe_bt": sharpe,
                    "return": ret,
                    "trades": trades,
                    "status": run_result.status,
                })
                elapsed = time.time() - t0
                logger.info(
                    "[%d/%d] 완료: Sharpe=%.2f, Return=%.1f%%, Trades=%d (경과 %.0f초)",
                    i + 1, len(factors), sharpe, ret, trades, elapsed,
                )

    # ── 결과 요약 ──
    logger.info("\n=== 결과 요약 ===")
    completed = [r for r in results if r["status"] == "COMPLETED"]
    survived = [r for r in completed if r["sharpe_bt"] > 0]

    logger.info("총 팩터: %d", len(factors))
    logger.info("완료: %d", len(completed))
    logger.info("비용 후 Sharpe > 0: %d (%d%%)", len(survived), len(survived) * 100 // max(len(completed), 1))

    if survived:
        logger.info("\n생존 팩터 (Sharpe 내림차순):")
        for r in sorted(survived, key=lambda x: x["sharpe_bt"], reverse=True):
            logger.info(
                "  %s: BT_Sharpe=%.2f, Return=%.1f%%, Mining_Sharpe=%.2f, IC=%.4f",
                r["name"], r["sharpe_bt"], r["return"], r["sharpe_mining"] or 0, r["ic"] or 0,
            )

    logger.info("\n판정:")
    if len(survived) >= 22:
        logger.info("  ✓ 시스템 유효 (22/52 이상)")
    elif len(survived) >= 5:
        logger.info("  △ 부분 유효 (5~21개)")
    else:
        logger.info("  ✗ 마이닝 개선 필요 (5개 미만)")

    logger.info("총 소요: %.1f분", (time.time() - t0) / 60)


if __name__ == "__main__":
    asyncio.run(main())
