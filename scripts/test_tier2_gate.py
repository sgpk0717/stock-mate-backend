"""기존 discovered/validated 팩터에 Tier 2 게이트를 일회성으로 적용하는 검증 스크립트.

Usage:
    docker exec stockmate-worker python -m scripts.test_tier2_gate
"""
from __future__ import annotations

import asyncio
import logging
import sys
import time

import polars as pl
import sympy
from sqlalchemy import select, text

from app.alpha.ast_converter import ensure_alpha_features, get_required_columns
from app.alpha.surrogate_eval import compute_robustness_gate
from app.core.database import async_session
from app.alpha.models import AlphaFactor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("test_tier2")

# ── 테스트 대상 팩터 ID ──
TARGET_FACTOR_IDS = [
    # 복합(MegaAlpha) 일봉 — Sharpe 0.58, IC 0.147
    "16803d4c-6768-45f0-a6d6-d0f2121e5261",
    # 복합 5분봉 — Sharpe 0.90, IC 0.081
    "83905249-760b-419b-9f65-79999eef9c5b",
    # 단일 일봉 상위 1 — Sharpe 8.81 (의심 과적합)
    "b6a9e6d3-6e55-4c53-a362-85a5a00b453f",
    # 단일 일봉 상위 2 — Sharpe 3.20
    "5b7d02a0-9718-492a-84b5-95cfe69f405b",
    # 단일 5분봉 상위 1 — Sharpe 2.19, IC 0.031
    "87fbac12-5b85-4e06-a37d-952d6ba3ce3f",
    # 단일 5분봉 상위 2 — Sharpe 1.77, IC 0.035
    "3030b88c-60af-43fe-a4a2-9bbc8885b6b8",
]


async def load_factor_expressions() -> list[dict]:
    """DB에서 팩터 수식 로드."""
    factors = []
    async with async_session() as session:
        for fid in TARGET_FACTOR_IDS:
            result = await session.execute(
                select(AlphaFactor).where(AlphaFactor.id == fid)
            )
            factor = result.scalar_one_or_none()
            if factor is None:
                logger.warning("Factor %s not found in DB — skip", fid)
                continue
            factors.append({
                "id": str(factor.id),
                "expression_str": factor.expression_str,
                "factor_type": factor.factor_type,
                "ic_mean": factor.ic_mean,
                "sharpe": factor.sharpe,
                "interval": factor.interval,
                "status": factor.status,
            })
    return factors


async def load_candle_data(interval: str) -> pl.DataFrame:
    """enriched 캔들 데이터 로드 — 마이닝과 동일한 파이프라인 사용."""
    from app.backtest.data_loader import load_enriched_candles
    from datetime import datetime, timedelta
    from app.core.timezone import KST

    # 마이닝에서 사용하는 대표 유니버스 (KOSPI 대형주)
    symbols = [
        "005930", "000660", "035420", "051910", "006400",  # 삼성전자, SK하이닉스, NAVER, LG화학, 삼성SDI
        "035720", "068270", "105560", "055550", "003670",  # 카카오, 셀트리온, KB금융, 신한지주, 포스코홀딩스
        "207940", "012330", "066570", "003490", "096770",  # 삼성바이오, 현대모비스, LG전자, 대한항공, SK이노
        "034730", "028260", "017670", "032830", "010130",  # SK, 삼성물산, SK텔레콤, 삼성생명, 고려아연
        "086790", "009150", "018260", "010950", "011200",  # 하나금융, 삼성전기, 삼성에스디에스, S-Oil, HMM
    ]

    if interval == "1d":
        start = datetime(2022, 1, 1, tzinfo=KST)
        end = datetime(2026, 4, 1, tzinfo=KST)
    else:
        start = datetime(2025, 10, 1, tzinfo=KST)
        end = datetime(2026, 4, 1, tzinfo=KST)

    logger.info("load_enriched_candles 호출: %s, %d종목, %s~%s", interval, len(symbols), start.date(), end.date())

    df = await load_enriched_candles(
        symbols=symbols,
        start_date=start,
        end_date=end,
        interval=interval,
    )

    logger.info(
        "Loaded %d rows (%d symbols, %s ~ %s) for interval=%s (%d columns)",
        df.height, df["symbol"].n_unique(),
        df["dt"].min(), df["dt"].max(), interval, df.width,
    )

    # enrichment: fwd_return 계산
    df = df.sort(["symbol", "dt"])
    df = df.with_columns(
        pl.col("close").shift(-1).over("symbol").alias("_next_close")
    )
    df = df.with_columns(
        ((pl.col("_next_close") - pl.col("close")) / pl.col("close"))
        .alias("fwd_return")
    ).drop("_next_close")
    df = df.filter(pl.col("fwd_return").is_not_null())

    return df


def extract_tier2_segment(df: pl.DataFrame) -> pl.DataFrame:
    """전체 데이터에서 Tier2 구간(50%~70%) 추출."""
    dates = sorted(df["dt"].unique().to_list())
    n = len(dates)
    start_idx = int(n * 0.50)
    end_idx = int(n * 0.70)
    tier2_start = dates[start_idx]
    tier2_end = dates[end_idx]
    tier2 = df.filter(
        (pl.col("dt") >= tier2_start) & (pl.col("dt") <= tier2_end)
    )
    logger.info(
        "Tier2 segment: %s ~ %s (%d rows, %d symbols)",
        tier2_start, tier2_end, tier2.height, tier2["symbol"].n_unique(),
    )
    return tier2


async def main():
    logger.info("=" * 70)
    logger.info("Tier 2 Surrogate Evaluation — 기존 팩터 검증 스크립트")
    logger.info("=" * 70)

    # 1. 팩터 로드
    factors = await load_factor_expressions()
    if not factors:
        logger.error("No factors loaded — exit")
        return

    logger.info("테스트 대상: %d개 팩터", len(factors))
    for f in factors:
        logger.info(
            "  [%s] %s | type=%s interval=%s IC=%.4f Sharpe=%.2f status=%s",
            f["id"][:8], f["expression_str"][:60],
            f["factor_type"], f["interval"],
            f["ic_mean"] or 0, f["sharpe"] or 0, f["status"],
        )

    # 2. 캔들 데이터 로드 (일봉 + 5분봉)
    data_cache: dict[str, pl.DataFrame] = {}

    intervals_needed = set(f["interval"] for f in factors)
    for iv in intervals_needed:
        df = await load_candle_data(iv)
        if df.height > 0:
            data_cache[iv] = extract_tier2_segment(df)

    # 3. 각 팩터에 Tier 2 게이트 적용
    logger.info("")
    logger.info("=" * 70)
    logger.info("Tier 2 평가 시작")
    logger.info("=" * 70)

    results = []
    for f in factors:
        iv = f["interval"]
        tier2_data = data_cache.get(iv)
        if tier2_data is None or tier2_data.height == 0:
            logger.warning("[%s] interval=%s 데이터 없음 — skip", f["id"][:8], iv)
            results.append({**f, "tier2_passed": None, "tier2_details": {}})
            continue

        try:
            expr = sympy.sympify(f["expression_str"])
        except Exception as e:
            logger.error("[%s] SymPy 파싱 실패: %s", f["id"][:8], e)
            results.append({**f, "tier2_passed": None, "tier2_details": {"error": str(e)}})
            continue

        logger.info("")
        logger.info("─── [%s] %s ───", f["id"][:8], f["expression_str"][:70])
        t0 = time.monotonic()

        passed, details = compute_robustness_gate(
            factor_expr=expr,
            tier2_data=tier2_data,
            stop_loss_grid=(0.0, 0.10, 0.15),
            trailing_stop_grid=(0.0, 0.30, 0.50),
        )

        elapsed = time.monotonic() - t0
        result_str = "PASS ✓" if passed else "FAIL ✗"
        median_sharpe = details.get("median_sharpe", 0)
        positive_pct = details.get("positive_pct", 0)
        n_symbols = details.get("n_symbols", 0)

        logger.info(
            "  결과: %s | median_sharpe=%.4f | positive=%.0f%% | symbols=%d | %.1fs",
            result_str, median_sharpe, positive_pct * 100, n_symbols, elapsed,
        )

        # 상세 조합 결과 출력
        for combo in details.get("combo_results", []):
            sl = combo.get("stop_loss", 0)
            ts = combo.get("trailing_stop", 0)
            sh = combo.get("sharpe", 0)
            mdd = combo.get("mdd", 0)
            ret = combo.get("total_return", 0)
            logger.info(
                "    SL=%.0f%% TS=%.0f%% → Sharpe=%.4f MDD=%.1f%% Return=%.1f%%",
                sl * 100, ts * 100, sh, mdd * 100, ret * 100,
            )

        results.append({**f, "tier2_passed": passed, "tier2_details": details})

    # 4. 최종 요약
    logger.info("")
    logger.info("=" * 70)
    logger.info("최종 요약")
    logger.info("=" * 70)
    logger.info("%-10s %-8s %-8s %-10s %-10s %-10s %s",
                "ID", "IV", "Type", "Sharpe", "Tier2", "Med.Sharpe", "Expression")
    logger.info("-" * 100)

    for r in results:
        status = "PASS" if r["tier2_passed"] is True else "FAIL" if r["tier2_passed"] is False else "SKIP"
        med_sh = r["tier2_details"].get("median_sharpe", 0) if r["tier2_details"] else 0
        logger.info(
            "%-10s %-8s %-8s %-10.2f %-10s %-10.4f %s",
            r["id"][:8], r["interval"], r["factor_type"],
            r["sharpe"] or 0, status, med_sh,
            r["expression_str"][:50],
        )

    passed_count = sum(1 for r in results if r["tier2_passed"] is True)
    failed_count = sum(1 for r in results if r["tier2_passed"] is False)
    skip_count = sum(1 for r in results if r["tier2_passed"] is None)
    logger.info("")
    logger.info("통과: %d | 탈락: %d | 스킵: %d | 총: %d", passed_count, failed_count, skip_count, len(results))


if __name__ == "__main__":
    asyncio.run(main())
