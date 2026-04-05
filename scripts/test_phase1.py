"""Phase 1 테스트: P0-1 인과검증 정합성 + P0-2 멀티호라이즌 IC.

Usage: docker exec stockmate-worker python -m scripts.test_phase1
"""
import asyncio
import sys
from datetime import datetime

from app.alpha.evaluator import evaluate_factor, _collapse_to_daily, compute_forward_returns
from app.alpha.ast_converter import ensure_alpha_features
from app.alpha.causal_runner import _prepare_factor_and_validate_sync
from app.alpha.confounders import load_confounders
from app.backtest.data_loader import load_enriched_candles
from app.core.config import settings
from app.core.timezone import KST

SYMBOLS = [
    "005930", "000660", "035420", "051910", "006400", "035720", "068270", "105560", "055550", "003670",
    "207940", "012330", "066570", "003490", "096770", "034730", "028260", "017670", "032830", "010130",
    "086790", "009150", "018260", "010950", "011200", "033780", "036570", "005380", "012450", "000270",
    "138040", "015760", "024110", "316140", "003550", "066910", "032640", "051900", "034020", "004020",
]
START = datetime(2025, 11, 1, tzinfo=KST)
END = datetime(2026, 3, 1, tzinfo=KST)
EXPR = "rsi"  # 단순 팩터


async def test_t1():
    """T1: 인과검증이 장중 수익률을 사용하는지 확인."""
    print("=" * 60)
    print("T1: 인과검증 장중 수익률 정합성")
    print("=" * 60)

    candles = await load_enriched_candles(
        symbols=SYMBOLS, start_date=START, end_date=END, interval="5m",
    )
    base_df = ensure_alpha_features(candles)
    base_df = compute_forward_returns(base_df, periods=1)

    confounders_df = await load_confounders(START.date(), END.date(), SYMBOLS)
    sector_map = {s: i for i, s in enumerate(SYMBOLS)}

    # 설정 확인
    from app.alpha.interval import is_intraday
    print(f"  is_intraday('5m'): {is_intraday('5m')}")
    print(f"  FWD_RETURN_MODE: {settings.ALPHA_FWD_RETURN_MODE}")
    intraday_active = is_intraday("5m") and settings.ALPHA_FWD_RETURN_MODE == "intraday"
    print(f"  장중 수익률 경로 활성: {intraday_active}")

    try:
        result = _prepare_factor_and_validate_sync(
            base_df, EXPR, confounders_df, sector_map, interval="5m",
        )
        print(f"  robust={result.is_causally_robust}")
        print(f"  ATE={result.causal_effect_size:.6f}")
        print(f"  p_value={result.p_value:.4f}")
        if intraday_active:
            print(f"  T1: PASS (장중 수익률 경로 확인)")
        else:
            print(f"  T1: FAIL (intraday 경로 비활성)")
    except Exception as e:
        print(f"  T1: FAIL — {type(e).__name__}: {e}")


async def test_t2():
    """T2: 멀티호라이즌 IC가 정상 산출되는지."""
    print("\n" + "=" * 60)
    print("T2: 멀티호라이즌 IC 산출")
    print("=" * 60)

    candles = await load_enriched_candles(
        symbols=SYMBOLS, start_date=START, end_date=END, interval="5m",
    )

    metrics = evaluate_factor(candles, EXPR, name="test_5m", interval="5m")
    print(f"  IC={metrics.ic_mean:.4f} Sharpe={metrics.sharpe:.2f}")
    print(f"  multi_horizon is None: {metrics.multi_horizon is None}")
    print(f"  optimal_horizon: {metrics.optimal_horizon}")

    if metrics.multi_horizon is not None:
        any_nonzero = False
        for h in [1, 3, 5, 10, 20]:
            if h in metrics.multi_horizon:
                info = metrics.multi_horizon[h]
                print(f"    H={h:2d}일: IC={info['ic_mean']:+.4f} ICIR={info['icir']:+.4f} n={info.get('n_obs',0)}")
                if abs(info["ic_mean"]) > 1e-6:
                    any_nonzero = True
        if any_nonzero:
            print(f"  T2: PASS (멀티호라이즌 IC 정상 계산)")
        else:
            print(f"  T2: WARNING (모든 IC=0 — 종목 수 부족 가능성. 30종목 필터 확인)")
    else:
        print(f"  T2: FAIL (multi_horizon is None)")


async def test_t3():
    """T3: 일봉 팩터에 영향 없는지 (회귀)."""
    print("\n" + "=" * 60)
    print("T3: 일봉 회귀 테스트")
    print("=" * 60)

    candles = await load_enriched_candles(
        symbols=SYMBOLS, start_date=START, end_date=END, interval="1d",
    )

    metrics = evaluate_factor(candles, EXPR, name="test_1d", interval="1d")
    print(f"  IC={metrics.ic_mean:.4f} Sharpe={metrics.sharpe:.2f}")
    print(f"  multi_horizon is None: {metrics.multi_horizon is None}")
    print(f"  optimal_horizon is None: {metrics.optimal_horizon is None}")

    if metrics.multi_horizon is None and metrics.optimal_horizon is None:
        if abs(metrics.ic_mean) > 1e-6:
            print(f"  T3: PASS (일봉 무영향 + IC 정상 계산)")
        else:
            print(f"  T3: WARNING (multi_horizon=None 확인, 하지만 IC=0 — 데이터/팩터 문제)")
    else:
        print(f"  T3: FAIL (일봉에 멀티호라이즌 적용됨)")


async def main():
    print(f"FWD_RETURN_MODE: {settings.ALPHA_FWD_RETURN_MODE}")
    print(f"Symbols: {SYMBOLS}")
    print(f"Period: {START.date()} ~ {END.date()}")

    await test_t1()
    await test_t2()
    await test_t3()

    print("\n" + "=" * 60)
    print("Phase 1 테스트 완료")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
