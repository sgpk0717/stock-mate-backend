"""분봉 fwd_return 장중 수익률 전환 — 전체 재검증 스크립트.

Usage: docker exec stockmate-worker python -m scripts.verify_fwd_return
"""
import asyncio
import polars as pl
import sympy
import numpy as np
from app.core.config import settings
from app.alpha.evaluator import (
    _collapse_to_daily, compute_ic_series, evaluate_factor,
    compute_long_only_returns,
)
from app.alpha.ast_converter import sympy_to_polars, ensure_alpha_features, _ALL_VARIABLES
from app.alpha.cpcv import cpcv_validate
from app.alpha.interval import is_intraday
from app.backtest.data_loader import load_enriched_candles
from app.core.timezone import KST
from app.core.database import async_session
from datetime import datetime
from sqlalchemy import text

SYMBOLS = [
    "005930","000660","035420","051910","006400","035720","068270","105560","055550","003670",
    "207940","012330","066570","003490","096770","034730","028260","017670","032830","010130",
    "086790","009150","018260","010950","011200","033780","036570","005380","012450","000270",
    "138040","015760","024110","316140","003550","066910","032640","051900","034020","004020",
]


async def main():
    print("=" * 70)
    print("전체 재검증 (5개 시나리오) — 실행 기반")
    print("=" * 70)

    # 데이터 로드
    print("\n[데이터 로드]")
    data_1d = await load_enriched_candles(
        symbols=SYMBOLS[:25],
        start_date=datetime(2024, 1, 1, tzinfo=KST),
        end_date=datetime(2026, 4, 1, tzinfo=KST),
        interval="1d",
    )
    data_5m = await load_enriched_candles(
        symbols=SYMBOLS,
        start_date=datetime(2025, 10, 1, tzinfo=KST),
        end_date=datetime(2026, 4, 1, tzinfo=KST),
        interval="5m",
    )
    print(f"  1d: {data_1d.height} rows, {data_1d['symbol'].n_unique()} syms")
    print(f"  5m: {data_5m.height} rows, {data_5m['symbol'].n_unique()} syms")

    # ── S1: 회귀 — 일봉 재계산 ──
    print("\n" + "=" * 70)
    print("S1: 회귀 — 일봉 팩터 재계산 비교")
    print("=" * 70)

    async with async_session() as session:
        r = await session.execute(text(
            "SELECT id::text, expression_str, ic_mean, sharpe FROM alpha_factors "
            "WHERE interval='1d' AND status='validated' ORDER BY sharpe DESC LIMIT 1"
        ))
        row = r.fetchone()

    if row:
        fid, expr_str, db_ic, db_sharpe = row
        print(f"  DB: [{fid[:8]}] IC={db_ic:.4f} Sharpe={db_sharpe:.2f}")
        try:
            metrics = evaluate_factor(data_1d, expr_str, interval="1d")
            print(f"  재계산: IC={metrics.ic_mean:.4f} Sharpe={metrics.sharpe:.2f}")
            sign_ok = (metrics.ic_mean > 0) == (db_ic > 0)
            print(f"  IC 부호 일치: {sign_ok}")
            print(f"  S1: {'PASS' if sign_ok else 'FAIL'}")
        except Exception as e:
            print(f"  S1: ERROR — {e}")
    else:
        print("  일봉 validated 없음 — SKIP")

    # ── S2: 정합성 — 40종목 IC ↔ Sharpe ──
    print("\n" + "=" * 70)
    print("S2: 정합성 — 40종목 분봉 IC ↔ Sharpe")
    print("=" * 70)

    settings.ALPHA_FWD_RETURN_MODE = "intraday"
    enriched = ensure_alpha_features(data_5m, required_cols={"rsi"})
    df = enriched.with_columns(sympy_to_polars(sympy.sympify("rsi")).alias("alpha_factor"))

    collapsed = _collapse_to_daily(df, "alpha_factor")
    collapsed = collapsed.drop_nulls(subset=["alpha_factor", "fwd_return"])

    n_null = collapsed["fwd_return"].is_null().sum()
    syms_per_date = collapsed.group_by("dt").agg(pl.col("symbol").n_unique().alias("n"))
    min_syms = syms_per_date["n"].min()
    max_syms = syms_per_date["n"].max()
    print(f"  collapsed: {collapsed.height} rows, null_fwd={n_null}")
    print(f"  종목수/일: min={min_syms} max={max_syms} (30개 필터 통과: {min_syms >= 30})")

    ic_series = compute_ic_series(collapsed, factor_col="alpha_factor")
    print(f"  IC days: {len(ic_series)}")

    if ic_series:
        mean_ic = np.mean(ic_series)
        lo = compute_long_only_returns(collapsed, factor_col="alpha_factor")
        sharpe = (np.mean(lo) / np.std(lo)) * np.sqrt(252) if lo and np.std(lo) > 0 else 0
        sign_ok = (mean_ic > 0) == (sharpe > 0) if abs(mean_ic) > 0.005 else "IC~0"
        print(f"  IC={mean_ic:.4f} Sharpe={sharpe:.2f} 부호일치={sign_ok}")
        print(f"  S2: {'PASS' if sign_ok == True else 'INCONCLUSIVE' if sign_ok == 'IC~0' else 'FAIL'}")
    else:
        print(f"  S2: FAIL (IC 계산 불가 — 종목수 부족?)")

    # ── S3: CPCV 실행 ──
    print("\n" + "=" * 70)
    print("S3: CPCV 실행 (interval=5m, intraday)")
    print("=" * 70)

    try:
        cpcv_result = cpcv_validate(
            data_5m, "rsi",
            n_groups=5, n_test=2, embargo_days=5,
            ic_threshold=0.01, interval="5m",
        )
        print(f"  passed={cpcv_result.passed} reason={cpcv_result.reason}")
        print(f"  mean_ic={cpcv_result.mean_ic:.4f} pbo={cpcv_result.pbo:.4f}")
        print(f"  paths={len(cpcv_result.paths_ic)}")
        print(f"  S3: PASS (에러 없이 완료)")
    except Exception as e:
        print(f"  S3: FAIL — {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    # ── S4: open 누락 fallback ──
    print("\n" + "=" * 70)
    print("S4: open 컬럼 누락 fallback")
    print("=" * 70)

    raw_no_open = data_5m.drop("open") if "open" in data_5m.columns else data_5m
    enr_no_open = ensure_alpha_features(raw_no_open, required_cols={"rsi"})
    df_no_open = enr_no_open.with_columns(sympy_to_polars(sympy.sympify("rsi")).alias("alpha_factor"))

    settings.ALPHA_FWD_RETURN_MODE = "intraday"
    try:
        fb = _collapse_to_daily(df_no_open, "alpha_factor")
        fb = fb.drop_nulls(subset=["alpha_factor", "fwd_return"])
        last_null = fb.sort("dt").group_by("symbol").tail(1)["fwd_return"].is_null().sum()
        total = fb.sort("dt").group_by("symbol").tail(1).height
        print(f"  결과: {fb.height} rows")
        print(f"  마지막 날 null: {last_null}/{total} (overnight fallback이면 >0)")
        if last_null > 0:
            print(f"  S4: PASS (overnight fallback 확인)")
        else:
            print(f"  S4: INVESTIGATE (null 없음 — intraday 경로?)")
    except Exception as e:
        print(f"  S4: FAIL — {type(e).__name__}: {e}")

    # ── S5: 모드 전환 ──
    print("\n" + "=" * 70)
    print("S5: 모드 전환 (동일 데이터, 동일 팩터)")
    print("=" * 70)

    enriched = ensure_alpha_features(data_5m, required_cols={"rsi"})
    df = enriched.with_columns(sympy_to_polars(sympy.sympify("rsi")).alias("alpha_factor"))

    results = {}
    for mode in ["overnight", "intraday"]:
        settings.ALPHA_FWD_RETURN_MODE = mode
        r = _collapse_to_daily(df, "alpha_factor")
        r = r.drop_nulls(subset=["alpha_factor", "fwd_return"])
        results[mode] = r
        m = r["fwd_return"].mean()
        s = r["fwd_return"].std()
        print(f"  {mode:10s}: rows={r.height} mean_fwd={m:.6f} std_fwd={s:.6f}")

    # 값 비교
    on_fwd = results["overnight"].sort(["symbol", "dt"]).head(3)["fwd_return"].to_list()
    id_fwd = results["intraday"].sort(["symbol", "dt"]).head(3)["fwd_return"].to_list()
    print(f"\n  Overnight 첫3행: {[f'{x:.6f}' for x in on_fwd]}")
    print(f"  Intraday  첫3행: {[f'{x:.6f}' for x in id_fwd]}")

    differ = on_fwd != id_fwd
    print(f"\n  값 차이: {differ}")
    print(f"  S5: {'PASS' if differ else 'FAIL'}")

    settings.ALPHA_FWD_RETURN_MODE = "intraday"

    print("\n" + "=" * 70)
    print("전체 재검증 완료")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
