"""IC Decay Test — fwd_return 시작점 이동에 따른 IC 변화 측정.

시그널-수익률 간격을 0봉(겹침)/1봉(수정)/2봉/5봉으로 변화시키며
IC가 급락하면 기계적 착시, 완만 감쇠면 진짜 알파.

Usage: docker exec stockmate-worker python -m scripts.ic_decay_test
"""
import asyncio
import polars as pl
import sympy
import numpy as np
from app.alpha.ast_converter import sympy_to_polars, ensure_alpha_features, _ALL_VARIABLES
from app.alpha.evaluator import compute_ic_series, _resolve_date_col, _filter_valid
from app.backtest.data_loader import load_enriched_candles
from app.core.timezone import KST
from datetime import datetime

SYMBOLS = [
    "005930","000660","035420","051910","006400","035720","068270","105560","055550","003670",
    "207940","012330","066570","003490","096770","034730","028260","017670","032830","010130",
    "086790","009150","018260","010950","011200","033780","036570","005380","012450","000270",
    "138040","015760","024110","316140","003550","066910","032640","051900","034020","004020",
]

# 테스트 팩터: 기존 마이닝에서 Sharpe 8+ 나온 수식
TEST_FACTORS = [
    ("price_momentum", "Cs_Rank_volume * price_change_pct"),
    ("simple_rsi", "rsi"),
    ("volume_rank", "Cs_Rank_volume"),
]

SKIP_BARS = [0, 1, 2, 5]  # 시그널 후 N봉 스킵


def collapse_with_skip(df: pl.DataFrame, factor_col: str, skip_bars: int) -> pl.DataFrame:
    """분봉 데이터를 일별 축소하되, fwd_return 시작점을 N봉 스킵."""
    date_col = _resolve_date_col(df)

    if df[date_col].dtype in (pl.Datetime, pl.Datetime("ns"), pl.Datetime("us"), pl.Datetime("ms")):
        df = df.with_columns(pl.col(date_col).dt.date().alias("_date"))
    else:
        df = df.with_columns(pl.col(date_col).alias("_date"))

    # 일별 집계
    # 팩터값: 첫 봉 (시그널 확정 시점)
    # 진입가: (skip_bars+1)번째 봉의 시가
    # 청산가: 마지막 봉의 종가
    entry_idx = skip_bars + 1  # 0-indexed에서 skip_bars+1번째

    agg_exprs = [
        pl.col(factor_col).first().alias(factor_col),
        pl.col("close").last().alias("close"),
        pl.col("open").sort_by(date_col).gather(entry_idx).first().alias("_entry"),
        pl.col(date_col).len().alias("_bar_count"),
    ]

    result = (
        df.sort(["symbol", date_col])
        .group_by(["symbol", "_date"])
        .agg(agg_exprs)
        .sort(["symbol", "_date"])
    )

    # 봉 수가 entry_idx보다 적은 날은 제외
    result = result.filter(pl.col("_bar_count") > entry_idx)

    # fwd_return = close(마지막봉) / open(entry봉) - 1
    result = result.with_columns(
        pl.when(pl.col("_entry") > 0)
        .then(pl.col("close") / pl.col("_entry") - 1.0)
        .otherwise(None)
        .alias("fwd_return")
    ).drop(["_entry", "_bar_count"]).rename({"_date": date_col})

    return result


async def main():
    print("=" * 70)
    print("IC Decay Test — fwd_return 시작점 이동에 따른 IC 변화")
    print("=" * 70)

    data = await load_enriched_candles(
        symbols=SYMBOLS,
        start_date=datetime(2025, 10, 1, tzinfo=KST),
        end_date=datetime(2026, 4, 1, tzinfo=KST),
        interval="5m",
    )
    print(f"Data: {data.height} rows, {data['symbol'].n_unique()} symbols\n")

    for name, expr_str in TEST_FACTORS:
        print(f"{'='*50}")
        print(f"팩터: {name} = {expr_str}")
        print(f"{'='*50}")

        try:
            expr = sympy.sympify(expr_str)
            req = {_ALL_VARIABLES.get(str(s), str(s)) for s in expr.free_symbols}
            enriched = ensure_alpha_features(data, required_cols=req)
            polars_expr = sympy_to_polars(expr)
            df = enriched.with_columns(polars_expr.alias("alpha_factor"))
        except Exception as e:
            print(f"  ERROR: {e}\n")
            continue

        print(f"  {'Skip':>6s}  {'Entry':>10s}  {'IC':>8s}  {'ICIR':>8s}  {'Days':>5s}  {'변화':>8s}")
        print(f"  {'-'*55}")

        base_ic = None
        for skip in SKIP_BARS:
            try:
                collapsed = collapse_with_skip(df, "alpha_factor", skip)
                collapsed = collapsed.drop_nulls(subset=["alpha_factor", "fwd_return"])

                ic_series = compute_ic_series(collapsed, factor_col="alpha_factor")
                if ic_series:
                    mean_ic = np.mean(ic_series)
                    std_ic = np.std(ic_series)
                    icir = mean_ic / std_ic if std_ic > 0 else 0
                else:
                    # 30종목 미만 필터 → 수동 계산
                    from scipy import stats
                    dates = collapsed["dt"].unique().sort().to_list()
                    manual_ic = []
                    for d in dates:
                        day = collapsed.filter(pl.col("dt") == d)
                        if day.height >= 5:
                            f_vals = day["alpha_factor"].to_numpy()
                            r_vals = day["fwd_return"].to_numpy()
                            mask = np.isfinite(f_vals) & np.isfinite(r_vals)
                            if mask.sum() >= 5:
                                corr, _ = stats.spearmanr(f_vals[mask], r_vals[mask])
                                if np.isfinite(corr):
                                    manual_ic.append(corr)
                    mean_ic = np.mean(manual_ic) if manual_ic else 0
                    std_ic = np.std(manual_ic) if manual_ic else 0
                    icir = mean_ic / std_ic if std_ic > 0 else 0
                    ic_series = manual_ic

                entry_time = f"09:{(skip+1)*5:02d}"
                if base_ic is None:
                    base_ic = mean_ic
                    change = ""
                else:
                    pct = ((mean_ic - base_ic) / abs(base_ic) * 100) if base_ic != 0 else 0
                    change = f"{pct:+.0f}%"

                print(f"  {skip:>5d}봉  {entry_time:>10s}  {mean_ic:>8.4f}  {icir:>8.4f}  {len(ic_series):>5d}  {change:>8s}")

            except Exception as e:
                print(f"  {skip:>5d}봉  ERROR: {e}")

        print()

    print("해석: skip 0→1에서 IC가 50% 이상 급락하면 기계적 착시")
    print("      완만한 감쇠(10~20%)면 진짜 알파 존재 가능")


if __name__ == "__main__":
    asyncio.run(main())
