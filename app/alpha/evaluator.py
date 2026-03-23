"""알파 팩터 IC(Information Coefficient) 평가기.

Polars 네이티브 연산으로 Spearman IC, Long-only/L-S 수익률, 턴오버 계산.
Spearman IC = Pearson(rank(factor), rank(return)) 동치 활용.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from app.alpha.ast_converter import sympy_to_polars, ensure_alpha_features, parse_expression

logger = logging.getLogger(__name__)


@dataclass
class FactorMetrics:
    """팩터 평가 메트릭."""

    ic_mean: float
    ic_std: float
    icir: float
    turnover: float            # 포지션 턴오버 (일별 상위 포트폴리오 변경 비율)
    sharpe: float              # Long-only 포트폴리오 연환산 Sharpe
    max_drawdown: float        # Long-only 포트폴리오 MDD (비율, 예: -0.15 = -15%)
    ic_series: list[float]
    long_only_returns: list[float] = field(default_factory=list)


def compute_forward_returns(
    df: pl.DataFrame, periods: int = 1
) -> pl.DataFrame:
    """T+periods 수익률 컬럼 추가."""
    return df.with_columns(
        (pl.col("close").shift(-periods) / pl.col("close") - 1.0)
        .alias("fwd_return")
    )


def _resolve_date_col(df: pl.DataFrame) -> str:
    """날짜 컬럼 이름 결정."""
    return "dt" if "dt" in df.columns else "date"


def _filter_valid(df: pl.DataFrame, factor_col: str, return_col: str = "fwd_return") -> pl.DataFrame:
    """유효한 행만 필터 (factor + return 모두 non-null/nan)."""
    return df.filter(
        pl.col(factor_col).is_not_null()
        & pl.col(factor_col).is_not_nan()
        & pl.col(return_col).is_not_null()
        & pl.col(return_col).is_not_nan()
    )


def compute_ic_series(
    df: pl.DataFrame,
    factor_col: str = "alpha_factor",
    return_col: str = "fwd_return",
) -> list[float]:
    """일별 cross-sectional Spearman IC 시리즈 계산.

    Spearman IC = Pearson(rank(factor), rank(return)).
    Polars rank().over(date) + pearson_corr() group_by로 벡터화.

    단일 종목일 경우 시계열 IC, 다종목일 경우 날짜별 횡단면 IC.

    Parameters
    ----------
    return_col : 수익률 컬럼명. 기본 "fwd_return". 다중 보유기간 사용 시 변경.
    """
    valid = _filter_valid(df, factor_col, return_col=return_col)

    if valid.height < 10:
        return []

    date_col = _resolve_date_col(valid)

    # symbol 컬럼이 있으면 날짜별 횡단면 IC
    if "symbol" in valid.columns:
        # Polars 벡터화: rank().over(date) → group_by + pearson_corr
        ranked = valid.with_columns([
            pl.col(factor_col).rank().over(date_col).alias("_f_rank"),
            pl.col(return_col).rank().over(date_col).alias("_r_rank"),
            pl.col(factor_col).count().over(date_col).alias("_cnt"),
            pl.col(factor_col).std().over(date_col).alias("_f_std"),
            pl.col(return_col).std().over(date_col).alias("_r_std"),
        ]).filter(
            (pl.col("_cnt") >= 30)
            & (pl.col("_f_std") > 1e-12)
            & (pl.col("_r_std") > 1e-12)
        )

        if ranked.height == 0:
            return []

        ic_df = (
            ranked.group_by(date_col)
            .agg(pl.corr("_f_rank", "_r_rank", method="pearson").alias("ic"))
            .filter(pl.col("ic").is_not_null() & pl.col("ic").is_not_nan())
            .sort(date_col)
        )

        return ic_df["ic"].to_list()

    # 단일 종목: 롤링 윈도우 IC (20일) — 데이터 소량이므로 기존 방식 유지
    window = min(20, valid.height // 3)
    if window < 5:
        factor_vals = valid[factor_col].to_numpy()
        return_vals = valid[return_col].to_numpy()
        corr, _ = spearmanr(factor_vals, return_vals)
        return [float(corr)] if not np.isnan(corr) else []

    ic_list: list[float] = []
    factor_arr = valid[factor_col].to_numpy()
    return_arr = valid[return_col].to_numpy()

    for i in range(window, len(factor_arr)):
        f_slice = factor_arr[i - window : i]
        r_slice = return_arr[i - window : i]

        if np.std(f_slice) < 1e-12 or np.std(r_slice) < 1e-12:
            continue

        corr, _ = spearmanr(f_slice, r_slice)
        if not np.isnan(corr):
            ic_list.append(float(corr))

    if 0 < len(ic_list) < 20:
        logger.warning(
            "IC series length %d < 20: statistical significance may be weak",
            len(ic_list),
        )

    return ic_list


def compute_ic_multi_horizon(
    df: pl.DataFrame,
    factor_col: str = "alpha_factor",
    horizons: list[int] | None = None,
) -> dict[int, dict]:
    """다중 보유기간별 IC 계산 → 최적 보유기간 식별.

    Parameters
    ----------
    horizons : 보유기간 리스트 (봉 수). 기본 [1, 6, 12, 39, 78]
               (5분봉 기준: 5분, 30분, 1시간, 반일, 전일)

    Returns
    -------
    {horizon: {"ic_mean", "icir", "ic_series"}}
    """
    if horizons is None:
        horizons = [1, 6, 12, 39, 78]

    results: dict[int, dict] = {}
    for h in horizons:
        ret_col = f"_fwd_ret_{h}"
        df_h = df.with_columns(
            (pl.col("close").shift(-h).over("symbol") / pl.col("close") - 1.0)
            .alias(ret_col)
        ) if "symbol" in df.columns else df.with_columns(
            (pl.col("close").shift(-h) / pl.col("close") - 1.0)
            .alias(ret_col)
        )

        ic_series = compute_ic_series(df_h, factor_col, return_col=ret_col)
        if ic_series:
            ic_mean = float(np.mean(ic_series))
            ic_std = float(np.std(ic_series, ddof=1)) if len(ic_series) > 1 else 0.0
            icir = ic_mean / ic_std if ic_std > 1e-12 else 0.0
        else:
            ic_mean = 0.0
            icir = 0.0

        results[h] = {
            "ic_mean": round(ic_mean, 6),
            "icir": round(icir, 4),
            "n_obs": len(ic_series),
        }

    # 최적 보유기간 (IC 절대값 기준)
    if results:
        best_h = max(results, key=lambda h: abs(results[h]["ic_mean"]))
        results["optimal_horizon"] = best_h

    return results


# ---------------------------------------------------------------------------
# 배치 IC 계산 (EvolutionEngine 최적화용)
# ---------------------------------------------------------------------------

_BATCH_IC_CHUNK_SIZE = 5  # 5분봉 대규모 데이터 OOM 방지 (rank().over() 메모리 비례)


def compute_ic_series_batch(
    df: pl.DataFrame,
    factor_cols: list[str],
    chunk_size: int = _BATCH_IC_CHUNK_SIZE,
) -> dict[str, list[float]]:
    """여러 팩터의 일별 cross-sectional Spearman IC를 배치 계산.

    compute_ic_series()와 **수학적으로 동치**이되, chunk 단위로
    rank + corr을 배치 실행하여 반복 호출 대비 ~15x 빠르다.

    핵심 설계:
    - 각 팩터의 null 패턴이 다르므로, fwd_return rank도 팩터별로 계산
      (per-factor masked return rank)
    - Polars rank()는 null 입력에 null rank 반환 → 마스킹 후 rank =
      필터 후 rank와 동일한 비-null 집합에서 순위 산출

    Parameters
    ----------
    df : pl.DataFrame
        symbol, dt(or date), fwd_return + 팩터 컬럼들이 포함된 DataFrame.
        fwd_return은 미리 계산되어 있어야 한다.
    factor_cols : list[str]
        IC를 계산할 팩터 컬럼 이름 리스트.
    chunk_size : int
        한 번에 처리할 팩터 수 (메모리 제어, 기본 50).

    Returns
    -------
    dict[str, list[float]]
        팩터 이름 → IC 시리즈 (날짜순 정렬, null/NaN 제거 후).
    """
    if not factor_cols:
        return {}

    if "fwd_return" not in df.columns:
        raise ValueError("fwd_return column required — call compute_forward_returns() first")

    # 단일 종목: 배치 최적화 불가, 개별 호출로 폴백
    if "symbol" not in df.columns:
        return {f: compute_ic_series(df, factor_col=f) for f in factor_cols}

    date_col = _resolve_date_col(df)

    # fwd_return null/NaN 사전 필터 (모든 팩터에 공통)
    base = df.filter(
        pl.col("fwd_return").is_not_null()
        & ~pl.col("fwd_return").is_nan()
    )

    if base.height < 10:
        return {f: [] for f in factor_cols}

    result: dict[str, list[float]] = {}

    for chunk_start in range(0, len(factor_cols), chunk_size):
        chunk = factor_cols[chunk_start: chunk_start + chunk_size]
        chunk_result = _compute_ic_chunk(base, chunk, date_col)
        result.update(chunk_result)

    return result


def _compute_ic_chunk(
    df: pl.DataFrame,
    factor_cols: list[str],
    date_col: str,
) -> dict[str, list[float]]:
    """단일 청크의 IC 배치 계산 (내부 헬퍼).

    Steps:
    1. factor rank + stats + per-factor masked return rank + stats (with_columns)
    2. validity 조건 미달 시 rank nullify (with_columns)
    3. group_by(date).agg([corr(...) for each factor])
    4. IC series 추출
    """
    # Step 1: 배치 rank + stats 계산 (한 번의 with_columns)
    rank_exprs: list[pl.Expr] = []
    for i, f in enumerate(factor_cols):
        valid_mask = pl.col(f).is_not_null() & ~pl.col(f).is_nan()
        masked_ret = pl.when(valid_mask).then(pl.col("fwd_return")).otherwise(None)

        rank_exprs.extend([
            # factor rank (null 입력 → null rank)
            pl.col(f).rank().over(date_col).alias(f"_fR_{i}"),
            # factor count per date (non-null only)
            pl.col(f).count().over(date_col).alias(f"_fC_{i}"),
            # factor std per date
            pl.col(f).std().over(date_col).alias(f"_fS_{i}"),
            # per-factor masked return rank
            masked_ret.rank().over(date_col).alias(f"_rR_{i}"),
            # per-factor masked return std
            masked_ret.std().over(date_col).alias(f"_rS_{i}"),
        ])

    ranked = df.with_columns(rank_exprs)

    # Step 2: validity 조건 미달 시 rank nullify
    nullify_exprs: list[pl.Expr] = []
    for i in range(len(factor_cols)):
        cond = (
            (pl.col(f"_fC_{i}") >= 30)
            & (pl.col(f"_fS_{i}") > 1e-12)
            & (pl.col(f"_rS_{i}") > 1e-12)
        )
        nullify_exprs.append(
            pl.when(cond).then(pl.col(f"_fR_{i}")).otherwise(None).alias(f"_fR_{i}")
        )
        nullify_exprs.append(
            pl.when(cond).then(pl.col(f"_rR_{i}")).otherwise(None).alias(f"_rR_{i}")
        )

    ranked = ranked.with_columns(nullify_exprs)

    # Step 3: 배치 corr (한 번의 group_by로 N개 IC 동시)
    corr_exprs = [
        pl.corr(f"_fR_{i}", f"_rR_{i}", method="pearson").alias(f"_ic_{i}")
        for i in range(len(factor_cols))
    ]

    ic_df = ranked.group_by(date_col).agg(corr_exprs).sort(date_col)

    # Step 4: IC series 추출
    chunk_result: dict[str, list[float]] = {}
    for i, f in enumerate(factor_cols):
        ic_col = ic_df[f"_ic_{i}"]
        valid_ic = ic_col.filter(ic_col.is_not_null() & ~ic_col.is_nan())
        chunk_result[f] = valid_ic.to_list()

    return chunk_result


def compute_quantile_returns(
    df: pl.DataFrame,
    factor_col: str = "alpha_factor",
    n_quantiles: int = 5,
) -> list[float]:
    """일별 Long-Short (Q5-Q1) 수익률 시리즈 계산.

    Polars rank().over(date)로 분위수 배정 → group_by 집계.
    """
    if "symbol" not in df.columns:
        return []

    date_col = _resolve_date_col(df)

    valid = _filter_valid(df, factor_col)

    if valid.height < 10:
        return []

    # rank로 분위수 배정 (0-indexed)
    # rank는 1-based, 0-indexed quantile = (rank-1) * n_quantiles // count
    ranked = valid.with_columns([
        pl.col(factor_col).rank().over(date_col).alias("_rank"),
        pl.col(factor_col).count().over(date_col).alias("_cnt"),
        pl.col(factor_col).std().over(date_col).alias("_f_std"),
    ]).filter(
        (pl.col("_cnt") >= n_quantiles)
        & (pl.col("_f_std") > 1e-12)
    ).with_columns(
        (((pl.col("_rank") - 1) * n_quantiles) // pl.col("_cnt")).alias("_quantile")
    )

    if ranked.height == 0:
        return []

    # Long (최상위) vs Short (최하위) 평균 수익률
    long_df = (
        ranked.filter(pl.col("_quantile") == (n_quantiles - 1))
        .group_by(date_col)
        .agg(pl.col("fwd_return").mean().alias("long_ret"))
    )
    short_df = (
        ranked.filter(pl.col("_quantile") == 0)
        .group_by(date_col)
        .agg(pl.col("fwd_return").mean().alias("short_ret"))
    )

    ls_df = (
        long_df.join(short_df, on=date_col, how="inner")
        .with_columns((pl.col("long_ret") - pl.col("short_ret")).alias("ls_ret"))
        .filter(
            pl.col("ls_ret").is_not_null()
            & pl.col("ls_ret").is_not_nan()
            & pl.col("ls_ret").is_finite()
        )
        .sort(date_col)
    )

    return ls_df["ls_ret"].to_list()


def compute_long_only_returns(
    df: pl.DataFrame,
    factor_col: str = "alpha_factor",
    top_pct: float = 0.2,
) -> list[float]:
    """일별 Long-only (상위 top_pct%) 수익률 시리즈 계산.

    Polars rank(descending=True).over(date)로 상위 종목 선택 → group_by 평균.
    """
    if "symbol" not in df.columns:
        return []

    date_col = _resolve_date_col(df)

    valid = _filter_valid(df, factor_col)

    if valid.height < 10:
        return []

    # 팩터 기준 내림차순 rank (1=최고) + 날짜별 종목 수
    ranked = valid.with_columns([
        pl.col(factor_col).rank(descending=True).over(date_col).alias("_rank"),
        pl.col(factor_col).count().over(date_col).alias("_cnt"),
        pl.col(factor_col).std().over(date_col).alias("_f_std"),
    ]).filter(
        (pl.col("_cnt") >= 3) & (pl.col("_f_std") > 1e-12)
    ).with_columns(
        (pl.col("_cnt").cast(pl.Float64) * top_pct)
        .floor()
        .cast(pl.Int64)
        .clip(lower_bound=1)
        .alias("_n_top")
    ).filter(pl.col("_rank") <= pl.col("_n_top"))

    if ranked.height == 0:
        return []

    lo_df = (
        ranked.group_by(date_col)
        .agg(pl.col("fwd_return").mean().alias("lo_ret"))
        .filter(
            pl.col("lo_ret").is_not_null()
            & pl.col("lo_ret").is_not_nan()
            & pl.col("lo_ret").is_finite()
        )
        .sort(date_col)
    )

    return lo_df["lo_ret"].to_list()


def compute_position_turnover(
    df: pl.DataFrame,
    factor_col: str = "alpha_factor",
    top_pct: float = 0.2,
) -> tuple[float, list[float]]:
    """일별 포지션 변경 비율 계산.

    Polars rank().over(date)로 상위 종목 추출 →
    self-join으로 연속 날짜 간 차이 계산.
    """
    if "symbol" not in df.columns:
        return 0.0, []

    date_col = _resolve_date_col(df)

    valid = df.filter(
        pl.col(factor_col).is_not_null()
        & pl.col(factor_col).is_not_nan()
    )

    if valid.height < 10:
        return 0.0, []

    # 상위 종목 추출
    ranked = valid.with_columns([
        pl.col(factor_col).rank(descending=True).over(date_col).alias("_rank"),
        pl.col(factor_col).count().over(date_col).alias("_cnt"),
    ]).filter(pl.col("_cnt") >= 3).with_columns(
        (pl.col("_cnt").cast(pl.Float64) * top_pct)
        .floor()
        .cast(pl.Int64)
        .clip(lower_bound=1)
        .alias("_n_top")
    ).filter(pl.col("_rank") <= pl.col("_n_top"))

    if ranked.height == 0:
        return 0.0, []

    # 날짜 순서 매핑 (연속 날짜 인덱스)
    dates_sorted = ranked.select(pl.col(date_col)).unique().sort(date_col)
    if dates_sorted.height < 2:
        return 0.0, []

    # 날짜별 상위 종목 집합 + n_top
    top_per_date = ranked.select([date_col, "symbol", "_n_top"])

    # 연속 날짜 쌍 생성
    date_list = dates_sorted[date_col].to_list()

    # 한 번에 모든 날짜별 상위 종목 집합을 dict로 추출
    grouped = top_per_date.group_by(date_col).agg([
        pl.col("symbol").alias("symbols"),
        pl.col("_n_top").first().alias("n_top"),
    ])
    date_to_symbols: dict = {}
    date_to_ntop: dict = {}
    for row in grouped.iter_rows(named=True):
        date_to_symbols[row[date_col]] = set(row["symbols"])
        date_to_ntop[row[date_col]] = row["n_top"]

    turnover_series: list[float] = []
    for i in range(1, len(date_list)):
        prev_dt = date_list[i - 1]
        curr_dt = date_list[i]
        prev_set = date_to_symbols.get(prev_dt)
        curr_set = date_to_symbols.get(curr_dt)
        if prev_set is None or curr_set is None:
            continue
        n_top = date_to_ntop.get(curr_dt, 1)
        changed = len(curr_set - prev_set) + len(prev_set - curr_set)
        daily_turnover = changed / (2 * max(n_top, 1))
        turnover_series.append(daily_turnover)

    avg_turnover = float(np.mean(turnover_series)) if turnover_series else 0.0
    return avg_turnover, turnover_series


def compute_net_sharpe(
    long_only_returns: list[float],
    turnover_series: list[float],
    round_trip_cost: float = 0.0043,
    annualize: float = 252.0,
) -> float:
    """거래 비용 차감 후 Net Sharpe.

    net_return[t] = long_only_return[t] - turnover[t] * round_trip_cost
    Net Sharpe = mean(net) / std(net) * sqrt(annualize)

    Parameters
    ----------
    long_only_returns : list[float]
        봉별 Long-only 수익률 시리즈.
    turnover_series : list[float]
        봉별 포지션 턴오버 시리즈.
    round_trip_cost : float
        왕복 거래 비용 (일봉 0.43%, 분봉 0.33%).
    annualize : float
        연환산 팩터 (일봉 252, 5분봉 19656 등).
    """
    if not long_only_returns or len(long_only_returns) < 2:
        return 0.0

    lo_arr = np.array(long_only_returns)

    # turnover_series 길이 맞춤 (첫 날은 turnover 없음)
    if turnover_series:
        # long_only_returns[0]은 첫 포트폴리오 → turnover 없음
        # long_only_returns[1:]에 turnover_series 매칭
        if len(turnover_series) >= len(lo_arr) - 1:
            to_arr = np.array(turnover_series[:len(lo_arr) - 1])
            costs = np.concatenate([[0.0], to_arr * round_trip_cost])
        else:
            # 길이 불일치 → 평균 턴오버 사용
            avg_to = float(np.mean(turnover_series))
            costs = np.full(len(lo_arr), avg_to * round_trip_cost)
            costs[0] = 0.0
    else:
        costs = np.zeros(len(lo_arr))

    net_returns = lo_arr - costs

    net_mean = float(np.mean(net_returns))
    net_std = float(np.std(net_returns, ddof=1))

    if net_std < 1e-12:
        return 0.0

    return float(net_mean / net_std * np.sqrt(annualize))


def _sanitize_float(value: float) -> float:
    """NaN/Inf를 0.0으로 클렌징."""
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return value


def compute_factor_metrics(
    ic_series: list[float],
    ls_returns: list[float] | None = None,
    long_only_returns: list[float] | None = None,
    position_turnover: float = 0.0,
    turnover_series: list[float] | None = None,
    round_trip_cost: float = 0.0043,
    annualize: float = 252.0,
) -> FactorMetrics:
    """IC 시리즈 + 수익률에서 집계 메트릭 산출.

    Parameters
    ----------
    ic_series : list[float]
        일별 IC 시리즈.
    ls_returns : list[float] | None
        일별 Long-Short 수익률 (참고용, 적합도에는 미사용).
    long_only_returns : list[float] | None
        일별 Long-only 수익률.
    position_turnover : float
        평균 포지션 턴오버 (0~1).
    turnover_series : list[float] | None
        일별 포지션 턴오버 시리즈 (Net Sharpe 계산용).
    """
    if not ic_series:
        return FactorMetrics(
            ic_mean=0.0, ic_std=0.0, icir=0.0,
            turnover=0.0, sharpe=0.0, max_drawdown=0.0,
            ic_series=[],
        )

    arr = np.array(ic_series)
    ic_mean = float(np.mean(arr))
    ic_std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    icir = ic_mean / ic_std if ic_std > 1e-12 else 0.0

    # Long-only 포트폴리오 기반 Sharpe & MDD
    sharpe = 0.0
    max_drawdown = 0.0

    if long_only_returns and len(long_only_returns) > 1:
        # Net Sharpe: 거래비용 차감 후 연환산 Sharpe
        sharpe = compute_net_sharpe(
            long_only_returns,
            turnover_series or [],
            round_trip_cost=round_trip_cost,
            annualize=annualize,
        )

        # MDD: 누적 수익률 곡선 기반 (비율)
        lo_arr = np.array(long_only_returns)
        cum_returns = np.cumprod(1.0 + lo_arr)
        peak = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - peak) / np.where(peak > 1e-12, peak, 1.0)
        max_drawdown = float(np.min(drawdown))
    elif ls_returns and len(ls_returns) > 1:
        # Long-only 데이터 없으면 L/S fallback (단일 종목 등)
        ls_arr = np.array(ls_returns)
        ls_mean = float(np.mean(ls_arr))
        ls_std = float(np.std(ls_arr, ddof=1))
        if ls_std > 1e-12:
            sharpe = ls_mean / ls_std * np.sqrt(annualize)
        cum_returns = np.cumprod(1.0 + ls_arr)
        peak = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - peak) / np.where(peak > 1e-12, peak, 1.0)
        max_drawdown = float(np.min(drawdown))

    return FactorMetrics(
        ic_mean=_sanitize_float(ic_mean),
        ic_std=_sanitize_float(ic_std),
        icir=_sanitize_float(icir),
        turnover=_sanitize_float(position_turnover),
        sharpe=_sanitize_float(sharpe),
        max_drawdown=_sanitize_float(max_drawdown),
        ic_series=ic_series,
        long_only_returns=long_only_returns or [],
    )


def _collapse_to_daily(
    df: pl.DataFrame,
    factor_col: str,
) -> pl.DataFrame:
    """분봉 데이터를 일별 스냅샷으로 축소.

    일별 첫 봉의 팩터값으로 종목 선택, 일별 close-to-close 수익률 사용.
    이는 '매일 장 시작에 리밸런스, 장 끝까지 보유' 전략과 동일하다.

    Returns
    -------
    pl.DataFrame  (dt=Date, symbol, factor_col, fwd_return)
        dt별 하루에 1행씩, fwd_return = 다음 거래일 종가/오늘 종가 - 1
    """
    date_col = _resolve_date_col(df)

    # dt → date 추출
    if df[date_col].dtype in (pl.Datetime, pl.Datetime("ns"), pl.Datetime("us"), pl.Datetime("ms")):
        df = df.with_columns(pl.col(date_col).dt.date().alias("_date"))
    else:
        df = df.with_columns(pl.col(date_col).alias("_date"))

    # 일별 첫 봉의 팩터값, 일별 마지막 봉의 종가
    first_factor = (
        df.sort(["symbol", date_col])
        .group_by(["symbol", "_date"])
        .agg([
            pl.col(factor_col).first().alias(factor_col),
            pl.col("close").last().alias("close"),
        ])
        .sort(["symbol", "_date"])
    )

    # 일별 forward return: close(T+1) / close(T) - 1 (종목별)
    first_factor = first_factor.with_columns(
        (pl.col("close").shift(-1).over("symbol") / pl.col("close") - 1.0)
        .alias("fwd_return")
    ).rename({"_date": date_col})

    return first_factor


def evaluate_factor(
    df: pl.DataFrame,
    expression_str: str,
    name: str = "alpha_factor",
    interval: str = "1d",
) -> FactorMetrics:
    """팩터 하나의 전체 평가 파이프라인.

    1. 수식 파싱 (SymPy)
    2. 기저 지표 보장
    3. 팩터 컬럼 추가 (Polars Expression)
    4. IC / Sharpe / Turnover 계산
    5. 메트릭 집계

    **분봉 팩터 평가 (핵심)**:
    _collapse_to_daily()로 일별 단위 축소 후 IC/Sharpe 모두 일별 기준 평가.

    이유 — Lo (2002), Kakushadze (2016):
    - 바 단위(5분) IC는 자기상관(autocorrelation)으로 과대평가됨
    - √19,656 연환산 → 약한 시그널도 Sharpe 5+ (실매매에서 무너짐)
    - 예) alpha_0_1: 바 단위 IC=0.15, Sharpe=6.32 → 실매매 4일 -131%
    - WorldQuant 등 프로 퀀트도 short-horizon alpha를 일별 수익률로 평가
    - 일별 IC 기준 0.03~0.05가 genuinely strong, Sharpe 2~3이 excellent
    """
    from app.alpha.interval import bars_per_year, default_round_trip_cost, is_intraday

    # 수식 파싱
    expr = parse_expression(expression_str)
    polars_expr = sympy_to_polars(expr)

    # 기저 지표 보장
    df = ensure_alpha_features(df)

    # 팩터 컬럼 추가
    df = df.with_columns(polars_expr.alias(name))

    # ── IC / Sharpe / Turnover 계산 ──
    if is_intraday(interval):
        # 분봉 → 일별 축소 후 평가 (Lo 2002, Kakushadze 2016)
        #
        # _collapse_to_daily: 일별 첫 봉의 팩터값, 당일 종가 → 익일 종가 수익률
        # 이는 "매일 장 시작에 팩터 기준으로 종목 선택, 장 끝까지 보유" 전략과 동일.
        #
        # 바 단위 평가를 하지 않는 이유:
        # - 78개/일의 5분봉은 독립 관측이 아님 (자기상관)
        # - √19,656 연환산은 Sharpe를 ~10배 과대평가
        # - 진정한 예측력이 있는 팩터는 일별 IC에서도 유의미하게 나옴
        df_daily = _collapse_to_daily(df, factor_col=name)
        df_daily = df_daily.drop_nulls(subset=[name, "fwd_return"])

        ic_series = compute_ic_series(df_daily, factor_col=name)
        ls_returns = compute_quantile_returns(df_daily, factor_col=name)
        long_only_returns = compute_long_only_returns(df_daily, factor_col=name)
        pos_turnover, turnover_series = compute_position_turnover(df_daily, factor_col=name)

        sharpe_annualize = 252.0  # 일별 수익률 → 연환산
        sharpe_cost = default_round_trip_cost(interval)
    else:
        # 일봉: 1-bar forward return으로 평가
        df = compute_forward_returns(df, periods=1)
        df = df.drop_nulls(subset=[name, "fwd_return"])

        ic_series = compute_ic_series(df, factor_col=name)
        ls_returns = compute_quantile_returns(df, factor_col=name)
        long_only_returns = compute_long_only_returns(df, factor_col=name)
        pos_turnover, turnover_series = compute_position_turnover(df, factor_col=name)

        sharpe_annualize = 252.0  # bars_per_year("1d") = 252
        sharpe_cost = default_round_trip_cost(interval)

    return compute_factor_metrics(
        ic_series,
        ls_returns=ls_returns,
        long_only_returns=long_only_returns,
        position_turnover=pos_turnover,
        turnover_series=turnover_series,
        round_trip_cost=sharpe_cost,
        annualize=sharpe_annualize,
    )
