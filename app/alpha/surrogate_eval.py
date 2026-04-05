"""경량 포트폴리오 시뮬레이터 — 마이닝 Phase 3 CPCV 전 pass/fail 게이트.

팩터의 실전 강건성을 9개 파라미터 조합(손절 × 트레일링)으로 빠르게
검증하여, CPCV에 투입할 가치가 있는 팩터만 통과시킨다.

- DB 접근 없음 (이미 메모리에 올라온 DataFrame 사용)
- 동기 함수 (to_thread 안에서 호출)
- Trade 객체 없음 — 집계 통계만 반환
"""

from __future__ import annotations

import logging
import math
import signal
import time
from itertools import product

import polars as pl
import sympy

_FACTOR_TIMEOUT_SEC = 30  # 팩터 1개당 최대 평가 시간

from app.alpha.ast_converter import (
    ensure_alpha_features,
    get_required_columns,
    sympy_to_polars,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_robustness_gate(
    factor_expr: sympy.Basic,
    tier2_data: pl.DataFrame,
    top_pct: float = 0.2,
    max_positions: int = 20,
    round_trip_cost: float = 0.0043,
    stop_loss_grid: tuple[float, ...] = (0.0, 0.10, 0.15),
    trailing_stop_grid: tuple[float, ...] = (0.0, 0.30, 0.50),
) -> tuple[bool, dict]:
    """팩터의 파라미터 강건성을 경량 포트폴리오 시뮬레이션으로 검증.

    Parameters
    ----------
    factor_expr : sympy.Basic
        알파 팩터 SymPy 수식.
    tier2_data : pl.DataFrame
        dt, symbol, close, fwd_return, volume + 기타 팩터 피처 컬럼.
    top_pct : float
        매수 대상 상위 퍼센타일 (0~1).
    max_positions : int
        동시 최대 보유 종목 수.
    round_trip_cost : float
        왕복 거래비용 비율 (매수+매도 합산).
    stop_loss_grid : tuple[float, ...]
        손절 비율 그리드 (0.0 = 손절 없음).
    trailing_stop_grid : tuple[float, ...]
        트레일링 스탑 비율 그리드 (0.0 = 트레일링 없음).

    Returns
    -------
    tuple[bool, dict]
        (passed, details). 9개 Sharpe의 중앙값 >= 0이면 passed=True.
    """
    t0 = time.monotonic()
    try:
        # ── 1. 팩터값 계산 + 횡단면 퍼센타일 랭크 ──
        ranked_df = _compute_factor_rank(factor_expr, tier2_data)
        if ranked_df.height == 0:
            logger.warning("Robustness gate: empty DataFrame after factor computation")
            return False, {}

        # ── 1b. 최소 종목 수 체크 ──
        n_symbols = ranked_df["symbol"].n_unique()
        if n_symbols < 10:
            logger.warning("Robustness gate: 종목 수 부족 (%d < 10) — skip", n_symbols)
            return False, {"skip_reason": "min_symbols", "n_symbols": n_symbols}

        # ── 2. 주간 리밸런싱 날짜 계산 ──
        rebalance_dates = _weekly_rebalance_dates(ranked_df)
        if not rebalance_dates:
            logger.warning("Robustness gate: no rebalance dates found")
            return False, {}

        # ── 3. 날짜별 데이터 dict 구축 (루프 성능 최적화) ──
        date_data = _build_date_data(ranked_df)

        # ── 4. 9개 파라미터 조합별 시뮬레이션 ──
        all_dates_sorted = sorted(date_data.keys())
        sharpe_list: list[float] = []
        combo_results: list[dict] = []

        for sl, ts in product(stop_loss_grid, trailing_stop_grid):
            # 타임아웃 체크
            elapsed = time.monotonic() - t0
            if elapsed > _FACTOR_TIMEOUT_SEC:
                logger.warning(
                    "Robustness gate timeout: %.1fs > %ds — evaluated %d/%d combos",
                    elapsed, _FACTOR_TIMEOUT_SEC, len(sharpe_list),
                    len(stop_loss_grid) * len(trailing_stop_grid),
                )
                break

            result = _simulate_portfolio(
                date_data=date_data,
                all_dates_sorted=all_dates_sorted,
                rebalance_dates=rebalance_dates,
                stop_loss_pct=sl,
                trailing_stop_pct=ts,
                round_trip_cost=round_trip_cost,
                max_positions=max_positions,
                top_pct=top_pct,
            )
            sharpe_list.append(result["sharpe"])
            combo_results.append({"stop_loss": sl, "trailing_stop": ts, **result})
            logger.debug(
                "Robustness combo SL=%.2f TS=%.2f → Sharpe=%.4f MDD=%.4f Return=%.4f",
                sl, ts, result["sharpe"], result["mdd"], result["total_return"],
            )

        if not sharpe_list:
            logger.warning("Robustness gate: no combo completed")
            return False, {"skip_reason": "no_combos"}

        # ── 5. 결과 집계 ──
        median_sharpe = float(sorted(sharpe_list)[len(sharpe_list) // 2])
        positive_pct = sum(1 for s in sharpe_list if s > 0) / len(sharpe_list)
        # pass 기준: median Sharpe ≥ 0 AND 67% 이상 조합이 양수
        passed = median_sharpe >= 0.0 and positive_pct >= 0.67

        details = {
            "median_sharpe": median_sharpe,
            "positive_pct": positive_pct,
            "sharpe_list": sharpe_list,
            "combo_results": combo_results,
            "n_combos": len(sharpe_list),
            "n_rebalance_dates": len(rebalance_dates),
            "n_dates": len(all_dates_sorted),
            "n_symbols": n_symbols,
            "elapsed_sec": round(time.monotonic() - t0, 2),
        }

        logger.info(
            "Robustness gate %s: median_sharpe=%.4f positive_pct=%.0f%% (%d combos, %.1fs)",
            "PASSED" if passed else "FAILED",
            median_sharpe,
            positive_pct * 100,
            len(sharpe_list),
            time.monotonic() - t0,
        )
        return passed, details

    except Exception:
        logger.exception("Robustness gate error — returning failed")
        return False, {"skip_reason": "exception", "elapsed_sec": round(time.monotonic() - t0, 2)}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_factor_rank(
    factor_expr: sympy.Basic,
    df: pl.DataFrame,
) -> pl.DataFrame:
    """팩터 수식을 Polars로 변환하고 횡단면 퍼센타일 랭크를 계산.

    NOTE: 실제 마이닝 파이프라인에서는 evolution_engine이 전체 DataFrame에
    ensure_alpha_features()를 이미 적용한 상태로 tier2_data를 넘겨준다.
    여기서는 누락된 피처가 있을 경우에만 보충한다.
    """

    polars_expr = sympy_to_polars(factor_expr)

    # 필요 컬럼 추출 — SymPy 심볼명 → Polars 내부 컬럼명 매핑
    required_cols: set[str] = set()
    for sym in factor_expr.free_symbols:
        sym_name = str(sym)
        # NAMED_VARIABLE_MAP에서 Polars 컬럼명 조회
        from app.alpha.ast_converter import _ALL_VARIABLES
        mapped = _ALL_VARIABLES.get(sym_name, sym_name)
        required_cols.add(mapped)

    # 10행 미만 종목 제거
    sym_counts = df.group_by("symbol").len()
    valid_syms = sym_counts.filter(pl.col("len") >= 10)["symbol"].to_list()
    full = df.filter(pl.col("symbol").is_in(valid_syms)).sort(["symbol", "dt"])

    if full.height == 0:
        return pl.DataFrame()

    # 전체 DataFrame에 피처 보충 (횡단면 피처 포함)
    try:
        full = ensure_alpha_features(full, required_cols=required_cols)
    except Exception as e:
        logger.warning("ensure_alpha_features failed: %s", e)
        return pl.DataFrame()

    # 팩터 수식 적용
    full = full.with_columns(polars_expr.alias("_raw_factor"))

    # Inf/NaN → null
    full = full.with_columns(
        pl.when(pl.col("_raw_factor").is_finite())
        .then(pl.col("_raw_factor"))
        .otherwise(None)
        .alias("_raw_factor")
    )

    # 횡단면 퍼센타일 랭크 (날짜별, 0~1)
    full = full.with_columns(
        pl.col("_raw_factor")
        .rank(method="average")
        .over("dt")
        .truediv(
            pl.col("_raw_factor")
            .count()
            .over("dt")
            .cast(pl.Float64)
            .clip(lower_bound=1.0)
        )
        .fill_null(0.5)
        .cast(pl.Float64)
        .alias("factor_rank")
    )

    return full.select(["dt", "symbol", "close", "factor_rank"])


def _weekly_rebalance_dates(df: pl.DataFrame) -> set:
    """unique dates에서 매주 첫 거래일을 추출."""
    dates_sorted = sorted(df["dt"].unique().to_list())
    if not dates_sorted:
        return set()

    rebalance: set = set()
    current_week: int | None = None

    for d in dates_sorted:
        # isocalendar week number
        iso_week = d.isocalendar()[1]
        iso_year = d.isocalendar()[0]
        week_key = iso_year * 100 + iso_week

        if week_key != current_week:
            rebalance.add(d)
            current_week = week_key

    return rebalance


def _build_date_data(df: pl.DataFrame) -> dict:
    """DataFrame을 {date: {symbol: {close, rank}}} 딕셔너리로 변환."""
    date_data: dict = {}

    for row in df.iter_rows(named=True):
        dt = row["dt"]
        sym = row["symbol"]
        if dt not in date_data:
            date_data[dt] = {}
        date_data[dt][sym] = {
            "close": row["close"],
            "rank": row["factor_rank"],
        }

    return date_data


def _simulate_portfolio(
    date_data: dict,
    all_dates_sorted: list,
    rebalance_dates: set,
    stop_loss_pct: float,
    trailing_stop_pct: float,
    round_trip_cost: float,
    max_positions: int,
    top_pct: float,
) -> dict:
    """간소 포트폴리오 시뮬레이션.

    Parameters
    ----------
    date_data : dict
        {date: {symbol: {close, rank}}} 형태의 딕셔너리.
    all_dates_sorted : list
        정렬된 전체 날짜 리스트.
    rebalance_dates : set
        리밸런싱 실행 날짜 집합.
    stop_loss_pct : float
        고정 손절 비율 (0.0이면 미적용).
    trailing_stop_pct : float
        트레일링 스탑 비율 (0.0이면 미적용).
    round_trip_cost : float
        왕복 거래비용 비율.
    max_positions : int
        동시 최대 보유 종목 수.
    top_pct : float
        매수 대상 상위 퍼센타일 (0~1).

    Returns
    -------
    dict
        {"sharpe": float, "mdd": float, "total_return": float}
    """
    # holdings: {symbol: {entry_price, high_price, weight}}
    holdings: dict[str, dict] = {}
    daily_returns: list[float] = []
    equity = 1.0
    peak_equity = 1.0
    max_drawdown = 0.0

    for date in all_dates_sorted:
        day_data = date_data.get(date)
        if not day_data:
            daily_returns.append(0.0)
            continue

        portfolio_return = 0.0
        n_holdings = len(holdings) if holdings else 1

        # ── 1. 보유 종목 평가 + 손절/트레일링 ──
        to_remove: list[str] = []

        for sym, h in holdings.items():
            sym_data = day_data.get(sym)
            if sym_data is None:
                # 종목 데이터 없음 → 보유 유지, 수익률 0
                continue

            close = sym_data["close"]
            entry_price = h["entry_price"]
            pnl_pct = (close - entry_price) / entry_price if entry_price > 0 else 0.0

            # 고정 손절
            if stop_loss_pct > 0.0 and pnl_pct <= -stop_loss_pct:
                cost = round_trip_cost * 0.5  # 매도 비용만
                portfolio_return += (pnl_pct - cost) / n_holdings
                to_remove.append(sym)
                continue

            # 트레일링 스탑
            if trailing_stop_pct > 0.0:
                high = h["high_price"]
                trail_pnl = (close - high) / high if high > 0 else 0.0
                if trail_pnl <= -trailing_stop_pct:
                    cost = round_trip_cost * 0.5
                    portfolio_return += (pnl_pct - cost) / n_holdings
                    to_remove.append(sym)
                    continue

            # high_price 갱신
            if close > h["high_price"]:
                h["high_price"] = close

            # 일별 수익률 누적
            prev_close = h.get("prev_close", entry_price)
            day_ret = (close - prev_close) / prev_close if prev_close > 0 else 0.0
            portfolio_return += day_ret / n_holdings
            h["prev_close"] = close

        for sym in to_remove:
            del holdings[sym]

        # ── 2. 리밸런싱 날 ──
        if date in rebalance_dates:
            # 목표 포트폴리오: 상위 top_pct 종목
            ranked = sorted(
                day_data.items(),
                key=lambda x: x[1]["rank"],
                reverse=True,
            )
            n_target = min(max_positions, max(1, int(len(ranked) * top_pct)))
            target_symbols = {item[0] for item in ranked[:n_target]}

            # 기존 보유 중 목표에 없는 종목 매도
            sell_syms = [s for s in holdings if s not in target_symbols]
            for sym in sell_syms:
                sym_data = day_data.get(sym)
                if sym_data is not None:
                    entry_price = holdings[sym]["entry_price"]
                    pnl_pct = (sym_data["close"] - entry_price) / entry_price if entry_price > 0 else 0.0
                    cost = round_trip_cost * 0.5
                    portfolio_return += (pnl_pct - cost) / max(n_holdings, 1)
                del holdings[sym]

            # 목표에 있지만 미보유인 종목 매수
            for sym in target_symbols:
                if sym not in holdings:
                    sym_data = day_data.get(sym)
                    if sym_data is not None and sym_data["close"] > 0:
                        buy_price = sym_data["close"] * (1 + round_trip_cost * 0.5)
                        holdings[sym] = {
                            "entry_price": buy_price,
                            "high_price": sym_data["close"],
                            "prev_close": sym_data["close"],
                        }

        daily_returns.append(portfolio_return)

        # 에쿼티 트래킹
        equity *= (1 + portfolio_return)
        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        if dd > max_drawdown:
            max_drawdown = dd

    # ── 최종 지표 계산 ──
    total_return = equity - 1.0

    if len(daily_returns) < 2:
        return {"sharpe": 0.0, "mdd": max_drawdown, "total_return": total_return}

    mean_ret = sum(daily_returns) / len(daily_returns)
    variance = sum((r - mean_ret) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
    std_ret = math.sqrt(variance) if variance > 0 else 0.0

    if std_ret > 0:
        sharpe = (mean_ret / std_ret) * math.sqrt(252)
    else:
        sharpe = 0.0

    return {
        "sharpe": sharpe,
        "mdd": max_drawdown,
        "total_return": total_return,
    }
