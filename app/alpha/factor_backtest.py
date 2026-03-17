"""횡단면 포트폴리오 기반 알파 팩터 백테스트.

기존 engine.py의 조건 기반 시그널 방식 대신,
매일 전체 종목을 팩터값으로 랭킹하여 상위 K종목을 매수/리밸런싱한다.
evaluator.py와 동일한 compute pipeline을 공유한다.
"""

from __future__ import annotations

import logging
import math
import uuid
from collections import defaultdict
from dataclasses import asdict
from datetime import date, datetime, timedelta
from typing import Awaitable, Callable

import numpy as np
import polars as pl
from sqlalchemy import update

from app.alpha.ast_converter import (
    ensure_alpha_features,
    get_required_columns,
    parse_expression,
    sympy_to_polars,
)
from app.alpha.interval import bars_per_year, warmup_days, is_intraday
from app.backtest.cost_model import CostConfig, default_cost_config, effective_buy_price, effective_sell_price
from app.backtest.data_loader import load_enriched_candles
from app.backtest.metrics import Trade, compute_metrics
from app.backtest.engine import BacktestResult
from app.backtest.models import BacktestRun
from app.core.database import async_session
from app.core.stock_master import get_stock_name
from app.services.ws_manager import manager

logger = logging.getLogger(__name__)

# ── 정확도 보강 상수 ──
MAX_VOLUME_PARTICIPATION = 0.10  # 봉 거래량의 최대 10%만 참여
MIN_DAILY_TURNOVER = 100_000_000  # 1억원 (일평균 거래대금 필터)
MAX_ADV_PARTICIPATION = 0.05     # ADV(평균일거래대금)의 최대 5%만 참여


def _clamp_qty_by_volume(intended_qty: int, bar_volume: int, price: float) -> int:
    """거래량 참여율 제한. 봉 거래량의 10% 초과 주문은 잘라낸다."""
    max_tradeable = int(bar_volume * MAX_VOLUME_PARTICIPATION)
    return min(intended_qty, max(max_tradeable, 0))


def _clamp_qty_by_adv(intended_qty: int, price: float, adv: float) -> int:
    """ADV 기반 일일 참여율 제한. ADV의 5% 초과 주문은 잘라낸다.

    Parameters
    ----------
    adv : 평균 일거래대금 (원 단위)
    """
    if adv <= 0 or price <= 0:
        return intended_qty
    max_value = adv * MAX_ADV_PARTICIPATION
    max_qty = int(max_value / price)
    return min(intended_qty, max(max_qty, 1))

ProgressCallback = Callable[[int, int, str], Awaitable[None]] | None


# ── 거래 상세 헬퍼 ──

def _dt_to_str(dt_val) -> str:
    """date/datetime → ISO 문자열."""
    if isinstance(dt_val, datetime):
        return dt_val.isoformat()
    if isinstance(dt_val, date):
        return dt_val.isoformat()
    return str(dt_val)


def _calc_holding_days(current, entry, intraday: bool) -> float:
    """보유 기간 (일 단위)."""
    if intraday and isinstance(current, datetime) and isinstance(entry, datetime):
        return max(0.01, round((current - entry).total_seconds() / 86400, 2))
    if isinstance(current, date) and isinstance(entry, date):
        return max(0, (current - entry).days)
    return 0


def _make_entry_reason(pos: dict) -> list[dict]:
    """진입 사유 상세 (UI 표시용).

    TradeConditionResult 호환 형식: {condition, column, actual, met}
    """
    rank = pos.get("entry_factor_rank")
    rank_pos = pos.get("entry_rank_pos")
    total = pos.get("entry_total_candidates")
    target_count = pos.get("entry_target_count")
    fv = pos.get("entry_factor_value")

    reasons = []
    if rank is not None and total:
        reasons.append({
            "condition": f"팩터 랭크 상위 {(1 - rank) * 100:.1f}% ({rank_pos or '?'}/{total}종목)",
            "column": "factor_rank",
            "actual": round(rank, 4),
            "met": True,
        })
    if target_count:
        reasons.append({
            "condition": f"매수 대상 {target_count}종목 포트폴리오에 포함",
            "column": "target",
            "actual": target_count,
            "met": True,
        })
    if fv is not None and isinstance(fv, (int, float)):
        reasons.append({
            "condition": f"팩터 값: {fv:.4f}",
            "column": "factor_value",
            "actual": round(fv, 4),
            "met": True,
        })
    return reasons or None


def _make_exit_reason_detail(reason_text: str, pos: dict) -> list[dict]:
    """퇴출 사유 상세 (UI 표시용).

    TradeConditionResult 호환 형식: {condition, column, actual, met}
    """
    entry_price = pos.get("avg_price", 0)
    details = [{
        "condition": reason_text,
        "column": "exit_trigger",
        "actual": None,
        "met": True,
    }]

    if entry_price > 0:
        high = pos.get("high_price", entry_price)
        low = pos.get("low_price", entry_price)
        max_gain = round((high / entry_price - 1) * 100, 2)
        max_loss = round((low / entry_price - 1) * 100, 2)
        details.append({
            "condition": f"보유 중 최고 수익: +{max_gain:.2f}%",
            "column": "max_gain",
            "actual": max_gain,
            "met": max_gain > 0,
        })
        details.append({
            "condition": f"보유 중 최대 손실: {max_loss:.2f}%",
            "column": "max_loss",
            "actual": max_loss,
            "met": max_loss >= 0,
        })

    return details


def _make_entry_snapshot(pos: dict) -> dict:
    """진입 시점 팩터 스냅샷."""
    rank = pos.get("entry_factor_rank")
    return {
        "factor_rank": rank,
        "factor_rank_pct": round((1 - rank) * 100, 1) if rank is not None else None,
        "rank_position": pos.get("entry_rank_pos"),
        "total_candidates": pos.get("entry_total_candidates"),
        "target_count": pos.get("entry_target_count"),
        "factor_value": pos.get("entry_factor_value"),
    }


def _make_exit_snapshot(pos: dict, today_sym_data: dict | None, exit_factor_rank: float | None = None) -> dict:
    """퇴출 시점 스냅샷."""
    entry_price = pos.get("avg_price", 0)
    high = pos.get("high_price", entry_price)
    low = pos.get("low_price", entry_price)

    snapshot: dict = {}
    if exit_factor_rank is not None:
        snapshot["factor_rank"] = exit_factor_rank
        snapshot["factor_rank_pct"] = round((1 - exit_factor_rank) * 100, 1)
    if today_sym_data:
        snapshot["exit_price_close"] = today_sym_data.get("close")
        if today_sym_data.get("factor_value") is not None:
            snapshot["factor_value"] = today_sym_data["factor_value"]
    if entry_price > 0:
        snapshot["max_gain_pct"] = round((high / entry_price - 1) * 100, 2)
        snapshot["max_loss_pct"] = round((low / entry_price - 1) * 100, 2)
    return snapshot


def _precompute_factor_values(
    df: pl.DataFrame, expression_str: str, interval: str = "1d"
) -> pl.DataFrame:
    """3-Phase 파이프라인: rolling 지표 → 횡단면 피처 보정 → 팩터 적용.

    Phase 1: 종목별 rolling 지표 계산 (RSI, SMA, MACD 등)
             — 종목 간 데이터 오염 방지를 위해 분할 필수.
    Phase 2: 전체 DF에서 횡단면 피처 (rank_close, zscore_volume 등)를
             .over("dt")로 덮어쓰기 — evaluator.py와 동일한 방식.
    Phase 3: 보정된 피처로 팩터 수식 적용 → 횡단면 퍼센타일 랭크.

    Returns
    -------
    pl.DataFrame
        symbol, dt, close, open, factor_rank (0~1) 컬럼 포함.
    """
    import time as _time

    expr = parse_expression(expression_str)
    polars_expr = sympy_to_polars(expr)
    required_cols = get_required_columns(expression_str)
    logger.info("Factor requires columns: %s", required_cols)

    # ── Phase 1: 종목별 rolling 지표 계산 ──
    # partition_by로 한 번에 분할 (filter × N 풀스캔 회피)
    partitions = df.sort(["symbol", "dt"]).partition_by("symbol", maintain_order=True)
    n_symbols = len(partitions)

    t0 = _time.monotonic()
    parts: list[pl.DataFrame] = []
    skipped = 0
    for i, sym_df in enumerate(partitions):
        if sym_df.height < 30:
            skipped += 1
            continue
        try:
            sym_df = ensure_alpha_features(sym_df, required_cols=required_cols)
            parts.append(sym_df)
        except Exception as e:
            sym = sym_df["symbol"][0] if sym_df.height > 0 else "?"
            logger.warning("Feature computation failed for %s: %s", sym, e)
            skipped += 1

        if (i + 1) % 100 == 0:
            elapsed = _time.monotonic() - t0
            logger.info(
                "Phase 1: %d/%d symbols done (%.1fs elapsed, %d skipped)",
                i + 1, n_symbols, elapsed, skipped,
            )

    elapsed = _time.monotonic() - t0
    logger.info(
        "Phase 1 complete: %d/%d symbols in %.1fs (%d skipped)",
        len(parts), n_symbols, elapsed, skipped,
    )

    if not parts:
        return pl.DataFrame()

    full = pl.concat(parts)

    # ── Phase 2: 횡단면 피처 덮어쓰기 (.over("dt")) ──
    # ensure_alpha_features()가 n_symbols=1로 호출되어 rolling fallback이
    # 적용된 rank_close, zscore_volume 등을 올바른 횡단면 값으로 보정한다.
    for col_name in ["close", "volume"]:
        rank_alias = f"rank_{col_name}"
        zscore_alias = f"zscore_{col_name}"

        if rank_alias in full.columns:
            full = full.with_columns(
                pl.col(col_name)
                .rank(method="average")
                .over("dt")
                .truediv(pl.col(col_name).count().over("dt"))
                .alias(rank_alias)
            )

        if zscore_alias in full.columns:
            full = full.with_columns(
                (
                    (pl.col(col_name) - pl.col(col_name).mean().over("dt"))
                    / pl.col(col_name).std().over("dt").clip(lower_bound=1e-10)
                ).alias(zscore_alias)
            )

    # ── Phase 3: 팩터 수식 적용 + 횡단면 퍼센타일 랭크 ──
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

    # 유동성 필터: 일평균 거래대금 기준 미달 종목 제거
    # 분봉은 봉당 거래대금이 작으므로 bars_per_day로 스케일링
    if "volume" in full.columns:
        from app.alpha.interval import bars_per_day as _bpd
        scale = _bpd(interval)
        # 봉당 평균 거래대금 × 일간 봉 수 = 추정 일간 거래대금
        avg_turnover = full.group_by("symbol").agg(
            (pl.col("close") * pl.col("volume")).mean().alias("avg_bar_turnover")
        ).with_columns(
            (pl.col("avg_bar_turnover") * scale).alias("est_daily_turnover")
        )
        liquid_symbols = avg_turnover.filter(
            pl.col("est_daily_turnover") >= MIN_DAILY_TURNOVER
        )
        removed_count = avg_turnover.height - liquid_symbols.height
        if removed_count > 0:
            logger.info(
                "Liquidity filter: removed %d/%d symbols (est daily turnover < %s, interval=%s)",
                removed_count, avg_turnover.height,
                f"{MIN_DAILY_TURNOVER:,.0f}", interval,
            )
        full = full.join(liquid_symbols.select("symbol"), on="symbol", how="inner")

    # 필요 컬럼만 남기기 (volume, factor_value 유지)
    select_cols = ["symbol", "dt", "close", "open", "factor_rank"]
    if "_raw_factor" in full.columns:
        full = full.rename({"_raw_factor": "factor_value"})
        select_cols.append("factor_value")
    if "volume" in full.columns:
        select_cols.append("volume")
    full = full.select(select_cols)
    return full


def _get_rebalance_dates(
    all_dates: list, freq: str
) -> list:
    """리밸런싱 날짜/시간 목록 생성.

    all_dates는 date 또는 datetime 리스트.
    """
    if not all_dates:
        return []

    if freq == "every_bar":
        return list(all_dates)

    if freq == "daily":
        # 분봉 데이터: 캘린더 날짜별 첫 바만 리밸런스 (일봉은 그대로 전체)
        result: list = []
        seen_days: set = set()
        for d in all_dates:
            d_date = d.date() if isinstance(d, datetime) else d
            if d_date not in seen_days:
                seen_days.add(d_date)
                result.append(d)
        return result

    if freq == "hourly":
        # 매시 첫 봉에서 리밸런싱
        result: list = []
        seen_hours: set[tuple] = set()
        for d in all_dates:
            if isinstance(d, datetime):
                hour_key = (d.date(), d.hour)
            else:
                hour_key = (d, 0)
            if hour_key not in seen_hours:
                seen_hours.add(hour_key)
                result.append(d)
        return result

    if freq == "weekly":
        result = []
        seen_weeks: set[tuple[int, int]] = set()
        for d in all_dates:
            d_date = d.date() if isinstance(d, datetime) else d
            iso = d_date.isocalendar()
            week_key = (iso[0], iso[1])
            if week_key not in seen_weeks:
                seen_weeks.add(week_key)
                result.append(d)
        return result

    if freq == "monthly":
        result = []
        seen_months: set[tuple[int, int]] = set()
        for d in all_dates:
            d_date = d.date() if isinstance(d, datetime) else d
            month_key = (d_date.year, d_date.month)
            if month_key not in seen_months:
                seen_months.add(month_key)
                result.append(d)
        return result

    return list(all_dates)


def _sanitize_for_json(obj):
    """NaN/Infinity를 None으로 변환."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


async def run_factor_backtest(
    expression_str: str,
    symbols: list[str],
    start_date: date,
    end_date: date,
    initial_capital: float = 100_000_000,
    top_pct: float = 0.2,
    max_positions: int = 20,
    rebalance_freq: str = "weekly",
    band_threshold: float = 0.05,
    cost_config: CostConfig | None = None,
    progress_cb: ProgressCallback = None,
    interval: str = "1d",
    stop_loss_pct: float = 0.0,
    max_drawdown_pct: float = 0.0,
) -> BacktestResult:
    """횡단면 포트폴리오 기반 알파 팩터 백테스트.

    Parameters
    ----------
    expression_str : 팩터 수식 문자열.
    symbols : 종목 리스트.
    start_date, end_date : 백테스트 기간.
    initial_capital : 초기 자본.
    top_pct : 상위 몇 %를 매수할지 (0.2 = 상위 20%).
    max_positions : 최대 동시 보유 종목 수.
    rebalance_freq : 리밸런싱 빈도 (every_bar/hourly/daily/weekly/monthly).
    band_threshold : 밴드 리밸런싱 임계값 (0 = 밴드 없음).
    cost_config : 거래 비용 설정.
    progress_cb : 진행률 콜백.
    interval : 데이터 인터벌 (1d, 5m 등).
    stop_loss_pct : 포지션별 손절 비율 (0=비활성, 0.05=5% 손절).
    max_drawdown_pct : 포트폴리오 최대 낙폭 서킷 브레이커 (0=비활성, 0.15=15%).
    """
    if cost_config is None:
        cost_config = default_cost_config(interval)

    if len(symbols) < 3:
        return BacktestResult(
            metrics={"error": "횡단면 백테스트는 최소 3종목 이상 필요합니다."}
        )

    # ── 1. 데이터 로딩 ──
    if progress_cb:
        await progress_cb(0, 100, "데이터 로딩 중...")

    # 워밍업: 지표 계산을 위해 시작일 이전부터 로딩
    warmup_start = start_date - timedelta(days=warmup_days(interval))
    df = await load_enriched_candles(symbols, warmup_start, end_date, interval)

    if df.is_empty():
        return BacktestResult(metrics={"error": "데이터가 없습니다."})

    loaded_symbols = df["symbol"].unique().sort().to_list()

    if progress_cb:
        await progress_cb(5, 100, f"{len(loaded_symbols)}개 종목 데이터 로딩 완료")

    # ── 2. 팩터 값 사전 계산 ──
    if progress_cb:
        await progress_cb(10, 100, "팩터 값 계산 중...")

    try:
        factor_df = _precompute_factor_values(df, expression_str, interval=interval)
    except Exception as e:
        return BacktestResult(
            metrics={"error": f"팩터 수식 계산 실패: {str(e)[:200]}"}
        )

    if factor_df.is_empty():
        return BacktestResult(
            metrics={"error": "팩터 값 계산 결과가 비어 있습니다."}
        )

    # ── 2.5. 종목별 ADV (평균일거래대금) 사전 계산 ──
    adv_by_symbol: dict[str, float] = {}
    if "volume" in factor_df.columns:
        from app.alpha.interval import bars_per_day as _bpd_fn
        _scale = _bpd_fn(interval)
        _adv_df = factor_df.group_by("symbol").agg(
            (pl.col("close") * pl.col("volume")).mean().alias("avg_bar_turnover")
        ).with_columns(
            (pl.col("avg_bar_turnover") * _scale).alias("adv")
        )
        for row in _adv_df.iter_rows(named=True):
            adv_by_symbol[row["symbol"]] = row["adv"]

    if progress_cb:
        await progress_cb(30, 100, "팩터 횡단면 랭킹 완료")

    # ── 3. 날짜별 데이터 인덱싱 ──
    intraday = is_intraday(interval)

    # 일봉: dt → date, 분봉: dt → datetime 유지
    if not intraday and factor_df["dt"].dtype == pl.Datetime:
        factor_df = factor_df.with_columns(pl.col("dt").dt.date().alias("dt"))

    all_dates_raw = factor_df["dt"].unique().sort().to_list()
    # 워밍업 기간 제외
    if intraday:
        all_dates = [d for d in all_dates_raw if (d.date() >= start_date if isinstance(d, datetime) else d >= start_date)]
    else:
        all_dates = [d for d in all_dates_raw if d >= start_date]

    if len(all_dates) < 2:
        return BacktestResult(
            metrics={"error": "시뮬레이션 가능한 거래일이 부족합니다."}
        )

    rebalance_dates_set = set(_get_rebalance_dates(all_dates, rebalance_freq))

    has_volume = "volume" in factor_df.columns
    has_factor_value = "factor_value" in factor_df.columns

    # 날짜/시간별 종목 데이터를 dict로 인덱싱 (성능)
    date_data: dict = defaultdict(dict)
    for row in factor_df.iter_rows(named=True):
        dt = row["dt"]
        sym = row["symbol"]
        entry = {
            "close": row["close"],
            "open": row["open"],
            "factor_rank": row["factor_rank"],
        }
        if has_factor_value:
            entry["factor_value"] = row.get("factor_value")
        if has_volume:
            entry["volume"] = row.get("volume", 0) or 0
        date_data[dt][sym] = entry

    if progress_cb:
        await progress_cb(40, 100, "포트폴리오 시뮬레이션 시작")

    # ── 4. 포트폴리오 시뮬레이션 ──
    cutoff = 1.0 - top_pct  # 예: top_pct=0.2 → cutoff=0.8

    cash = initial_capital
    # holdings: {symbol: {qty, avg_price, entry_date, last_close,
    #   entry_factor_rank, entry_factor_value, entry_rank_pos,
    #   entry_total_candidates, entry_target_count, high_price, low_price}}
    holdings: dict[str, dict] = {}
    trades: list[Trade] = []
    equity_curve: list[dict] = []

    total_buys = 0
    total_sells = 0
    rebalance_count = 0
    stop_loss_count = 0
    circuit_breaker_triggered = False
    peak_equity = initial_capital
    prev_day_data: dict[str, dict] | None = None  # T-1 시그널용

    for day_idx, current_date in enumerate(all_dates):
        today = date_data.get(current_date, {})

        if not today:
            # 거래일이지만 데이터 없음 → 전일 포트폴리오 유지
            if equity_curve:
                dt_str = current_date.isoformat() if isinstance(current_date, datetime) else current_date.isoformat()
                equity_curve.append({
                    "date": dt_str,
                    "equity": equity_curve[-1]["equity"],
                })
            continue

        is_rebalance_day = current_date in rebalance_dates_set

        # ── 포지션별 고가/저가 추적 + 포지션 손절 ──
        for sym in list(holdings.keys()):
            pos = holdings[sym]
            if sym in today:
                price = today[sym]["close"]
                pos["high_price"] = max(pos.get("high_price", price), price)
                pos["low_price"] = min(pos.get("low_price", price), price)
                pos["last_close"] = price

                # 포지션 손절 체크 (매 바마다)
                if stop_loss_pct > 0 and not circuit_breaker_triggered:
                    drawdown = (price - pos["avg_price"]) / pos["avg_price"]
                    if drawdown <= -stop_loss_pct:
                        _sv = today.get(sym, {}).get("volume", 0) if has_volume else 0
                        sell_price = effective_sell_price(price, cost_config, order_qty=pos["qty"], bar_volume=_sv)
                        pnl = (sell_price - pos["avg_price"]) * pos["qty"]
                        pnl_pct = (sell_price / pos["avg_price"] - 1) * 100
                        holding_days = _calc_holding_days(current_date, pos["entry_date"], intraday)
                        cash += sell_price * pos["qty"]
                        dt_str = _dt_to_str(current_date)
                        trades.append(Trade(
                            symbol=sym,
                            name=get_stock_name(sym),
                            entry_date=_dt_to_str(pos["entry_date"]),
                            entry_price=pos["avg_price"],
                            exit_date=dt_str,
                            exit_price=sell_price,
                            qty=pos["qty"],
                            pnl=pnl,
                            pnl_pct=pnl_pct,
                            holding_days=holding_days,
                            scale_step="STOP-LOSS",
                            exit_reason="손절",
                            entry_reason=_make_entry_reason(pos),
                            exit_reason_detail=_make_exit_reason_detail(
                                f"포지션 손절: 진입가 대비 {drawdown*100:.1f}% 하락 (기준: -{stop_loss_pct*100:.0f}%)",
                                pos,
                            ),
                            entry_snapshot=_make_entry_snapshot(pos),
                            exit_snapshot=_make_exit_snapshot(pos, today.get(sym)),
                        ))
                        total_sells += 1
                        stop_loss_count += 1
                        del holdings[sym]
                        logger.debug("Stop-loss triggered: %s (%.1f%%)", sym, drawdown * 100)

        # ── 포트폴리오 서킷 브레이커 ──
        if max_drawdown_pct > 0 and not circuit_breaker_triggered:
            current_equity = cash + sum(
                today.get(s, {}).get("close", h.get("last_close", h["avg_price"])) * h["qty"]
                for s, h in holdings.items()
            )
            peak_equity = max(peak_equity, current_equity)
            if current_equity < peak_equity * (1 - max_drawdown_pct):
                logger.warning(
                    "Circuit breaker triggered: equity=%.0f, peak=%.0f, drawdown=%.1f%%",
                    current_equity, peak_equity,
                    (1 - current_equity / peak_equity) * 100,
                )
                circuit_breaker_triggered = True
                # 모든 포지션 즉시 청산
                for sym in list(holdings.keys()):
                    pos = holdings.pop(sym)
                    price = today.get(sym, {}).get("close", pos.get("last_close", pos["avg_price"]))
                    _sv2 = today.get(sym, {}).get("close_volume", today.get(sym, {}).get("volume", 0)) if has_volume else 0
                    sell_price = effective_sell_price(price, cost_config, order_qty=pos["qty"], bar_volume=_sv2)
                    pnl = (sell_price - pos["avg_price"]) * pos["qty"]
                    pnl_pct = (sell_price / pos["avg_price"] - 1) * 100 if pos["avg_price"] > 0 else 0
                    cash += sell_price * pos["qty"]
                    trades.append(Trade(
                        symbol=sym,
                        name=get_stock_name(sym),
                        entry_date=_dt_to_str(pos["entry_date"]),
                        entry_price=pos["avg_price"],
                        exit_date=_dt_to_str(current_date),
                        exit_price=sell_price,
                        qty=pos["qty"],
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        holding_days=_calc_holding_days(current_date, pos["entry_date"], intraday),
                        scale_step="CIRCUIT-BREAKER",
                        exit_reason="서킷 브레이커",
                        entry_reason=_make_entry_reason(pos),
                        exit_reason_detail=_make_exit_reason_detail(
                            f"포트폴리오 서킷 브레이커 발동: 최고점 대비 {(1-current_equity/peak_equity)*100:.1f}% 하락 (기준: -{max_drawdown_pct*100:.0f}%)",
                            pos,
                        ),
                        entry_snapshot=_make_entry_snapshot(pos),
                        exit_snapshot=_make_exit_snapshot(pos, today.get(sym)),
                    ))
                    total_sells += 1

        # ── 고아 포지션 강제 청산 ──
        # 보유 종목이 팩터 데이터에서 사라진 경우 (상장폐지, 데이터 공백 등)
        if holdings and not circuit_breaker_triggered:
            orphan_syms = [
                sym for sym in holdings
                if sym not in today and (not prev_day_data or sym not in prev_day_data)
            ]
            for sym in orphan_syms:
                pos = holdings.pop(sym)
                last_price = pos.get("last_close", pos["avg_price"])
                sell_price = effective_sell_price(last_price, cost_config, order_qty=pos["qty"])
                pnl = (sell_price - pos["avg_price"]) * pos["qty"]
                pnl_pct = (sell_price / pos["avg_price"] - 1) * 100 if pos["avg_price"] > 0 else 0

                cash += sell_price * pos["qty"]
                trades.append(Trade(
                    symbol=sym,
                    name=get_stock_name(sym),
                    entry_date=_dt_to_str(pos["entry_date"]),
                    entry_price=pos["avg_price"],
                    exit_date=_dt_to_str(current_date),
                    exit_price=sell_price,
                    qty=pos["qty"],
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    holding_days=_calc_holding_days(current_date, pos["entry_date"], intraday),
                    scale_step="ORPHAN-SELL",
                    exit_reason="데이터 소실 강제 청산",
                    entry_reason=_make_entry_reason(pos),
                    exit_reason_detail=_make_exit_reason_detail(
                        "2일 연속 팩터 데이터 없음 → 강제 청산",
                        pos,
                    ),
                    entry_snapshot=_make_entry_snapshot(pos),
                    exit_snapshot=_make_exit_snapshot(pos, None),
                ))
                total_sells += 1

        # 리밸런싱: T-1 시그널 → T 시가 실행 (look-ahead 방지)
        # prev_day_data의 factor_rank로 랭킹, today의 open으로 매매
        if is_rebalance_day and prev_day_data and len(prev_day_data) >= 3 and not circuit_breaker_triggered:
            # ── 리밸런싱 ──
            # 전일(T-1) factor_rank 기준으로 상위 종목 결정
            ranked = sorted(prev_day_data.items(), key=lambda x: x[1]["factor_rank"], reverse=True)
            n_top = max(1, int(len(ranked) * top_pct))
            n_top = min(n_top, max_positions)
            total_candidates = len(ranked)

            target_symbols: set[str] = set()
            for sym, data in ranked[:n_top]:
                if data["factor_rank"] is not None:
                    target_symbols.add(sym)

            # 리밸런싱: 보유 종목 유지/퇴출 결정 (전일 factor_rank 기준)
            sell_list: list[str] = []
            for sym in list(holdings.keys()):
                if sym not in prev_day_data:
                    if sym not in target_symbols and sym in today:
                        sell_list.append(sym)
                    continue
                if sym not in today:
                    continue
                if sym not in target_symbols:
                    sell_list.append(sym)

            # 신규 매수 결정
            buy_list: list[str] = []
            for sym in target_symbols:
                if sym in holdings:
                    continue
                if sym not in today:
                    continue
                buy_list.append(sym)

            current_after_sell = len(holdings) - len(sell_list)
            max_new_buys = max(0, max_positions - current_after_sell)
            buy_list.sort(key=lambda s: prev_day_data[s]["factor_rank"], reverse=True)
            buy_list = buy_list[:max_new_buys]

            # ── 매도 실행 (당일 시가) ──
            for sym in sell_list:
                pos = holdings.pop(sym, None)
                if pos is None:
                    continue
                sell_qty = pos["qty"]
                if has_volume:
                    sell_qty = _clamp_qty_by_volume(sell_qty, today[sym].get("volume", 0), today[sym]["open"])
                if sell_qty <= 0:
                    holdings[sym] = pos
                    continue

                _sell_vol = today[sym].get("volume", 0) if has_volume else 0
                sell_price = effective_sell_price(today[sym]["open"], cost_config, order_qty=sell_qty, bar_volume=_sell_vol)
                actual_pnl = (sell_price - pos["avg_price"]) * sell_qty
                pnl_pct = (sell_price / pos["avg_price"] - 1) * 100 if pos["avg_price"] > 0 else 0

                cash += sell_price * sell_qty

                remaining = pos["qty"] - sell_qty
                if remaining > 0:
                    holdings[sym] = {**pos, "qty": remaining}

                # 퇴출 상세: 진입/퇴출 시점 랭크 비교
                exit_rank = prev_day_data.get(sym, {}).get("factor_rank")
                exit_rank_pct = f"상위 {(1 - exit_rank) * 100:.1f}%" if exit_rank is not None else "N/A"
                entry_rank = pos.get("entry_factor_rank")
                entry_rank_pct = f"상위 {(1 - entry_rank) * 100:.1f}%" if entry_rank is not None else "N/A"

                trades.append(Trade(
                    symbol=sym,
                    name=get_stock_name(sym),
                    entry_date=_dt_to_str(pos["entry_date"]),
                    entry_price=pos["avg_price"],
                    exit_date=_dt_to_str(current_date),
                    exit_price=sell_price,
                    qty=sell_qty,
                    pnl=actual_pnl,
                    pnl_pct=pnl_pct,
                    holding_days=_calc_holding_days(current_date, pos["entry_date"], intraday),
                    scale_step="REBAL-SELL",
                    exit_reason="리밸런싱 퇴출",
                    entry_reason=_make_entry_reason(pos),
                    exit_reason_detail=_make_exit_reason_detail(
                        f"팩터 랭크 {entry_rank_pct} → {exit_rank_pct} 하락, 상위 {top_pct*100:.0f}% 이탈",
                        pos,
                    ),
                    entry_snapshot=_make_entry_snapshot(pos),
                    exit_snapshot=_make_exit_snapshot(pos, today.get(sym), exit_factor_rank=exit_rank),
                ))
                total_sells += 1

            # ── 매수 실행 (당일 시가) ──
            if buy_list and cash > 0:
                per_stock_budget = cash / max(len(buy_list), 1)

                for sym in buy_list:
                    _bar_vol = today[sym].get("volume", 0) if has_volume else 0
                    # 예비 qty 계산 (고정 슬리피지로 추정)
                    _est_price = today[sym]["open"] * (1 + cost_config.slippage_pct) * (1 + cost_config.buy_commission)
                    _est_qty = int(per_stock_budget / _est_price) if _est_price > 0 else 0
                    if has_volume:
                        _est_qty = _clamp_qty_by_volume(_est_qty, _bar_vol, _est_price)
                    # VolumeShare 슬리피지 적용
                    buy_price = effective_buy_price(
                        today[sym]["open"], cost_config,
                        order_qty=_est_qty, bar_volume=_bar_vol,
                    )
                    if buy_price <= 0:
                        continue
                    qty = int(per_stock_budget / buy_price)
                    if qty <= 0:
                        continue

                    if has_volume:
                        qty = _clamp_qty_by_volume(qty, _bar_vol, buy_price)
                    # ADV 참여율 제한 (일일 총 주문이 ADV의 5% 이내)
                    _sym_adv = adv_by_symbol.get(sym, 0)
                    if _sym_adv > 0:
                        qty = _clamp_qty_by_adv(qty, buy_price, _sym_adv)
                    if qty <= 0:
                        continue

                    cost = buy_price * qty
                    if cost > cash:
                        qty = int(cash / buy_price)
                        if qty <= 0:
                            continue
                        cost = buy_price * qty

                    # 랭크 순위 계산
                    sym_rank = prev_day_data[sym]["factor_rank"]
                    rank_pos = next(
                        (i + 1 for i, (s, _) in enumerate(ranked) if s == sym),
                        None,
                    )

                    cash -= cost
                    holdings[sym] = {
                        "qty": qty,
                        "avg_price": buy_price,
                        "entry_date": current_date,
                        "last_close": buy_price,
                        "high_price": buy_price,
                        "low_price": buy_price,
                        # 진입 시점 팩터 정보 (거래 상세용)
                        "entry_factor_rank": sym_rank,
                        "entry_factor_value": prev_day_data[sym].get("factor_value"),
                        "entry_rank_pos": rank_pos,
                        "entry_total_candidates": total_candidates,
                        "entry_target_count": len(target_symbols),
                    }
                    total_buys += 1

            if sell_list or buy_list:
                rebalance_count += 1

        prev_day_data = today

        # ── 포트폴리오 평가 ──
        portfolio_value = cash
        for sym, pos in holdings.items():
            if sym in today:
                portfolio_value += today[sym]["close"] * pos["qty"]
                # 최종 알려진 종가 추적 (고아 청산 시 사용)
                pos["last_close"] = today[sym]["close"]
            else:
                # 데이터 없는 날은 최종 알려진 종가로 평가
                last_close = pos.get("last_close", pos["avg_price"])
                portfolio_value += last_close * pos["qty"]

        dt_key = current_date.isoformat() if isinstance(current_date, datetime) else current_date.isoformat()
        equity_curve.append({
            "date": dt_key,
            "equity": portfolio_value,
        })

        # 진행률
        if progress_cb and (day_idx + 1) % 50 == 0:
            pct = 40 + int(50 * (day_idx + 1) / len(all_dates))
            await progress_cb(pct, 100, f"시뮬레이션 {day_idx + 1}/{len(all_dates)}")

    # ── 5. 잔여 포지션 강제 청산 ──
    last_date = all_dates[-1] if all_dates else start_date
    last_day = date_data.get(last_date, {})

    for sym, pos in list(holdings.items()):
        close_price = last_day.get(sym, {}).get("close", pos.get("last_close", pos["avg_price"]))
        sell_price = effective_sell_price(close_price, cost_config, order_qty=pos["qty"])
        pnl = (sell_price - pos["avg_price"]) * pos["qty"]
        pnl_pct = (sell_price / pos["avg_price"] - 1) * 100 if pos["avg_price"] > 0 else 0

        trades.append(Trade(
            symbol=sym,
            name=get_stock_name(sym),
            entry_date=_dt_to_str(pos["entry_date"]),
            entry_price=pos["avg_price"],
            exit_date=_dt_to_str(last_date),
            exit_price=sell_price,
            qty=pos["qty"],
            pnl=pnl,
            pnl_pct=pnl_pct,
            holding_days=_calc_holding_days(last_date, pos["entry_date"], intraday),
            scale_step="FINAL",
            exit_reason="백테스트 종료 청산",
            entry_reason=_make_entry_reason(pos),
            exit_reason_detail=_make_exit_reason_detail("백테스트 기간 종료", pos),
            entry_snapshot=_make_entry_snapshot(pos),
            exit_snapshot=_make_exit_snapshot(pos, last_day.get(sym)),
        ))

    holdings.clear()

    if progress_cb:
        await progress_cb(95, 100, "성과 지표 산출 중...")

    # ── 6. 성과 지표 ──
    metrics = compute_metrics(
        trades, equity_curve, initial_capital,
        annualize=bars_per_year(interval),
    )

    # 추가 메트릭
    metrics["total_buys"] = total_buys
    metrics["total_sells"] = total_sells
    metrics["rebalance_count"] = rebalance_count
    metrics["stop_loss_count"] = stop_loss_count
    metrics["circuit_breaker_triggered"] = circuit_breaker_triggered
    metrics["backtest_mode"] = "cross_sectional_portfolio"
    metrics["top_pct"] = top_pct
    metrics["max_positions"] = max_positions
    metrics["rebalance_freq"] = rebalance_freq
    metrics["band_threshold"] = band_threshold
    metrics["stop_loss_pct"] = stop_loss_pct
    metrics["max_drawdown_pct"] = max_drawdown_pct
    metrics["symbols_count"] = len(loaded_symbols)
    metrics["interval"] = interval

    if progress_cb:
        await progress_cb(100, 100, "완료")

    return BacktestResult(
        trades=trades,
        equity_curve=equity_curve,
        metrics=metrics,
    )


async def execute_factor_backtest(
    run_id: uuid.UUID,
    expression_str: str,
    symbols: list[str],
    start_date: date,
    end_date: date,
    initial_capital: float = 100_000_000,
    top_pct: float = 0.2,
    max_positions: int = 20,
    rebalance_freq: str = "weekly",
    band_threshold: float = 0.05,
    cost_config: CostConfig | None = None,
    interval: str = "1d",
    stop_loss_pct: float = 0.0,
    max_drawdown_pct: float = 0.0,
) -> None:
    """DB 래퍼: BacktestRun에 결과를 저장한다."""
    channel = f"backtest:{run_id}"

    async def progress_cb(current: int, total: int, msg: str) -> None:
        async with async_session() as db:
            await db.execute(
                update(BacktestRun)
                .where(BacktestRun.id == run_id)
                .values(progress=current)
            )
            await db.commit()

        await manager.broadcast(channel, {
            "type": "progress",
            "current": current,
            "total": total,
            "percent": current,
            "message": msg,
        })

    try:
        # RUNNING
        async with async_session() as db:
            await db.execute(
                update(BacktestRun)
                .where(BacktestRun.id == run_id)
                .values(status="RUNNING", progress=0)
            )
            await db.commit()

        result = await run_factor_backtest(
            expression_str=expression_str,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            top_pct=top_pct,
            max_positions=max_positions,
            rebalance_freq=rebalance_freq,
            band_threshold=band_threshold,
            cost_config=cost_config,
            progress_cb=progress_cb,
            interval=interval,
            stop_loss_pct=stop_loss_pct,
            max_drawdown_pct=max_drawdown_pct,
        )

        if "error" in result.metrics:
            async with async_session() as db:
                await db.execute(
                    update(BacktestRun)
                    .where(BacktestRun.id == run_id)
                    .values(
                        status="FAILED",
                        error_message=str(result.metrics["error"])[:500],
                        completed_at=datetime.utcnow(),
                    )
                )
                await db.commit()

            await manager.broadcast(channel, {
                "type": "failed",
                "error": str(result.metrics["error"])[:200],
            })
            logger.warning("Factor backtest %s failed: %s", run_id, result.metrics["error"])
            return

        trades_list = [asdict(t) for t in result.trades]

        # 분봉: equity_curve를 일봉 단위로 다운샘플 (DB/프론트엔드 부담 경감)
        equity_for_db = result.equity_curve
        if is_intraday(interval) and result.equity_curve:
            daily_equity: dict[str, float] = {}
            for pt in result.equity_curve:
                d = pt["date"][:10]  # YYYY-MM-DD
                daily_equity[d] = pt["equity"]  # 마지막 바의 equity = 일말 기준
            equity_for_db = [{"date": d, "equity": eq} for d, eq in sorted(daily_equity.items())]

        async with async_session() as db:
            await db.execute(
                update(BacktestRun)
                .where(BacktestRun.id == run_id)
                .values(
                    status="COMPLETED",
                    progress=100,
                    metrics=_sanitize_for_json(result.metrics),
                    equity_curve=_sanitize_for_json(equity_for_db),
                    trades_summary=_sanitize_for_json(trades_list),
                    symbol_count=len(set(t.symbol for t in result.trades)),
                    completed_at=datetime.utcnow(),
                )
            )
            await db.commit()

        await manager.broadcast(channel, {
            "type": "completed",
            "metrics": result.metrics,
        })

        logger.info(
            "Factor backtest %s completed: return=%.2f%%, trades=%d",
            run_id,
            result.metrics.get("total_return", 0),
            result.metrics.get("total_trades", 0),
        )

    except Exception as e:
        logger.exception("Factor backtest %s failed", run_id)
        async with async_session() as db:
            await db.execute(
                update(BacktestRun)
                .where(BacktestRun.id == run_id)
                .values(
                    status="FAILED",
                    error_message=str(e)[:500],
                    completed_at=datetime.utcnow(),
                )
            )
            await db.commit()

        await manager.broadcast(channel, {
            "type": "failed",
            "error": str(e)[:200],
        })
