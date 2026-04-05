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
from app.trading.tick_size import get_tick_size

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


def _default_limit_ttl(interval: str) -> int:
    """인터벌별 기본 지정가 TTL (약 30분 상당). 0이면 파라미터 값 사용."""
    return {"1m": 30, "3m": 10, "5m": 6, "15m": 2, "30m": 1, "1h": 1, "1d": 2}.get(interval, 2)


def _calc_holding_days(current, entry, intraday: bool) -> float:
    """보유 기간 (일 단위). date/datetime 혼용 안전 처리."""
    if intraday and isinstance(current, datetime) and isinstance(entry, datetime):
        return max(0.01, round((current - entry).total_seconds() / 86400, 2))
    # date/datetime 혼용 방어: datetime → date 변환
    c = current.date() if isinstance(current, datetime) else current
    e = entry.date() if isinstance(entry, datetime) else entry
    try:
        if isinstance(c, date) and isinstance(e, date):
            return max(0, (c - e).days)
    except TypeError:
        pass
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


def _extract_factor_variables(sym_data: dict | None, required_cols: set[str] | None = None) -> dict:
    """종목 데이터에서 팩터 수식 개별 변수 값을 추출."""
    if not sym_data or not required_cols:
        return {}
    variables: dict = {}
    _skip = {"close", "open", "high", "low", "volume", "factor_rank", "factor_value",
             "intraday_rank", "dt", "symbol", "dt_date"}
    for col in required_cols:
        if col in _skip:
            continue
        val = sym_data.get(col)
        if val is not None:
            try:
                variables[col] = round(float(val), 4)
            except (TypeError, ValueError):
                variables[col] = val
    return variables


def _make_entry_snapshot(pos: dict) -> dict:
    """진입 시점 팩터 스냅샷 + 개별 변수 값."""
    rank = pos.get("entry_factor_rank")
    snapshot = {
        "factor_rank": rank,
        "factor_rank_pct": round((1 - rank) * 100, 1) if rank is not None else None,
        "rank_position": pos.get("entry_rank_pos"),
        "total_candidates": pos.get("entry_total_candidates"),
        "target_count": pos.get("entry_target_count"),
        "factor_value": pos.get("entry_factor_value"),
    }
    # 매수 시점에 저장된 팩터 변수 스냅샷
    if pos.get("_factor_vars"):
        snapshot["factor_variables"] = pos["_factor_vars"]
    return snapshot


def _make_exit_snapshot(pos: dict, today_sym_data: dict | None, exit_factor_rank: float | None = None) -> dict:
    """퇴출 시점 스냅샷 + 개별 변수 값."""
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
    # 퇴출 시점 팩터 변수 스냅샷
    if today_sym_data:
        variables = _extract_factor_variables(today_sym_data, pos.get("_required_cols"))
        if variables:
            snapshot["factor_variables"] = variables
    return snapshot


def _extract_limit_info(pos: dict) -> dict:
    """포지션 dict에서 지정가 체결 정보를 Trade kwargs로 추출."""
    info: dict = {}
    if pos.get("_order_type"):
        info["order_type"] = pos["_order_type"]
    if pos.get("_limit_price") is not None:
        info["limit_price"] = pos["_limit_price"]
    if pos.get("_fill_price") is not None:
        info["fill_price"] = pos["_fill_price"]
    if pos.get("_wait_bars"):
        info["wait_bars"] = pos["_wait_bars"]
    if pos.get("_fill_method"):
        info["fill_method"] = pos["_fill_method"]
    return info


def _process_pending_orders_bt(
    pending_orders: list[dict],
    bar_data: dict[str, dict],  # {symbol: {close, open, high, low, volume, ...}}
    holdings: dict[str, dict],
    trades: list,  # list[Trade]
    cash: float,
    cost_config: CostConfig,
    dt_str: str,
    current_date,
    intraday: bool,
    strict_fill: bool = False,
    get_stock_name_fn=None,
    limit_stats: dict | None = None,
) -> tuple[float, list[dict]]:
    """pending order 체결/만료 확인. (updated_cash, remaining_orders) 반환.

    Parameters
    ----------
    pending_orders : 대기 중인 주문 리스트. 각 dict에는 symbol, side, price, qty,
                     ttl_bars, elapsed_bars, reserved_cash, entry_info 등이 포함.
    bar_data : 현재 봉 데이터 {symbol: {close, open, high, low, volume, ...}}.
    holdings : 보유 포지션 dict.
    trades : Trade 객체 리스트 (체결 시 append).
    cash : 현재 가용 현금.
    cost_config : 거래 비용 설정.
    dt_str : 현재 봉의 날짜/시간 문자열 (Trade 기록용).
    current_date : 현재 봉의 date/datetime 객체.
    intraday : 장중 전략 여부.
    strict_fill : Strict 모드 (한 호가 관통 시에만 체결).
    get_stock_name_fn : 종목명 조회 함수.
    limit_stats : 지정가 통계 dict (fill_count, market_count, total_wait 키).
    """
    if get_stock_name_fn is None:
        get_stock_name_fn = get_stock_name
    if limit_stats is None:
        limit_stats = {}

    still_pending: list[dict] = []

    for order in pending_orders:
        sym = order["symbol"]
        bar = bar_data.get(sym)

        if bar is None:
            # 해당 종목 데이터 없음 → 대기 유지
            still_pending.append(order)
            continue

        order["elapsed_bars"] = order.get("elapsed_bars", 0) + 1
        bar_low = bar.get("low", float("inf"))
        bar_high = bar.get("high", 0)
        bar_close = bar.get("close", 0)
        bar_volume = bar.get("volume", 0)

        if order["side"] == "BUY":
            # ── 매수 체결 판정 ──
            limit_price = order["price"]
            if strict_fill:
                filled = bar_low < limit_price - get_tick_size(int(limit_price))
            else:
                filled = bar_low <= limit_price

            if filled:
                # 지정가 체결
                buy_price = effective_buy_price(
                    limit_price, cost_config,
                    order_qty=order["qty"], bar_volume=bar_volume,
                )
                cost = buy_price * order["qty"]
                reserved = order.get("reserved_cash", 0)
                cash_diff = cost - reserved
                if cash_diff > 0 and cash_diff > cash:
                    # 현금 부족 시 수량 축소
                    order["qty"] = int((reserved + cash * 0.95) / buy_price)
                if order["qty"] <= 0:
                    cash += reserved
                    continue

                cost = buy_price * order["qty"]
                cash_diff = cost - reserved
                cash -= max(0, cash_diff)

                entry_info = order.get("entry_info", {})
                holdings[sym] = {
                    "qty": order["qty"],
                    "avg_price": buy_price,
                    "entry_date": order.get("entry_date", current_date),
                    "last_close": buy_price,
                    "high_price": buy_price,
                    "low_price": buy_price,
                    # 지정가 체결 정보 (퇴출 시 Trade에 반영)
                    "_order_type": "limit",
                    "_limit_price": limit_price,
                    "_fill_price": buy_price,
                    "_wait_bars": order["elapsed_bars"],
                    "_fill_method": "limit_fill",
                    **entry_info,
                }
                limit_stats["fill_count"] = limit_stats.get("fill_count", 0) + 1
                limit_stats["total_wait"] = limit_stats.get("total_wait", 0) + order["elapsed_bars"]
                continue

            # TTL 만료 → 시장가 체결
            if order["elapsed_bars"] >= order.get("ttl_bars", 2):
                market_price = effective_buy_price(
                    bar_close, cost_config,
                    order_qty=order["qty"], bar_volume=bar_volume,
                )
                reserved = order.get("reserved_cash", 0)
                cost = market_price * order["qty"]
                cash_diff = cost - reserved
                if cash_diff > 0 and cash_diff > cash:
                    order["qty"] = int((reserved + cash * 0.95) / market_price)
                if order["qty"] <= 0:
                    cash += reserved
                    continue

                cost = market_price * order["qty"]
                cash_diff = cost - reserved
                cash -= max(0, cash_diff)

                entry_info = order.get("entry_info", {})
                holdings[sym] = {
                    "qty": order["qty"],
                    "avg_price": market_price,
                    "entry_date": order.get("entry_date", current_date),
                    "last_close": market_price,
                    "high_price": market_price,
                    "low_price": market_price,
                    # TTL 만료 시장가 체결 정보
                    "_order_type": "limit",
                    "_limit_price": order["price"],
                    "_fill_price": market_price,
                    "_wait_bars": order["elapsed_bars"],
                    "_fill_method": "market_ttl",
                    **entry_info,
                }
                limit_stats["market_count"] = limit_stats.get("market_count", 0) + 1
                limit_stats["total_wait"] = limit_stats.get("total_wait", 0) + order["elapsed_bars"]
                continue

            # 아직 대기 중
            still_pending.append(order)

        elif order["side"] == "SELL":
            # ── 매도 체결 판정 ──
            limit_price = order["price"]
            if strict_fill:
                filled = bar_high > limit_price + get_tick_size(int(limit_price))
            else:
                filled = bar_high >= limit_price

            pos = holdings.get(sym)
            if pos is None:
                # 이미 손절/다른 경로로 청산됨 → 주문 폐기
                continue

            if filled:
                # 지정가 체결
                sell_price = effective_sell_price(
                    limit_price, cost_config,
                    order_qty=pos["qty"], bar_volume=bar_volume,
                )
                pnl = (sell_price - pos["avg_price"]) * pos["qty"]
                pnl_pct = (sell_price / pos["avg_price"] - 1) * 100 if pos["avg_price"] > 0 else 0
                holding_days = _calc_holding_days(current_date, pos["entry_date"], intraday)
                cash += sell_price * pos["qty"]

                trades.append(Trade(
                    symbol=sym,
                    name=get_stock_name_fn(sym),
                    entry_date=_dt_to_str(pos["entry_date"]),
                    entry_price=pos["avg_price"],
                    exit_date=dt_str,
                    exit_price=sell_price,
                    qty=pos["qty"],
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    holding_days=holding_days,
                    scale_step="LIMIT-SELL",
                    exit_reason="지정가 매도 체결",
                    entry_reason=_make_entry_reason(pos),
                    exit_reason_detail=_make_exit_reason_detail(
                        f"지정가 매도 체결 (지정가: {limit_price:,.0f}, 대기: {order['elapsed_bars']}봉)",
                        pos,
                    ),
                    entry_snapshot=_make_entry_snapshot(pos),
                    exit_snapshot=_make_exit_snapshot(pos, bar),
                    order_type="limit",
                    limit_price=limit_price,
                    fill_price=sell_price,
                    wait_bars=order["elapsed_bars"],
                    fill_method="limit_fill",
                ))
                limit_stats["fill_count"] = limit_stats.get("fill_count", 0) + 1
                limit_stats["total_wait"] = limit_stats.get("total_wait", 0) + order["elapsed_bars"]
                del holdings[sym]
                continue

            # TTL 만료 → 시장가 체결
            if order["elapsed_bars"] >= order.get("ttl_bars", 2):
                sell_price = effective_sell_price(
                    bar_close, cost_config,
                    order_qty=pos["qty"], bar_volume=bar_volume,
                )
                pnl = (sell_price - pos["avg_price"]) * pos["qty"]
                pnl_pct = (sell_price / pos["avg_price"] - 1) * 100 if pos["avg_price"] > 0 else 0
                holding_days = _calc_holding_days(current_date, pos["entry_date"], intraday)
                cash += sell_price * pos["qty"]

                trades.append(Trade(
                    symbol=sym,
                    name=get_stock_name_fn(sym),
                    entry_date=_dt_to_str(pos["entry_date"]),
                    entry_price=pos["avg_price"],
                    exit_date=dt_str,
                    exit_price=sell_price,
                    qty=pos["qty"],
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    holding_days=holding_days,
                    scale_step="LIMIT-SELL-TTL",
                    exit_reason="매도 TTL 만료 시장가 체결",
                    entry_reason=_make_entry_reason(pos),
                    exit_reason_detail=_make_exit_reason_detail(
                        f"매도 TTL 만료 시장가 체결 (지정가: {limit_price:,.0f} → 시장가: {sell_price:,.0f}, 대기: {order['elapsed_bars']}봉)",
                        pos,
                    ),
                    entry_snapshot=_make_entry_snapshot(pos),
                    exit_snapshot=_make_exit_snapshot(pos, bar),
                    order_type="limit",
                    limit_price=limit_price,
                    fill_price=sell_price,
                    wait_bars=order["elapsed_bars"],
                    fill_method="market_ttl",
                ))
                limit_stats["market_count"] = limit_stats.get("market_count", 0) + 1
                limit_stats["total_wait"] = limit_stats.get("total_wait", 0) + order["elapsed_bars"]
                del holdings[sym]
                continue

            # 아직 대기 중
            still_pending.append(order)

    return cash, still_pending


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

    # ── Phase 2b: 섹터 횡단면 피처 보정 ──
    # Phase 1에서 단일 종목 파티션으로 계산된 sector 피처를
    # 전체 DF에서 올바른 횡단면으로 재계산한다.
    if "sector_id" in full.columns and "price_change_pct" in full.columns:
        if "sector_return" in full.columns:
            full = full.with_columns(
                pl.col("price_change_pct")
                .mean()
                .over(["dt", "sector_id"])
                .alias("sector_return")
            )
        if "sector_rel_strength" in full.columns:
            full = full.with_columns(
                (pl.col("price_change_pct") - pl.col("sector_return"))
                .alias("sector_rel_strength")
            )
        if "sector_rank" in full.columns:
            full = full.with_columns(
                pl.col("price_change_pct")
                .rank()
                .over(["dt", "sector_id"])
                .truediv(
                    pl.col("price_change_pct")
                    .count()
                    .over(["dt", "sector_id"])
                    .clip(lower_bound=1)
                )
                .alias("sector_rank")
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

    # 필요 컬럼만 남기기 (volume, factor_value + 팩터 변수 컬럼 유지)
    select_cols = ["symbol", "dt", "close", "open", "factor_rank"]
    for _hilo in ("high", "low"):
        if _hilo in full.columns:
            select_cols.append(_hilo)
    if "_raw_factor" in full.columns:
        full = full.rename({"_raw_factor": "factor_value"})
        select_cols.append("factor_value")
    if "volume" in full.columns:
        select_cols.append("volume")
    # 팩터 수식에 사용된 개별 변수도 보존 (매수/매도 사유 스냅샷용)
    for rc in required_cols:
        if rc in full.columns and rc not in select_cols:
            select_cols.append(rc)
    full = full.select([c for c in select_cols if c in full.columns])
    return full


def _get_rebalance_dates(
    all_dates: list, freq: str, skip_opening_minutes: int = 0,
) -> list:
    """리밸런싱 날짜/시간 목록 생성.

    all_dates는 date 또는 datetime 리스트.
    """
    if not all_dates:
        return []

    if freq == "every_bar":
        return list(all_dates)

    if freq == "daily":
        # 분봉 데이터: 캘린더 날짜별 첫 바(또는 skip_opening_minutes 이후) 리밸런스
        result: list = []
        # skip 적용: 각 날짜의 첫 봉 시각 + skip_opening_minutes 이후 첫 봉
        day_bars: dict = {}  # date → list[datetime]
        for d in all_dates:
            d_date = d.date() if isinstance(d, datetime) else d
            day_bars.setdefault(d_date, []).append(d)
        for d_date in sorted(day_bars.keys()):
            bars = day_bars[d_date]
            if skip_opening_minutes > 0 and isinstance(bars[0], datetime):
                earliest = bars[0]
                threshold = earliest + timedelta(minutes=skip_opening_minutes)
                picked = next((b for b in bars if b >= threshold), bars[-1])
                result.append(picked)
            else:
                result.append(bars[0])
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
    trailing_stop_pct: float = 0.0,
    max_drawdown_pct: float = 0.0,
    eod_liquidation: bool = True,
    skip_opening_minutes: int = 0,
    engine: str = "loop",
    use_limit_orders: bool = True,
    strict_fill: bool = False,
    limit_ttl_bars: int = 2,
    collect_daily_snapshots: bool = False,
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
    trailing_stop_pct : 포지션별 트레일링 스탑 (0=비활성, 0.20=고점 대비 20% 하락 시 매도).
    max_drawdown_pct : 포트폴리오 최대 낙폭 서킷 브레이커 (0=비활성, 0.15=15%).
    use_limit_orders : 지정가 매매 시뮬레이션 (False=즉시 체결, 기존 방식).
    strict_fill : Strict 모드 (한 호가 관통 시에만 체결).
    limit_ttl_bars : 미체결 대기 봉 수.
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

    # 팩터 수식의 개별 변수 목록 (스냅샷용)
    required_cols = get_required_columns(expression_str)

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

    rebalance_dates_set = set(_get_rebalance_dates(all_dates, rebalance_freq, skip_opening_minutes))

    # EOD 강제 청산용: 각 캘린더 날짜의 마지막 봉 사전 계산
    eod_bar_set: set = set()
    if intraday and eod_liquidation:
        for i in range(len(all_dates) - 1):
            d_cur = all_dates[i]
            d_next = all_dates[i + 1]
            if d_cur.date() != d_next.date():
                eod_bar_set.add(d_cur)
        if all_dates:
            eod_bar_set.add(all_dates[-1])

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
            "high": row.get("high", row["close"]),
            "low": row.get("low", row["close"]),
            "factor_rank": row["factor_rank"],
        }
        if has_factor_value:
            entry["factor_value"] = row.get("factor_value")
        if has_volume:
            entry["volume"] = row.get("volume", 0) or 0
        # 팩터 수식 개별 변수 보존 (매수/매도 사유 스냅샷)
        for rc in required_cols:
            if rc in row and rc not in entry:
                v = row[rc]
                if v is not None:
                    entry[rc] = v
        date_data[dt][sym] = entry

    if progress_cb:
        await progress_cb(40, 100, "포트폴리오 시뮬레이션 시작")

    # ── VectorBT 엔진 분기 ──
    if engine == "vectorbt":
        from app.alpha.factor_backtest_vbt import run_factor_backtest_vbt
        result = run_factor_backtest_vbt(
            all_dates=all_dates,
            date_data=date_data,
            symbols=loaded_symbols,
            initial_capital=initial_capital,
            top_pct=top_pct,
            max_positions=max_positions,
            rebalance_dates_set=rebalance_dates_set,
            eod_bar_set=eod_bar_set,
            cost_config=cost_config,
            stop_loss_pct=stop_loss_pct,
            max_drawdown_pct=max_drawdown_pct,
            band_threshold=band_threshold,
            eod_liquidation=eod_liquidation,
            intraday=intraday,
            get_stock_name=get_stock_name,
            interval=interval,
        )
        # 추가 메트릭 (engine 공통)
        result.metrics["backtest_mode"] = "cross_sectional_portfolio"
        result.metrics["top_pct"] = top_pct
        result.metrics["max_positions"] = max_positions
        result.metrics["rebalance_freq"] = rebalance_freq
        result.metrics["band_threshold"] = band_threshold
        result.metrics["stop_loss_pct"] = stop_loss_pct
        result.metrics["trailing_stop_pct"] = trailing_stop_pct
        result.metrics["max_drawdown_pct"] = max_drawdown_pct
        result.metrics["symbols_count"] = len(loaded_symbols)
        result.metrics["interval"] = interval
        result.metrics["skip_opening_minutes"] = skip_opening_minutes
        if progress_cb:
            await progress_cb(100, 100, "완료")
        return result

    # ── 4. 포트폴리오 시뮬레이션 (loop 엔진) ──
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
    trailing_stop_count = 0
    eod_close_count = 0
    total_band_trades_saved = 0
    circuit_breaker_triggered = False
    peak_equity = initial_capital
    prev_day_data: dict[str, dict] | None = None  # T-1 시그널용
    _accumulating_day_data: dict[str, dict] = {}  # 당일 봉 누적 (종목별 최신)
    _current_calendar_day = None

    # ── 지정가 매매 상태 ──
    pending_orders: list[dict] = []
    limit_stats: dict = {"fill_count": 0, "market_count": 0, "total_wait": 0}

    # ── 일별 스냅샷 수집 (타임라인용) ──
    # key: (symbol, entry_date_str) → snapshots list
    _pos_daily_snapshots: dict[tuple[str, str], list[dict]] = {}

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

        # ── pending order 체결 처리 (손절 체크 전) ──
        if use_limit_orders and pending_orders:
            dt_str_pending = _dt_to_str(current_date)
            _sells_before = len([o for o in pending_orders if o["side"] == "SELL"])
            _buys_before = len([o for o in pending_orders if o["side"] == "BUY"])
            cash, pending_orders = _process_pending_orders_bt(
                pending_orders=pending_orders,
                bar_data=today,
                holdings=holdings,
                trades=trades,
                cash=cash,
                cost_config=cost_config,
                dt_str=dt_str_pending,
                current_date=current_date,
                intraday=intraday,
                strict_fill=strict_fill,
                get_stock_name_fn=get_stock_name,
                limit_stats=limit_stats,
            )
            _buys_filled = _buys_before - len([o for o in pending_orders if o["side"] == "BUY"])
            _sells_filled = _sells_before - len([o for o in pending_orders if o["side"] == "SELL"])
            total_buys += _buys_filled
            total_sells += _sells_filled

        # ── 포지션별 고가/저가 추적 + 포지션 손절 ──
        _snap_cal_day = current_date.date() if isinstance(current_date, datetime) else current_date
        for sym in list(holdings.keys()):
            pos = holdings[sym]
            if sym in today:
                price = today[sym]["close"]
                pos["high_price"] = max(pos.get("high_price", price), price)
                pos["low_price"] = min(pos.get("low_price", price), price)
                pos["last_close"] = price

                # 일별 스냅샷 수집 (하루 1번만)
                if collect_daily_snapshots:
                    _last_snap = pos.get("_last_snapshot_date")
                    if _last_snap != _snap_cal_day:
                        _snap_key = (sym, _dt_to_str(pos["entry_date"]))
                        _pos_daily_snapshots.setdefault(_snap_key, []).append({
                            "date": _snap_cal_day,
                            "close": price,
                            "variables": _extract_factor_variables(today.get(sym), required_cols),
                        })
                        pos["_last_snapshot_date"] = _snap_cal_day

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
                            **_extract_limit_info(pos),
                        ))
                        total_sells += 1
                        stop_loss_count += 1
                        del holdings[sym]
                        logger.debug("Stop-loss triggered: %s (%.1f%%)", sym, drawdown * 100)

                # 트레일링 스탑 체크 (매 바마다, 손절과 독립 실행)
                if sym in holdings and trailing_stop_pct > 0 and not circuit_breaker_triggered:
                    high_price = pos.get("high_price", pos["avg_price"])
                    trail_drawdown = (price - high_price) / high_price
                    if trail_drawdown <= -trailing_stop_pct:
                        _tv = today.get(sym, {}).get("volume", 0) if has_volume else 0
                        sell_price = effective_sell_price(price, cost_config, order_qty=pos["qty"], bar_volume=_tv)
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
                            scale_step="S-TRAIL",
                            exit_reason="트레일링 스탑",
                            entry_reason=_make_entry_reason(pos),
                            exit_reason_detail=_make_exit_reason_detail(
                                f"트레일링 스탑: 고점({high_price:,.0f}) 대비 {trail_drawdown*100:.1f}% 하락 (기준: -{trailing_stop_pct*100:.0f}%)",
                                pos,
                            ),
                            entry_snapshot=_make_entry_snapshot(pos),
                            exit_snapshot=_make_exit_snapshot(pos, today.get(sym)),
                            **_extract_limit_info(pos),
                        ))
                        total_sells += 1
                        trailing_stop_count += 1
                        del holdings[sym]
                        logger.debug("Trailing-stop triggered: %s (high=%.0f, cur=%.0f, %.1f%%)", sym, high_price, price, trail_drawdown * 100)

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
                # pending 매수 주문 취소 → 예약 현금 반환
                for _cb_order in pending_orders:
                    if _cb_order["side"] == "BUY":
                        cash += _cb_order.get("reserved_cash", 0)
                pending_orders.clear()
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
                        **_extract_limit_info(pos),
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
                    **_extract_limit_info(pos),
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
            # band_threshold > 0: 기존 보유 종목은 rank가 exit_rank_threshold 미만일 때만 매도
            # (핑퐁 방지: 상위 20% 진입 → threshold 미만 이탈 시에만 퇴출)
            exit_rank_threshold = (1.0 - top_pct) * (1.0 - band_threshold) if band_threshold > 0 else None
            band_trades_saved = 0

            sell_list: list[str] = []
            for sym in list(holdings.keys()):
                if sym not in prev_day_data:
                    if sym not in target_symbols and sym in today:
                        sell_list.append(sym)
                    continue
                if sym not in today:
                    continue
                if sym not in target_symbols:
                    # band 적용: rank가 exit_rank_threshold 이상이면 유지
                    if exit_rank_threshold is not None:
                        sym_rank = prev_day_data[sym].get("factor_rank", 0)
                        if sym_rank >= exit_rank_threshold:
                            band_trades_saved += 1
                            continue
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
                if use_limit_orders:
                    # 지정가 매도: pending에 추가 (포지션은 유지, 현금 변동 없음)
                    pos = holdings.get(sym)
                    if pos is None:
                        continue
                    # 이미 pending 매도가 있는 종목은 건너뜀
                    if any(o["symbol"] == sym and o["side"] == "SELL" for o in pending_orders):
                        continue
                    pending_orders.append({
                        "symbol": sym,
                        "side": "SELL",
                        "price": today[sym]["open"],  # 시가를 지정가로 사용
                        "qty": pos["qty"],
                        "ttl_bars": limit_ttl_bars,
                        "elapsed_bars": 0,
                        "entry_date": pos.get("entry_date", current_date),
                    })
                else:
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
                        **_extract_limit_info(pos),
                    ))
                    total_sells += 1

            # ── 매수 실행 (당일 시가) ──
            # [2026-04-06] 장 마감 매수 컷오프: 분봉에서 장 마감 N분 전 매수 금지
            _buy_cutoff_blocked = False
            if intraday and isinstance(current_date, datetime):
                _market_close_min = 15 * 60 + 30  # 15:30
                _cur_min = current_date.hour * 60 + current_date.minute
                _cutoff = settings.ALPHA_INTRADAY_BUY_CUTOFF_MINUTES
                if _cur_min >= _market_close_min - _cutoff:
                    _buy_cutoff_blocked = True

            if buy_list and cash > 0 and not _buy_cutoff_blocked:
                per_stock_budget = cash / max(len(buy_list), 1)

                for sym in buy_list:
                    # 이미 pending 매수가 있는 종목은 건너뜀
                    if use_limit_orders and any(o["symbol"] == sym and o["side"] == "BUY" for o in pending_orders):
                        continue

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

                    if use_limit_orders:
                        # 지정가 매수: pending에 추가 + 현금 예약
                        reserved_cash = cost
                        cash -= reserved_cash
                        pending_orders.append({
                            "symbol": sym,
                            "side": "BUY",
                            "price": today[sym]["open"],  # 시가를 지정가로 사용
                            "qty": qty,
                            "ttl_bars": limit_ttl_bars,
                            "elapsed_bars": 0,
                            "reserved_cash": reserved_cash,
                            "entry_date": current_date,
                            "entry_info": {
                                "entry_factor_rank": sym_rank,
                                "entry_factor_value": prev_day_data[sym].get("factor_value"),
                                "entry_rank_pos": rank_pos,
                                "entry_total_candidates": total_candidates,
                                "entry_target_count": len(target_symbols),
                            },
                        })
                    else:
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
                            # 팩터 변수 스냅샷 (매수/매도 사유 분석용)
                            "_factor_vars": _extract_factor_variables(prev_day_data.get(sym), required_cols),
                            "_required_cols": required_cols,
                        }
                        total_buys += 1

            if sell_list or buy_list:
                rebalance_count += 1
            total_band_trades_saved += band_trades_saved

        # ── 장 종료 강제 청산 (장중 단타 원칙) ──
        if intraday and eod_liquidation and current_date in eod_bar_set and not circuit_breaker_triggered:
            # EOD 시 pending 매수 취소 → 예약 현금 반환
            if pending_orders:
                for _eod_order in pending_orders:
                    if _eod_order["side"] == "BUY":
                        cash += _eod_order.get("reserved_cash", 0)
                pending_orders.clear()
        if intraday and eod_liquidation and current_date in eod_bar_set and holdings and not circuit_breaker_triggered:
            for sym in list(holdings.keys()):
                pos = holdings.pop(sym)
                price = today.get(sym, {}).get("close", pos.get("last_close", pos["avg_price"]))
                _sv_eod = today.get(sym, {}).get("volume", 0) if has_volume else 0
                sell_price = effective_sell_price(price, cost_config, order_qty=pos["qty"], bar_volume=_sv_eod)
                pnl = (sell_price - pos["avg_price"]) * pos["qty"]
                pnl_pct = (sell_price / pos["avg_price"] - 1) * 100 if pos["avg_price"] > 0 else 0
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
                    scale_step="S-EOD",
                    exit_reason="장 종료 강제 청산",
                    entry_reason=_make_entry_reason(pos),
                    exit_reason_detail=_make_exit_reason_detail(
                        "장중 단타 원칙: 장 마감 시 전량 청산",
                        pos,
                    ),
                    entry_snapshot=_make_entry_snapshot(pos),
                    exit_snapshot=_make_exit_snapshot(pos, today.get(sym)),
                    **_extract_limit_info(pos),
                ))
                total_sells += 1
                eod_close_count += 1

        # prev_day_data 누적: 장후 시간외(15:35) 봉에 7종목만 있어도
        # 정규 장중(09:00~15:30) 전체 종목 데이터가 유지되도록 한다.
        current_cal_day = current_date.date() if isinstance(current_date, datetime) else current_date
        if _current_calendar_day is not None and current_cal_day != _current_calendar_day:
            prev_day_data = _accumulating_day_data
            _accumulating_day_data = {}
        _current_calendar_day = current_cal_day
        _accumulating_day_data.update(today)

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
        # pending 매수의 예약 현금도 포트폴리오 가치에 포함
        # (예약 현금은 아직 사용된 게 아니라 체결 대기 중이므로 자산에서 빠지면 안 됨)
        for _pend in pending_orders:
            if _pend.get("side") == "BUY":
                portfolio_value += _pend.get("reserved_cash", 0)

        dt_key = current_date.isoformat() if isinstance(current_date, datetime) else current_date.isoformat()
        equity_curve.append({
            "date": dt_key,
            "equity": portfolio_value,
        })

        # 진행률
        if progress_cb and (day_idx + 1) % 50 == 0:
            pct = 40 + int(50 * (day_idx + 1) / len(all_dates))
            await progress_cb(pct, 100, f"시뮬레이션 {day_idx + 1}/{len(all_dates)}")

    # ── 5. 미체결 주문 정리 + 잔여 포지션 강제 청산 ──
    # pending 매수 주문 취소 → 예약 현금 반환
    for order in pending_orders:
        if order["side"] == "BUY":
            cash += order.get("reserved_cash", 0)
    pending_orders.clear()

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
            **_extract_limit_info(pos),
        ))

    holdings.clear()

    if progress_cb:
        await progress_cb(95, 100, "성과 지표 산출 중...")

    # ── 6. 성과 지표 ──
    metrics = compute_metrics(
        trades, equity_curve, initial_capital,
        annualize=bars_per_year(interval),
        intraday=intraday,
    )

    # 추가 메트릭
    metrics["total_buys"] = total_buys
    metrics["total_sells"] = total_sells
    metrics["rebalance_count"] = rebalance_count
    metrics["stop_loss_count"] = stop_loss_count
    metrics["trailing_stop_count"] = trailing_stop_count
    metrics["eod_close_count"] = eod_close_count
    metrics["eod_liquidation"] = eod_liquidation and intraday
    metrics["band_trades_saved"] = total_band_trades_saved
    metrics["circuit_breaker_triggered"] = circuit_breaker_triggered
    metrics["backtest_mode"] = "cross_sectional_portfolio"
    metrics["top_pct"] = top_pct
    metrics["max_positions"] = max_positions
    metrics["rebalance_freq"] = rebalance_freq
    metrics["band_threshold"] = band_threshold
    metrics["stop_loss_pct"] = stop_loss_pct
    metrics["trailing_stop_pct"] = trailing_stop_pct
    metrics["max_drawdown_pct"] = max_drawdown_pct
    metrics["symbols_count"] = len(loaded_symbols)
    metrics["interval"] = interval
    metrics["skip_opening_minutes"] = skip_opening_minutes
    metrics["engine"] = "loop"
    # 거래 비용 설정 기록 (UI 표시용)
    metrics["buy_commission"] = cost_config.buy_commission
    metrics["sell_commission"] = cost_config.sell_commission
    metrics["slippage_pct"] = cost_config.slippage_pct

    # ── 지정가 매매 메트릭 ──
    if use_limit_orders:
        _lf = limit_stats.get("fill_count", 0)
        _lm = limit_stats.get("market_count", 0)
        _lt = limit_stats.get("total_wait", 0)
        metrics["limit_fill_rate"] = _lf / max(_lf + _lm, 1)
        metrics["limit_unfilled_count"] = _lm
        metrics["limit_avg_wait_bars"] = _lt / max(_lf + _lm, 1)
        metrics["use_limit_orders"] = True
        metrics["strict_fill"] = strict_fill
        metrics["limit_ttl_bars"] = limit_ttl_bars
    else:
        metrics["use_limit_orders"] = False

    # ── 장중 전용 메트릭 (4-B, 4-C, 4-D) ──
    if intraday and equity_curve:
        # 4-B: 시간대별 수익률 분해
        pnl_by_session = {"morning": 0.0, "midday": 0.0, "afternoon": 0.0}
        for t in trades:
            if not t.exit_date:
                continue
            try:
                exit_dt = datetime.fromisoformat(t.exit_date)
                hour = exit_dt.hour
                if hour < 10:
                    pnl_by_session["morning"] += t.pnl
                elif hour < 14:
                    pnl_by_session["midday"] += t.pnl
                else:
                    pnl_by_session["afternoon"] += t.pnl
            except (ValueError, AttributeError):
                pass
        metrics["pnl_by_session"] = {
            k: round(v) for k, v in pnl_by_session.items()
        }

        # 4-C: 장중 MDD (일별 봉간 고점-저점)
        day_intraday_mdds: list[float] = []
        day_equities: dict[str, list[float]] = {}
        for pt in equity_curve:
            d = pt["date"][:10]
            day_equities.setdefault(d, []).append(pt["equity"])
        for d_key in sorted(day_equities.keys()):
            eqs = day_equities[d_key]
            peak = eqs[0]
            worst_dd = 0.0
            for eq in eqs:
                if eq > peak:
                    peak = eq
                dd = (peak - eq) / peak * 100 if peak > 0 else 0
                worst_dd = max(worst_dd, dd)
            day_intraday_mdds.append(worst_dd)
        metrics["intraday_mdd_avg"] = round(sum(day_intraday_mdds) / max(len(day_intraday_mdds), 1), 2)
        metrics["intraday_mdd_worst"] = round(max(day_intraday_mdds) if day_intraday_mdds else 0, 2)

        # 4-D: Gross vs Net 수익률
        gross_pnl = sum(abs(t.pnl_pct) * (1 if t.pnl > 0 else -1) for t in trades if t.exit_date)
        net_pnl = metrics.get("total_return", 0)
        metrics["total_return_gross"] = round(net_pnl, 2)  # 현재 이미 비용 포함
        # 거래 비용 추정: 총 매수/매도 × 비용율
        total_traded_value = sum(
            t.entry_price * t.qty + (t.exit_price or t.entry_price) * t.qty
            for t in trades if t.exit_date
        )
        estimated_cost = total_traded_value * (cost_config.buy_commission + cost_config.sell_commission + cost_config.slippage_pct * 2)
        cost_drag_pct = estimated_cost / initial_capital * 100 if initial_capital > 0 else 0
        metrics["cost_drag_pct"] = round(cost_drag_pct, 2)

    if progress_cb:
        await progress_cb(100, 100, "완료")

    # ── 일별 스냅샷 후처리 (다운샘플링 + 매핑) ──
    daily_snapshots_raw: list[dict] | None = None
    if collect_daily_snapshots and _pos_daily_snapshots:
        daily_snapshots_raw = []
        for trade_idx, t in enumerate(trades):
            snap_key = (t.symbol, t.entry_date)
            snapshots = _pos_daily_snapshots.get(snap_key)
            if not snapshots:
                continue
            # 다운샘플링: 너무 긴 보유기간은 간격 조정
            if len(snapshots) > 120:
                snapshots = snapshots[::10]
            elif len(snapshots) > 60:
                snapshots = snapshots[::5]
            for snap in snapshots:
                daily_snapshots_raw.append({
                    "backtest_run_id": None,  # execute에서 설정
                    "trade_index": trade_idx,
                    "symbol": t.symbol,
                    "snapshot_date": snap["date"],
                    "close": snap.get("close"),
                    "variables": snap.get("variables", {}),
                })

    return BacktestResult(
        trades=trades,
        equity_curve=equity_curve,
        metrics=metrics,
        daily_snapshots_raw=daily_snapshots_raw,
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
    trailing_stop_pct: float = 0.0,
    max_drawdown_pct: float = 0.0,
    eod_liquidation: bool = True,
    skip_opening_minutes: int = 0,
    engine: str = "loop",
    use_limit_orders: bool = True,
    strict_fill: bool = False,
    limit_ttl_bars: int = 2,
    collect_daily_snapshots: bool = False,
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
            trailing_stop_pct=trailing_stop_pct,
            max_drawdown_pct=max_drawdown_pct,
            eod_liquidation=eod_liquidation,
            skip_opening_minutes=skip_opening_minutes,
            engine=engine,
            use_limit_orders=use_limit_orders,
            strict_fill=strict_fill,
            limit_ttl_bars=limit_ttl_bars,
            collect_daily_snapshots=collect_daily_snapshots,
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

        # ── daily_snapshots INSERT (별도 트랜잭션, 실패해도 백테스트 결과 보존) ──
        if collect_daily_snapshots and result.daily_snapshots_raw:
            try:
                from app.backtest.models import BacktestDailySnapshot

                async with async_session() as db2:
                    for item in result.daily_snapshots_raw:
                        item["backtest_run_id"] = run_id
                        db2.add(BacktestDailySnapshot(**item))
                    await db2.commit()
                logger.info("Daily snapshots saved: %d rows for run %s", len(result.daily_snapshots_raw), run_id)
            except Exception as e:
                logger.warning("Daily snapshots save failed (non-critical): %s", e)

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


# ── 듀얼 팩터 백테스트 (일봉 선별 + 분봉 진입/퇴출) ──


async def _load_intraday_for_date(
    symbols: list[str],
    trade_date: date,
    interval: str = "5m",
) -> pl.DataFrame | None:
    """특정 날짜의 분봉 데이터를 온디맨드로 로딩 + 외부 데이터 enrichment.

    DART 재무, 투자자 수급, 신용/공매도 등 일별 데이터를 분봉에 JOIN하여
    분봉 팩터에서도 earnings_yield, foreign_net_norm 등을 사용할 수 있게 한다.
    데이터가 없으면 None 반환 (일봉 폴백 신호).
    """
    from sqlalchemy import text

    async with async_session() as db:
        result = await db.execute(
            text("""
                SELECT dt, symbol, open, high, low, close, volume
                FROM stock_candles
                WHERE interval = :interval
                  AND symbol = ANY(:symbols)
                  AND dt::date = :trade_date
                ORDER BY symbol, dt
            """),
            {"interval": interval, "symbols": symbols, "trade_date": trade_date},
        )
        rows = result.fetchall()

    if not rows:
        return None

    df = pl.DataFrame({
        "dt": [r.dt for r in rows],
        "symbol": [r.symbol for r in rows],
        "open": [float(r.open) for r in rows],
        "high": [float(r.high) for r in rows],
        "low": [float(r.low) for r in rows],
        "close": [float(r.close) for r in rows],
        "volume": [int(r.volume) for r in rows],
    })

    # 외부 데이터 enrichment (일별 데이터를 분봉에 JOIN)
    try:
        from app.backtest.data_loader import (
            _load_dart_financials,
            _load_investor_trading,
            _load_margin_short,
            _load_sector_mapping,
        )
        from datetime import timedelta

        unique_syms = df["symbol"].unique().to_list()
        # T-1 ~ T 범위로 로드 (T+1 shift 적용되므로)
        ext_start = trade_date - timedelta(days=7)
        ext_end = trade_date + timedelta(days=1)

        df = df.with_columns(pl.col("dt").cast(pl.Date).alias("dt_date"))

        # DART 재무 (backward join_asof)
        dart_df = await _load_dart_financials(unique_syms, ext_start, ext_end)
        if not dart_df.is_empty():
            df = df.sort(["symbol", "dt_date"])
            dart_df = dart_df.sort(["symbol", "disclosure_date"])
            df = df.join_asof(dart_df, left_on="dt_date", right_on="disclosure_date", by="symbol", strategy="backward")
        for col in ["eps", "bps", "operating_margin", "debt_to_equity"]:
            if col not in df.columns:
                df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

        # 투자자 수급 (T+1 shift)
        inv_df = await _load_investor_trading(unique_syms, ext_start, ext_end)
        if not inv_df.is_empty():
            inv_shifted = inv_df.with_columns(
                (pl.col("dt").cast(pl.Date) + pl.duration(days=1)).alias("dt_next")
            )
            df = df.join(inv_shifted, left_on=["symbol", "dt_date"], right_on=["symbol", "dt_next"], how="left")
            if "dt_right" in df.columns:
                df = df.drop("dt_right")

        # 신용/공매도 (T+1 shift)
        ms_df = await _load_margin_short(unique_syms, ext_start, ext_end)
        if not ms_df.is_empty():
            ms_shifted = ms_df.with_columns(
                (pl.col("dt").cast(pl.Date) + pl.duration(days=1)).alias("dt_next")
            )
            df = df.join(ms_shifted, left_on=["symbol", "dt_date"], right_on=["symbol", "dt_next"], how="left")
            if "dt_right" in df.columns:
                df = df.drop("dt_right")

        # 섹터
        sector_df = await _load_sector_mapping(unique_syms)
        if not sector_df.is_empty():
            df = df.join(sector_df, on="symbol", how="left")

        # dt_date 헬퍼 컬럼 제거
        if "dt_date" in df.columns:
            df = df.drop("dt_date")

    except Exception as e:
        logger.debug("Intraday enrichment failed (non-critical): %s", e)

    return df


async def _get_intraday_start_date(interval: str) -> date | None:
    """분봉 데이터가 존재하는 최초 날짜를 조회."""
    from sqlalchemy import text

    async with async_session() as db:
        result = await db.execute(
            text("SELECT MIN(dt::date) FROM stock_candles WHERE interval = :iv LIMIT 1"),
            {"iv": interval},
        )
        return result.scalar()


def _compute_intraday_factor_ranks(
    intraday_df: pl.DataFrame,
    polars_expr: pl.Expr,
    required_cols: set[str],
) -> pl.DataFrame:
    """분봉 DataFrame에 팩터 수식을 적용하고 봉별 횡단면 랭크를 계산.

    Returns
    -------
    pl.DataFrame
        dt, symbol, close, open, volume, intraday_rank (0~1) 컬럼 포함.
    """
    # 종목별로 rolling 지표 계산
    partitions = intraday_df.sort(["symbol", "dt"]).partition_by("symbol", maintain_order=True)
    parts: list[pl.DataFrame] = []
    for sym_df in partitions:
        if sym_df.height < 2:
            continue
        try:
            sym_df = ensure_alpha_features(sym_df, required_cols=required_cols)
            parts.append(sym_df)
        except Exception:
            continue

    if not parts:
        return pl.DataFrame()

    full = pl.concat(parts)

    # 팩터 수식 적용
    full = full.with_columns(polars_expr.alias("_raw_intraday"))

    # Inf/NaN → null
    full = full.with_columns(
        pl.when(pl.col("_raw_intraday").is_finite())
        .then(pl.col("_raw_intraday"))
        .otherwise(None)
        .alias("_raw_intraday")
    )

    # 봉별 횡단면 퍼센타일 랭크 (0~1)
    full = full.with_columns(
        pl.col("_raw_intraday")
        .rank(method="average")
        .over("dt")
        .truediv(
            pl.col("_raw_intraday")
            .count()
            .over("dt")
            .cast(pl.Float64)
            .clip(lower_bound=1.0)
        )
        .fill_null(0.5)
        .cast(pl.Float64)
        .alias("intraday_rank")
    )

    select_cols = ["dt", "symbol", "close", "open", "intraday_rank"]
    for _hilo in ("high", "low"):
        if _hilo in full.columns:
            select_cols.append(_hilo)
    if "volume" in full.columns:
        select_cols.append("volume")
    return full.select([c for c in select_cols if c in full.columns])


async def run_dual_factor_backtest(
    daily_expression_str: str,
    intraday_expression_str: str,
    symbols: list[str],
    start_date: date,
    end_date: date,
    initial_capital: float = 100_000_000,
    top_pct: float = 0.2,
    max_positions: int = 20,
    intraday_interval: str = "5m",
    intraday_entry_threshold: float = 0.8,
    intraday_exit_threshold: float = 0.2,
    stop_loss_pct: float = 0.15,
    trailing_stop_pct: float = 0.20,
    cost_config: CostConfig | None = None,
    progress_cb: ProgressCallback = None,
    use_limit_orders: bool = True,
    strict_fill: bool = False,
    limit_ttl_bars: int = 2,
    collect_daily_snapshots: bool = False,
    **kwargs,
) -> BacktestResult:
    """듀얼 팩터 백테스트.

    일봉 팩터로 매수 후보 종목(watchlist)을 선정하고,
    분봉 팩터로 정확한 매수/매도 시점을 결정한다.
    분봉 데이터가 없는 기간은 일봉 전용 리밸런싱으로 폴백한다.

    Parameters
    ----------
    daily_expression_str : 일봉 팩터 수식 (종목 선별용).
    intraday_expression_str : 분봉 팩터 수식 (타이밍 결정용).
    symbols : 유니버스 종목 리스트.
    start_date, end_date : 백테스트 기간.
    initial_capital : 초기 자본.
    top_pct : 일봉 팩터 상위 비율 (0.2 = 상위 20%).
    max_positions : 최대 동시 보유 종목 수.
    intraday_interval : 분봉 인터벌 (기본 5m).
    intraday_entry_threshold : 분봉 팩터 랭크 진입 기준 (기본 0.8, 상위 20%).
    intraday_exit_threshold : 분봉 팩터 랭크 퇴출 기준 (기본 0.2, 하위 20%).
    stop_loss_pct : 포지션별 손절 비율.
    trailing_stop_pct : 트레일링 스탑 비율.
    cost_config : 거래 비용 설정.
    progress_cb : 진행률 콜백.
    use_limit_orders : 지정가 매매 시뮬레이션 (False=즉시 체결, 기존 방식).
    strict_fill : Strict 모드 (한 호가 관통 시에만 체결).
    limit_ttl_bars : 미체결 대기 봉 수.
    """
    if cost_config is None:
        cost_config = default_cost_config(intraday_interval)

    if len(symbols) < 3:
        return BacktestResult(
            metrics={"error": "듀얼 팩터 백테스트는 최소 3종목 이상 필요합니다."}
        )

    # TTL 자동 조정: 기본값(2)이면 인터벌에 맞게 동적 결정
    if limit_ttl_bars == 2 and intraday_interval != "1d":
        limit_ttl_bars = _default_limit_ttl(intraday_interval)
        logger.info("TTL 자동 조정: %s → %d봉", intraday_interval, limit_ttl_bars)

    # ── 1. 일봉 데이터 로딩 + 팩터 사전 계산 ──
    if progress_cb:
        await progress_cb(0, 100, "일봉 데이터 로딩 중...")

    warmup_start = start_date - timedelta(days=warmup_days("1d"))
    daily_df = await load_enriched_candles(symbols, warmup_start, end_date, "1d")

    if daily_df.is_empty():
        return BacktestResult(metrics={"error": "일봉 데이터가 없습니다."})

    loaded_symbols = daily_df["symbol"].unique().sort().to_list()

    if progress_cb:
        await progress_cb(5, 100, f"{len(loaded_symbols)}개 종목 일봉 로딩 완료")

    # ── 2. 일봉 팩터 값 사전 계산 ──
    if progress_cb:
        await progress_cb(10, 100, "일봉 팩터 값 계산 중...")

    try:
        daily_factor_df = _precompute_factor_values(daily_df, daily_expression_str, interval="1d")
    except Exception as e:
        return BacktestResult(
            metrics={"error": f"일봉 팩터 수식 계산 실패: {str(e)[:200]}"}
        )

    if daily_factor_df.is_empty():
        return BacktestResult(
            metrics={"error": "일봉 팩터 값 계산 결과가 비어 있습니다."}
        )

    # dt → date 변환
    if daily_factor_df["dt"].dtype == pl.Datetime:
        daily_factor_df = daily_factor_df.with_columns(pl.col("dt").dt.date().alias("dt"))

    if progress_cb:
        await progress_cb(20, 100, "일봉 팩터 계산 완료")

    # 일봉 팩터 수식의 개별 변수 목록 (스냅샷용)
    daily_required_cols = get_required_columns(daily_expression_str)

    # ── 3. 분봉 팩터 수식 파싱 (데이터는 날짜별 온디맨드) ──
    intraday_expr = parse_expression(intraday_expression_str)
    intraday_polars_expr = sympy_to_polars(intraday_expr)
    intraday_required_cols = get_required_columns(intraday_expression_str)

    # 분봉 데이터 존재하는 최초 날짜 조회
    intraday_start = await _get_intraday_start_date(intraday_interval)

    if progress_cb:
        await progress_cb(25, 100, f"분봉 데이터 시작: {intraday_start or '없음'}")

    # ── 4. 날짜별 일봉 데이터 인덱싱 ──
    has_factor_value = "factor_value" in daily_factor_df.columns

    daily_date_data: dict[date, dict[str, dict]] = defaultdict(dict)
    for row in daily_factor_df.iter_rows(named=True):
        dt_val = row["dt"]
        sym = row["symbol"]
        entry = {
            "close": row["close"],
            "open": row["open"],
            "high": row.get("high", row["close"]),
            "low": row.get("low", row["close"]),
            "factor_rank": row["factor_rank"],
        }
        if has_factor_value:
            entry["factor_value"] = row.get("factor_value")
        if "volume" in row:
            entry["volume"] = row.get("volume", 0) or 0
        daily_date_data[dt_val][sym] = entry

    all_daily_dates = sorted(
        d for d in daily_date_data.keys() if d >= start_date
    )

    if len(all_daily_dates) < 2:
        return BacktestResult(
            metrics={"error": "시뮬레이션 가능한 거래일이 부족합니다."}
        )

    if progress_cb:
        await progress_cb(30, 100, "포트폴리오 시뮬레이션 시작")

    # ── 5. 시뮬레이션 루프 ──
    cutoff = 1.0 - top_pct
    cash = initial_capital
    holdings: dict[str, dict] = {}
    trades: list[Trade] = []
    equity_curve: list[dict] = []

    total_buys = 0
    total_sells = 0
    daily_rebal_count = 0
    intraday_trade_days = 0
    stop_loss_count = 0
    trailing_stop_count = 0

    # ── 지정가 매매 상태 ──
    pending_orders: list[dict] = []
    limit_stats: dict = {"fill_count": 0, "market_count": 0, "total_wait": 0}

    # ── 일별 스냅샷 수집 (타임라인용) ──
    _pos_daily_snapshots: dict[tuple[str, str], list[dict]] = {}

    prev_day_data: dict[str, dict] | None = None

    for day_idx, trade_date in enumerate(all_daily_dates):
        today_daily = daily_date_data.get(trade_date, {})
        if not today_daily:
            if equity_curve:
                equity_curve.append({
                    "date": trade_date.isoformat(),
                    "equity": equity_curve[-1]["equity"],
                })
            continue

        # 일봉 팩터 랭킹 → watchlist (T-1 시그널 기반)
        if prev_day_data and len(prev_day_data) >= 3:
            ranked = sorted(
                prev_day_data.items(),
                key=lambda x: x[1]["factor_rank"],
                reverse=True,
            )
            n_top = max(1, int(len(ranked) * top_pct))
            n_top = min(n_top, max_positions)
            watchlist = set()
            for sym, data in ranked[:n_top]:
                if data["factor_rank"] is not None:
                    watchlist.add(sym)
        else:
            watchlist = set()

        # 분봉 데이터 사용 가능 여부 판별
        use_intraday = (
            intraday_start is not None
            and trade_date >= intraday_start
            and (watchlist or holdings)
        )

        if use_intraday:
            # ── 분봉 모드: 온디맨드 로딩 ──
            targets = list(watchlist | set(holdings.keys()))
            intraday_df = await _load_intraday_for_date(
                targets, trade_date, intraday_interval,
            )

            if intraday_df is not None and intraday_df.height > 0:
                # 분봉 팩터 랭크 계산
                ranked_intraday = _compute_intraday_factor_ranks(
                    intraday_df, intraday_polars_expr, intraday_required_cols,
                )

                if ranked_intraday.height > 0:
                    intraday_trade_days += 1
                    # 봉별 데이터 인덱싱: {dt: {symbol: {close, open, intraday_rank, volume}}}
                    bar_data: dict[datetime, dict[str, dict]] = defaultdict(dict)
                    for row in ranked_intraday.iter_rows(named=True):
                        bar_entry: dict = {
                            "close": row["close"],
                            "open": row["open"],
                            "high": row.get("high", row["close"]),
                            "low": row.get("low", row["close"]),
                            "intraday_rank": row["intraday_rank"],
                        }
                        if "volume" in row:
                            bar_entry["volume"] = row.get("volume", 0) or 0
                        bar_data[row["dt"]][row["symbol"]] = bar_entry

                    sorted_bars = sorted(bar_data.keys())

                    for bar_dt in sorted_bars:
                        bar = bar_data[bar_dt]

                        # ── pending order 체결 처리 (손절 체크 전) ──
                        if use_limit_orders and pending_orders:
                            _dt_str_bar = _dt_to_str(bar_dt)
                            _sb = len([o for o in pending_orders if o["side"] == "SELL"])
                            _bb = len([o for o in pending_orders if o["side"] == "BUY"])
                            cash, pending_orders = _process_pending_orders_bt(
                                pending_orders=pending_orders,
                                bar_data=bar,
                                holdings=holdings,
                                trades=trades,
                                cash=cash,
                                cost_config=cost_config,
                                dt_str=_dt_str_bar,
                                current_date=bar_dt,
                                intraday=True,
                                strict_fill=strict_fill,
                                get_stock_name_fn=get_stock_name,
                                limit_stats=limit_stats,
                            )
                            total_buys += _bb - len([o for o in pending_orders if o["side"] == "BUY"])
                            total_sells += _sb - len([o for o in pending_orders if o["side"] == "SELL"])

                        # ── 포지션 관리: 손절/트레일링 + 분봉 매도 ──
                        for sym in list(holdings.keys()):
                            pos = holdings[sym]
                            if sym not in bar:
                                continue

                            price = bar[sym]["close"]
                            pos["high_price"] = max(pos.get("high_price", price), price)
                            pos["low_price"] = min(pos.get("low_price", price), price)
                            pos["last_close"] = price

                            # 손절 체크
                            if stop_loss_pct > 0:
                                drawdown = (price - pos["avg_price"]) / pos["avg_price"]
                                if drawdown <= -stop_loss_pct:
                                    sell_price = effective_sell_price(price, cost_config, order_qty=pos["qty"], bar_volume=bar[sym].get("volume", 0))
                                    pnl = (sell_price - pos["avg_price"]) * pos["qty"]
                                    pnl_pct = (sell_price / pos["avg_price"] - 1) * 100
                                    cash += sell_price * pos["qty"]
                                    trades.append(Trade(
                                        symbol=sym,
                                        name=get_stock_name(sym),
                                        entry_date=_dt_to_str(pos["entry_date"]),
                                        entry_price=pos["avg_price"],
                                        exit_date=_dt_to_str(bar_dt),
                                        exit_price=sell_price,
                                        qty=pos["qty"],
                                        pnl=pnl,
                                        pnl_pct=pnl_pct,
                                        holding_days=_calc_holding_days(bar_dt, pos["entry_date"], True),
                                        scale_step="STOP-LOSS",
                                        exit_reason="손절",
                                        entry_reason=_make_entry_reason(pos),
                                        exit_reason_detail=_make_exit_reason_detail(
                                            f"포지션 손절: 진입가 대비 {drawdown*100:.1f}% 하락 (기준: -{stop_loss_pct*100:.0f}%)",
                                            pos,
                                        ),
                                        entry_snapshot=_make_entry_snapshot(pos),
                                        exit_snapshot={},
                                        **_extract_limit_info(pos),
                                    ))
                                    total_sells += 1
                                    stop_loss_count += 1
                                    del holdings[sym]
                                    continue

                            # 트레일링 스탑 체크 (손절과 독립)
                            if sym in holdings and trailing_stop_pct > 0:
                                high_price = pos.get("high_price", pos["avg_price"])
                                trail_dd = (price - high_price) / high_price
                                if trail_dd <= -trailing_stop_pct:
                                    sell_price = effective_sell_price(price, cost_config, order_qty=pos["qty"], bar_volume=bar[sym].get("volume", 0))
                                    pnl = (sell_price - pos["avg_price"]) * pos["qty"]
                                    pnl_pct = (sell_price / pos["avg_price"] - 1) * 100
                                    cash += sell_price * pos["qty"]
                                    trades.append(Trade(
                                        symbol=sym,
                                        name=get_stock_name(sym),
                                        entry_date=_dt_to_str(pos["entry_date"]),
                                        entry_price=pos["avg_price"],
                                        exit_date=_dt_to_str(bar_dt),
                                        exit_price=sell_price,
                                        qty=pos["qty"],
                                        pnl=pnl,
                                        pnl_pct=pnl_pct,
                                        holding_days=_calc_holding_days(bar_dt, pos["entry_date"], True),
                                        scale_step="S-TRAIL",
                                        exit_reason="트레일링 스탑",
                                        entry_reason=_make_entry_reason(pos),
                                        exit_reason_detail=_make_exit_reason_detail(
                                            f"트레일링 스탑: 고점({high_price:,.0f}) 대비 {trail_dd*100:.1f}% 하락",
                                            pos,
                                        ),
                                        entry_snapshot=_make_entry_snapshot(pos),
                                        exit_snapshot={},
                                        **_extract_limit_info(pos),
                                    ))
                                    total_sells += 1
                                    trailing_stop_count += 1
                                    del holdings[sym]
                                    continue

                            # 분봉 팩터 매도 시그널: 랭크가 exit_threshold 이하
                            intraday_rank = bar[sym].get("intraday_rank", 0.5)
                            if intraday_rank <= intraday_exit_threshold:
                                if use_limit_orders:
                                    # 지정가 매도: pending에 추가
                                    if not any(o["symbol"] == sym and o["side"] == "SELL" for o in pending_orders):
                                        pending_orders.append({
                                            "symbol": sym,
                                            "side": "SELL",
                                            "price": price,
                                            "qty": pos["qty"],
                                            "ttl_bars": limit_ttl_bars,
                                            "elapsed_bars": 0,
                                            "entry_date": pos.get("entry_date", bar_dt),
                                        })
                                else:
                                    sell_price = effective_sell_price(price, cost_config, order_qty=pos["qty"], bar_volume=bar[sym].get("volume", 0))
                                    pnl = (sell_price - pos["avg_price"]) * pos["qty"]
                                    pnl_pct = (sell_price / pos["avg_price"] - 1) * 100
                                    cash += sell_price * pos["qty"]
                                    trades.append(Trade(
                                        symbol=sym,
                                        name=get_stock_name(sym),
                                        entry_date=_dt_to_str(pos["entry_date"]),
                                        entry_price=pos["avg_price"],
                                        exit_date=_dt_to_str(bar_dt),
                                        exit_price=sell_price,
                                        qty=pos["qty"],
                                        pnl=pnl,
                                        pnl_pct=pnl_pct,
                                        holding_days=_calc_holding_days(bar_dt, pos["entry_date"], True),
                                        scale_step="INTRADAY-SELL",
                                        exit_reason="분봉 팩터 매도",
                                        entry_reason=_make_entry_reason(pos),
                                        exit_reason_detail=_make_exit_reason_detail(
                                            f"분봉 팩터 랭크 {intraday_rank:.2f} <= 퇴출 기준 {intraday_exit_threshold:.2f}",
                                            pos,
                                        ),
                                        entry_snapshot=_make_entry_snapshot(pos),
                                        exit_snapshot={},
                                        **_extract_limit_info(pos),
                                    ))
                                    total_sells += 1
                                    del holdings[sym]

                        # ── 분봉 매수: watchlist 중 미보유 + 랭크 충족 ──
                        # [2026-04-06] 장 마감 매수 컷오프
                        _intraday_buy_ok = True
                        if isinstance(bar_dt, datetime):
                            _mc = 15 * 60 + 30
                            _cm = bar_dt.hour * 60 + bar_dt.minute
                            if _cm >= _mc - settings.ALPHA_INTRADAY_BUY_CUTOFF_MINUTES:
                                _intraday_buy_ok = False

                        if len(holdings) < max_positions and cash > 0 and _intraday_buy_ok:
                            buy_candidates = []
                            for sym in watchlist:
                                if sym in holdings or sym not in bar:
                                    continue
                                intraday_rank = bar[sym].get("intraday_rank", 0.5)
                                if intraday_rank >= intraday_entry_threshold:
                                    buy_candidates.append((sym, intraday_rank))

                            # 랭크 높은 순으로 매수
                            buy_candidates.sort(key=lambda x: x[1], reverse=True)
                            slots = max_positions - len(holdings)
                            for sym, rank_val in buy_candidates[:slots]:
                                if cash <= 0:
                                    break
                                # 이미 pending 매수가 있는 종목은 건너뜀
                                if use_limit_orders and any(o["symbol"] == sym and o["side"] == "BUY" for o in pending_orders):
                                    continue
                                buy_price = effective_buy_price(
                                    bar[sym]["close"], cost_config,
                                    order_qty=0, bar_volume=bar[sym].get("volume", 0),
                                )
                                if buy_price <= 0:
                                    continue
                                per_stock = cash / max(slots, 1)
                                qty = int(per_stock / buy_price)
                                if qty <= 0:
                                    continue
                                cost = buy_price * qty
                                if cost > cash:
                                    qty = int(cash / buy_price)
                                    if qty <= 0:
                                        continue
                                    cost = buy_price * qty

                                if use_limit_orders:
                                    # 지정가 매수: pending에 추가 + 현금 예약
                                    cash -= cost
                                    pending_orders.append({
                                        "symbol": sym,
                                        "side": "BUY",
                                        "price": bar[sym]["close"],
                                        "qty": qty,
                                        "ttl_bars": limit_ttl_bars,
                                        "elapsed_bars": 0,
                                        "reserved_cash": cost,
                                        "entry_date": bar_dt,
                                        "entry_info": {
                                            "entry_factor_rank": rank_val,
                                            "entry_factor_value": None,
                                            "entry_rank_pos": None,
                                            "entry_total_candidates": len(watchlist),
                                            "entry_target_count": len(watchlist),
                                        },
                                    })
                                else:
                                    cash -= cost
                                    holdings[sym] = {
                                        "qty": qty,
                                        "avg_price": buy_price,
                                        "entry_date": bar_dt,
                                        "last_close": buy_price,
                                        "high_price": buy_price,
                                        "low_price": buy_price,
                                        "entry_factor_rank": rank_val,
                                        "entry_factor_value": None,
                                        "entry_rank_pos": None,
                                        "entry_total_candidates": len(watchlist),
                                        "entry_target_count": len(watchlist),
                                    }
                                    total_buys += 1
                                slots -= 1

                    # 분봉 처리 완료 → 일봉 폴백 불필요
                    # 다음 날짜로 진행
                else:
                    # ranked_intraday가 빈 경우 → 일봉 폴백
                    use_intraday = False
            else:
                # 분봉 데이터 없음 → 일봉 폴백
                use_intraday = False

        if not use_intraday:
            # ── 일봉 pending order 체결 처리 ──
            if use_limit_orders and pending_orders:
                _dt_str_daily = trade_date.isoformat()
                _sb_d = len([o for o in pending_orders if o["side"] == "SELL"])
                _bb_d = len([o for o in pending_orders if o["side"] == "BUY"])
                cash, pending_orders = _process_pending_orders_bt(
                    pending_orders=pending_orders,
                    bar_data=today_daily,
                    holdings=holdings,
                    trades=trades,
                    cash=cash,
                    cost_config=cost_config,
                    dt_str=_dt_str_daily,
                    current_date=trade_date,
                    intraday=False,
                    strict_fill=strict_fill,
                    get_stock_name_fn=get_stock_name,
                    limit_stats=limit_stats,
                )
                total_buys += _bb_d - len([o for o in pending_orders if o["side"] == "BUY"])
                total_sells += _sb_d - len([o for o in pending_orders if o["side"] == "SELL"])

            # ── 일봉 폴백: 포지션 손절/트레일링 스탑 (매일 체크) ──
            for sym in list(holdings.keys()):
                pos = holdings[sym]
                if sym not in today_daily:
                    continue

                price = today_daily[sym]["close"]
                pos["high_price"] = max(pos.get("high_price", price), price)
                pos["low_price"] = min(pos.get("low_price", price), price)
                pos["last_close"] = price

                # 손절 체크
                if stop_loss_pct > 0:
                    drawdown = (price - pos["avg_price"]) / pos["avg_price"]
                    if drawdown <= -stop_loss_pct:
                        sell_price = effective_sell_price(price, cost_config, order_qty=pos["qty"])
                        pnl = (sell_price - pos["avg_price"]) * pos["qty"]
                        pnl_pct = (sell_price / pos["avg_price"] - 1) * 100
                        cash += sell_price * pos["qty"]
                        trades.append(Trade(
                            symbol=sym,
                            name=get_stock_name(sym),
                            entry_date=_dt_to_str(pos["entry_date"]),
                            entry_price=pos["avg_price"],
                            exit_date=trade_date.isoformat(),
                            exit_price=sell_price,
                            qty=pos["qty"],
                            pnl=pnl,
                            pnl_pct=pnl_pct,
                            holding_days=_calc_holding_days(trade_date, pos["entry_date"], False),
                            scale_step="STOP-LOSS",
                            exit_reason="손절",
                            entry_reason=_make_entry_reason(pos),
                            exit_reason_detail=_make_exit_reason_detail(
                                f"포지션 손절: 진입가 대비 {drawdown*100:.1f}% 하락 (기준: -{stop_loss_pct*100:.0f}%) [일봉 폴백]",
                                pos,
                            ),
                            entry_snapshot=_make_entry_snapshot(pos),
                            exit_snapshot={},
                            **_extract_limit_info(pos),
                        ))
                        total_sells += 1
                        stop_loss_count += 1
                        del holdings[sym]
                        continue

                # 트레일링 스탑 체크 (손절과 독립)
                if sym in holdings and trailing_stop_pct > 0:
                    high_price = pos.get("high_price", pos["avg_price"])
                    trail_dd = (price - high_price) / high_price
                    if trail_dd <= -trailing_stop_pct:
                        sell_price = effective_sell_price(price, cost_config, order_qty=pos["qty"])
                        pnl = (sell_price - pos["avg_price"]) * pos["qty"]
                        pnl_pct = (sell_price / pos["avg_price"] - 1) * 100
                        cash += sell_price * pos["qty"]
                        trades.append(Trade(
                            symbol=sym,
                            name=get_stock_name(sym),
                            entry_date=_dt_to_str(pos["entry_date"]),
                            entry_price=pos["avg_price"],
                            exit_date=trade_date.isoformat(),
                            exit_price=sell_price,
                            qty=pos["qty"],
                            pnl=pnl,
                            pnl_pct=pnl_pct,
                            holding_days=_calc_holding_days(trade_date, pos["entry_date"], False),
                            scale_step="S-TRAIL",
                            exit_reason="트레일링 스탑",
                            entry_reason=_make_entry_reason(pos),
                            exit_reason_detail=_make_exit_reason_detail(
                                f"트레일링 스탑: 고점({high_price:,.0f}) 대비 {trail_dd*100:.1f}% 하락 [일봉 폴백]",
                                pos,
                            ),
                            entry_snapshot=_make_entry_snapshot(pos),
                            exit_snapshot={},
                            **_extract_limit_info(pos),
                        ))
                        total_sells += 1
                        trailing_stop_count += 1
                        del holdings[sym]
                        continue

            # ── 일봉 리밸런싱 폴백 (주간 리밸런싱) ──
            # 매주 첫 거래일에만 리밸런싱 (간략화)
            is_weekly_rebal = False
            if prev_day_data and len(prev_day_data) >= 3:
                if day_idx == 0:
                    is_weekly_rebal = True
                else:
                    prev_trade_date = all_daily_dates[day_idx - 1] if day_idx > 0 else None
                    if prev_trade_date:
                        prev_iso = prev_trade_date.isocalendar()
                        cur_iso = trade_date.isocalendar()
                        if (prev_iso[0], prev_iso[1]) != (cur_iso[0], cur_iso[1]):
                            is_weekly_rebal = True

            if is_weekly_rebal and watchlist and prev_day_data:
                # 매도: 보유 중 watchlist 아닌 종목
                for sym in list(holdings.keys()):
                    if sym in watchlist:
                        continue
                    if sym not in today_daily:
                        continue
                    if use_limit_orders:
                        # 지정가 매도: pending에 추가
                        if not any(o["symbol"] == sym and o["side"] == "SELL" for o in pending_orders):
                            pending_orders.append({
                                "symbol": sym,
                                "side": "SELL",
                                "price": today_daily[sym]["open"],
                                "qty": holdings[sym]["qty"],
                                "ttl_bars": limit_ttl_bars,
                                "elapsed_bars": 0,
                                "entry_date": holdings[sym].get("entry_date", trade_date),
                            })
                    else:
                        pos = holdings.pop(sym)
                        sell_price = effective_sell_price(today_daily[sym]["open"], cost_config, order_qty=pos["qty"])
                        pnl = (sell_price - pos["avg_price"]) * pos["qty"]
                        pnl_pct = (sell_price / pos["avg_price"] - 1) * 100 if pos["avg_price"] > 0 else 0
                        cash += sell_price * pos["qty"]
                        trades.append(Trade(
                            symbol=sym,
                            name=get_stock_name(sym),
                            entry_date=_dt_to_str(pos["entry_date"]),
                            entry_price=pos["avg_price"],
                            exit_date=trade_date.isoformat(),
                            exit_price=sell_price,
                            qty=pos["qty"],
                            pnl=pnl,
                            pnl_pct=pnl_pct,
                            holding_days=_calc_holding_days(trade_date, pos["entry_date"], False),
                            scale_step="REBAL-SELL",
                            exit_reason="일봉 리밸런싱 퇴출",
                            entry_reason=_make_entry_reason(pos),
                            exit_reason_detail=_make_exit_reason_detail(
                                "일봉 팩터 watchlist 이탈 (분봉 데이터 없음, 일봉 폴백)",
                                pos,
                            ),
                            entry_snapshot=_make_entry_snapshot(pos),
                            exit_snapshot={},
                            **_extract_limit_info(pos),
                        ))
                        total_sells += 1

                # 매수: watchlist 중 미보유
                buy_list = [
                    sym for sym in watchlist
                    if sym not in holdings and sym in today_daily
                ]
                buy_list.sort(
                    key=lambda s: prev_day_data.get(s, {}).get("factor_rank", 0),
                    reverse=True,
                )
                max_new = max(0, max_positions - len(holdings))
                buy_list = buy_list[:max_new]

                if buy_list and cash > 0:
                    per_stock = cash / max(len(buy_list), 1)
                    for sym in buy_list:
                        # 이미 pending 매수가 있는 종목은 건너뜀
                        if use_limit_orders and any(o["symbol"] == sym and o["side"] == "BUY" for o in pending_orders):
                            continue
                        buy_price = effective_buy_price(
                            today_daily[sym]["open"], cost_config, order_qty=0,
                        )
                        if buy_price <= 0:
                            continue
                        qty = int(per_stock / buy_price)
                        if qty <= 0:
                            continue
                        cost = buy_price * qty
                        if cost > cash:
                            qty = int(cash / buy_price)
                            if qty <= 0:
                                continue
                            cost = buy_price * qty

                        sym_rank = prev_day_data.get(sym, {}).get("factor_rank")
                        if use_limit_orders:
                            # 지정가 매수: pending에 추가 + 현금 예약
                            cash -= cost
                            pending_orders.append({
                                "symbol": sym,
                                "side": "BUY",
                                "price": today_daily[sym]["open"],
                                "qty": qty,
                                "ttl_bars": limit_ttl_bars,
                                "elapsed_bars": 0,
                                "reserved_cash": cost,
                                "entry_date": trade_date,
                                "entry_info": {
                                    "entry_factor_rank": sym_rank,
                                    "entry_factor_value": prev_day_data.get(sym, {}).get("factor_value"),
                                    "entry_rank_pos": None,
                                    "entry_total_candidates": len(prev_day_data),
                                    "entry_target_count": len(watchlist),
                                },
                            })
                        else:
                            cash -= cost
                            holdings[sym] = {
                                "qty": qty,
                                "avg_price": buy_price,
                                "entry_date": trade_date,
                                "last_close": buy_price,
                                "high_price": buy_price,
                                "low_price": buy_price,
                                "entry_factor_rank": sym_rank,
                                "entry_factor_value": prev_day_data.get(sym, {}).get("factor_value"),
                                "entry_rank_pos": None,
                                "entry_total_candidates": len(prev_day_data),
                                "entry_target_count": len(watchlist),
                            }
                            total_buys += 1

                daily_rebal_count += 1

        # ── 포지션 고가/저가 업데이트 (일봉 기준) ──
        for sym, pos in holdings.items():
            if sym in today_daily:
                price = today_daily[sym]["close"]
                pos["high_price"] = max(pos.get("high_price", price), price)
                pos["low_price"] = min(pos.get("low_price", price), price)
                pos["last_close"] = price

                # 일별 스냅샷 수집 (하루 1번)
                if collect_daily_snapshots:
                    _snap_key = (sym, _dt_to_str(pos["entry_date"]))
                    _pos_daily_snapshots.setdefault(_snap_key, []).append({
                        "date": trade_date,
                        "close": price,
                        "variables": _extract_factor_variables(today_daily.get(sym), daily_required_cols),
                    })

        # ── 포트폴리오 평가 ──
        portfolio_value = cash
        for sym, pos in holdings.items():
            if sym in today_daily:
                portfolio_value += today_daily[sym]["close"] * pos["qty"]
            else:
                portfolio_value += pos.get("last_close", pos["avg_price"]) * pos["qty"]
        # pending 매수의 예약 현금도 포트폴리오 가치에 포함
        for _pend in pending_orders:
            if _pend.get("side") == "BUY":
                portfolio_value += _pend.get("reserved_cash", 0)

        equity_curve.append({
            "date": trade_date.isoformat(),
            "equity": portfolio_value,
        })

        prev_day_data = today_daily

        # 진행률
        if progress_cb and (day_idx + 1) % 20 == 0:
            pct = 30 + int(60 * (day_idx + 1) / len(all_daily_dates))
            await progress_cb(pct, 100, f"시뮬레이션 {day_idx + 1}/{len(all_daily_dates)}")

    # ── 6. 미체결 주문 정리 + 잔여 포지션 강제 청산 ──
    # pending 매수 주문 취소 → 예약 현금 반환
    for order in pending_orders:
        if order["side"] == "BUY":
            cash += order.get("reserved_cash", 0)
    pending_orders.clear()

    last_date = all_daily_dates[-1] if all_daily_dates else start_date
    last_day = daily_date_data.get(last_date, {})

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
            holding_days=_calc_holding_days(last_date, pos["entry_date"], False),
            scale_step="FINAL",
            exit_reason="백테스트 종료 청산",
            entry_reason=_make_entry_reason(pos),
            exit_reason_detail=_make_exit_reason_detail("백테스트 기간 종료", pos),
            entry_snapshot=_make_entry_snapshot(pos),
            exit_snapshot={},
            **_extract_limit_info(pos),
        ))

    holdings.clear()

    if progress_cb:
        await progress_cb(95, 100, "성과 지표 산출 중...")

    # ── 7. 성과 지표 ──
    metrics = compute_metrics(
        trades, equity_curve, initial_capital,
        annualize=bars_per_year("1d"),
        intraday=False,
    )

    metrics["backtest_mode"] = "dual_factor"
    metrics["daily_expression"] = daily_expression_str
    metrics["intraday_expression"] = intraday_expression_str
    metrics["intraday_interval"] = intraday_interval
    metrics["intraday_entry_threshold"] = intraday_entry_threshold
    metrics["intraday_exit_threshold"] = intraday_exit_threshold
    metrics["top_pct"] = top_pct
    metrics["max_positions"] = max_positions
    metrics["stop_loss_pct"] = stop_loss_pct
    metrics["trailing_stop_pct"] = trailing_stop_pct
    metrics["symbols_count"] = len(loaded_symbols)
    metrics["total_buys"] = total_buys
    metrics["total_sells"] = total_sells
    metrics["daily_rebal_count"] = daily_rebal_count
    metrics["intraday_trade_days"] = intraday_trade_days
    metrics["stop_loss_count"] = stop_loss_count
    metrics["trailing_stop_count"] = trailing_stop_count
    metrics["buy_commission"] = cost_config.buy_commission
    metrics["sell_commission"] = cost_config.sell_commission
    metrics["slippage_pct"] = cost_config.slippage_pct

    # ── 지정가 매매 메트릭 ──
    if use_limit_orders:
        _lf = limit_stats.get("fill_count", 0)
        _lm = limit_stats.get("market_count", 0)
        _lt = limit_stats.get("total_wait", 0)
        metrics["limit_fill_rate"] = _lf / max(_lf + _lm, 1)
        metrics["limit_unfilled_count"] = _lm
        metrics["limit_avg_wait_bars"] = _lt / max(_lf + _lm, 1)
        metrics["use_limit_orders"] = True
        metrics["strict_fill"] = strict_fill
        metrics["limit_ttl_bars"] = limit_ttl_bars
    else:
        metrics["use_limit_orders"] = False

    if progress_cb:
        await progress_cb(100, 100, "완료")

    # ── 일별 스냅샷 후처리 (다운샘플링 + 매핑) ──
    daily_snapshots_raw: list[dict] | None = None
    if collect_daily_snapshots and _pos_daily_snapshots:
        daily_snapshots_raw = []
        for trade_idx, t in enumerate(trades):
            snap_key = (t.symbol, t.entry_date)
            snapshots = _pos_daily_snapshots.get(snap_key)
            if not snapshots:
                continue
            if len(snapshots) > 120:
                snapshots = snapshots[::10]
            elif len(snapshots) > 60:
                snapshots = snapshots[::5]
            for snap in snapshots:
                daily_snapshots_raw.append({
                    "backtest_run_id": None,
                    "trade_index": trade_idx,
                    "symbol": t.symbol,
                    "snapshot_date": snap["date"],
                    "close": snap.get("close"),
                    "variables": snap.get("variables", {}),
                })

    return BacktestResult(
        trades=trades,
        equity_curve=equity_curve,
        metrics=metrics,
        daily_snapshots_raw=daily_snapshots_raw,
    )


async def execute_dual_factor_backtest(
    run_id: uuid.UUID,
    daily_expression_str: str,
    intraday_expression_str: str,
    symbols: list[str],
    start_date: date,
    end_date: date,
    initial_capital: float = 100_000_000,
    top_pct: float = 0.2,
    max_positions: int = 20,
    intraday_interval: str = "5m",
    intraday_entry_threshold: float = 0.8,
    intraday_exit_threshold: float = 0.2,
    stop_loss_pct: float = 0.15,
    trailing_stop_pct: float = 0.20,
    cost_config: CostConfig | None = None,
    use_limit_orders: bool = True,
    strict_fill: bool = False,
    limit_ttl_bars: int = 2,
    collect_daily_snapshots: bool = False,
) -> None:
    """DB 래퍼: 듀얼 팩터 백테스트 결과를 BacktestRun에 저장한다."""
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
        async with async_session() as db:
            await db.execute(
                update(BacktestRun)
                .where(BacktestRun.id == run_id)
                .values(status="RUNNING", progress=0)
            )
            await db.commit()

        result = await run_dual_factor_backtest(
            daily_expression_str=daily_expression_str,
            intraday_expression_str=intraday_expression_str,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            top_pct=top_pct,
            max_positions=max_positions,
            intraday_interval=intraday_interval,
            intraday_entry_threshold=intraday_entry_threshold,
            intraday_exit_threshold=intraday_exit_threshold,
            stop_loss_pct=stop_loss_pct,
            trailing_stop_pct=trailing_stop_pct,
            cost_config=cost_config,
            progress_cb=progress_cb,
            use_limit_orders=use_limit_orders,
            strict_fill=strict_fill,
            limit_ttl_bars=limit_ttl_bars,
            collect_daily_snapshots=collect_daily_snapshots,
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
            logger.warning("Dual factor backtest %s failed: %s", run_id, result.metrics["error"])
            return

        trades_list = [asdict(t) for t in result.trades]

        async with async_session() as db:
            await db.execute(
                update(BacktestRun)
                .where(BacktestRun.id == run_id)
                .values(
                    status="COMPLETED",
                    progress=100,
                    metrics=_sanitize_for_json(result.metrics),
                    equity_curve=_sanitize_for_json(result.equity_curve),
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
            "Dual factor backtest %s completed: return=%.2f%%, trades=%d, intraday_days=%d",
            run_id,
            result.metrics.get("total_return", 0),
            result.metrics.get("total_trades", 0),
            result.metrics.get("intraday_trade_days", 0),
        )

        # ── daily_snapshots INSERT (별도 트랜잭션, 실패해도 백테스트 결과 보존) ──
        if collect_daily_snapshots and result.daily_snapshots_raw:
            try:
                from app.backtest.models import BacktestDailySnapshot

                async with async_session() as db2:
                    for item in result.daily_snapshots_raw:
                        item["backtest_run_id"] = run_id
                        db2.add(BacktestDailySnapshot(**item))
                    await db2.commit()
                logger.info("Daily snapshots saved: %d rows for dual run %s", len(result.daily_snapshots_raw), run_id)
            except Exception as e:
                logger.warning("Daily snapshots save failed (non-critical): %s", e)

    except Exception as e:
        logger.exception("Dual factor backtest %s failed", run_id)
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
