"""SymPy AST → Polars Expression 재귀 변환기.

PySR 또는 Claude가 생성한 SymPy 수식을
Polars DataFrame에 적용 가능한 Expression으로 변환한다.
"""

from __future__ import annotations

import hashlib

import sympy
import polars as pl


# ── 커스텀 SymPy 함수 ──

class Clip(sympy.Function):
    """clip(x, lo, hi) — x를 [lo, hi] 범위로 제한."""

    nargs = 3

from app.backtest.indicators import (
    add_atr,
    add_bb,
    add_ema,
    add_lag,
    add_macd,
    add_return_nd,
    add_rsi,
    add_sma,
    add_volume_ratio,
    add_price_change_pct,
)

# PySR 변수 → Polars 컬럼 매핑
VARIABLE_MAP: dict[str, str] = {
    "x0": "close",
    "x1": "open",
    "x2": "high",
    "x3": "low",
    "x4": "volume",
    "x5": "sma_20",
    "x6": "rsi",
    "x7": "volume_ratio",
    "x8": "atr_14",
    "x9": "macd_hist",
    "x10": "bb_upper",
    "x11": "bb_lower",
    "x12": "price_change_pct",
}

# 사람 읽기용 이름 → Polars 컬럼 (Claude-only 모드에서 사용)
NAMED_VARIABLE_MAP: dict[str, str] = {
    # OHLCV 기본
    "close": "close",
    "open": "open",
    "high": "high",
    "low": "low",
    "volume": "volume",
    # 기존 지표
    "sma_20": "sma_20",
    "sma": "sma_20",
    "rsi": "rsi",
    "volume_ratio": "volume_ratio",
    "vol_ratio": "volume_ratio",
    "atr": "atr_14",
    "atr_14": "atr_14",
    "macd_hist": "macd_hist",
    "macd": "macd_hist",
    "bb_upper": "bb_upper",
    "bb_lower": "bb_lower",
    "bb_width": "bb_width",
    "price_change_pct": "price_change_pct",
    "pct_change": "price_change_pct",
    "ema_20": "ema_20",
    # 멀티 윈도우 이동평균
    "sma_5": "sma_5",
    "sma_10": "sma_10",
    "sma_60": "sma_60",
    "ema_5": "ema_5",
    "ema_10": "ema_10",
    "ema_60": "ema_60",
    # 멀티 윈도우 RSI/ATR
    "rsi_7": "rsi_7",
    "rsi_21": "rsi_21",
    "atr_7": "atr_7",
    "atr_21": "atr_21",
    # 시차 피처
    "close_lag_1": "close_lag_1",
    "close_lag_5": "close_lag_5",
    "close_lag_20": "close_lag_20",
    "volume_lag_1": "volume_lag_1",
    "volume_lag_5": "volume_lag_5",
    # N일 수익률
    "return_5d": "return_5d",
    "return_20d": "return_20d",
    # 파생 피처
    "bb_position": "bb_position",
    # 횡단면 피처 (Cs_: Cross-sectional)
    "rank_close": "rank_close",
    "rank_volume": "rank_volume",
    "zscore_close": "zscore_close",
    "zscore_volume": "zscore_volume",
    # 명시적 횡단면 접두사 (별칭)
    "Cs_Rank_close": "rank_close",
    "Cs_Rank_volume": "rank_volume",
    "Cs_ZScore_close": "zscore_close",
    "Cs_ZScore_volume": "zscore_volume",
    # 시계열 피처 (Ts_: Time-series, 60일 롤링)
    "Ts_Rank_close": "ts_rank_close",
    "Ts_Rank_volume": "ts_rank_volume",
    "Ts_ZScore_close": "ts_zscore_close",
    "Ts_ZScore_volume": "ts_zscore_volume",
    "ts_rank_close": "ts_rank_close",
    "ts_rank_volume": "ts_rank_volume",
    "ts_zscore_close": "ts_zscore_close",
    "ts_zscore_volume": "ts_zscore_volume",
    # 투자자 수급 (거래량 대비 정규화)
    "foreign_net_buy": "foreign_net_norm",
    "foreign_net_norm": "foreign_net_norm",
    "inst_net_buy": "inst_net_norm",
    "inst_net_norm": "inst_net_norm",
    "retail_net_buy": "retail_net_norm",
    "retail_net_norm": "retail_net_norm",
    # 투자자 매수 강도 비율 (buy / (buy + sell))
    "foreign_buy_ratio": "foreign_buy_ratio",
    "inst_buy_ratio": "inst_buy_ratio",
    "retail_buy_ratio": "retail_buy_ratio",
    # DART 재무 피처
    "eps": "eps",
    "bps": "bps",
    "earnings_per_share": "eps",
    "book_value_per_share": "bps",
    "operating_margin": "operating_margin",
    "debt_to_equity": "debt_to_equity",
    "earnings_yield": "earnings_yield",
    "book_yield": "book_yield",
    # 뉴스 감성 피처 (enriched candles에서 제공)
    "sentiment_score": "sentiment_score",
    "article_count": "article_count",
    "event_score": "event_score",
    # 섹터 횡단면 피처
    "sector_return": "sector_return",
    "sector_rel_strength": "sector_rel_strength",
    "sector_rank": "sector_rank",
    # 신용잔고/공매도 피처 (enriched candles에서 제공)
    "margin_rate": "margin_rate",
    "short_balance_rate": "short_balance_rate",
    "short_volume_ratio": "short_volume_ratio",
    # 프로그램 매매 피처 (enriched candles에서 제공)
    "pgm_net_norm": "pgm_net_norm",
    "pgm_buy_ratio": "pgm_buy_ratio",
    # ── 이벤트 감지 피처 (OHLCV에서 자동 계산) ──
    # 거래량 이벤트
    "vol_spike_5d": "vol_spike_5d",
    "vol_spike_20d": "vol_spike_20d",
    "consec_low_vol_5": "consec_low_vol_5",
    "vol_dry_then_spike": "vol_dry_then_spike",
    # 가격 이벤트
    "consec_up": "consec_up",
    "consec_down": "consec_down",
    "gap_up_pct": "gap_up_pct",
    "gap_down_pct": "gap_down_pct",
    "range_breakout": "range_breakout",
    "range_breakdown": "range_breakdown",
    # 모멘텀 이벤트
    "rsi_oversold_bounce": "rsi_oversold_bounce",
    "macd_cross_up": "macd_cross_up",
    "bb_squeeze": "bb_squeeze",
    "bb_breakout_upper": "bb_breakout_upper",
    # 수급 이벤트 (enriched candles에서 제공)
    "foreign_accumulate_5d": "foreign_accumulate_5d",
    "inst_accumulate_5d": "inst_accumulate_5d",
}

# 모든 유효한 변수명 합집합
_ALL_VARIABLES = {**VARIABLE_MAP, **NAMED_VARIABLE_MAP}


# ── Feature Family Taxonomy (Niche 분류용) ──
_FAMILY_DEFINITIONS: dict[str, list[str]] = {
    "price": [
        "close", "open", "high", "low",
        "close_lag_1", "close_lag_5", "close_lag_20",
        "bb_upper", "bb_lower", "bb_middle", "bb_position", "bb_width",
        "sma_5", "sma_10", "sma_20", "sma_60",
        "ema_5", "ema_10", "ema_20", "ema_60",
        "price_change_pct",
        "gap_up_pct", "gap_down_pct",
        "range_breakout", "range_breakdown",
        "rank_close", "zscore_close", "ts_rank_close", "ts_zscore_close",
    ],
    "volume": [
        "volume", "volume_ratio",
        "volume_lag_1", "volume_lag_5",
        "rank_volume", "zscore_volume", "ts_rank_volume", "ts_zscore_volume",
        "vol_spike_5d", "vol_spike_20d", "consec_low_vol_5", "vol_dry_then_spike",
    ],
    "momentum": [
        "rsi", "rsi_7", "rsi_21",
        "macd_hist", "macd_line", "macd_signal",
        "macd_cross_up", "rsi_oversold_bounce",
        "return_5d", "return_20d",
        "consec_up", "consec_down",
    ],
    "volatility": [
        "atr_7", "atr_14", "atr_21",
        "bb_squeeze", "bb_breakout_upper",
    ],
    "supply": [
        "foreign_net_norm", "inst_net_norm", "retail_net_norm",
        "foreign_buy_ratio", "inst_buy_ratio", "retail_buy_ratio",
        "foreign_accumulate_5d", "inst_accumulate_5d",
    ],
    "fundamental": [
        "eps", "bps", "operating_margin", "debt_to_equity",
        "earnings_yield", "book_yield",
    ],
    "sentiment": [
        "sentiment_score", "article_count", "event_score",
    ],
    "market_micro": [
        "margin_rate", "short_balance_rate", "short_volume_ratio",
        "pgm_net_norm", "pgm_buy_ratio",
        "sector_return", "sector_rel_strength", "sector_rank",
    ],
}

FEATURE_FAMILY_MAP: dict[str, str] = {}
for _family, _cols in _FAMILY_DEFINITIONS.items():
    for _col in _cols:
        FEATURE_FAMILY_MAP[_col] = _family


def classify_niche(expr: "sympy.Basic") -> str:
    """수식의 dominant feature family(니치)를 반환.

    free_symbols를 패밀리로 분류하고, 가장 많이 등장한 패밀리를 반환한다.
    동점 시 알파벳 순서로 결정 (결정적).
    인식된 피처가 없으면 'unknown' 반환.
    """
    family_counts: dict[str, int] = {}
    for sym in expr.free_symbols:
        col_name = _ALL_VARIABLES.get(str(sym), str(sym))
        family = FEATURE_FAMILY_MAP.get(col_name, "unknown")
        family_counts[family] = family_counts.get(family, 0) + 1

    if not family_counts:
        return "unknown"

    return max(
        family_counts.keys(),
        key=lambda f: (family_counts[f], f),  # count 우선, 동점 시 이름순
    )


class ASTConversionError(Exception):
    """SymPy → Polars 변환 실패."""


def _resolve_column(name: str) -> str:
    """변수명을 Polars 컬럼명으로 해석."""
    col = _ALL_VARIABLES.get(name)
    if col is None:
        raise ASTConversionError(f"Unknown variable: {name}")
    return col


def sympy_to_polars(expr: sympy.Basic) -> pl.Expr:
    """SymPy 표현식을 Polars Expression으로 재귀 변환."""

    # Symbol → pl.col
    if isinstance(expr, sympy.Symbol):
        return pl.col(_resolve_column(str(expr)))

    # 정수/실수 → pl.lit
    if isinstance(expr, (sympy.Integer, sympy.Float, sympy.Rational)):
        return pl.lit(float(expr))

    # python int/float (sympify 결과에서 발생 가능)
    if isinstance(expr, (int, float)):
        return pl.lit(float(expr))

    # 덧셈: Add(a, b, c, ...)
    if isinstance(expr, sympy.Add):
        args = [sympy_to_polars(a) for a in expr.args]
        result = args[0]
        for a in args[1:]:
            result = result + a
        return result

    # 곱셈: Mul(a, b, c, ...)
    if isinstance(expr, sympy.Mul):
        args = [sympy_to_polars(a) for a in expr.args]
        result = args[0]
        for a in args[1:]:
            result = result * a
        return result

    # 거듭제곱: Pow(base, exp)
    if isinstance(expr, sympy.Pow):
        base = sympy_to_polars(expr.args[0])
        exp_val = expr.args[1]

        # sqrt: Pow(x, 1/2)
        if exp_val == sympy.Rational(1, 2) or exp_val == sympy.Float(0.5):
            return base.sqrt()

        # 정수 지수
        if isinstance(exp_val, (sympy.Integer, sympy.Float, sympy.Rational)):
            return base.pow(float(exp_val))

        # 일반 지수 (변수)
        exp_expr = sympy_to_polars(exp_val)
        return base.pow(exp_expr)

    # log
    if isinstance(expr, sympy.log):
        return sympy_to_polars(expr.args[0]).log()

    # exp
    if isinstance(expr, sympy.exp):
        return sympy_to_polars(expr.args[0]).exp()

    # abs
    if isinstance(expr, sympy.Abs):
        return sympy_to_polars(expr.args[0]).abs()

    # sign(x) → 방향 추출: +1 / -1 / 0
    if isinstance(expr, sympy.sign):
        inner = sympy_to_polars(expr.args[0])
        return (
            pl.when(inner > 0).then(pl.lit(1.0))
            .when(inner < 0).then(pl.lit(-1.0))
            .otherwise(pl.lit(0.0))
        )

    # Heaviside(x) → step 함수: x>0이면 1, 아니면 0
    if isinstance(expr, sympy.Heaviside):
        inner = sympy_to_polars(expr.args[0])
        return pl.when(inner > 0).then(pl.lit(1.0)).otherwise(pl.lit(0.0))

    # Max(x, y, ...)
    if isinstance(expr, sympy.Max):
        args = [sympy_to_polars(a) for a in expr.args]
        return pl.max_horizontal(*args)

    # Min(x, y, ...)
    if isinstance(expr, sympy.Min):
        args = [sympy_to_polars(a) for a in expr.args]
        return pl.min_horizontal(*args)

    # clip(x, lo, hi) → x를 [lo, hi]로 제한
    if isinstance(expr, Clip):
        x = sympy_to_polars(expr.args[0])
        lo = sympy_to_polars(expr.args[1])
        hi = sympy_to_polars(expr.args[2])
        return pl.max_horizontal(lo, pl.min_horizontal(hi, x))

    # NegativeOne: -1 (Mul(-1, x)로 처리됨)
    if expr == sympy.S.NegativeOne:
        return pl.lit(-1.0)

    # One, Zero 등 상수
    if expr.is_number:
        return pl.lit(float(expr))

    raise ASTConversionError(
        f"Unsupported SymPy node: {type(expr).__name__} ({expr})"
    )


def sympy_to_code_string(expr: sympy.Basic) -> str:
    """SymPy 표현식을 Polars 코드 문자열로 변환 (DB 저장용)."""

    if isinstance(expr, sympy.Symbol):
        return f'pl.col("{_resolve_column(str(expr))}")'

    if isinstance(expr, (sympy.Integer, sympy.Float, sympy.Rational)):
        return f"pl.lit({float(expr)})"

    if isinstance(expr, (int, float)):
        return f"pl.lit({float(expr)})"

    if isinstance(expr, sympy.Add):
        parts = [sympy_to_code_string(a) for a in expr.args]
        return " + ".join(f"({p})" for p in parts)

    if isinstance(expr, sympy.Mul):
        parts = [sympy_to_code_string(a) for a in expr.args]
        return " * ".join(f"({p})" for p in parts)

    if isinstance(expr, sympy.Pow):
        base = sympy_to_code_string(expr.args[0])
        exp_val = expr.args[1]
        if exp_val == sympy.Rational(1, 2) or exp_val == sympy.Float(0.5):
            return f"({base}).sqrt()"
        return f"({base}).pow({float(exp_val)})"

    if isinstance(expr, sympy.log):
        return f"({sympy_to_code_string(expr.args[0])}).log()"

    if isinstance(expr, sympy.exp):
        return f"({sympy_to_code_string(expr.args[0])}).exp()"

    if isinstance(expr, sympy.Abs):
        return f"({sympy_to_code_string(expr.args[0])}).abs()"

    if isinstance(expr, sympy.sign):
        inner = sympy_to_code_string(expr.args[0])
        return f"pl.when(({inner}) > 0).then(pl.lit(1.0)).when(({inner}) < 0).then(pl.lit(-1.0)).otherwise(pl.lit(0.0))"

    if isinstance(expr, sympy.Heaviside):
        inner = sympy_to_code_string(expr.args[0])
        return f"pl.when(({inner}) > 0).then(pl.lit(1.0)).otherwise(pl.lit(0.0))"

    if isinstance(expr, sympy.Max):
        parts = [sympy_to_code_string(a) for a in expr.args]
        return f"pl.max_horizontal({', '.join(parts)})"

    if isinstance(expr, sympy.Min):
        parts = [sympy_to_code_string(a) for a in expr.args]
        return f"pl.min_horizontal({', '.join(parts)})"

    if isinstance(expr, Clip):
        x = sympy_to_code_string(expr.args[0])
        lo = sympy_to_code_string(expr.args[1])
        hi = sympy_to_code_string(expr.args[2])
        return f"pl.max_horizontal({lo}, pl.min_horizontal({hi}, {x}))"

    if expr.is_number:
        return f"pl.lit({float(expr)})"

    raise ASTConversionError(
        f"Cannot convert to code string: {type(expr).__name__} ({expr})"
    )


# 기저 지표가 이미 추가되었는지 확인하기 위한 컬럼명 집합
_ALPHA_FEATURE_COLUMNS = {
    # 기존
    "sma_20", "rsi", "volume_ratio", "atr_14",
    "macd_hist", "macd_line", "macd_signal",
    "bb_upper", "bb_lower", "bb_middle",
    "price_change_pct", "ema_20",
    # 멀티 윈도우
    "sma_5", "sma_10", "sma_60",
    "ema_5", "ema_10", "ema_60",
    "rsi_7", "rsi_21",
    "atr_7", "atr_21",
    # 시차
    "close_lag_1", "close_lag_5", "close_lag_20",
    "volume_lag_1", "volume_lag_5",
    "return_5d", "return_20d",
    # 파생
    "bb_position",
    # 횡단면
    "rank_close", "rank_volume",
    "zscore_close", "zscore_volume",
    # 투자자 수급 (enriched candles에서 제공)
    "foreign_net_norm", "inst_net_norm", "retail_net_norm",
    "foreign_buy_ratio", "inst_buy_ratio", "retail_buy_ratio",
    # DART 재무 (enriched candles에서 제공)
    "eps", "bps", "operating_margin", "debt_to_equity",
    "earnings_yield", "book_yield",
    # 뉴스 감성 (enriched candles에서 제공)
    "sentiment_score", "article_count", "event_score",
    # 섹터 횡단면
    "sector_return", "sector_rel_strength", "sector_rank",
    # 신용잔고/공매도 (enriched candles에서 제공)
    "margin_rate", "short_balance_rate", "short_volume_ratio",
    # 프로그램 매매 (enriched candles에서 제공)
    "pgm_net_norm", "pgm_buy_ratio",
    # 이벤트 감지 피처
    "vol_spike_5d", "vol_spike_20d", "consec_low_vol_5", "vol_dry_then_spike",
    "consec_up", "consec_down", "gap_up_pct", "gap_down_pct",
    "range_breakout", "range_breakdown",
    "rsi_oversold_bounce", "macd_cross_up", "bb_squeeze", "bb_breakout_upper",
    "foreign_accumulate_5d", "inst_accumulate_5d",
}


def ensure_alpha_features(
    df: pl.DataFrame,
    required_cols: set[str] | None = None,
) -> pl.DataFrame:
    """PySR 변수에 필요한 기저 지표를 DataFrame에 추가.

    이미 존재하는 컬럼은 건너뛴다.

    Parameters
    ----------
    required_cols : 필요한 컬럼명 집합. None이면 전부 계산 (기존 동작).
                    지정하면 해당 컬럼과 그 의존성만 계산 (고속 모드).
    """
    existing = set(df.columns)

    # ── 전이적 의존성 확장 ──
    # sector 피처는 price_change_pct → sector_return → sector_rel_strength 체인 필요
    if required_cols is not None:
        expanded = set(required_cols)
        _SECTOR_COLS = {"sector_return", "sector_rel_strength", "sector_rank"}
        if expanded & _SECTOR_COLS:
            expanded.add("price_change_pct")
        if "sector_rel_strength" in expanded:
            expanded.add("sector_return")
        required_cols = expanded

    def _need(col: str) -> bool:
        """이 컬럼을 계산해야 하는지 판단."""
        if col in existing:
            return False
        if required_cols is None:
            return True
        return col in required_cols

    # bb_width가 필요하면 bb_upper/bb_lower도 필요
    _bb_needed = _need("bb_upper") or _need("bb_lower") or _need("bb_width") or _need("bb_position")

    # ── 기존 기본 지표 ──
    if _need("sma_20"):
        df = add_sma(df, period=20)

    if _need("ema_20"):
        df = add_ema(df, period=20)

    if _need("rsi"):
        df = add_rsi(df, period=14)

    if _need("volume_ratio"):
        df = add_volume_ratio(df, period=20)

    if _need("atr_14"):
        df = add_atr(df, period=14)

    if _need("macd_hist") or _need("macd_line") or _need("macd_signal"):
        if "macd_hist" not in existing:
            df = add_macd(df, fast=12, slow=26, signal=9)

    if _bb_needed:
        if "bb_upper" not in existing or "bb_lower" not in existing:
            df = add_bb(df, period=20, std=2.0)

    if _need("price_change_pct"):
        df = add_price_change_pct(df, period=1)

    # bb_width: bb_upper - bb_lower (편의용)
    if _need("bb_width") and "bb_width" not in existing and "bb_upper" in df.columns:
        df = df.with_columns(
            (pl.col("bb_upper") - pl.col("bb_lower")).alias("bb_width")
        )

    # ── 멀티 윈도우 이동평균 ──
    for period in [5, 10, 60]:
        if _need(f"sma_{period}"):
            df = add_sma(df, period=period)
        if _need(f"ema_{period}"):
            df = add_ema(df, period=period)

    # ── 멀티 윈도우 RSI (inline — add_rsi는 항상 "rsi" alias) ──
    for period in [7, 21]:
        col_name = f"rsi_{period}"
        if _need(col_name):
            delta = pl.col("close").diff()
            gain = delta.clip(lower_bound=0.0)
            loss = (-delta).clip(lower_bound=0.0)
            avg_gain = gain.ewm_mean(
                alpha=1.0 / period, adjust=False, min_periods=period
            )
            avg_loss = loss.ewm_mean(
                alpha=1.0 / period, adjust=False, min_periods=period
            )
            rs = avg_gain / avg_loss
            rsi_expr = pl.lit(100.0) - (pl.lit(100.0) / (pl.lit(1.0) + rs))
            df = df.with_columns(rsi_expr.alias(col_name))

    # ── 멀티 윈도우 ATR ──
    for period in [7, 21]:
        col_name = f"atr_{period}"
        if _need(col_name):
            prev_close = pl.col("close").shift(1)
            tr = pl.max_horizontal(
                pl.col("high") - pl.col("low"),
                (pl.col("high") - prev_close).abs(),
                (pl.col("low") - prev_close).abs(),
            )
            atr_expr = tr.ewm_mean(span=period, adjust=False)
            df = df.with_columns(atr_expr.alias(col_name))

    # ── 시차 피처 ──
    for lag in [1, 5, 20]:
        if _need(f"close_lag_{lag}"):
            df = add_lag(df, "close", lag)
    for lag in [1, 5]:
        if _need(f"volume_lag_{lag}"):
            df = add_lag(df, "volume", lag)

    # ── N일 수익률 ──
    for period in [5, 20]:
        if _need(f"return_{period}d"):
            df = add_return_nd(df, period)

    # ── 파생 피처: bb_position ──
    if _need("bb_position") and "bb_upper" in df.columns:
        df = df.with_columns(
            (
                (pl.col("close") - pl.col("bb_lower"))
                / (pl.col("bb_upper") - pl.col("bb_lower")).clip(
                    lower_bound=1e-10
                )
            ).alias("bb_position")
        )

    # ── 횡단면 피처 (symbol 컬럼 있을 때만) ──
    _CS_ROLLING = 60
    _CS_MIN_PERIODS = 10
    if "symbol" in df.columns:
        n_symbols = df.select("symbol").n_unique()
        if n_symbols > 1:
            # 다종목: 날짜 기준 횡단면 rank/z-score
            for col_name in ["close", "volume"]:
                rank_alias = f"rank_{col_name}"
                zscore_alias = f"zscore_{col_name}"
                if _need(rank_alias):
                    df = df.with_columns(
                        pl.col(col_name)
                        .rank(method="average")
                        .over("dt")
                        .truediv(pl.col(col_name).count().over("dt"))
                        .alias(rank_alias)
                    )
                if _need(zscore_alias):
                    df = df.with_columns(
                        (
                            (
                                pl.col(col_name)
                                - pl.col(col_name).mean().over("dt")
                            )
                            / pl.col(col_name)
                            .std()
                            .over("dt")
                            .clip(lower_bound=1e-10)
                        ).alias(zscore_alias)
                    )
        else:
            # 단일 종목 fallback: 시계열 롤링 rank/z-score
            for col_name in ["close", "volume"]:
                rank_alias = f"rank_{col_name}"
                zscore_alias = f"zscore_{col_name}"
                if _need(rank_alias):
                    _min = pl.col(col_name).rolling_min(
                        window_size=_CS_ROLLING, min_periods=_CS_MIN_PERIODS,
                    )
                    _max = pl.col(col_name).rolling_max(
                        window_size=_CS_ROLLING, min_periods=_CS_MIN_PERIODS,
                    )
                    _range = (_max - _min).clip(lower_bound=1e-10)
                    df = df.with_columns(
                        ((pl.col(col_name) - _min) / _range)
                        .clip(0.0, 1.0)
                        .fill_null(0.5)
                        .alias(rank_alias)
                    )
                if _need(zscore_alias):
                    df = df.with_columns(
                        (
                            (
                                pl.col(col_name)
                                - pl.col(col_name).rolling_mean(
                                    window_size=_CS_ROLLING,
                                    min_periods=_CS_MIN_PERIODS,
                                )
                            )
                            / pl.col(col_name)
                            .rolling_std(
                                window_size=_CS_ROLLING,
                                min_periods=_CS_MIN_PERIODS,
                            )
                            .clip(lower_bound=1e-10)
                        )
                        .fill_null(0.0)
                        .alias(zscore_alias)
                    )

    # ── 시계열 피처 (Ts_: Time-series) ──
    existing = set(df.columns)  # 갱신
    _TS_ROLLING = 60
    _TS_MIN_PERIODS = 10
    for col_name in ["close", "volume"]:
        ts_rank_alias = f"ts_rank_{col_name}"
        ts_zscore_alias = f"ts_zscore_{col_name}"
        if _need(ts_rank_alias):
            _min = pl.col(col_name).rolling_min(
                window_size=_TS_ROLLING, min_periods=_TS_MIN_PERIODS,
            )
            _max = pl.col(col_name).rolling_max(
                window_size=_TS_ROLLING, min_periods=_TS_MIN_PERIODS,
            )
            _range = (_max - _min).clip(lower_bound=1e-10)
            df = df.with_columns(
                ((pl.col(col_name) - _min) / _range)
                .clip(0.0, 1.0)
                .fill_null(0.5)
                .alias(ts_rank_alias)
            )
        if _need(ts_zscore_alias):
            df = df.with_columns(
                (
                    (
                        pl.col(col_name)
                        - pl.col(col_name).rolling_mean(
                            window_size=_TS_ROLLING,
                            min_periods=_TS_MIN_PERIODS,
                        )
                    )
                    / pl.col(col_name)
                    .rolling_std(
                        window_size=_TS_ROLLING,
                        min_periods=_TS_MIN_PERIODS,
                    )
                    .clip(lower_bound=1e-10)
                )
                .fill_null(0.0)
                .alias(ts_zscore_alias)
            )

    # ── 투자자 수급 정규화 (enriched candles에서 raw 컬럼이 존재할 때만) ──
    existing = set(df.columns)  # 갱신
    if "foreign_net" in existing:
        for col_name, src in [
            ("foreign_net_norm", "foreign_net"),
            ("inst_net_norm", "inst_net"),
            ("retail_net_norm", "retail_net"),
        ]:
            if _need(col_name) and src in existing:
                df = df.with_columns(
                    (pl.col(src).cast(pl.Float64) / pl.col("volume").cast(pl.Float64).clip(lower_bound=1.0))
                    .alias(col_name)
                )

    # ── 투자자 매수 강도 비율 (buy / (buy + sell)) ──
    existing = set(df.columns)
    for prefix in ["foreign", "inst", "retail"]:
        buy_col = f"{prefix}_buy_vol"
        sell_col = f"{prefix}_sell_vol"
        ratio_col = f"{prefix}_buy_ratio"
        if _need(ratio_col) and buy_col in existing and sell_col in existing:
            df = df.with_columns(
                (
                    pl.col(buy_col).cast(pl.Float64)
                    / (pl.col(buy_col) + pl.col(sell_col)).cast(pl.Float64).clip(lower_bound=1.0)
                ).alias(ratio_col)
            )

    # ── DART 재무 파생 피처 (enriched candles에서 raw 컬럼이 존재할 때만) ──
    existing = set(df.columns)
    if _need("earnings_yield") and "eps" in existing:
        df = df.with_columns(
            (pl.col("eps") / pl.col("close").clip(lower_bound=1.0)).alias("earnings_yield")
        )
    if _need("book_yield") and "bps" in existing:
        df = df.with_columns(
            (pl.col("bps") / pl.col("close").clip(lower_bound=1.0)).alias("book_yield")
        )

    # ── 섹터 횡단면 모멘텀 (sector_id + price_change_pct 필요) ──
    existing = set(df.columns)
    _sector_needed = _need("sector_return") or _need("sector_rel_strength") or _need("sector_rank")
    if _sector_needed and "sector_id" in existing and "price_change_pct" in existing:
        if _need("sector_return"):
            df = df.with_columns(
                pl.col("price_change_pct").mean().over(["dt", "sector_id"]).alias("sector_return")
            )
        if _need("sector_rel_strength"):
            sr = "sector_return" if "sector_return" in df.columns else None
            if sr:
                df = df.with_columns(
                    (pl.col("price_change_pct") - pl.col("sector_return")).alias("sector_rel_strength")
                )
        if _need("sector_rank"):
            df = df.with_columns(
                pl.col("price_change_pct").rank().over(["dt", "sector_id"])
                .truediv(
                    pl.col("price_change_pct").count().over(["dt", "sector_id"]).clip(lower_bound=1)
                )
                .alias("sector_rank")
            )

    # ── 신용/공매도 파생 피처 ──
    existing = set(df.columns)
    if _need("short_volume_ratio") and "short_volume" in existing:
        df = df.with_columns(
            (pl.col("short_volume").cast(pl.Float64) / pl.col("volume").cast(pl.Float64).clip(lower_bound=1.0))
            .alias("short_volume_ratio")
        )

    # ── 프로그램 매매 파생 피처 ──
    existing = set(df.columns)
    if "pgm_net_qty" in existing:
        if _need("pgm_net_norm"):
            df = df.with_columns(
                (pl.col("pgm_net_qty").cast(pl.Float64) / pl.col("volume").cast(pl.Float64).clip(lower_bound=1.0))
                .alias("pgm_net_norm")
            )
        if _need("pgm_buy_ratio") and "pgm_buy_qty" in existing:
            df = df.with_columns(
                (
                    pl.col("pgm_buy_qty").cast(pl.Float64)
                    / (pl.col("pgm_buy_qty") + pl.col("pgm_sell_qty")).cast(pl.Float64).clip(lower_bound=1.0)
                ).alias("pgm_buy_ratio")
            )

    # ══════════════════════════════════════════════════════
    # ── 이벤트 감지 피처 (OHLCV 기반, 자동 계산) ──
    # ══════════════════════════════════════════════════════
    existing = set(df.columns)

    # ── 거래량 이벤트 ──
    if _need("vol_spike_5d"):
        df = df.with_columns(
            (
                pl.col("volume").cast(pl.Float64)
                / pl.col("volume").cast(pl.Float64)
                .rolling_mean(window_size=5, min_periods=1)
                .clip(lower_bound=1.0)
            ).alias("vol_spike_5d")
        )

    if _need("vol_spike_20d"):
        df = df.with_columns(
            (
                pl.col("volume").cast(pl.Float64)
                / pl.col("volume").cast(pl.Float64)
                .rolling_mean(window_size=20, min_periods=1)
                .clip(lower_bound=1.0)
            ).alias("vol_spike_20d")
        )

    if _need("consec_low_vol_5"):
        # 5일 윈도우 내 거래량이 20일 평균 미만인 일수
        df = df.with_columns(
            (pl.col("volume") < pl.col("volume").cast(pl.Float64).rolling_mean(window_size=20, min_periods=5))
            .cast(pl.Int8)
            .rolling_sum(window_size=5, min_periods=1)
            .alias("consec_low_vol_5")
        )

    if _need("vol_dry_then_spike"):
        # 의존 컬럼 보장
        if "consec_low_vol_5" not in df.columns:
            df = df.with_columns(
                (pl.col("volume") < pl.col("volume").cast(pl.Float64).rolling_mean(window_size=20, min_periods=5))
                .cast(pl.Int8)
                .rolling_sum(window_size=5, min_periods=1)
                .alias("consec_low_vol_5")
            )
        if "vol_spike_5d" not in df.columns:
            df = df.with_columns(
                (
                    pl.col("volume").cast(pl.Float64)
                    / pl.col("volume").cast(pl.Float64)
                    .rolling_mean(window_size=5, min_periods=1)
                    .clip(lower_bound=1.0)
                ).alias("vol_spike_5d")
            )
        df = df.with_columns(
            ((pl.col("consec_low_vol_5") >= 3) & (pl.col("vol_spike_5d") >= 3.0))
            .cast(pl.Int8)
            .alias("vol_dry_then_spike")
        )

    # ── 가격 이벤트 ──
    if _need("consec_up"):
        # 5일 윈도우 내 상승일 수
        df = df.with_columns(
            (pl.col("close") > pl.col("close").shift(1))
            .cast(pl.Int8)
            .rolling_sum(window_size=5, min_periods=1)
            .alias("consec_up")
        )

    if _need("consec_down"):
        # 5일 윈도우 내 하락일 수
        df = df.with_columns(
            (pl.col("close") < pl.col("close").shift(1))
            .cast(pl.Int8)
            .rolling_sum(window_size=5, min_periods=1)
            .alias("consec_down")
        )

    if _need("gap_up_pct"):
        prev_close = pl.col("close").shift(1)
        df = df.with_columns(
            ((pl.col("open") - prev_close) / prev_close.clip(lower_bound=1e-10) * 100)
            .clip(lower_bound=0.0)
            .alias("gap_up_pct")
        )

    if _need("gap_down_pct"):
        prev_close = pl.col("close").shift(1)
        df = df.with_columns(
            ((prev_close - pl.col("open")) / prev_close.clip(lower_bound=1e-10) * 100)
            .clip(lower_bound=0.0)
            .alias("gap_down_pct")
        )

    if _need("range_breakout"):
        df = df.with_columns(
            (pl.col("close") > pl.col("high").rolling_max(window_size=20, min_periods=5).shift(1))
            .cast(pl.Int8)
            .alias("range_breakout")
        )

    if _need("range_breakdown"):
        df = df.with_columns(
            (pl.col("close") < pl.col("low").rolling_min(window_size=20, min_periods=5).shift(1))
            .cast(pl.Int8)
            .alias("range_breakdown")
        )

    # ── 모멘텀 이벤트 (기술적 지표 의존 — 위에서 이미 계산됨) ──
    existing = set(df.columns)

    if _need("rsi_oversold_bounce"):
        if "rsi" not in existing:
            df = add_rsi(df, period=14)
        df = df.with_columns(
            ((pl.col("rsi").shift(1) < 30.0) & (pl.col("rsi") >= 30.0))
            .cast(pl.Int8)
            .alias("rsi_oversold_bounce")
        )

    if _need("macd_cross_up"):
        if "macd_hist" not in existing:
            df = add_macd(df, fast=12, slow=26, signal=9)
        df = df.with_columns(
            ((pl.col("macd_hist").shift(1) <= 0.0) & (pl.col("macd_hist") > 0.0))
            .cast(pl.Int8)
            .alias("macd_cross_up")
        )

    if _need("bb_squeeze"):
        if "bb_upper" not in existing or "bb_lower" not in existing:
            df = add_bb(df, period=20, std=2.0)
        bb_w = pl.col("bb_upper") - pl.col("bb_lower")
        df = df.with_columns(
            (bb_w <= bb_w.rolling_min(window_size=20, min_periods=5))
            .cast(pl.Int8)
            .alias("bb_squeeze")
        )

    if _need("bb_breakout_upper"):
        if "bb_upper" not in existing:
            df = add_bb(df, period=20, std=2.0)
        df = df.with_columns(
            (pl.col("close") > pl.col("bb_upper"))
            .cast(pl.Int8)
            .alias("bb_breakout_upper")
        )

    # ── 수급 이벤트 (enriched candles 조건부) ──
    existing = set(df.columns)

    if _need("foreign_accumulate_5d") and "foreign_net_norm" in existing:
        df = df.with_columns(
            (pl.col("foreign_net_norm") > 0)
            .cast(pl.Int8)
            .rolling_sum(window_size=5, min_periods=5)
            .eq(5)
            .cast(pl.Int8)
            .alias("foreign_accumulate_5d")
        )

    if _need("inst_accumulate_5d") and "inst_net_norm" in existing:
        df = df.with_columns(
            (pl.col("inst_net_norm") > 0)
            .cast(pl.Int8)
            .rolling_sum(window_size=5, min_periods=5)
            .eq(5)
            .cast(pl.Int8)
            .alias("inst_accumulate_5d")
        )

    return df


def expression_hash(expr: sympy.Basic) -> str:
    """Merkle-style 바텀업 구조 해시. 상수값 무시, 구조만 해싱.

    동일 구조(상수만 다른) 수식은 같은 해시를 반환한다.
    예: close * 2.0 + rsi == close * 3.0 + rsi
    """

    def _hash_node(node: sympy.Basic) -> str:
        # 리프: 숫자 → "N" (값 무시)
        if isinstance(node, (sympy.Integer, sympy.Float, sympy.Rational)):
            return "N"
        if isinstance(node, (int, float)):
            return "N"
        if node.is_number:
            return "N"

        # 리프: 심볼 → "S:name"
        if isinstance(node, sympy.Symbol):
            return f"S:{node.name}"

        # 내부 노드: "TypeName(child1,child2,...)"
        type_name = type(node).__name__
        if hasattr(node, "args") and node.args:
            child_hashes = sorted(_hash_node(a) for a in node.args)
            content = f"{type_name}({','.join(child_hashes)})"
        else:
            content = type_name

        return hashlib.sha256(content.encode()).hexdigest()[:16]

    return _hash_node(expr)


def tree_depth(expr: sympy.Basic) -> int:
    """AST 깊이 계산. 리프=0."""
    if not hasattr(expr, "args") or not expr.args:
        return 0
    return 1 + max(tree_depth(a) for a in expr.args)


def tree_size(expr: sympy.Basic) -> int:
    """AST 노드 수 계산. 모든 노드 카운트."""
    if not hasattr(expr, "args") or not expr.args:
        return 1
    return 1 + sum(tree_size(a) for a in expr.args)


def get_required_columns(expr_str: str) -> set[str]:
    """수식 문자열에서 필요한 Polars 컬럼명 집합을 추출."""
    expr = parse_expression(expr_str)
    result: set[str] = set()
    for sym in expr.free_symbols:
        col = _ALL_VARIABLES.get(str(sym))
        if col:
            result.add(col)
    return result


def parse_expression(expr_str: str) -> sympy.Basic:
    """문자열을 SymPy 표현식으로 파싱.

    Claude가 생성한 수식 문자열을 sympify로 변환한다.
    변수명을 SymPy Symbol로 인식시키기 위해 local_dict를 제공.
    """
    local_dict = {name: sympy.Symbol(name) for name in _ALL_VARIABLES}
    # 자주 쓰이는 함수 추가
    local_dict["log"] = sympy.log
    local_dict["exp"] = sympy.exp
    local_dict["sqrt"] = sympy.sqrt
    local_dict["abs"] = sympy.Abs
    # 조건부 함수
    local_dict["sign"] = sympy.sign
    local_dict["step"] = sympy.Heaviside
    local_dict["Heaviside"] = sympy.Heaviside
    local_dict["Max"] = sympy.Max
    local_dict["Min"] = sympy.Min
    local_dict["clip"] = Clip

    try:
        return sympy.sympify(expr_str, locals=local_dict)
    except (sympy.SympifyError, SyntaxError, TypeError) as e:
        raise ASTConversionError(f"Failed to parse expression: {expr_str}") from e
