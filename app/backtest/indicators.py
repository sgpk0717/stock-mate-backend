"""Polars 네이티브 기술적 지표 계산.

모든 함수는 단일 종목의 OHLCV DataFrame 을 입력받아
지표 컬럼이 추가된 DataFrame 을 반환한다.
"""

from __future__ import annotations

import polars as pl


# ── helpers ──────────────────────────────────────────────


def _ewm(col: str, span: int) -> pl.Expr:
    """지수 이동 평균 (EWM) Polars 표현식."""
    return pl.col(col).ewm_mean(span=span, adjust=False)


# ── 지표 함수 ────────────────────────────────────────────


def add_sma(df: pl.DataFrame, period: int = 20, *, col: str = "close") -> pl.DataFrame:
    name = f"sma_{period}"
    return df.with_columns(pl.col(col).rolling_mean(window_size=period).alias(name))


def add_ema(df: pl.DataFrame, period: int = 20, *, col: str = "close") -> pl.DataFrame:
    name = f"ema_{period}"
    return df.with_columns(_ewm(col, period).alias(name))


def add_rsi(df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    delta = pl.col("close").diff()
    gain = delta.clip(lower_bound=0.0)
    loss = (-delta).clip(lower_bound=0.0)

    # Wilder's RSI: alpha=1/period (업계 표준, talipp과 동일)
    # 기존 span=period는 alpha=2/(period+1)이라 talipp/TradingView와 다른 값 산출
    avg_gain = gain.ewm_mean(alpha=1.0 / period, adjust=False, min_periods=period)
    avg_loss = loss.ewm_mean(alpha=1.0 / period, adjust=False, min_periods=period)

    rs = avg_gain / avg_loss
    rsi = pl.lit(100.0) - (pl.lit(100.0) / (pl.lit(1.0) + rs))

    return df.with_columns(rsi.alias("rsi"))


def add_macd(
    df: pl.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pl.DataFrame:
    ema_fast = _ewm("close", fast)
    ema_slow = _ewm("close", slow)
    macd_line = ema_fast - ema_slow

    df = df.with_columns(macd_line.alias("macd_line"))
    df = df.with_columns(
        pl.col("macd_line").ewm_mean(span=signal, adjust=False).alias("macd_signal")
    )
    df = df.with_columns(
        (pl.col("macd_line") - pl.col("macd_signal")).alias("macd_hist")
    )
    return df


def add_bb(df: pl.DataFrame, period: int = 20, std: float = 2.0) -> pl.DataFrame:
    middle = pl.col("close").rolling_mean(window_size=period)
    roll_std = pl.col("close").rolling_std(window_size=period)
    return df.with_columns(
        middle.alias("bb_middle"),
        (middle + roll_std * std).alias("bb_upper"),
        (middle - roll_std * std).alias("bb_lower"),
    )


def add_volume_ratio(df: pl.DataFrame, period: int = 20) -> pl.DataFrame:
    avg_vol = pl.col("volume").rolling_mean(window_size=period)
    return df.with_columns(
        (pl.col("volume").cast(pl.Float64) / avg_vol).alias("volume_ratio")
    )


def add_price_change_pct(df: pl.DataFrame, period: int = 1) -> pl.DataFrame:
    return df.with_columns(
        (pl.col("close").pct_change(n=period) * 100.0).alias("price_change_pct")
    )


def add_consec_decline(df: pl.DataFrame, days: int = 3) -> pl.DataFrame:
    """N일 연속 종가 하락 감지.

    bar N 에서 값이 1이면: close[N-1] < close[N-2] < ... < close[N-days] 성립
    (어제까지 days일 연속 하락)
    """
    conds = []
    for i in range(1, days + 1):
        conds.append(pl.col("close").shift(i) < pl.col("close").shift(i + 1))
    result = conds[0]
    for c in conds[1:]:
        result = result & c
    return df.with_columns(result.cast(pl.Int8).alias(f"consec_decline_{days}"))


def add_open_gap_pct(df: pl.DataFrame) -> pl.DataFrame:
    """시가 갭 퍼센트 = (open - prev_close) / prev_close * 100."""
    prev_close = pl.col("close").shift(1)
    gap_pct = (pl.col("open") - prev_close) / prev_close * 100
    return df.with_columns(gap_pct.alias("open_gap_pct"))


def add_lag(df: pl.DataFrame, col_name: str = "close", lag: int = 1) -> pl.DataFrame:
    """시차 피처: col_name을 lag만큼 shift."""
    return df.with_columns(
        pl.col(col_name).shift(lag).alias(f"{col_name}_lag_{lag}")
    )


def add_return_nd(df: pl.DataFrame, period: int = 5) -> pl.DataFrame:
    """N일 수익률: (close / close[t-period]) - 1."""
    return df.with_columns(
        (pl.col("close") / pl.col("close").shift(period) - 1.0).alias(f"return_{period}d")
    )


def add_atr(df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    prev_close = pl.col("close").shift(1)
    tr = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - prev_close).abs(),
        (pl.col("low") - prev_close).abs(),
    )
    atr = tr.ewm_mean(span=period, adjust=False)
    return df.with_columns(
        tr.alias("true_range"),
        atr.alias(f"atr_{period}"),
    )


# ── 디스패처 ─────────────────────────────────────────────


_INDICATOR_FN = {
    "rsi": lambda df, p: add_rsi(df, period=p.get("period", 14)),
    "sma": lambda df, p: add_sma(df, period=p.get("period", 20)),
    "ema": lambda df, p: add_ema(df, period=p.get("period", 20)),
    "macd": lambda df, p: add_macd(df, fast=p.get("fast", 12), slow=p.get("slow", 26), signal=p.get("signal", 9)),
    "macd_hist": lambda df, p: add_macd(df, fast=p.get("fast", 12), slow=p.get("slow", 26), signal=p.get("signal", 9)),
    "macd_cross": lambda df, p: add_macd(df, fast=p.get("fast", 12), slow=p.get("slow", 26), signal=p.get("signal", 9)),
    "bb": lambda df, p: add_bb(df, period=p.get("period", 20), std=p.get("std", 2.0)),
    "bb_upper": lambda df, p: add_bb(df, period=p.get("period", 20), std=p.get("std", 2.0)),
    "bb_lower": lambda df, p: add_bb(df, period=p.get("period", 20), std=p.get("std", 2.0)),
    "volume_ratio": lambda df, p: add_volume_ratio(df, period=p.get("period", 20)),
    "price_change_pct": lambda df, p: add_price_change_pct(df, period=p.get("period", 1)),
    "atr": lambda df, p: add_atr(df, period=p.get("period", 14)),
    "consec_decline": lambda df, p: add_consec_decline(df, days=p.get("days", 3)),
    "open_gap_pct": lambda df, p: add_open_gap_pct(df),
    # 뉴스 감성 지표 (외부 join으로 추가되며, 여기서는 no-op)
    "sentiment_score": lambda df, p: df,
    "article_count": lambda df, p: df,
    "event_score": lambda df, p: df,
}

# 뉴스 감성 지표 (engine.py에서 join으로 추가됨, ensure_indicators에서는 skip)
_SENTIMENT_INDICATORS = {"sentiment_score", "article_count", "event_score"}

# golden_cross / dead_cross 는 SMA 기반이므로 별도 처리
_CROSS_INDICATORS = {"golden_cross", "dead_cross"}


def add_indicator(df: pl.DataFrame, indicator: str, params: dict | None = None) -> pl.DataFrame:
    """지표 이름과 파라미터로 DataFrame 에 컬럼을 추가한다."""
    params = params or {}

    if indicator in _CROSS_INDICATORS:
        fast = params.get("fast_period", 5)
        slow = params.get("slow_period", 20)
        df = add_sma(df, period=fast)
        df = add_sma(df, period=slow)
        # 크로스 감지: 전봉에서 fast<=slow → 현봉에서 fast>slow (golden)
        fast_col = f"sma_{fast}"
        slow_col = f"sma_{slow}"
        df = df.with_columns(
            ((pl.col(fast_col) > pl.col(slow_col))
             & (pl.col(fast_col).shift(1) <= pl.col(slow_col).shift(1)))
            .alias("golden_cross"),
            ((pl.col(fast_col) < pl.col(slow_col))
             & (pl.col(fast_col).shift(1) >= pl.col(slow_col).shift(1)))
            .alias("dead_cross"),
        )
        return df

    fn = _INDICATOR_FN.get(indicator)
    if fn is None:
        raise ValueError(f"Unknown indicator: {indicator}")

    return fn(df, params)


def register_custom_indicator(name: str, fn) -> None:
    """동적으로 커스텀 지표를 등록한다.

    fn 시그니처: (df: pl.DataFrame, params: dict) -> pl.DataFrame
    """
    _INDICATOR_FN[name] = fn


def ensure_indicators(df: pl.DataFrame, conditions: list[dict]) -> pl.DataFrame:
    """전략 조건 목록에서 필요한 지표를 모두 추가한다."""
    added: set[str] = set()
    for cond in conditions:
        ind = cond["indicator"]
        params = cond.get("params", {})
        key = f"{ind}:{sorted(params.items())}"
        if key not in added:
            df = add_indicator(df, ind, params)
            added.add(key)
    return df
