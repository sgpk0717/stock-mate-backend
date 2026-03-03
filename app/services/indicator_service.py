"""기술적 지표 계산 서비스 — talipp 기반.

캔들 종가 데이터로 RSI, MACD, 볼린저밴드를 계산하여 반환.
각 지표는 캔들 배열과 동일한 길이의 배열로 반환되며,
데이터 부족 구간은 None으로 채워진다.
"""

from talipp.indicators import BB, MACD, RSI, SMA


def compute_indicators(
    candles: list[dict], indicators: list[str]
) -> dict[str, list]:
    """지표 계산.

    Args:
        candles: [{"time", "open", "high", "low", "close", "volume"}, ...]
        indicators: ["rsi", "macd", "bb"]

    Returns:
        {
            "rsi": [{"value": float|null}, ...],
            "macd": [{"macd", "signal", "histogram"}, ...],
            "bb": [{"upper", "middle", "lower"}, ...],
        }
    """
    if not candles:
        return {}

    closes = [c["close"] for c in candles]
    n = len(closes)
    result: dict[str, list] = {}

    if "rsi" in indicators:
        rsi = RSI(period=14, input_values=closes)
        values = list(rsi)
        padding = n - len(values)
        result["rsi"] = (
            [{"value": None}] * padding
            + [
                {"value": round(float(v), 2) if v is not None else None}
                for v in values
            ]
        )

    if "macd" in indicators:
        macd = MACD(
            fast_period=12, slow_period=26, signal_period=9, input_values=closes
        )
        values = list(macd)
        padding = n - len(values)
        result["macd"] = (
            [{"macd": None, "signal": None, "histogram": None}] * padding
            + [
                {
                    "macd": round(float(v.macd), 2) if v and v.macd is not None else None,
                    "signal": round(float(v.signal), 2) if v and v.signal is not None else None,
                    "histogram": round(float(v.histogram), 2) if v and v.histogram is not None else None,
                }
                for v in values
            ]
        )

    for ind in indicators:
        if ind.startswith("sma_"):
            try:
                period = int(ind.split("_")[1])
            except (IndexError, ValueError):
                continue
            if period < 2 or period > 500:
                continue
            sma = SMA(period=period, input_values=closes)
            values = list(sma)
            padding = n - len(values)
            result[ind] = (
                [{"value": None}] * padding
                + [
                    {"value": round(float(v), 2) if v is not None else None}
                    for v in values
                ]
            )

    if "bb" in indicators:
        bb = BB(period=20, std_dev_mult=2.0, input_values=closes)
        values = list(bb)
        padding = n - len(values)
        result["bb"] = (
            [{"upper": None, "middle": None, "lower": None}] * padding
            + [
                {
                    "upper": round(float(v.ub), 2) if v and v.ub is not None else None,
                    "middle": round(float(v.cb), 2) if v and v.cb is not None else None,
                    "lower": round(float(v.lb), 2) if v and v.lb is not None else None,
                }
                for v in values
            ]
        )

    return result
