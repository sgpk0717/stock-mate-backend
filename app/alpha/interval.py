"""인터벌 유틸리티.

annualization factor, warmup 기간, 비용 설정 등
interval 관련 매직넘버를 중앙 관리한다.
"""

from __future__ import annotations

_KRX_MINUTES_PER_DAY = 390  # 09:00~15:30 KST
_TRADING_DAYS_PER_YEAR = 252

_INTERVAL_MINUTES: dict[str, int] = {
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
}

SUPPORTED_ALPHA_INTERVALS = {"1m", "3m", "5m", "15m", "30m", "1h", "1d"}


def bars_per_year(interval: str) -> float:
    """연간 봉 수. annualization factor로 사용."""
    if interval == "1d":
        return float(_TRADING_DAYS_PER_YEAR)
    m = _INTERVAL_MINUTES.get(interval)
    if m is None:
        raise ValueError(f"Unsupported interval: {interval}")
    return (_KRX_MINUTES_PER_DAY / m) * _TRADING_DAYS_PER_YEAR


def bars_per_day(interval: str) -> float:
    """일간 봉 수."""
    if interval == "1d":
        return 1.0
    m = _INTERVAL_MINUTES.get(interval)
    if m is None:
        raise ValueError(f"Unsupported interval: {interval}")
    return _KRX_MINUTES_PER_DAY / m


def warmup_days(interval: str) -> int:
    """지표 계산용 워밍업 기간 (캘린더 일수).

    분봉도 SMA(20), BB(20), MACD(26) 등 일간 기반 지표를 사용하므로
    최소 60 캘린더일(약 40 거래일)의 워밍업이 필요.
    """
    return 130 if interval == "1d" else 90


def default_round_trip_cost(interval: str) -> float:
    """왕복 거래비용.

    일봉: 0.43% (매수 0.015% + 매도 0.215% + 슬리피지 0.10%×2)
    분봉: 0.33% (매수 0.015% + 매도 0.215% + 슬리피지 0.05%×2)
    """
    return 0.0043 if interval == "1d" else 0.0033


def is_intraday(interval: str) -> bool:
    """일중(분봉) 인터벌 여부."""
    return interval != "1d"
