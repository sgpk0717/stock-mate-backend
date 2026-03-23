"""개별 전략 필터 함수.

모든 필터는 동일한 시그니처를 따른다:
    filter_fn(signal, row, context, state) -> dict

반환값:
    {"skip": False}  — 통과
    {"skip": True, "filter": "이름", "reason": "사유"}  — 거부
"""
from __future__ import annotations

import logging
from datetime import datetime, time, timezone, timedelta
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)

_KST = timezone(timedelta(hours=9))


def time_filter(
    signal: int,
    row: dict[str, Any],
    context: Any,
    state: dict[str, Any],
) -> dict[str, Any]:
    """장 초반 / 장 막판 매수 차단.

    - 매수(signal=1)만 필터링. 매도는 항상 통과.
    - 09:00~09:00+HOLD분: 개장 노이즈 회피
    - 15:30-BLOCK분 이후: 동시호가 리스크 회피
    """
    if signal != 1:
        return {"skip": False}

    dt_val = row.get("dt")
    if dt_val is None:
        return {"skip": False}

    if isinstance(dt_val, str):
        try:
            dt_val = datetime.fromisoformat(dt_val)
        except (ValueError, TypeError):
            return {"skip": False}

    # KST 변환
    if dt_val.tzinfo is None:
        dt_val = dt_val.replace(tzinfo=_KST)
    t = dt_val.astimezone(_KST).time()

    hold_min = settings.STRATEGY_MARKET_OPEN_HOLD_MINUTES
    block_min = settings.STRATEGY_MARKET_CLOSE_BLOCK_MINUTES

    # 장 초반 차단
    open_cutoff = time(9, hold_min)
    if t < open_cutoff:
        return {
            "skip": True,
            "filter": "time_open",
            "reason": f"장 초반 {hold_min}분 매수 차단 ({t.strftime('%H:%M')})",
        }

    # 장 막판 차단
    close_hour = 15
    close_minute = 30 - block_min
    if close_minute < 0:
        close_hour -= 1
        close_minute += 60
    close_cutoff = time(close_hour, close_minute)
    if t >= close_cutoff:
        return {
            "skip": True,
            "filter": "time_close",
            "reason": f"장 마감 {block_min}분 전 매수 차단 ({t.strftime('%H:%M')})",
        }

    return {"skip": False}


def volume_filter(
    signal: int,
    row: dict[str, Any],
    context: Any,
    state: dict[str, Any],
) -> dict[str, Any]:
    """거래량 확인 필터 — volume_ratio가 임계값 미만이면 매수 차단."""
    if signal != 1:
        return {"skip": False}

    min_ratio = settings.STRATEGY_MIN_VOLUME_RATIO
    if min_ratio <= 0:
        return {"skip": False}

    vol_ratio = row.get("volume_ratio")
    if vol_ratio is None:
        # volume_ratio 컬럼이 없으면 필터 비활성화
        return {"skip": False}

    if vol_ratio < min_ratio:
        return {
            "skip": True,
            "filter": "volume",
            "reason": f"거래량비 {vol_ratio:.2f} < {min_ratio} (기준 미달)",
        }

    return {"skip": False}


def trade_limit_filter(
    signal: int,
    row: dict[str, Any],
    context: Any,
    state: dict[str, Any],
) -> dict[str, Any]:
    """일일 매매 횟수 제한 — 매수만 차단, 매도는 항상 허용."""
    if signal != 1:
        return {"skip": False}

    max_trades = settings.STRATEGY_MAX_DAILY_TRADES
    if max_trades <= 0:
        return {"skip": False}

    daily_count = state.get("daily_buy_count", 0)
    if daily_count >= max_trades:
        return {
            "skip": True,
            "filter": "trade_limit",
            "reason": f"일일 매수 {daily_count}/{max_trades} 초과",
        }

    return {"skip": False}
