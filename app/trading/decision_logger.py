"""매매 의사결정 로깅 공통 모듈.

live_runner, sim_engine, backtest 모두 이 함수를 사용하여
동일한 포맷의 의사결정 기록을 남긴다.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.core.stock_master import get_stock_name

_KST = timezone(timedelta(hours=9))


def log_decision(
    log_store: list[dict],
    symbol: str,
    action: str,
    reason: str,
    *,
    signal: int = 0,
    conditions: dict | None = None,
    snapshot: dict | None = None,
    sizing: dict | None = None,
    risk: dict | None = None,
    timestamp: str | None = None,
) -> dict:
    """매매 의사결정 기록.

    Parameters
    ----------
    log_store : 기록을 추가할 리스트 (session.decision_log, runner.decisions 등)
    symbol : 종목 코드
    action : BUY, SELL, SKIP_BUY, RISK_STOP, RISK_TRAIL, RISK_ATR_STOP,
             PARTIAL_EXIT, SCALE_IN, SKIP_DATA, SKIP_ERROR, SKIP_PENDING,
             SKIP_STRATEGY_FILTER, SUBMIT, FILL, EXPIRE 등
    reason : 판단 사유 (사람이 읽을 수 있는 문자열)
    signal : 시그널 값 (1=매수, -1=매도, 0=중립)
    conditions : 매수/매도 조건 충족 상세
    snapshot : 당시 지표 스냅샷
    sizing : 포지션 사이징 정보
    risk : 리스크 관리 정보
    timestamp : 지정 시각 (None이면 현재 KST)

    Returns
    -------
    dict : 추가된 엔트리 (SSE 스트림 등에서 재사용 가능)
    """
    entry = {
        "timestamp": timestamp or datetime.now(_KST).isoformat(),
        "symbol": symbol,
        "name": get_stock_name(symbol),
        "action": action,
        "reason": reason,
        "signal": signal,
        "conditions": conditions,
        "snapshot": snapshot,
        "sizing": sizing,
        "risk": risk,
    }
    log_store.append(entry)
    return entry
