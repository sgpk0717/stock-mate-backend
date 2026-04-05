"""타임존 헬퍼 — 프로젝트 전체 단일 규칙.

규칙:
  - 비즈니스 로직은 항상 KST aware datetime
  - DB는 TIMESTAMPTZ (PostgreSQL이 UTC로 자동 변환)
  - DB 조회 결과는 to_kst()로 변환 후 사용
  - datetime.now() (naive) 금지
  - timezone(timedelta(hours=9)) 직접 선언 금지

사용법:
  from app.core.timezone import KST, now_kst, to_kst, today_kst, to_iso_kst
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

KST = timezone(timedelta(hours=9))


def now_kst() -> datetime:
    """현재 시각 (KST aware)."""
    return datetime.now(KST)


def to_kst(dt: datetime) -> datetime:
    """datetime을 KST aware로 변환.

    - naive → KST로 간주
    - UTC aware → KST로 변환
    - KST aware → 그대로
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=KST)
    return dt.astimezone(KST)


def today_kst() -> date:
    """오늘 날짜 (KST 기준)."""
    return now_kst().date()


def to_iso_kst(dt: datetime) -> str:
    """KST ISO 8601 문자열."""
    return to_kst(dt).isoformat()
