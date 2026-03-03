"""강화된 익명화 — 날짜/티커 추가 마스킹.

기존 news/anonymizer.py의 EntityAnonymizer를 확장.
LLM 에이전트에 전달할 시장 맥락에서 민감 정보 제거.
"""

from __future__ import annotations

import re
from typing import Sequence

from app.news.anonymizer import EntityAnonymizer


class EnhancedAnonymizer(EntityAnonymizer):
    """기업명 + 날짜 + 티커 마스킹."""

    # YYYY-MM-DD, YYYY/MM/DD, YYYY.MM.DD
    _DATE_PATTERN = re.compile(r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}")
    # 6자리 종목코드
    _TICKER_PATTERN = re.compile(r"\b\d{6}\b")

    def __init__(self, entity_names: Sequence[str]):
        super().__init__(entity_names)

    def anonymize_full(self, text: str) -> str:
        """기업명 + 날짜 + 티커 전체 익명화."""
        text = self.anonymize(text)  # 기업명
        text = self._DATE_PATTERN.sub("[DATE]", text)
        text = self._TICKER_PATTERN.sub("[TICKER]", text)
        return text
