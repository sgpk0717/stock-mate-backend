"""개체명 익명화 — 백테스트 전용.

뉴스 기사에서 종목명을 [COMPANY_A] 등으로 치환하여
Look-ahead bias를 추가 방지한다.

실거래에서는 사용하지 않음 (실시간 분석에 불필요).
"""

from __future__ import annotations

import logging
import re
from typing import Sequence

logger = logging.getLogger(__name__)

# 익명화 라벨 순서
_LABELS = [f"[COMPANY_{chr(65 + i)}]" for i in range(26)]  # A~Z


class EntityAnonymizer:
    """종목명 → 익명 라벨 변환기."""

    def __init__(self, entity_names: Sequence[str]):
        """
        Args:
            entity_names: 종목명 리스트 (예: ["삼성전자", "SK하이닉스", ...])
        """
        # 긴 이름 우선 매칭 (삼성전자우선주 before 삼성전자)
        sorted_names = sorted(entity_names, key=len, reverse=True)
        self._mapping: dict[str, str] = {}
        self._reverse: dict[str, str] = {}
        label_idx = 0

        for name in sorted_names:
            name = name.strip()
            if not name or name in self._mapping:
                continue
            if label_idx >= len(_LABELS):
                break
            label = _LABELS[label_idx]
            self._mapping[name] = label
            self._reverse[label] = name
            label_idx += 1

        # 정규식 패턴: 모든 종목명을 OR로 결합
        if self._mapping:
            escaped = [re.escape(n) for n in self._mapping]
            self._pattern = re.compile("|".join(escaped))
        else:
            self._pattern = None

    def anonymize(self, text: str) -> str:
        """텍스트에서 종목명을 익명 라벨로 치환."""
        if not self._pattern or not text:
            return text
        return self._pattern.sub(lambda m: self._mapping.get(m.group(), m.group()), text)

    def deanonymize(self, text: str) -> str:
        """익명 라벨을 원래 종목명으로 복원."""
        if not self._reverse or not text:
            return text
        for label, name in self._reverse.items():
            text = text.replace(label, name)
        return text

    @property
    def mapping(self) -> dict[str, str]:
        return dict(self._mapping)


async def build_anonymizer_from_db() -> EntityAnonymizer:
    """DB stock_masters에서 종목명 로딩 → 익명화기 생성."""
    from sqlalchemy import select

    from app.core.database import async_session
    from app.models.base import StockMaster

    async with async_session() as session:
        result = await session.execute(select(StockMaster.name))
        names = [row[0] for row in result.fetchall() if row[0]]

    logger.info("익명화 사전 구축: %d개 종목명", len(names))
    return EntityAnonymizer(names)


def anonymize_articles(
    articles: list[dict],
    anonymizer: EntityAnonymizer,
) -> list[dict]:
    """기사 리스트의 title/content를 익명화."""
    result = []
    for article in articles:
        a = dict(article)
        if "title" in a:
            a["title"] = anonymizer.anonymize(a["title"])
        if "content" in a:
            a["content"] = anonymizer.anonymize(a["content"])
        result.append(a)
    return result
