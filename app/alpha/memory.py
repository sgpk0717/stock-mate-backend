"""벡터 기반 경험 메모리.

sector/embedder.py의 ko-sroberta 768-dim 임베딩을 활용하여
과거 성공/실패 팩터를 벡터 유사도로 조회한다.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.alpha.models import AlphaExperience

logger = logging.getLogger(__name__)


def _encode_text_safe(text: str) -> list[float] | None:
    """임베딩 생성. 모델 미로드 시 None 반환."""
    try:
        from app.sector.embedder import encode_text

        return encode_text(text)
    except Exception as e:
        logger.warning("Embedding encode failed: %s", e)
        return None


def _cosine_similarity_batch_safe(
    query_vec: list[float], candidate_vecs: list[list[float]]
) -> list[float]:
    """배치 코사인 유사도. 실패 시 빈 리스트."""
    try:
        from app.sector.embedder import cosine_similarity_batch

        return cosine_similarity_batch(query_vec, candidate_vecs)
    except Exception as e:
        logger.warning("Cosine similarity batch failed: %s", e)
        return []


@dataclass
class CachedExperience:
    """메모리 캐시된 경험 항목."""

    id: str
    expression_str: str
    hypothesis: str
    ic_mean: float
    success: bool
    generation: int
    embedding: list[float]
    parent_ids: list[str] | None = None


class ExperienceVectorMemory:
    """벡터 기반 경험 메모리.

    초기화 시 DB에서 모든 경험을 메모리 캐시로 로드.
    add() 시 DB 저장 + 캐시 업데이트.
    retrieve_similar() 시 인메모리 코사인 유사도 검색.
    """

    def __init__(self) -> None:
        self._cache: list[CachedExperience] = []
        self._embeddings: np.ndarray | None = None  # shape (N, 768)
        self._pending_embeddings: list[np.ndarray] = []  # lazy rebuild용

    async def load_cache(self, db: AsyncSession) -> None:
        """DB에서 모든 경험을 메모리로 로드."""
        result = await db.execute(
            select(AlphaExperience)
            .where(AlphaExperience.embedding.isnot(None))
            .order_by(AlphaExperience.created_at.desc())
        )
        experiences = result.scalars().all()

        self._cache = []
        embeddings_list: list[list[float]] = []

        for exp in experiences:
            self._cache.append(
                CachedExperience(
                    id=str(exp.id),
                    expression_str=exp.expression_str,
                    hypothesis=exp.hypothesis or "",
                    ic_mean=exp.ic_mean or 0.0,
                    success=exp.success,
                    generation=exp.generation,
                    embedding=exp.embedding,
                    parent_ids=exp.parent_ids,
                )
            )
            embeddings_list.append(exp.embedding)

        self._pending_embeddings.clear()
        if embeddings_list:
            self._embeddings = np.array(embeddings_list, dtype=np.float32)
        else:
            self._embeddings = None

        logger.info(
            "Loaded %d experiences into vector memory (%d successes)",
            len(self._cache),
            sum(1 for e in self._cache if e.success),
        )

    async def add(
        self,
        db: AsyncSession,
        expression_str: str,
        hypothesis: str,
        ic_mean: float,
        generation: int,
        success: bool,
        factor_id: uuid.UUID | None = None,
        parent_ids: list[str] | None = None,
        failure_reason: str | None = None,
    ) -> None:
        """경험을 DB에 저장하고 인메모리 캐시를 업데이트."""
        text = f"{hypothesis} {expression_str}"
        embedding = _encode_text_safe(text)

        exp = AlphaExperience(
            factor_id=factor_id,
            expression_str=expression_str,
            hypothesis=hypothesis,
            embedding=embedding,
            ic_mean=ic_mean,
            success=success,
            generation=generation,
            parent_ids=parent_ids,
            failure_reason=failure_reason,
        )
        db.add(exp)
        await db.flush()

        if embedding is None:
            logger.debug("Experience saved without embedding (model unavailable)")
            return

        cached = CachedExperience(
            id=str(exp.id),
            expression_str=expression_str,
            hypothesis=hypothesis,
            ic_mean=ic_mean,
            success=success,
            generation=generation,
            embedding=embedding,
            parent_ids=parent_ids,
        )
        self._cache.append(cached)

        emb_arr = np.array(embedding, dtype=np.float32).reshape(1, -1)
        self._pending_embeddings.append(emb_arr)

    def _rebuild_embeddings_if_needed(self) -> None:
        """pending embeddings를 메인 배열에 병합."""
        if not self._pending_embeddings:
            return
        parts: list[np.ndarray] = []
        if self._embeddings is not None:
            parts.append(self._embeddings)
        parts.extend(self._pending_embeddings)
        self._embeddings = np.vstack(parts)
        self._pending_embeddings.clear()

    def retrieve_similar(
        self,
        query: str,
        k: int = 10,
        success_only: bool | None = None,
    ) -> list[CachedExperience]:
        """코사인 유사도 기반으로 유사 경험을 검색."""
        self._rebuild_embeddings_if_needed()
        if not self._cache or self._embeddings is None:
            return []

        query_vec = _encode_text_safe(query)
        if query_vec is None:
            return []

        similarities = _cosine_similarity_batch_safe(
            query_vec, self._embeddings.tolist()
        )
        if not similarities:
            return []

        indexed = list(enumerate(similarities))
        if success_only is not None:
            indexed = [
                (i, s)
                for i, s in indexed
                if self._cache[i].success == success_only
            ]

        indexed.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, _sim in indexed[:k]:
            results.append(self._cache[idx])

        return results

    def check_orthogonality(
        self,
        expression_str: str,
        hypothesis: str,
        threshold: float = 0.7,
    ) -> bool:
        """새 팩터가 기존 성공 팩터들과 충분히 직교적인지 확인.

        cos_sim < threshold이면 True (직교적).
        """
        self._rebuild_embeddings_if_needed()
        if not self._cache or self._embeddings is None:
            return True

        text = f"{hypothesis} {expression_str}"
        query_vec = _encode_text_safe(text)
        if query_vec is None:
            return True

        success_indices = [
            i for i, e in enumerate(self._cache) if e.success
        ]
        if not success_indices:
            return True

        success_embeddings = self._embeddings[success_indices].tolist()
        similarities = _cosine_similarity_batch_safe(
            query_vec, success_embeddings
        )

        max_sim = max(similarities) if similarities else 0.0
        return max_sim < threshold

    def format_rag_context(self, query: str) -> str:
        """Claude 프롬프트용: 유사 성공 5개 + 유사 실패 3개."""
        lines: list[str] = []

        top_success = self.retrieve_similar(query, k=5, success_only=True)
        if top_success:
            lines.append("=== 유사한 성공 팩터 (RAG 검색) ===")
            for i, exp in enumerate(top_success, 1):
                lines.append(
                    f"{i}. {exp.expression_str} "
                    f"(IC={exp.ic_mean:.4f}, gen={exp.generation})"
                )
                if exp.hypothesis:
                    lines.append(f"   가설: {exp.hypothesis[:100]}")

        top_failure = self.retrieve_similar(query, k=3, success_only=False)
        if top_failure:
            lines.append("\n=== 유사하지만 실패한 팩터 (회피) ===")
            for i, exp in enumerate(top_failure, 1):
                lines.append(
                    f"{i}. {exp.expression_str} "
                    f"(IC={exp.ic_mean:.4f})"
                )

        if not lines:
            return "아직 탐색 이력이 없습니다."

        return "\n".join(lines)
