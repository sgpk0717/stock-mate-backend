"""Sentence Transformers 기반 종목 임베딩.

jhgan/ko-sroberta-multitask 모델 (768차원, CPU 동작, ~400MB)을 사용하여
종목 정보를 벡터로 변환한다.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)

MODEL_NAME = "jhgan/ko-sroberta-multitask"
_model = None


def _get_model():
    """모델을 지연 로딩한다 (싱글턴)."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer

            logger.info("임베딩 모델 로딩: %s", MODEL_NAME)
            _model = SentenceTransformer(MODEL_NAME)
            logger.info("임베딩 모델 로딩 완료 (dim=%d)", _model.get_sentence_embedding_dimension())
        except ImportError:
            raise ImportError(
                "sentence-transformers 패키지가 필요합니다: "
                "pip install sentence-transformers"
            )
    return _model


def build_stock_text(
    name: str,
    sector: str | None = None,
    sub_sector: str | None = None,
    description: str | None = None,
) -> str:
    """종목 정보를 임베딩용 텍스트로 조합한다.

    예: "삼성전자 반도체 메모리반도체 DRAM NAND 파운드리 시스템LSI"
    """
    parts = [name]
    if sector:
        parts.append(sector)
    if sub_sector:
        parts.append(sub_sector)
    if description:
        parts.append(description)
    return " ".join(parts)


def encode_text(text: str) -> list[float]:
    """단일 텍스트를 768차원 벡터로 변환한다."""
    model = _get_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def encode_texts(texts: list[str]) -> list[list[float]]:
    """여러 텍스트를 배치로 인코딩한다."""
    if not texts:
        return []
    model = _get_model()
    embeddings = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
    return embeddings.tolist()


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """두 벡터 간 코사인 유사도를 계산한다.

    정규화된 벡터이므로 내적 = 코사인 유사도.
    """
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    return float(np.dot(a, b))


def cosine_similarity_batch(
    query_vec: list[float],
    candidate_vecs: list[list[float]],
) -> list[float]:
    """쿼리 벡터와 후보 벡터들 간 코사인 유사도를 배치 계산한다."""
    q = np.array(query_vec, dtype=np.float32).reshape(1, -1)
    c = np.array(candidate_vecs, dtype=np.float32)
    if c.ndim == 1:
        c = c.reshape(1, -1)
    scores = (q @ c.T).flatten()
    return scores.tolist()
