"""K-means 기반 테마 클러스터링.

임베딩된 종목들을 자동으로 테마 그룹으로 분류한다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.base import StockMaster

logger = logging.getLogger(__name__)


@dataclass
class Cluster:
    """클러스터 정보."""

    cluster_id: int
    stocks: list[dict]  # [{symbol, name, sector}]
    centroid: list[float]
    top_sectors: list[str]  # 클러스터 내 가장 많은 섹터


async def cluster_stocks(
    session: AsyncSession,
    n_clusters: int = 30,
) -> list[Cluster]:
    """임베딩된 종목들을 K-means로 클러스터링한다.

    Args:
        session: DB 세션
        n_clusters: 클러스터 수

    Returns:
        Cluster 리스트
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        raise ImportError("scikit-learn 패키지가 필요합니다: pip install scikit-learn")

    # 임베딩 있는 종목 조회
    stmt = select(StockMaster).where(StockMaster.embedding.isnot(None))
    result = await session.execute(stmt)
    stocks = result.scalars().all()

    if len(stocks) < n_clusters:
        logger.warning("종목 수(%d)가 클러스터 수(%d)보다 적습니다.", len(stocks), n_clusters)
        n_clusters = max(2, len(stocks) // 5)

    # 임베딩 행렬
    embeddings = np.array([s.embedding for s in stocks], dtype=np.float32)

    # K-means 클러스터링
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    # 클러스터별 정리
    clusters: dict[int, list] = {}
    for stock, label in zip(stocks, labels):
        label_int = int(label)
        if label_int not in clusters:
            clusters[label_int] = []
        clusters[label_int].append({
            "symbol": stock.symbol,
            "name": stock.name,
            "sector": stock.sector,
        })

    result_clusters: list[Cluster] = []
    for cid, stock_list in sorted(clusters.items()):
        # 가장 많은 섹터 추출
        sector_counts: dict[str, int] = {}
        for s in stock_list:
            sec = s.get("sector") or "미분류"
            sector_counts[sec] = sector_counts.get(sec, 0) + 1

        top_sectors = sorted(sector_counts, key=sector_counts.get, reverse=True)[:3]

        result_clusters.append(
            Cluster(
                cluster_id=cid,
                stocks=stock_list,
                centroid=kmeans.cluster_centers_[cid].tolist(),
                top_sectors=top_sectors,
            )
        )

    logger.info("클러스터링 완료: %d개 클러스터, %d개 종목", len(result_clusters), len(stocks))
    return result_clusters
