"""미검증 알파 팩터 일괄 인과 검증 스크립트.

IC 기반 필터링 + asyncio.Semaphore 병렬 처리로 빠르게 검증한다.
mining_run_id별로 그룹화하여 교란 변수 캐시를 재사용한다.
앱 lifespan 없이 실행되므로 stock_master 캐시를 수동 로드한다.

Usage:
    docker-compose run --rm app python -m scripts.validate_all_factors
    docker-compose run --rm app python -m scripts.validate_all_factors --dry-run
    docker-compose run --rm app python -m scripts.validate_all_factors --min-ic 0.05
    docker-compose run --rm app python -m scripts.validate_all_factors --workers 8
    docker-compose run --rm app python -m scripts.validate_all_factors --simulations 30
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
import warnings

# DoWhy / pandas FutureWarning 억제
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*dowhy.*")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("dowhy").setLevel(logging.ERROR)
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

logger = logging.getLogger("validate_all")


async def main(
    dry_run: bool = False,
    min_ic: float = 0.03,
    workers: int = 4,
    simulations: int = 50,
) -> None:
    from collections import defaultdict

    from sqlalchemy import select, update

    from app.alpha.causal_runner import validate_single_factor
    from app.alpha.models import AlphaFactor
    from app.core.config import settings
    from app.core.database import async_session
    from app.core.stock_master import load_stock_cache

    # DoWhy 시뮬레이션 횟수 오버라이드
    settings.CAUSAL_NUM_SIMULATIONS = simulations

    # stock_master 캐시 로드 (유니버스 폴백에 필요)
    async with async_session() as db:
        await load_stock_cache(db)
    logger.info("Stock master 캐시 로드 완료")

    # 미검증 팩터 조회 (IC 포함)
    async with async_session() as db:
        result = await db.execute(
            select(AlphaFactor.id, AlphaFactor.mining_run_id, AlphaFactor.ic_mean)
            .where(AlphaFactor.causal_robust.is_(None))
            .order_by(AlphaFactor.mining_run_id.asc().nulls_last(), AlphaFactor.created_at.asc())
        )
        rows = result.all()

    # IC 기준 분류
    to_validate: list[tuple] = []  # (factor_id, run_id)
    to_skip_ids: list = []  # IC 미달 → 바로 mirage

    for factor_id, run_id, ic_mean in rows:
        if ic_mean is not None and ic_mean < min_ic:
            to_skip_ids.append(factor_id)
        else:
            to_validate.append((factor_id, run_id))

    total_unvalidated = len(rows)
    skip_count = len(to_skip_ids)
    validate_count = len(to_validate)

    logger.info(
        "미검증 팩터: %d개 (IC >= %.3f: %d개 검증 대상, IC 미달: %d개 → mirage 마킹)",
        total_unvalidated, min_ic, validate_count, skip_count,
    )

    if dry_run:
        # 그룹별 상세
        groups: dict[str | None, list] = defaultdict(list)
        for factor_id, run_id in to_validate:
            groups[str(run_id) if run_id else None].append(factor_id)
        for run_id, ids in groups.items():
            logger.info("  run %s: %d개", run_id, len(ids))
        return

    if total_unvalidated == 0:
        logger.info("모든 팩터가 이미 검증되었습니다.")
        return

    # IC 미달 팩터 일괄 mirage 마킹
    if to_skip_ids:
        logger.info("IC 미달 %d개 팩터를 mirage로 일괄 마킹...", skip_count)
        batch_size = 500
        for i in range(0, len(to_skip_ids), batch_size):
            batch = to_skip_ids[i : i + batch_size]
            async with async_session() as db:
                await db.execute(
                    update(AlphaFactor)
                    .where(AlphaFactor.id.in_(batch))
                    .values(causal_robust=False, status="mirage")
                )
                await db.commit()
        logger.info("IC 미달 마킹 완료: %d개", skip_count)

    if validate_count == 0:
        logger.info("검증 대상 팩터가 없습니다.")
        return

    # mining_run_id별 그룹화
    groups: dict[str | None, list] = defaultdict(list)
    for factor_id, run_id in to_validate:
        groups[str(run_id) if run_id else None].append(factor_id)

    group_count = len(groups)
    logger.info(
        "검증 시작: %d개 팩터, %d개 run 그룹, workers=%d, simulations=%d",
        validate_count, group_count, workers, simulations,
    )

    validated = 0
    failed = 0
    done = 0
    start_time = time.time()
    semaphore = asyncio.Semaphore(workers)

    for g_idx, (run_id, factor_ids) in enumerate(groups.items(), 1):
        logger.info(
            "=== Run %s (%d/%d, %d 팩터) ===",
            run_id, g_idx, group_count, len(factor_ids),
        )

        # 교란 변수 캐시: 같은 run 내에서 재사용
        confounders_cache: dict = {}

        async def validate_one(factor_id, cache: dict) -> bool:
            async with semaphore:
                try:
                    async with async_session() as db:
                        await validate_single_factor(
                            factor_id, db, confounders_cache=cache
                        )
                    return True
                except Exception as e:
                    err_msg = str(e)
                    if "SVD" not in err_msg and "No candle" not in err_msg:
                        logger.warning("팩터 %s 실패: %s", str(factor_id)[:8], err_msg[:100])
                    return False

        # run 그룹 내 병렬 실행
        results = await asyncio.gather(
            *[validate_one(fid, confounders_cache) for fid in factor_ids],
            return_exceptions=True,
        )

        for r in results:
            done += 1
            if r is True:
                validated += 1
            else:
                failed += 1

        elapsed = time.time() - start_time
        rate = done / elapsed if elapsed > 0 else 0
        eta = (validate_count - done) / rate if rate > 0 else 0
        logger.info(
            "[%d/%d] 검증=%d 실패=%d (%.1f건/분, ETA %.0f분)",
            done, validate_count, validated, failed,
            rate * 60, eta / 60,
        )

    elapsed = time.time() - start_time
    logger.info(
        "=== 완료: %d/%d 검증, %d 실패, %d IC 미달 스킵 (%.1f분 소요) ===",
        validated, validate_count, failed, skip_count, elapsed / 60,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="미검증 알파 팩터 일괄 인과 검증")
    parser.add_argument("--dry-run", action="store_true", help="개수만 확인")
    parser.add_argument("--min-ic", type=float, default=0.03, help="IC 기준 (기본 0.03)")
    parser.add_argument("--workers", type=int, default=4, help="동시 처리 수 (기본 4)")
    parser.add_argument("--simulations", type=int, default=50, help="DoWhy 시뮬레이션 횟수 (기본 50)")
    args = parser.parse_args()

    asyncio.run(main(
        dry_run=args.dry_run,
        min_ic=args.min_ic,
        workers=args.workers,
        simulations=args.simulations,
    ))
