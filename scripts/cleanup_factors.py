"""알파 팩터 정리 스크립트.

누적된 저품질 팩터를 배치 삭제한다.
보호 대상(LiveFeedback/WorkflowRun/TradingContext 참조)은 자동 제외.

사용법:
    docker-compose exec app python -m scripts.cleanup_factors --dry-run
    docker-compose exec app python -m scripts.cleanup_factors --execute
    docker-compose exec app python -m scripts.cleanup_factors --execute --keep-top=500
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from uuid import UUID

from sqlalchemy import delete, func, select, text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cleanup_factors")

# ── 보호 대상 팩터 ID 수집 ──────────────────────────────────


async def _collect_protected_ids() -> set[UUID]:
    """삭제 불가 팩터 ID: LiveFeedback/WorkflowRun/TradingContext에서 참조."""
    from app.core.database import async_session

    protected: set[UUID] = set()
    async with async_session() as db:
        # LiveFeedback
        rows = (await db.execute(text(
            "SELECT DISTINCT factor_id FROM live_feedback"
        ))).fetchall()
        for r in rows:
            if r[0]:
                protected.add(r[0])

        # WorkflowRun
        rows = (await db.execute(text(
            "SELECT DISTINCT selected_factor_id FROM workflow_runs "
            "WHERE selected_factor_id IS NOT NULL"
        ))).fetchall()
        for r in rows:
            if r[0]:
                protected.add(r[0])

        # TradingContext (source_factor_id — JSON 내부)
        rows = (await db.execute(text(
            "SELECT DISTINCT "
            "  CASE WHEN strategy->>'factor_id' IS NOT NULL "
            "       THEN (strategy->>'factor_id')::uuid END "
            "FROM trading_contexts "
            "WHERE strategy->>'factor_id' IS NOT NULL"
        ))).fetchall()
        for r in rows:
            if r[0]:
                protected.add(r[0])

    logger.info("보호 팩터 ID: %d개", len(protected))
    return protected


# ── 삭제 대상 집계 ──────────────────────────────────────────


async def _count_targets() -> dict:
    """삭제 대상 수를 상태/조건별로 집계."""
    from app.core.database import async_session

    stats: dict = {}
    async with async_session() as db:
        stats["total"] = (await db.execute(text(
            "SELECT COUNT(*) FROM alpha_factors"
        ))).scalar()

        stats["mirage"] = (await db.execute(text(
            "SELECT COUNT(*) FROM alpha_factors WHERE status = 'mirage'"
        ))).scalar()

        stats["pop_bad_ic"] = (await db.execute(text(
            "SELECT COUNT(*) FROM alpha_factors "
            "WHERE status = 'population' AND (ic_mean < 0.03 OR ic_mean IS NULL)"
        ))).scalar()

        stats["pop_bad_sharpe"] = (await db.execute(text(
            "SELECT COUNT(*) FROM alpha_factors "
            "WHERE status = 'population' AND ic_mean >= 0.03 "
            "AND (sharpe < 0.3 OR sharpe IS NULL)"
        ))).scalar()

        stats["pop_good"] = (await db.execute(text(
            "SELECT COUNT(*) FROM alpha_factors "
            "WHERE status = 'population' AND ic_mean >= 0.03 AND sharpe >= 0.3"
        ))).scalar()

        stats["validated"] = (await db.execute(text(
            "SELECT COUNT(*) FROM alpha_factors WHERE status = 'validated'"
        ))).scalar()

        # 중복 수식
        stats["duplicate_hashes"] = (await db.execute(text(
            "SELECT COUNT(*) FROM ("
            "  SELECT expression_hash FROM alpha_factors "
            "  WHERE expression_hash IS NOT NULL "
            "  GROUP BY expression_hash HAVING COUNT(*) > 1"
            ") sub"
        ))).scalar()

        stats["duplicate_excess"] = (await db.execute(text(
            "SELECT COALESCE(SUM(cnt - 1), 0) FROM ("
            "  SELECT expression_hash, COUNT(*) as cnt FROM alpha_factors "
            "  WHERE expression_hash IS NOT NULL "
            "  GROUP BY expression_hash HAVING COUNT(*) > 1"
            ") sub"
        ))).scalar()

        stats["mining_runs"] = (await db.execute(text(
            "SELECT COUNT(*) FROM alpha_mining_runs"
        ))).scalar()

        stats["empty_runs"] = (await db.execute(text(
            "SELECT COUNT(*) FROM alpha_mining_runs r "
            "WHERE NOT EXISTS (SELECT 1 FROM alpha_factors f WHERE f.mining_run_id = r.id)"
        ))).scalar()

        stats["experiences"] = (await db.execute(text(
            "SELECT COUNT(*) FROM alpha_experiences"
        ))).scalar()

        stats["orphan_experiences"] = (await db.execute(text(
            "SELECT COUNT(*) FROM alpha_experiences e "
            "WHERE e.factor_id IS NOT NULL "
            "AND NOT EXISTS (SELECT 1 FROM alpha_factors f WHERE f.id = e.factor_id)"
        ))).scalar()

    return stats


# ── 실행 ────────────────────────────────────────────────────


async def _execute_cleanup(keep_top: int | None = None) -> dict:
    """실제 삭제 실행. 반환: 삭제 통계."""
    from app.core.database import async_session

    protected = await _collect_protected_ids()
    result: dict = {"protected": len(protected)}
    BATCH = 1000

    async with async_session() as db:
        # 1) mirage 삭제
        mirage_ids = [r[0] for r in (await db.execute(text(
            "SELECT id FROM alpha_factors WHERE status = 'mirage'"
        ))).fetchall()]
        mirage_ids = [fid for fid in mirage_ids if fid not in protected]
        deleted = 0
        for i in range(0, len(mirage_ids), BATCH):
            batch = mirage_ids[i:i + BATCH]
            await db.execute(text(
                "DELETE FROM alpha_factors WHERE id = ANY(:ids)"
            ), {"ids": batch})
            deleted += len(batch)
        await db.commit()
        result["mirage_deleted"] = deleted
        logger.info("mirage 삭제: %d개", deleted)

        # 2) population + IC 미달
        bad_ic_ids = [r[0] for r in (await db.execute(text(
            "SELECT id FROM alpha_factors "
            "WHERE status = 'population' AND (ic_mean < 0.03 OR ic_mean IS NULL)"
        ))).fetchall()]
        bad_ic_ids = [fid for fid in bad_ic_ids if fid not in protected]
        deleted = 0
        for i in range(0, len(bad_ic_ids), BATCH):
            batch = bad_ic_ids[i:i + BATCH]
            await db.execute(text(
                "DELETE FROM alpha_factors WHERE id = ANY(:ids)"
            ), {"ids": batch})
            deleted += len(batch)
        await db.commit()
        result["pop_bad_ic_deleted"] = deleted
        logger.info("population IC 미달 삭제: %d개", deleted)

        # 3) population + Sharpe 미달 (IC는 통과)
        bad_sharpe_ids = [r[0] for r in (await db.execute(text(
            "SELECT id FROM alpha_factors "
            "WHERE status = 'population' AND ic_mean >= 0.03 "
            "AND (sharpe < 0.3 OR sharpe IS NULL)"
        ))).fetchall()]
        bad_sharpe_ids = [fid for fid in bad_sharpe_ids if fid not in protected]
        deleted = 0
        for i in range(0, len(bad_sharpe_ids), BATCH):
            batch = bad_sharpe_ids[i:i + BATCH]
            await db.execute(text(
                "DELETE FROM alpha_factors WHERE id = ANY(:ids)"
            ), {"ids": batch})
            deleted += len(batch)
        await db.commit()
        result["pop_bad_sharpe_deleted"] = deleted
        logger.info("population Sharpe 미달 삭제: %d개", deleted)

        # 4) 중복 수식 정리 (expression_hash 기준, fitness 최고만 보존)
        dup_rows = (await db.execute(text(
            "SELECT expression_hash, COUNT(*) as cnt "
            "FROM alpha_factors "
            "WHERE expression_hash IS NOT NULL "
            "GROUP BY expression_hash HAVING COUNT(*) > 1"
        ))).fetchall()

        dup_deleted = 0
        for row in dup_rows:
            ehash = row[0]
            # fitness 최고 1개만 보존, 나머지 삭제
            keep_id = (await db.execute(text(
                "SELECT id FROM alpha_factors "
                "WHERE expression_hash = :h "
                "ORDER BY fitness_composite DESC NULLS LAST, created_at DESC "
                "LIMIT 1"
            ), {"h": ehash})).scalar()
            if keep_id:
                del_ids = [r[0] for r in (await db.execute(text(
                    "SELECT id FROM alpha_factors "
                    "WHERE expression_hash = :h AND id != :keep"
                ), {"h": ehash, "keep": keep_id})).fetchall()]
                del_ids = [fid for fid in del_ids if fid not in protected]
                if del_ids:
                    await db.execute(text(
                        "DELETE FROM alpha_factors WHERE id = ANY(:ids)"
                    ), {"ids": del_ids})
                    dup_deleted += len(del_ids)
        await db.commit()
        result["duplicate_deleted"] = dup_deleted
        logger.info("중복 수식 삭제: %d개", dup_deleted)

        # 5) keep_top: population 상위 N개만 보존
        if keep_top is not None:
            pop_excess_ids = [r[0] for r in (await db.execute(text(
                "SELECT id FROM alpha_factors "
                "WHERE status = 'population' "
                "ORDER BY fitness_composite DESC NULLS LAST "
                "OFFSET :offset"
            ), {"offset": keep_top})).fetchall()]
            pop_excess_ids = [fid for fid in pop_excess_ids if fid not in protected]
            deleted = 0
            for i in range(0, len(pop_excess_ids), BATCH):
                batch = pop_excess_ids[i:i + BATCH]
                await db.execute(text(
                    "DELETE FROM alpha_factors WHERE id = ANY(:ids)"
                ), {"ids": batch})
                deleted += len(batch)
            await db.commit()
            result["pop_excess_deleted"] = deleted
            logger.info("population 상위 %d개 이후 삭제: %d개", keep_top, deleted)

        # 6) 고아 경험 메모리 정리
        orphan_del = (await db.execute(text(
            "DELETE FROM alpha_experiences e "
            "WHERE e.factor_id IS NOT NULL "
            "AND NOT EXISTS (SELECT 1 FROM alpha_factors f WHERE f.id = e.factor_id)"
        ))).rowcount
        await db.commit()
        result["orphan_experiences_deleted"] = orphan_del
        logger.info("고아 경험 메모리 삭제: %d개", orphan_del)

        # 7) 빈 마이닝 런 정리
        empty_run_del = (await db.execute(text(
            "DELETE FROM alpha_mining_runs r "
            "WHERE NOT EXISTS (SELECT 1 FROM alpha_factors f WHERE f.mining_run_id = r.id)"
        ))).rowcount
        await db.commit()
        result["empty_runs_deleted"] = empty_run_del
        logger.info("빈 마이닝 런 삭제: %d개", empty_run_del)

        # 최종 카운트
        result["remaining_total"] = (await db.execute(text(
            "SELECT COUNT(*) FROM alpha_factors"
        ))).scalar()
        result["remaining_validated"] = (await db.execute(text(
            "SELECT COUNT(*) FROM alpha_factors WHERE status = 'validated'"
        ))).scalar()
        result["remaining_population"] = (await db.execute(text(
            "SELECT COUNT(*) FROM alpha_factors WHERE status = 'population'"
        ))).scalar()

    return result


async def main() -> None:
    parser = argparse.ArgumentParser(description="알파 팩터 정리")
    parser.add_argument("--dry-run", action="store_true", help="삭제 대상만 집계")
    parser.add_argument("--execute", action="store_true", help="실제 삭제 실행")
    parser.add_argument("--keep-top", type=int, default=None,
                        help="population 상위 N개만 보존 (나머지 삭제)")
    args = parser.parse_args()

    if not args.dry_run and not args.execute:
        parser.print_help()
        sys.exit(1)

    if args.dry_run:
        logger.info("=== DRY RUN: 삭제 대상 집계 ===")
        stats = await _count_targets()
        protected = await _collect_protected_ids()

        print("\n" + "=" * 60)
        print(f"총 팩터: {stats['total']:,}")
        print(f"보호 대상 (참조 중): {len(protected)}")
        print("-" * 60)
        print(f"삭제 대상:")
        print(f"  mirage (인과 실패):          {stats['mirage']:,}")
        print(f"  population IC 미달:          {stats['pop_bad_ic']:,}")
        print(f"  population Sharpe 미달:      {stats['pop_bad_sharpe']:,}")
        print(f"  중복 수식 초과분:             {stats['duplicate_excess']:,}")
        total_del = stats["mirage"] + stats["pop_bad_ic"] + stats["pop_bad_sharpe"]
        print(f"  ─────────────────────────────")
        print(f"  소계 (중복 제외):             ~{total_del:,}")
        print("-" * 60)
        print(f"보존 대상:")
        print(f"  validated:                   {stats['validated']:,}")
        print(f"  population 품질 통과:         {stats['pop_good']:,}")
        print(f"  ─────────────────────────────")
        print(f"  예상 잔여:                    ~{stats['validated'] + stats['pop_good']:,}")
        print("-" * 60)
        print(f"마이닝 런: {stats['mining_runs']:,} (빈 런: {stats['empty_runs']:,})")
        print(f"경험 메모리: {stats['experiences']:,} (고아: {stats['orphan_experiences']:,})")
        print("=" * 60)

    if args.execute:
        logger.info("=== EXECUTE: 팩터 정리 시작 ===")
        result = await _execute_cleanup(keep_top=args.keep_top)

        print("\n" + "=" * 60)
        print("삭제 결과:")
        print(f"  보호 팩터:              {result['protected']}")
        print(f"  mirage 삭제:            {result['mirage_deleted']:,}")
        print(f"  population IC 미달:     {result['pop_bad_ic_deleted']:,}")
        print(f"  population Sharpe 미달: {result['pop_bad_sharpe_deleted']:,}")
        print(f"  중복 수식:              {result['duplicate_deleted']:,}")
        if "pop_excess_deleted" in result:
            print(f"  population 초과분:      {result['pop_excess_deleted']:,}")
        print(f"  고아 경험 메모리:       {result['orphan_experiences_deleted']:,}")
        print(f"  빈 마이닝 런:           {result['empty_runs_deleted']:,}")
        print("-" * 60)
        total_del = (result["mirage_deleted"] + result["pop_bad_ic_deleted"]
                     + result["pop_bad_sharpe_deleted"] + result["duplicate_deleted"]
                     + result.get("pop_excess_deleted", 0))
        print(f"  총 삭제 팩터:           {total_del:,}")
        print(f"  잔여 총 팩터:           {result['remaining_total']:,}")
        print(f"    validated:            {result['remaining_validated']:,}")
        print(f"    population:           {result['remaining_population']:,}")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
