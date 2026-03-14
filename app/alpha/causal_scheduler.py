"""미검증 알파 팩터 주기적 인과 검증 스케줄러.

1시간마다 실행:
- 현재 검증 진행 중이거나 대기열이 있으면 스킵
- IC 기준 상위 30% 미검증 팩터만 선별하여 순차 검증
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from sqlalchemy import func, select, update

from app.alpha.causal_runner import validate_single_factor
from app.alpha.models import AlphaFactor
from app.core.database import async_session

logger = logging.getLogger(__name__)

_INTERVAL_SECONDS = 3600  # 1시간
_TOP_PERCENTILE = 0.3  # 상위 30%

# 모듈 레벨 상태
_running = False
_validating = False  # 현재 검증 진행 중 여부
_task: asyncio.Task | None = None


def is_validating() -> bool:
    """현재 검증이 진행 중인지 반환."""
    return _validating


async def _get_pending_factor_ids() -> list:
    """미검증 팩터 중 IC 상위 30%의 ID 목록을 반환한다."""
    async with async_session() as db:
        # 미검증 팩터 전체 수
        count_result = await db.execute(
            select(func.count(AlphaFactor.id)).where(
                AlphaFactor.causal_robust.is_(None),
                AlphaFactor.status == "discovered",
                AlphaFactor.ic_mean.is_not(None),
            )
        )
        total = count_result.scalar() or 0

        if total == 0:
            return []

        # IC 상위 30% 임계값 계산
        # percentile_disc(0.7) = 하위 70% 지점 = 상위 30% 시작점
        threshold_result = await db.execute(
            select(
                func.percentile_disc(0.7).within_group(AlphaFactor.ic_mean)
            ).where(
                AlphaFactor.causal_robust.is_(None),
                AlphaFactor.status == "discovered",
                AlphaFactor.ic_mean.is_not(None),
            )
        )
        ic_threshold = threshold_result.scalar()

        if ic_threshold is None:
            return []

        # 상위 30% 팩터 조회 (IC 높은 순)
        result = await db.execute(
            select(AlphaFactor.id)
            .where(
                AlphaFactor.causal_robust.is_(None),
                AlphaFactor.status == "discovered",
                AlphaFactor.ic_mean >= ic_threshold,
            )
            .order_by(AlphaFactor.ic_mean.desc())
        )
        factor_ids = [row[0] for row in result.fetchall()]

        logger.info(
            "Causal scheduler: %d unvalidated factors, IC threshold=%.4f, "
            "selected %d (top 30%%)",
            total, ic_threshold, len(factor_ids),
        )
        return factor_ids


async def _run_cycle() -> None:
    """1회 검증 사이클: 미검증 상위 30% 팩터를 순차 검증."""
    global _validating

    if _validating:
        logger.debug("Causal scheduler: validation already in progress, skipping")
        return

    _validating = True
    try:
        factor_ids = await _get_pending_factor_ids()
        if not factor_ids:
            logger.info("Causal scheduler: no pending factors to validate")
            return

        logger.info("Causal scheduler: starting validation of %d factors", len(factor_ids))

        confounders_cache: dict = {}
        candles_cache: dict = {}
        validated = 0
        failed = 0

        for i, fid in enumerate(factor_ids):
            if not _running:
                logger.info("Causal scheduler: stopped mid-cycle")
                break

            try:
                async with async_session() as db:
                    # 이미 검증된 경우 스킵
                    check = await db.execute(
                        select(AlphaFactor.causal_robust, AlphaFactor.status)
                        .where(AlphaFactor.id == fid)
                    )
                    row = check.fetchone()
                    if row and row[0] is not None:
                        logger.debug(
                            "Factor %s already validated, skipping", str(fid)[:8]
                        )
                        continue

                    await validate_single_factor(
                        fid, db,
                        confounders_cache=confounders_cache,
                        candles_cache=candles_cache,
                    )
                validated += 1
            except Exception as e:
                failed += 1
                logger.error(
                    "Causal scheduler [%d/%d] factor %s failed: %s",
                    i + 1, len(factor_ids), str(fid)[:8], str(e)[:200],
                )
                # causal_failed 마킹
                try:
                    async with async_session() as err_db:
                        await err_db.execute(
                            update(AlphaFactor)
                            .where(AlphaFactor.id == fid)
                            .values(causal_robust=False, status="causal_failed")
                        )
                        await err_db.commit()
                except Exception:
                    pass

            # 이벤트 루프 yield
            await asyncio.sleep(0)

        logger.info(
            "Causal scheduler cycle done: %d validated, %d failed, %d total",
            validated, failed, len(factor_ids),
        )
    finally:
        _validating = False


async def _scheduler_loop() -> None:
    """1시간마다 검증 사이클을 실행하는 메인 루프."""
    global _running
    _running = True
    logger.info("Causal validation scheduler started (interval=%ds)", _INTERVAL_SECONDS)

    while _running:
        try:
            await _run_cycle()
        except Exception as e:
            logger.error("Causal scheduler cycle error: %s", e)

        # 다음 사이클까지 대기 (1초 단위로 체크하여 빠른 종료 지원)
        for _ in range(_INTERVAL_SECONDS):
            if not _running:
                break
            await asyncio.sleep(1)


def start_causal_scheduler() -> asyncio.Task:
    """스케줄러 백그라운드 태스크 시작."""
    global _task
    _task = asyncio.create_task(_scheduler_loop())
    return _task


def stop_causal_scheduler() -> None:
    """스케줄러 중지."""
    global _running
    _running = False
    logger.info("Causal validation scheduler stopping...")
