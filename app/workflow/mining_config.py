"""워크플로우 마이닝 설정 (Redis runtime override).

interval 변경 시 CPCV/모집단/세대 수가 프리셋에 따라 자동 조정된다.
Redis에 설정이 없으면 settings.WORKFLOW_DATA_INTERVAL 기반 fallback.
"""

from __future__ import annotations

import logging

from app.core import redis as rd
from app.core.config import settings

logger = logging.getLogger(__name__)

REDIS_KEY = "workflow:mining_config"

INTERVAL_PRESETS: dict[str, dict] = {
    "5m": {
        "cpcv_n_groups": 5,
        "cpcv_n_test": 2,
        "cpcv_embargo_days": 5,
        "population_size": 300,
        "max_iterations": 30,
    },
    "1d": {
        "cpcv_n_groups": 10,
        "cpcv_n_test": 3,
        "cpcv_embargo_days": 10,
        "population_size": 750,
        "max_iterations": 100,
    },
}


async def get_mining_config() -> dict:
    """현재 마이닝 설정 반환 (Redis → Settings fallback)."""
    raw = await rd.hgetall(REDIS_KEY)
    if raw and "interval" in raw:
        return {
            "interval": raw["interval"],
            "cpcv_n_groups": int(raw.get("cpcv_n_groups", 5)),
            "cpcv_n_test": int(raw.get("cpcv_n_test", 2)),
            "cpcv_embargo_days": int(raw.get("cpcv_embargo_days", 5)),
            "population_size": int(raw.get("population_size", 300)),
            "max_iterations": int(raw.get("max_iterations", 30)),
        }
    # fallback: .env의 WORKFLOW_DATA_INTERVAL 기반 프리셋
    fallback_interval = settings.WORKFLOW_DATA_INTERVAL
    preset = INTERVAL_PRESETS.get(fallback_interval, INTERVAL_PRESETS["5m"])
    return {"interval": fallback_interval, **preset}


async def set_mining_config(interval: str) -> dict:
    """마이닝 interval 변경 → 프리셋 자동 적용 → Redis 저장."""
    if interval not in INTERVAL_PRESETS:
        raise ValueError(f"지원하지 않는 interval: {interval}. 가능: {list(INTERVAL_PRESETS)}")
    preset = INTERVAL_PRESETS[interval]
    config = {"interval": interval, **preset}
    await rd.hset(REDIS_KEY, config)
    logger.info("마이닝 설정 변경: interval=%s → %s", interval, config)

    # 이전 interval 로그 초기화 (DB factory_status.log_lines 클리어)
    try:
        from app.core.database import async_session
        from app.models.base import WorkerState
        from sqlalchemy import select, update as sa_update

        async with async_session() as db:
            result = await db.execute(select(WorkerState).where(WorkerState.id == 1))
            state = result.scalar_one_or_none()
            if state and state.factory_status:
                fs = dict(state.factory_status)
                fs["log_lines"] = []
                await db.execute(
                    sa_update(WorkerState).where(WorkerState.id == 1).values(factory_status=fs)
                )
                await db.commit()
    except Exception as e:
        logger.warning("log_lines 초기화 실패: %s", e)

    return config
