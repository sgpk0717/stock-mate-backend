"""알파 팩토리 전체 파라미터 조합 테스트.

화면에서 입력 가능한 모든 파라미터 경우의 수를 조합하여
팩토리 1사이클을 실행하고 에러 없이 완료되는지 검증한다.

Usage:
    docker-compose run --rm app python -m scripts.test_factory_params
    docker-compose run --rm app python -m scripts.test_factory_params --quick
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import logging
import sys
import time
import traceback
from datetime import date, timedelta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
# SQLAlchemy 로그 억제
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
logging.getLogger("pykrx").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("test_factory")


# ── 테스트 파라미터 공간 ──

UNIVERSES = ["KOSPI200", "KOSDAQ150", "KRX300", "ALL"]
IC_THRESHOLDS = [0.0, 0.03, 1.0]
ENABLE_CROSSOVER = [True, False]
MAX_CYCLES_OPTIONS = [1]  # null은 1로 제한 (무한 루프 방지)

# 날짜 범위
DATE_RANGES = [
    ("2025-11-01", "2025-12-01"),  # 짧은 기간 (30일)
    ("2024-01-01", "2025-12-31"),  # 정상 기간 (2년)
]

# 빠른 테스트용 (핵심 조합만)
QUICK_COMBOS = [
    # (universe, ic_threshold, crossover, start, end, description)
    ("KOSPI200", 0.03, True, "2024-01-01", "2025-12-31", "기본 설정"),
    ("KOSPI200", 0.0, True, "2024-01-01", "2025-12-31", "IC=0 (극히 관대)"),
    ("KOSPI200", 1.0, True, "2024-01-01", "2025-12-31", "IC=1.0 (극히 엄격)"),
    ("KOSPI200", 0.03, False, "2024-01-01", "2025-12-31", "교차 비활성"),
    ("KOSDAQ150", 0.03, True, "2024-01-01", "2025-12-31", "KOSDAQ150"),
    ("KRX300", 0.03, True, "2024-01-01", "2025-12-31", "KRX300"),
    ("ALL", 0.03, True, "2024-01-01", "2025-12-31", "ALL 유니버스"),
    ("KOSPI200", 0.03, True, "2025-11-01", "2025-12-01", "짧은 기간 (30일)"),
    ("KOSPI200", 0.03, False, "2025-11-01", "2025-12-01", "짧은+교차X"),
]


async def run_single_test(
    test_num: int,
    total: int,
    universe: str,
    ic_threshold: float,
    enable_crossover: bool,
    start_date: str,
    end_date: str,
    description: str = "",
) -> dict:
    """단일 팩토리 테스트 실행."""
    from app.alpha.scheduler import AlphaFactoryScheduler

    desc = description or f"u={universe} ic={ic_threshold} cx={enable_crossover} {start_date}~{end_date}"
    logger.info("=" * 60)
    logger.info("[%d/%d] %s", test_num, total, desc)
    logger.info("=" * 60)

    # 매 테스트마다 새 스케줄러 인스턴스 (싱글턴 오염 방지)
    scheduler = AlphaFactoryScheduler()

    t0 = time.time()
    result = {
        "test_num": test_num,
        "description": desc,
        "universe": universe,
        "ic_threshold": ic_threshold,
        "enable_crossover": enable_crossover,
        "start_date": start_date,
        "end_date": end_date,
        "status": "UNKNOWN",
        "error": None,
        "elapsed_sec": 0,
        "factors_found": 0,
    }

    try:
        started = await scheduler.start(
            context="파라미터 조합 테스트",
            universe=universe,
            start_date=start_date,
            end_date=end_date,
            interval_minutes=0,  # 연속 (바로 종료)
            max_iterations=3,    # 빠른 테스트를 위해 3회
            ic_threshold=ic_threshold,
            orthogonality_threshold=0.7,
            enable_crossover=enable_crossover,
            max_cycles=1,        # 1사이클만
        )

        if not started:
            result["status"] = "SKIP_ALREADY_RUNNING"
            logger.warning("[%d/%d] 스킵: 이미 실행 중", test_num, total)
            return result

        # 사이클 완료 대기 (최대 10분)
        timeout = 600
        elapsed = 0
        while scheduler._state.running and elapsed < timeout:
            await asyncio.sleep(2)
            elapsed += 2

        if elapsed >= timeout:
            await scheduler.stop()
            result["status"] = "TIMEOUT"
            result["error"] = f"Timeout after {timeout}s"
            logger.error("[%d/%d] TIMEOUT", test_num, total)
        else:
            status = scheduler.get_status()
            result["status"] = "PASS"
            result["factors_found"] = status.get("factors_discovered_total", 0)
            logger.info(
                "[%d/%d] PASS — factors=%d, gen=%d, pop=%d",
                test_num, total,
                status.get("factors_discovered_total", 0),
                status.get("generation", 0),
                status.get("population_size", 0),
            )

    except Exception as e:
        result["status"] = "FAIL"
        result["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        logger.error("[%d/%d] FAIL: %s", test_num, total, result["error"])
        traceback.print_exc()
        # 스케줄러 정리
        try:
            await scheduler.stop()
        except Exception:
            pass

    result["elapsed_sec"] = round(time.time() - t0, 1)
    return result


async def _init_stock_cache() -> None:
    """테스트 전 stock_master 캐시 로딩."""
    from app.core.database import async_session
    from app.core.stock_master import load_stock_cache

    async with async_session() as db:
        await load_stock_cache(db)


async def main(quick: bool = False) -> None:
    """전체 파라미터 조합 테스트 실행."""
    # stock_master 캐시 로딩 (universe 리졸버 DB fallback 지원)
    await _init_stock_cache()

    if quick:
        combos = [
            (i + 1, len(QUICK_COMBOS), *combo) for i, combo in enumerate(QUICK_COMBOS)
        ]
    else:
        # 전체 조합 생성
        all_combos = list(itertools.product(
            UNIVERSES, IC_THRESHOLDS, ENABLE_CROSSOVER, DATE_RANGES,
        ))
        combos = []
        for i, (univ, ic, cx, (sd, ed)) in enumerate(all_combos):
            desc = f"u={univ} ic={ic} cx={cx} {sd}~{ed}"
            combos.append((i + 1, len(all_combos), univ, ic, cx, sd, ed, desc))

    total = len(combos)
    logger.info("=" * 60)
    logger.info("알파 팩토리 파라미터 조합 테스트 시작: %d 개 조합", total)
    logger.info("모드: %s", "QUICK" if quick else "FULL")
    logger.info("=" * 60)

    results = []
    for combo in combos:
        r = await run_single_test(*combo)
        results.append(r)

    # 결과 요약
    logger.info("")
    logger.info("=" * 60)
    logger.info("테스트 결과 요약")
    logger.info("=" * 60)

    passed = [r for r in results if r["status"] == "PASS"]
    failed = [r for r in results if r["status"] == "FAIL"]
    timeout = [r for r in results if r["status"] == "TIMEOUT"]
    skipped = [r for r in results if r["status"] == "SKIP_ALREADY_RUNNING"]

    logger.info("PASS: %d / %d", len(passed), total)
    logger.info("FAIL: %d / %d", len(failed), total)
    logger.info("TIMEOUT: %d / %d", len(timeout), total)
    logger.info("SKIPPED: %d / %d", len(skipped), total)

    if failed:
        logger.error("")
        logger.error("--- 실패 상세 ---")
        for r in failed:
            logger.error("[#%d] %s", r["test_num"], r["description"])
            logger.error("     Error: %s", r["error"])

    if timeout:
        logger.warning("")
        logger.warning("--- 타임아웃 ---")
        for r in timeout:
            logger.warning("[#%d] %s (%.1fs)", r["test_num"], r["description"], r["elapsed_sec"])

    total_elapsed = sum(r["elapsed_sec"] for r in results)
    logger.info("")
    logger.info("총 소요 시간: %.1f초", total_elapsed)

    if failed:
        sys.exit(1)
    else:
        logger.info("모든 테스트 통과!")
        sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="알파 팩토리 파라미터 조합 테스트")
    parser.add_argument(
        "--quick", action="store_true",
        help="빠른 테스트 (핵심 10개 조합만)",
    )
    args = parser.parse_args()
    asyncio.run(main(quick=args.quick))
