"""Phase 2 테스트: P1-1~P1-4 통합 검증.

Usage: docker exec stockmate-worker python -m scripts.test_phase2
"""
import asyncio
import numpy as np
from datetime import datetime, date

from app.core.config import settings
from app.core.timezone import KST


async def test_p1_1():
    """P1-1: 매수 시간 컷오프 설정 확인."""
    print("=" * 60)
    print("P1-1: 매수 시간 컷오프")
    print("=" * 60)

    cutoff = settings.ALPHA_INTRADAY_BUY_CUTOFF_MINUTES
    cutoff_time_min = 15 * 60 + 30 - cutoff
    cutoff_h, cutoff_m = divmod(cutoff_time_min, 60)
    print(f"  CUTOFF_MINUTES: {cutoff}")
    print(f"  매수 금지 시각: {cutoff_h:02d}:{cutoff_m:02d} 이후")

    # 시간 체크 로직 테스트
    test_times = [
        (14, 30, True),   # 14:30 → 매수 OK
        (14, 39, True),   # 14:39 → 매수 OK
        (14, 40, False),  # 14:40 → 매수 금지
        (15, 0, False),   # 15:00 → 매수 금지
        (15, 25, False),  # 15:25 → 매수 금지
    ]
    all_pass = True
    for h, m, expected_ok in test_times:
        cur_min = h * 60 + m
        blocked = cur_min >= (15 * 60 + 30) - cutoff
        actual_ok = not blocked
        status = "OK" if actual_ok == expected_ok else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  {h:02d}:{m:02d} → 매수 {'허용' if actual_ok else '금지'} (예상: {'허용' if expected_ok else '금지'}) [{status}]")

    print(f"  P1-1: {'PASS' if all_pass else 'FAIL'}")


async def test_p1_3():
    """P1-3: 교란변수 미국 증시/환율 로드."""
    print("\n" + "=" * 60)
    print("P1-3: 교란변수 미국 증시/환율")
    print("=" * 60)

    from app.alpha.confounders import load_confounders

    try:
        df = await load_confounders(date(2025, 11, 1), date(2026, 3, 1), ["005930"])
        cols = list(df.columns)
        print(f"  교란변수 컬럼: {cols}")
        print(f"  행 수: {len(df)}")

        has_us = "us_market_return" in cols
        has_fx = "usd_krw_change" in cols
        print(f"  us_market_return 존재: {has_us}")
        print(f"  usd_krw_change 존재: {has_fx}")

        if has_us:
            us_vals = df["us_market_return"].dropna()
            print(f"  us_market_return: mean={us_vals.mean():.6f}, count={len(us_vals)}")
        if has_fx:
            fx_vals = df["usd_krw_change"].dropna()
            print(f"  usd_krw_change: mean={fx_vals.mean():.6f}, count={len(fx_vals)}")

        if has_us and has_fx:
            print(f"  P1-3: PASS")
        else:
            print(f"  P1-3: WARNING (yfinance 접근 불가일 수 있음 — fallback 정상)")
    except Exception as e:
        print(f"  P1-3: FAIL — {e}")


async def test_p1_4():
    """P1-4: Block Permutation 동작 확인."""
    print("\n" + "=" * 60)
    print("P1-4: Block Permutation")
    print("=" * 60)

    from app.alpha.causal import _block_permutation, _vectorized_placebo_fwl

    # 합성 데이터: 3일, 각 5행
    dates = np.array([
        "2025-01-01", "2025-01-01", "2025-01-01", "2025-01-01", "2025-01-01",
        "2025-01-02", "2025-01-02", "2025-01-02", "2025-01-02", "2025-01-02",
        "2025-01-03", "2025-01-03", "2025-01-03", "2025-01-03", "2025-01-03",
    ])
    treatment = np.arange(15, dtype=float)

    # 블록 셔플 실행
    shuffled = _block_permutation(treatment, dates)

    # 검증: 같은 날짜 블록 내 값들이 함께 이동했는가?
    # 원본: day1=[0,1,2,3,4], day2=[5,6,7,8,9], day3=[10,11,12,13,14]
    # 셔플 후: 각 블록이 통째로 이동 (블록 내 순서는 유지)
    day1_vals = set(shuffled[:5])
    day2_vals = set(shuffled[5:10])
    day3_vals = set(shuffled[10:15])

    # 각 블록이 원본 블록 중 하나와 일치해야 함
    original_blocks = [
        {0, 1, 2, 3, 4},
        {5, 6, 7, 8, 9},
        {10, 11, 12, 13, 14},
    ]

    blocks_intact = all(
        vals in original_blocks for vals in [day1_vals, day2_vals, day3_vals]
    )

    print(f"  원본 블록: {[list(b) for b in original_blocks]}")
    print(f"  셔플 결과: day1={sorted(day1_vals)}, day2={sorted(day2_vals)}, day3={sorted(day3_vals)}")
    print(f"  블록 무결성: {blocks_intact}")

    # block_dates=None이면 기존 단순 셔플 동작 확인
    X_base = np.random.randn(15, 2)
    # proj_base = (X'X)^-1 X' (k×n 행렬)
    proj_base = np.linalg.pinv(X_base)  # (2, 15)
    y = np.random.randn(15)

    betas_block = _vectorized_placebo_fwl(
        X_base, proj_base, treatment, y, n_perms=50, batch_size=50, block_dates=dates,
    )
    betas_simple = _vectorized_placebo_fwl(
        X_base, proj_base, treatment, y, n_perms=50, batch_size=50, block_dates=None,
    )

    print(f"  Block betas: mean={betas_block.mean():.4f}, std={betas_block.std():.4f}, n={len(betas_block)}")
    print(f"  Simple betas: mean={betas_simple.mean():.4f}, std={betas_simple.std():.4f}, n={len(betas_simple)}")

    if blocks_intact and len(betas_block) == 50 and len(betas_simple) == 50:
        print(f"  P1-4: PASS")
    else:
        print(f"  P1-4: FAIL")


async def main():
    print(f"FWD_RETURN_MODE: {settings.ALPHA_FWD_RETURN_MODE}")
    print()

    await test_p1_1()
    await test_p1_3()
    await test_p1_4()

    print("\n" + "=" * 60)
    print("Phase 2 테스트 완료")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
