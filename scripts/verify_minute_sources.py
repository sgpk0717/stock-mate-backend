"""키움 vs KIS 분봉 데이터 일치 검증.

키움이 수집하여 DB에 저장한 분봉 데이터와, KIS API에서 동일 기간을
새로 조회한 분봉 데이터를 비교하여 OHLCV 일치 여부를 확인한다.

Usage:
    docker-compose run --rm app python -m scripts.verify_minute_sources \
        --symbols 000660,000150 --start 20250714 --end 20250718
"""

import argparse
import asyncio
import logging
from datetime import datetime, timedelta, timezone

import polars as pl
from sqlalchemy import text

from app.core.database import async_session
from app.trading.kis_client import get_kis_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9))


# ── DB 조회 (키움이 넣은 데이터) ─────────────────────────

async def load_db_candles(symbol: str, start: str, end: str) -> pl.DataFrame:
    """DB에서 키움이 수집한 1분봉을 Polars DataFrame으로 반환."""
    start_dt = datetime.strptime(start, "%Y%m%d").replace(tzinfo=KST)
    # end 날짜의 장마감 시각(16:00)까지 포함
    end_dt = datetime.strptime(end, "%Y%m%d").replace(
        hour=16, minute=0, second=0, tzinfo=KST,
    )

    async with async_session() as db:
        result = await db.execute(
            text("""
                SELECT dt, open, high, low, close, volume
                FROM stock_candles
                WHERE symbol = :symbol
                  AND interval = '1m'
                  AND dt >= :start
                  AND dt <= :end
                ORDER BY dt
            """),
            {"symbol": symbol, "start": start_dt, "end": end_dt},
        )
        rows = result.fetchall()

    if not rows:
        return pl.DataFrame(schema={
            "dt": pl.Utf8, "open": pl.Float64, "high": pl.Float64,
            "low": pl.Float64, "close": pl.Float64, "volume": pl.Int64,
        })

    data = []
    for r in rows:
        dt_val = r[0]
        if hasattr(dt_val, "astimezone"):
            # DB는 UTC로 저장 → KST로 변환 후 포맷
            dt_kst = dt_val.astimezone(KST)
            dt_str = dt_kst.strftime("%Y%m%d%H%M%S")
        elif hasattr(dt_val, "strftime"):
            dt_str = dt_val.strftime("%Y%m%d%H%M%S")
        else:
            dt_str = str(dt_val)
        data.append({
            "dt": dt_str,
            "open": float(r[1]),
            "high": float(r[2]),
            "low": float(r[3]),
            "close": float(r[4]),
            "volume": int(r[5]),
        })

    return pl.DataFrame(data).with_columns(
        pl.col("volume").cast(pl.Int64),
    )


# ── KIS API 조회 ─────────────────────────────────────────

async def load_kis_candles(
    client, symbol: str, start: str, end: str,
) -> pl.DataFrame:
    """KIS API에서 동일 기간의 1분봉을 조회하여 Polars DataFrame으로 반환."""
    all_rows: list[dict] = []
    date = end
    hour = "160000"

    while True:
        # retry on transient errors (500, timeout)
        candles = None
        for attempt in range(3):
            try:
                candles, next_date, next_hour = await client.get_minute_candles(
                    symbol, date, hour,
                )
                break
            except Exception as e:
                logger.warning("  KIS API error (attempt %d): %s", attempt + 1, e)
                if attempt < 2:
                    await asyncio.sleep(1)
                else:
                    return pl.DataFrame(schema={
                        "dt": pl.Utf8, "open": pl.Float64, "high": pl.Float64,
                        "low": pl.Float64, "close": pl.Float64, "volume": pl.Int64,
                    })
        if not candles:
            break

        for c in candles:
            dt_str = c.get("stck_bsop_date", "") + c.get("stck_cntg_hour", "")
            if len(dt_str) != 14:
                continue
            cl = float(c.get("stck_prpr", 0) or 0)
            if cl <= 0:
                continue

            # 시작일 이전이면 중단
            if dt_str[:8] < start:
                # 이 배치에서 start 이전 데이터도 들어올 수 있으므로 필터
                continue

            all_rows.append({
                "dt": dt_str,
                "open": float(c.get("stck_oprc", 0) or 0),
                "high": float(c.get("stck_hgpr", 0) or 0),
                "low": float(c.get("stck_lwpr", 0) or 0),
                "close": cl,
                "volume": int(c.get("cntg_vol", 0) or 0),
            })

        # 페이지네이션 중단 조건
        if not next_date or not next_hour:
            break
        if next_date < start:
            break

        date, hour = next_date, next_hour

    if not all_rows:
        return pl.DataFrame(schema={
            "dt": pl.Utf8, "open": pl.Float64, "high": pl.Float64,
            "low": pl.Float64, "close": pl.Float64, "volume": pl.Int64,
        })

    return pl.DataFrame(all_rows).with_columns(
        pl.col("volume").cast(pl.Int64),
    ).sort("dt").unique(subset=["dt"], keep="first")


# ── 비교 로직 ────────────────────────────────────────────

def compare_sources(
    db_df: pl.DataFrame, kis_df: pl.DataFrame, symbol: str,
) -> bool:
    """두 DataFrame을 비교하고 불일치를 출력. 일치하면 True."""
    logger.info("=" * 60)
    logger.info("검증: %s", symbol)
    logger.info("  DB (키움):  %d건", len(db_df))
    logger.info("  KIS API:    %d건", len(kis_df))

    if len(db_df) == 0:
        logger.warning("  DB에 데이터 없음 — 이 종목은 아직 키움 수집 안됨")
        return False

    if len(kis_df) == 0:
        logger.warning("  KIS API 데이터 없음")
        return False

    # 건수 차이
    count_diff = abs(len(db_df) - len(kis_df))
    if count_diff > 0:
        logger.warning("  건수 차이: %d건", count_diff)

    # dt 기준으로 inner join
    merged = db_df.join(
        kis_df,
        on="dt",
        how="inner",
        suffix="_kis",
    )
    logger.info("  공통 타임스탬프: %d건", len(merged))

    # DB에만 있는 dt
    db_dts = set(db_df["dt"].to_list())
    kis_dts = set(kis_df["dt"].to_list())
    db_only_dts = db_dts - kis_dts
    kis_only_dts = kis_dts - db_dts

    db_only = db_df.filter(pl.col("dt").is_in(list(db_only_dts)))
    if len(db_only) > 0:
        logger.warning("  DB에만 있는 봉: %d건", len(db_only))
        logger.warning("    예시: %s", db_only.head(5).to_dicts())

    # KIS에만 있는 dt
    kis_only = kis_df.filter(pl.col("dt").is_in(list(kis_only_dts)))
    if len(kis_only) > 0:
        logger.warning("  KIS에만 있는 봉: %d건", len(kis_only))
        logger.warning("    예시: %s", kis_only.head(5).to_dicts())

    if len(merged) == 0:
        logger.error("  공통 타임스탬프가 없음 — 완전 불일치!")
        return False

    # OHLCV 비교
    all_match = True
    for col in ["open", "high", "low", "close"]:
        diff = merged.filter(pl.col(col) != pl.col(f"{col}_kis"))
        if len(diff) > 0:
            all_match = False
            logger.warning(
                "  %s 불일치: %d건 (%.1f%%)",
                col.upper(), len(diff), len(diff) / len(merged) * 100,
            )
            sample = diff.select([
                "dt",
                pl.col(col).alias(f"db_{col}"),
                pl.col(f"{col}_kis").alias(f"kis_{col}"),
                (pl.col(col) - pl.col(f"{col}_kis")).alias("diff"),
            ]).head(10)
            logger.warning("    예시:\n%s", sample)

    vol_diff = merged.filter(pl.col("volume") != pl.col("volume_kis"))
    if len(vol_diff) > 0:
        all_match = False
        logger.warning(
            "  VOLUME 불일치: %d건 (%.1f%%)",
            len(vol_diff), len(vol_diff) / len(merged) * 100,
        )
        sample = vol_diff.select([
            "dt",
            pl.col("volume").alias("db_vol"),
            pl.col("volume_kis").alias("kis_vol"),
            (pl.col("volume") - pl.col("volume_kis")).alias("diff"),
        ]).head(10)
        logger.warning("    예시:\n%s", sample)

    # 최종 결과
    if all_match and count_diff == 0 and len(db_only) == 0 and len(kis_only) == 0:
        logger.info("  ✅ 완전 일치!")
        return True
    elif all_match and (len(db_only) > 0 or len(kis_only) > 0):
        logger.info("  ⚠️ OHLCV 값은 일치하나 건수/타임스탬프 차이 있음")
        return True  # 값은 일치하므로 경계선 병합에는 문제 없음
    else:
        logger.error("  ❌ OHLCV 값 불일치 발견!")
        return False


# ── 메인 ─────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="키움 vs KIS 분봉 데이터 일치 검증")
    parser.add_argument(
        "--symbols", required=True,
        help="검증할 종목코드 (콤마 구분, 예: 000660,000150)",
    )
    parser.add_argument("--start", required=True, help="시작일 YYYYMMDD")
    parser.add_argument("--end", required=True, help="종료일 YYYYMMDD")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]
    client = get_kis_client(is_mock=False)

    logger.info("=== 키움 vs KIS 분봉 데이터 검증 ===")
    logger.info("  종목: %s", symbols)
    logger.info("  기간: %s ~ %s", args.start, args.end)

    results = {}
    for sym in symbols:
        logger.info("\n%s 데이터 로딩...", sym)

        # 1. DB에서 키움 데이터 로드
        db_df = await load_db_candles(sym, args.start, args.end)
        logger.info("  DB 로드 완료: %d건", len(db_df))

        # 2. KIS API에서 데이터 로드
        kis_df = await load_kis_candles(client, sym, args.start, args.end)
        logger.info("  KIS 로드 완료: %d건", len(kis_df))

        # 3. 비교
        ok = compare_sources(db_df, kis_df, sym)
        results[sym] = ok

    await client.close()

    # 최종 요약
    logger.info("\n" + "=" * 60)
    logger.info("=== 최종 결과 ===")
    all_ok = True
    for sym, ok in results.items():
        status = "PASS" if ok else "FAIL"
        logger.info("  %s: %s", sym, status)
        if not ok:
            all_ok = False

    if all_ok:
        logger.info("\n모든 종목 검증 통과! 경계선 병합 안전합니다.")
    else:
        logger.error("\n불일치 발견! 원인 분석 후 보정 필요.")


if __name__ == "__main__":
    asyncio.run(main())
