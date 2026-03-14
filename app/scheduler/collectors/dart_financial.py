"""DART 재무 데이터 증분 수집.

분기 공시 기반 EPS/BPS/영업이익률/부채비율을 수집한다.
매일 실행되지만, 현재 연도의 최신 분기만 처리하므로
이미 수집 완료된 종목은 UPSERT로 스킵된다.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import date, timedelta
from typing import Callable

from sqlalchemy import text

from app.core.config import settings
from app.core.database import async_session
from app.scheduler.circuit_breaker import CircuitBreaker
from app.scheduler.schemas import CollectionResult

logger = logging.getLogger(__name__)

ProgressCb = Callable[[int, int, str], object] | None


def _current_quarter() -> tuple[int, str, str]:
    """현재 날짜 기준 가장 최근 공시 가능 분기 반환.

    Returns:
        (year, quarter_label, reprt_code)
    """
    today = date.today()
    month = today.month

    # 공시 타이밍 고려: 분기 마감 후 ~45일 후 공시
    # 1Q(3월말) → 5월 공시, 2Q(6월말) → 8월, 3Q(9월말) → 11월, 4Q(12월말) → 3월
    if month >= 11:
        return today.year, "3Q", "11014"
    elif month >= 8:
        return today.year, "2Q", "11012"
    elif month >= 5:
        return today.year, "1Q", "11013"
    else:
        return today.year - 1, "4Q", "11011"


async def _get_symbols_without_data(year: int, quarter: str) -> list[str]:
    """해당 분기 데이터가 아직 없는 종목 조회."""
    async with async_session() as db:
        result = await db.execute(
            text("""
                SELECT sm.symbol
                FROM stock_masters sm
                WHERE sm.market IN ('KOSPI', 'KOSDAQ')
                  AND NOT EXISTS (
                      SELECT 1 FROM dart_financials df
                      WHERE df.symbol = sm.symbol
                        AND df.fiscal_year = :year
                        AND df.fiscal_quarter = :quarter
                  )
                ORDER BY sm.symbol
            """),
            {"year": str(year), "quarter": quarter},
        )
        return [r[0] for r in result.fetchall()]


def _fetch_single(dart, symbol: str, year: int, reprt_code: str, quarter: str) -> dict | None:
    """단일 종목 재무 데이터 추출 (동기, thread에서 실행)."""
    import pandas as pd

    try:
        fs = dart.finstate(symbol, year, reprt_code=reprt_code)
        if fs is None or fs.empty:
            return None

        rcept_dt_str = fs["rcept_dt"].iloc[0] if "rcept_dt" in fs.columns else None
        if rcept_dt_str:
            disclosure_date = date(
                int(rcept_dt_str[:4]),
                int(rcept_dt_str[4:6]),
                int(rcept_dt_str[6:8]),
            )
        else:
            quarter_end_months = {"1Q": 3, "2Q": 6, "3Q": 9, "4Q": 12}
            month = quarter_end_months[quarter]
            fiscal_end = date(year, month, 28)
            disclosure_date = fiscal_end + timedelta(days=45)

        def _extract(df, *names):
            for name in names:
                mask = df["account_nm"].str.contains(name, na=False)
                rows = df[mask]
                if not rows.empty:
                    for col in ["thstrm_amount", "thstrm_dt"]:
                        if col in rows.columns:
                            val = rows[col].iloc[0]
                            if pd.notna(val):
                                try:
                                    return float(str(val).replace(",", ""))
                                except (ValueError, TypeError):
                                    continue
            return None

        eps = _extract(fs, "당기순이익", "주당순이익")
        bps = _extract(fs, "자본총계", "주당순자산")

        revenue = _extract(fs, "매출액", "수익(매출액)")
        operating_income = _extract(fs, "영업이익")
        operating_margin = None
        if revenue and operating_income and revenue != 0:
            operating_margin = operating_income / revenue

        total_debt = _extract(fs, "부채총계")
        total_equity = _extract(fs, "자본총계")
        debt_to_equity = None
        if total_equity and total_debt and total_equity != 0:
            debt_to_equity = total_debt / total_equity

        return {
            "symbol": symbol,
            "disclosure_date": disclosure_date,
            "fiscal_year": str(year),
            "fiscal_quarter": quarter,
            "eps": eps,
            "bps": bps,
            "operating_margin": operating_margin,
            "debt_to_equity": debt_to_equity,
        }

    except Exception as e:
        err_msg = str(e)
        if "조회된 데이터가 없습니다" not in err_msg:
            logger.debug("DART %s year=%d %s: %s", symbol, year, quarter, err_msg[:80])
        return None


async def collect_dart_financials(
    date_str: str,
    *,
    progress_cb: ProgressCb = None,
    cb: CircuitBreaker,
) -> CollectionResult:
    """DART 재무 데이터 증분 수집.

    현재 연도의 최신 분기만 수집. 이미 데이터가 있는 종목은 스킵.
    """
    if not settings.DART_API_KEY:
        logger.info("[DART 재무] API 키 미설정 — 스킵")
        return CollectionResult(job="dart_financial")

    try:
        import OpenDartReader  # noqa: N811
    except ImportError:
        logger.warning("[DART 재무] opendartreader 미설치 — 스킵")
        return CollectionResult(job="dart_financial", error="opendartreader 미설치")

    year, quarter, reprt_code = _current_quarter()
    symbols = await _get_symbols_without_data(year, quarter)

    if not symbols:
        logger.info("[DART 재무] 이미 모든 종목 수집 완료 (year=%d %s)", year, quarter)
        return CollectionResult(job="dart_financial", total=0, completed=0)

    logger.info(
        "[DART 재무] 수집 시작: year=%d %s, 미수집 %d종목",
        year, quarter, len(symbols),
    )

    dart = OpenDartReader(settings.DART_API_KEY)
    completed = 0
    failed = 0

    for i, sym in enumerate(symbols):
        try:
            record = await cb.call(
                asyncio.to_thread,
                _fetch_single, dart, sym, year, reprt_code, quarter,
            )

            if record:
                async with async_session() as db:
                    await db.execute(
                        text("""
                            INSERT INTO dart_financials
                                (symbol, disclosure_date, fiscal_year, fiscal_quarter,
                                 eps, bps, operating_margin, debt_to_equity)
                            VALUES (:symbol, :disclosure_date, :fiscal_year, :fiscal_quarter,
                                    :eps, :bps, :operating_margin, :debt_to_equity)
                            ON CONFLICT (symbol, fiscal_year, fiscal_quarter) DO UPDATE
                            SET disclosure_date = EXCLUDED.disclosure_date,
                                eps = EXCLUDED.eps,
                                bps = EXCLUDED.bps,
                                operating_margin = EXCLUDED.operating_margin,
                                debt_to_equity = EXCLUDED.debt_to_equity
                        """),
                        record,
                    )
                    await db.commit()
                completed += 1

            # DART API 쓰로틀링
            await asyncio.sleep(0.15)

        except Exception as e:
            failed += 1
            logger.debug("[DART 재무] %s 실패: %s", sym, e)

        if progress_cb and (i + 1) % 100 == 0:
            await progress_cb(len(symbols), i + 1, sym)

    logger.info(
        "[DART 재무] 완료: %d/%d 성공, %d 실패 (year=%d %s)",
        completed, len(symbols), failed, year, quarter,
    )
    return CollectionResult(
        job="dart_financial",
        total=len(symbols),
        completed=completed,
        failed=failed,
    )
