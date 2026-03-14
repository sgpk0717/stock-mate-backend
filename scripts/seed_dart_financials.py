"""DART 재무 데이터 시딩.

OpenDartReader를 사용하여 상장사의 EPS/BPS/영업이익률/부채비율을 수집한다.
disclosure_date(공시일)를 기준으로 저장하여 look-ahead bias를 방지한다.

Usage:
    docker-compose run --rm app python -m scripts.seed_dart_financials
    docker-compose run --rm app python -m scripts.seed_dart_financials --year 2024
    docker-compose run --rm app python -m scripts.seed_dart_financials --symbols 005930,000660

Requirements:
    pip install opendartreader
    DART_API_KEY 환경변수 설정
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from datetime import date

import asyncpg

from app.core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("seed_dart")


def _dsn() -> str:
    s = settings
    return (
        f"postgresql://{s.POSTGRES_USER}:{s.POSTGRES_PASSWORD}"
        f"@{s.POSTGRES_HOST}:{s.POSTGRES_PORT}/{s.POSTGRES_DB}"
    )


async def seed(
    years: list[int],
    symbols: list[str] | None = None,
    batch_size: int = 100,
) -> None:
    """DART 재무 데이터 시딩."""
    api_key = settings.DART_API_KEY
    if not api_key:
        logger.error("DART_API_KEY가 설정되지 않았습니다. .env에 DART_API_KEY를 추가하세요.")
        return

    try:
        import OpenDartReader
    except ImportError:
        logger.error("opendartreader가 설치되지 않았습니다: pip install opendartreader")
        return

    dart = OpenDartReader(api_key)
    conn = await asyncpg.connect(_dsn())

    # 테이블 생성
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS dart_financials (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            disclosure_date DATE NOT NULL,
            fiscal_year VARCHAR(4) NOT NULL,
            fiscal_quarter VARCHAR(2) NOT NULL,
            fiscal_period_end DATE,
            eps FLOAT,
            bps FLOAT,
            operating_margin FLOAT,
            debt_to_equity FLOAT,
            CONSTRAINT uq_dart_financial UNIQUE (symbol, fiscal_year, fiscal_quarter)
        )
    """)
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_dart_financials_lookup ON dart_financials (symbol, disclosure_date)"
    )

    # 대상 종목 목록
    if symbols is None:
        rows = await conn.fetch(
            "SELECT symbol FROM stock_masters WHERE market IN ('KOSPI', 'KOSDAQ') ORDER BY symbol"
        )
        symbols = [r["symbol"] for r in rows]

    logger.info("Target: %d symbols, years: %s", len(symbols), years)

    total_inserted = 0
    report_types = {
        "11013": "1Q",  # 1분기
        "11012": "2Q",  # 반기
        "11014": "3Q",  # 3분기
        "11011": "4Q",  # 사업보고서
    }

    for year in years:
        for reprt_code, quarter in report_types.items():
            logger.info("Fetching year=%d quarter=%s...", year, quarter)

            for i in range(0, len(symbols), 50):
                batch_symbols = symbols[i : i + 50]

                for symbol in batch_symbols:
                    try:
                        # 단일기업 재무제표 조회
                        fs = dart.finstate(symbol, year, reprt_code=reprt_code)
                        if fs is None or fs.empty:
                            continue

                        # rcept_dt = 공시접수일 (look-ahead bias 방지)
                        rcept_dt_str = fs["rcept_dt"].iloc[0] if "rcept_dt" in fs.columns else None
                        if rcept_dt_str:
                            disclosure_date = date(
                                int(rcept_dt_str[:4]),
                                int(rcept_dt_str[4:6]),
                                int(rcept_dt_str[6:8]),
                            )
                        else:
                            # fallback: 분기말 + 45일 (보수적 추정)
                            quarter_end_months = {"1Q": 3, "2Q": 6, "3Q": 9, "4Q": 12}
                            month = quarter_end_months[quarter]
                            from datetime import timedelta
                            fiscal_end = date(year, month, 28)
                            disclosure_date = fiscal_end + timedelta(days=45)

                        # EPS, BPS 추출
                        eps = _extract_value(fs, "당기순이익", "주당순이익")
                        bps = _extract_value(fs, "자본총계", "주당순자산")

                        # 영업이익률 계산
                        revenue = _extract_value(fs, "매출액", "수익(매출액)")
                        operating_income = _extract_value(fs, "영업이익")
                        operating_margin = None
                        if revenue and operating_income and revenue != 0:
                            operating_margin = operating_income / revenue

                        # 부채비율 계산
                        total_debt = _extract_value(fs, "부채총계")
                        total_equity = _extract_value(fs, "자본총계")
                        debt_to_equity = None
                        if total_equity and total_debt and total_equity != 0:
                            debt_to_equity = total_debt / total_equity

                        await conn.execute(
                            """
                            INSERT INTO dart_financials
                                (symbol, disclosure_date, fiscal_year, fiscal_quarter, eps, bps, operating_margin, debt_to_equity)
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                            ON CONFLICT (symbol, fiscal_year, fiscal_quarter) DO UPDATE
                            SET disclosure_date = EXCLUDED.disclosure_date,
                                eps = EXCLUDED.eps,
                                bps = EXCLUDED.bps,
                                operating_margin = EXCLUDED.operating_margin,
                                debt_to_equity = EXCLUDED.debt_to_equity
                            """,
                            symbol, disclosure_date, str(year), quarter,
                            eps, bps, operating_margin, debt_to_equity,
                        )
                        total_inserted += 1

                    except Exception as e:
                        err_msg = str(e)
                        if "조회된 데이터가 없습니다" not in err_msg:
                            logger.debug("Symbol %s year %d %s: %s", symbol, year, quarter, err_msg[:80])

                    time.sleep(0.15)  # DART API 쓰로틀링

                logger.info(
                    "  year=%d %s: processed %d/%d symbols (total %d)",
                    year, quarter, min(i + 50, len(symbols)), len(symbols), total_inserted,
                )

    await conn.close()
    logger.info("=== Seed complete: %d records inserted/updated ===", total_inserted)


def _extract_value(df, *account_names: str) -> float | None:
    """재무제표에서 특정 계정과목의 값을 추출."""
    import pandas as pd

    for name in account_names:
        mask = df["account_nm"].str.contains(name, na=False)
        rows = df[mask]
        if not rows.empty:
            # 당기 값 우선
            for col in ["thstrm_amount", "thstrm_dt"]:
                if col in rows.columns:
                    val = rows[col].iloc[0]
                    if pd.notna(val):
                        try:
                            # 쉼표 제거 후 float 변환
                            return float(str(val).replace(",", ""))
                        except (ValueError, TypeError):
                            continue
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DART 재무 데이터 시딩")
    parser.add_argument(
        "--year", type=int, default=0,
        help="단일 연도 (기본: 최근 3년)",
    )
    parser.add_argument(
        "--symbols", type=str, default="",
        help="종목 코드 (쉼표 구분, 기본: 전체)",
    )
    args = parser.parse_args()

    if args.year:
        years = [args.year]
    else:
        current_year = date.today().year
        years = list(range(current_year - 2, current_year + 1))

    symbols = args.symbols.split(",") if args.symbols else None

    asyncio.run(seed(years, symbols))
