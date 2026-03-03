"""KRX 종목 설명 + 임베딩을 stock_masters에 업데이트.

기존 stock_masters 데이터(name, market)를 기반으로 설명 텍스트를 만들고,
Sentence Transformers로 임베딩을 생성한다.

Usage:
    docker-compose run --rm app python -m scripts.seed_sectors

주의: sentence-transformers 모델 다운로드 (~400MB) + 인코딩에 약 10분 소요.
"""

from __future__ import annotations

import json
import logging
import sys

from sqlalchemy import create_engine, text

from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# KRX 업종 분류 (대표 종목 기반 수동 매핑)
SECTOR_MAP: dict[str, str] = {
    # 대형주 섹터
    "005930": "반도체", "000660": "반도체", "042700": "반도체",
    "005380": "자동차", "012330": "자동차", "000270": "자동차",
    "055550": "금융", "105560": "금융", "086790": "금융",
    "005490": "철강", "004020": "화학",
    "035420": "인터넷/IT", "035720": "인터넷/IT", "259960": "인터넷/IT",
    "006400": "전자/전기", "051910": "화학", "034730": "IT서비스",
    "068270": "바이오", "207940": "바이오", "091990": "바이오",
    "028260": "건설", "000720": "건설",
    "015760": "통신", "030200": "통신", "017670": "통신",
    "032830": "유통", "004170": "유통",
    "034020": "엔터/미디어", "352820": "엔터/미디어",
    "003550": "식품", "097950": "식품",
    "010950": "에너지", "096770": "에너지",
    "009150": "운송", "003490": "운송",
    "036570": "게임", "251270": "게임", "293490": "게임",
}


def main():
    engine = create_engine(settings.sync_database_url)

    # 1. 기존 종목 조회
    with engine.connect() as conn:
        result = conn.execute(text("SELECT symbol, name, market FROM stock_masters"))
        stocks = [dict(r._mapping) for r in result]

    logger.info("기존 종목: %d개", len(stocks))

    if not stocks:
        logger.error("stock_masters가 비어있습니다. seed_stock_masters.py를 먼저 실행하세요.")
        sys.exit(1)

    # 2. description 생성 (name + market + sector)
    for stock in stocks:
        sym = stock["symbol"]
        sector = SECTOR_MAP.get(sym)
        stock["sector"] = sector
        stock["sub_sector"] = None

        parts = [stock["name"]]
        if stock["market"]:
            parts.append(stock["market"])
        if sector:
            parts.append(sector)
        stock["description"] = " ".join(parts)

    mapped_count = sum(1 for s in stocks if s["sector"])
    logger.info("섹터 매핑: %d/%d개 종목 (나머지는 name+market 기반)", mapped_count, len(stocks))

    # 3. 임베딩 생성
    logger.info("임베딩 생성 시작 (모델 다운로드 포함 시 ~10분)...")
    from app.sector.embedder import encode_texts

    texts = [s["description"] for s in stocks]
    embeddings = encode_texts(texts)
    logger.info("임베딩 생성 완료: %d개", len(embeddings))

    # 4. DB 업데이트
    logger.info("DB 업데이트 중...")
    with engine.begin() as conn:
        for stock, emb in zip(stocks, embeddings):
            conn.execute(
                text(
                    "UPDATE stock_masters "
                    "SET sector = :sector, sub_sector = :sub_sector, "
                    "    description = :description, embedding = :embedding "
                    "WHERE symbol = :symbol"
                ),
                {
                    "symbol": stock["symbol"],
                    "sector": stock["sector"],
                    "sub_sector": stock["sub_sector"],
                    "description": stock["description"],
                    "embedding": json.dumps(emb),
                },
            )

    logger.info("완료! %d개 종목에 설명+임베딩 업데이트", len(stocks))

    # 통계
    sectors = set(s["sector"] for s in stocks if s["sector"])
    logger.info("고유 섹터: %d개", len(sectors))
    for sec in sorted(sectors):
        count = sum(1 for s in stocks if s["sector"] == sec)
        logger.info("  %s: %d개", sec, count)


if __name__ == "__main__":
    main()
