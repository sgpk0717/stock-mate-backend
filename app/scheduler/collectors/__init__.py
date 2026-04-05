"""수집기 공통 타입 정의."""

from __future__ import annotations

from typing import Awaitable, Callable

# 진행률 콜백: (total, completed, last_symbol)
ProgressCb = Callable[[int, int, str], object] | None

# 자유 텍스트 로그 콜백: (message) → UI 로그에 직접 표시
LogCb = Callable[[str], Awaitable[None]] | None


async def load_symbol_name_map() -> dict[str, str]:
    """stock_masters에서 symbol→name 매핑 로드."""
    from sqlalchemy import text

    from app.core.database import async_session

    async with async_session() as db:
        result = await db.execute(
            text("SELECT symbol, name FROM stock_masters ORDER BY symbol"),
        )
        return {r[0]: r[1] for r in result.fetchall()}
