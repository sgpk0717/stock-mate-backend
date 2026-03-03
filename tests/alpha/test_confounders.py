"""교란 변수 로더 단위 테스트 (C01-C04)."""

from __future__ import annotations

import math
from datetime import date

import pandas as pd
import pytest

from app.alpha.confounders import _load_base_rate, _BOK_JSON_PATH


class TestBaseRate:
    """BOK 기준금리 로더 테스트."""

    def test_c01_base_rate_parsing_and_ffill(self):
        """C01: BOK 기준금리 JSON 파싱 + forward-fill 정확성."""
        start = date(2023, 1, 1)
        end = date(2023, 12, 31)
        df = _load_base_rate(start, end)

        assert isinstance(df, pd.DataFrame)
        assert "dt" in df.columns
        assert "base_rate" in df.columns
        assert len(df) > 0

        # NaN이 없어야 (forward-fill 완료)
        assert df["base_rate"].isna().sum() == 0

        # 2023-01-13에 3.50으로 인상됨 → 그 이후는 3.50이어야
        after_hike = df[df["dt"] >= date(2023, 1, 16)]
        if len(after_hike) > 0:
            assert all(after_hike["base_rate"] == 3.50)

    def test_c02_base_rate_date_range(self):
        """C02: 날짜 범위 필터 정확성."""
        start = date(2024, 6, 1)
        end = date(2024, 6, 30)
        df = _load_base_rate(start, end)

        # 모든 날짜가 범위 내
        for dt_val in df["dt"]:
            assert start <= dt_val <= end

    def test_c03_base_rate_no_nan(self):
        """C03: forward-fill 후 NaN 없음."""
        start = date(2022, 1, 1)
        end = date(2025, 12, 31)
        df = _load_base_rate(start, end)

        assert df["base_rate"].isna().sum() == 0

    def test_c04_base_rate_json_exists(self):
        """C04: BOK 기준금리 JSON 파일 존재."""
        assert _BOK_JSON_PATH.exists(), f"BOK JSON not found: {_BOK_JSON_PATH}"
