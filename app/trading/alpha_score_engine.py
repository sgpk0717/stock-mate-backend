"""실시간 알파 스코어 엔진.

5분마다 전체 종목의 알파 팩터 스코어를 일괄 계산하여 Redis에 저장.
매매 tick과 프론트엔드 랭킹은 Redis에서 읽기만 한다.

핵심 최적화:
- 인메모리 슬라이딩 윈도우 (60일 데이터 캐시)
- Polars 벡터화 (.over("symbol") 시계열 + .over("dt") 횡단면)
- 증분 갱신 (신규 950행만 append + 최신 시점 횡단면만 재계산)
- Redis Sorted Set + Hash 이원화 (랭킹 + 스냅샷)
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from collections import deque
from datetime import date, datetime, timedelta, timezone
from typing import Any

import polars as pl
import sympy

from app.alpha.ast_converter import (
    ensure_alpha_features,
    parse_expression,
    sympy_to_polars,
)
from app.core.config import settings
from app.core.stock_master import get_stock_name

logger = logging.getLogger(__name__)

_KST = timezone(timedelta(hours=9))
_RANK_WINDOW = 60  # 롤링 퍼센타일 윈도우 (거래일)


class AlphaScoreEngine:
    """실시간 알파 스코어 엔진."""

    def __init__(self) -> None:
        self._cache: pl.DataFrame | None = None
        self._symbols: list[str] = []
        self._factor_configs: list[dict] = []  # [{id, expression_str, buy_threshold, sell_threshold}]
        self._version: int = 0
        self._last_update: datetime | None = None
        self._running: bool = False
        # 롤링 퍼센타일용 캐시 (종목별 deque)
        self._percentile_cache: dict[str, dict[str, deque]] = {}  # {factor_id: {symbol: deque(60)}}

    async def cold_start(
        self,
        symbols: list[str],
        factor_configs: list[dict],
        days: int = 60,
        interval: str = "5m",
    ) -> int:
        """장 시작: 전체 데이터 로드 + 지표 계산 + Redis 저장.

        Returns
        -------
        int : 스코어링된 종목 수
        """
        t0 = time.perf_counter()
        self._symbols = symbols
        self._factor_configs = factor_configs

        # 1. 데이터 로딩 (5분봉 직접 로드 — 1분봉 리샘플링 우회)
        from app.backtest.data_loader import _load_raw_candles

        end = date.today()
        start = end - timedelta(days=days + 30)  # 여유분 포함

        df = await _load_raw_candles(symbols, start, end, db_interval=interval, as_datetime=True)
        if df.is_empty():
            logger.warning("AlphaScoreEngine cold_start: 데이터 없음")
            return 0

        t_load = time.perf_counter() - t0
        logger.info(
            "ScoreEngine cold_start: %d행, %d종목 로드 (%.1f초)",
            df.height, df["symbol"].n_unique(), t_load,
        )

        # 2. 기저 지표 일괄 계산 (벡터화)
        t1 = time.perf_counter()
        df = self._compute_indicators(df)
        t_ind = time.perf_counter() - t1

        # 3. 알파 수식 적용 + 스코어 계산
        t2 = time.perf_counter()
        df = self._compute_alpha_scores(df)
        t_alpha = time.perf_counter() - t2

        # 4. 캐시 저장
        self._cache = df
        self._last_update = datetime.now(_KST)

        # 5. Redis 저장
        t3 = time.perf_counter()
        scored = await self._publish_to_redis(df)
        t_redis = time.perf_counter() - t3

        total = time.perf_counter() - t0
        logger.warning(
            "ScoreEngine cold_start 완료: %d종목 스코어링 (로드=%.1fs, 지표=%.2fs, 알파=%.2fs, Redis=%.3fs, 총=%.1fs)",
            scored, t_load, t_ind, t_alpha, t_redis, total,
        )

        self._running = True
        return scored

    async def update_tick(self) -> int:
        """5분마다: 신규 봉 로드 → append → 최신 횡단면 계산 → Redis.

        Returns
        -------
        int : 스코어링된 종목 수
        """
        if self._cache is None:
            logger.warning("ScoreEngine update_tick: 캐시 없음 (cold_start 필요)")
            return 0

        t0 = time.perf_counter()

        # 1. DB에서 최신 5분봉 로드 (직접 로드)
        from app.backtest.data_loader import _load_raw_candles

        end = date.today()
        start = end  # 오늘만
        new_bars = await _load_raw_candles(self._symbols, start, end, db_interval="5m", as_datetime=True)

        if new_bars.is_empty():
            return 0

        # 최신 시점만 추출
        latest_dt = new_bars["dt"].max()
        new_slice = new_bars.filter(pl.col("dt") == latest_dt)

        # 이미 캐시에 있는 시점이면 스킵
        if self._cache["dt"].max() >= latest_dt:
            return 0

        # 2. 캐시에 append
        # 컬럼 맞추기 (new_slice에 없는 지표 컬럼은 null로)
        for col in self._cache.columns:
            if col not in new_slice.columns:
                new_slice = new_slice.with_columns(pl.lit(None).cast(pl.Float64).alias(col))
        new_slice = new_slice.select(self._cache.columns)

        self._cache = pl.concat([self._cache, new_slice])

        # 60일 초과분 trim
        cutoff = datetime.now(_KST).replace(tzinfo=None) - timedelta(days=65)
        self._cache = self._cache.filter(pl.col("dt") >= cutoff)

        # 3. 시계열 지표 전체 재계산 (벡터화라 ~0.3초)
        self._cache = self._compute_indicators(self._cache)

        # 4. 알파 수식 적용
        self._cache = self._compute_alpha_scores(self._cache)

        # 5. Redis 저장
        scored = await self._publish_to_redis(self._cache)

        total = time.perf_counter() - t0
        self._last_update = datetime.now(_KST)
        self._version += 1

        logger.info(
            "ScoreEngine update_tick: %d종목, %.2f초 (v%d)",
            scored, total, self._version,
        )

        return scored

    def _compute_indicators(self, df: pl.DataFrame) -> pl.DataFrame:
        """시계열 + 횡단면 지표 일괄 벡터화 계산."""
        # 시계열 지표 (종목별)
        df = df.sort(["symbol", "dt"])
        df = df.with_columns([
            pl.col("close").rolling_mean(20).over("symbol").alias("sma_20"),
            pl.col("close").rolling_mean(5).over("symbol").alias("sma_5"),
            pl.col("close").rolling_mean(60).over("symbol").alias("sma_60"),
            pl.col("close").ewm_mean(span=12).over("symbol").alias("ema_12"),
            pl.col("close").ewm_mean(span=20).over("symbol").alias("ema_20"),
            pl.col("close").ewm_mean(span=26).over("symbol").alias("ema_26"),
        ])

        # RSI (14)
        delta = pl.col("close") - pl.col("close").shift(1).over("symbol")
        gain = pl.when(delta > 0).then(delta).otherwise(0.0)
        loss = pl.when(delta < 0).then(-delta).otherwise(0.0)
        df = df.with_columns([
            gain.ewm_mean(span=14).over("symbol").alias("_avg_gain"),
            loss.ewm_mean(span=14).over("symbol").alias("_avg_loss"),
        ])
        df = df.with_columns(
            (100.0 - 100.0 / (1.0 + pl.col("_avg_gain") / pl.col("_avg_loss").clip(lower_bound=1e-10)))
            .alias("rsi")
        )

        # ATR (14)
        tr = pl.max_horizontal(
            pl.col("high") - pl.col("low"),
            (pl.col("high") - pl.col("close").shift(1).over("symbol")).abs(),
            (pl.col("low") - pl.col("close").shift(1).over("symbol")).abs(),
        )
        df = df.with_columns(
            tr.ewm_mean(span=14).over("symbol").alias("atr_14")
        )

        # MACD
        df = df.with_columns(
            (pl.col("ema_12") - pl.col("ema_26")).alias("macd_line")
        )
        df = df.with_columns(
            pl.col("macd_line").ewm_mean(span=9).over("symbol").alias("macd_signal")
        )
        df = df.with_columns(
            (pl.col("macd_line") - pl.col("macd_signal")).alias("macd_hist")
        )

        # BB
        df = df.with_columns([
            pl.col("close").rolling_mean(20).over("symbol").alias("bb_middle"),
            pl.col("close").rolling_std(20).over("symbol").alias("_bb_std"),
        ])
        df = df.with_columns([
            (pl.col("bb_middle") + 2 * pl.col("_bb_std")).alias("bb_upper"),
            (pl.col("bb_middle") - 2 * pl.col("_bb_std")).alias("bb_lower"),
        ])

        # Volume ratio
        df = df.with_columns(
            (pl.col("volume") / pl.col("volume").rolling_mean(20).over("symbol").clip(lower_bound=1))
            .alias("volume_ratio")
        )

        # 시차 피처
        df = df.with_columns([
            pl.col("close").shift(1).over("symbol").alias("close_lag_1"),
            pl.col("close").shift(5).over("symbol").alias("close_lag_5"),
            pl.col("close").shift(20).over("symbol").alias("close_lag_20"),
            pl.col("volume").shift(1).over("symbol").alias("volume_lag_1"),
            pl.col("volume").shift(5).over("symbol").alias("volume_lag_5"),
        ])

        # N일 수익률
        df = df.with_columns([
            (pl.col("close") / pl.col("close_lag_5") - 1).alias("return_5d"),
            (pl.col("close") / pl.col("close_lag_20") - 1).alias("return_20d"),
        ])

        # 가격 변화율
        df = df.with_columns(
            (pl.col("close").pct_change().over("symbol")).alias("price_change_pct")
        )

        # BB position
        df = df.with_columns(
            ((pl.col("close") - pl.col("bb_lower")) /
             (pl.col("bb_upper") - pl.col("bb_lower")).clip(lower_bound=1e-10))
            .clip(0.0, 1.0).alias("bb_position")
        )
        df = df.with_columns(
            (pl.col("bb_upper") - pl.col("bb_lower")).alias("bb_width")
        )

        # 시계열 롤링 피처 (60일 윈도우)
        df = df.with_columns([
            pl.col("close").rolling_min(60).over("symbol").alias("_ts_min_close"),
            pl.col("close").rolling_max(60).over("symbol").alias("_ts_max_close"),
            pl.col("volume").rolling_min(60).over("symbol").alias("_ts_min_vol"),
            pl.col("volume").rolling_max(60).over("symbol").alias("_ts_max_vol"),
            pl.col("close").rolling_mean(60).over("symbol").alias("_ts_mean_close"),
            pl.col("close").rolling_std(60).over("symbol").alias("_ts_std_close"),
            pl.col("volume").rolling_mean(60).over("symbol").alias("_ts_mean_vol"),
            pl.col("volume").rolling_std(60).over("symbol").alias("_ts_std_vol"),
        ])
        df = df.with_columns([
            ((pl.col("close") - pl.col("_ts_min_close")) /
             (pl.col("_ts_max_close") - pl.col("_ts_min_close")).clip(lower_bound=1e-10))
            .alias("ts_rank_close"),
            ((pl.col("volume") - pl.col("_ts_min_vol")) /
             (pl.col("_ts_max_vol") - pl.col("_ts_min_vol")).clip(lower_bound=1e-10))
            .alias("ts_rank_volume"),
            ((pl.col("close") - pl.col("_ts_mean_close")) /
             pl.col("_ts_std_close").clip(lower_bound=1e-10))
            .alias("ts_zscore_close"),
            ((pl.col("volume") - pl.col("_ts_mean_vol")) /
             pl.col("_ts_std_vol").clip(lower_bound=1e-10))
            .alias("ts_zscore_volume"),
        ])

        # gap 피처
        prev_close = pl.col("close").shift(1).over("symbol")
        df = df.with_columns([
            ((pl.col("open") - prev_close) / prev_close.clip(lower_bound=1) * 100)
            .clip(-50, 50).fill_null(0).alias("gap_up_pct"),
            ((prev_close - pl.col("open")) / prev_close.clip(lower_bound=1) * 100)
            .clip(-50, 50).fill_null(0).alias("gap_down_pct"),
        ])

        # 횡단면 피처 (시점별)
        # 컬럼명은 NAMED_VARIABLE_MAP의 value(Polars 컬럼명)와 일치시킴
        df = df.with_columns([
            pl.col("volume").rank().over("dt").alias("rank_volume"),
            pl.col("close").rank().over("dt").alias("rank_close"),
            ((pl.col("volume") - pl.col("volume").mean().over("dt")) /
             pl.col("volume").std().over("dt").clip(lower_bound=1e-10))
            .alias("zscore_volume"),  # NAMED_VARIABLE_MAP: Cs_ZScore_volume → zscore_volume
            ((pl.col("close") - pl.col("close").mean().over("dt")) /
             pl.col("close").std().over("dt").clip(lower_bound=1e-10))
            .alias("zscore_close"),
        ])

        # 임시 컬럼 정리
        drop_cols = [c for c in df.columns if c.startswith("_")]
        if drop_cols:
            df = df.drop(drop_cols)

        return df

    def _compute_alpha_scores(self, df: pl.DataFrame) -> pl.DataFrame:
        """알파 팩터 수식 적용 + 롤링 퍼센타일 정규화."""
        for fc in self._factor_configs:
            factor_id = fc["id"][:8]
            expr_str = fc["expression_str"]
            col_name = f"alpha_{factor_id}"

            try:
                expr = parse_expression(expr_str)
                polars_expr = sympy_to_polars(expr)

                # 원시 값 계산
                raw_col = f"_raw_{col_name}"
                df = df.with_columns(polars_expr.alias(raw_col))

                # Inf/NaN 정리
                df = df.with_columns(
                    pl.when(pl.col(raw_col).is_finite())
                    .then(pl.col(raw_col))
                    .otherwise(None)
                    .alias(raw_col)
                )

                # 롤링 퍼센타일 (종목별 60일)
                df = df.with_columns([
                    pl.col(raw_col).rolling_min(_RANK_WINDOW).over("symbol").alias(f"_min_{col_name}"),
                    pl.col(raw_col).rolling_max(_RANK_WINDOW).over("symbol").alias(f"_max_{col_name}"),
                ])

                range_col = (
                    pl.col(f"_max_{col_name}") - pl.col(f"_min_{col_name}")
                ).clip(lower_bound=1e-10)

                df = df.with_columns(
                    ((pl.col(raw_col) - pl.col(f"_min_{col_name}")) / range_col)
                    .clip(0.0, 1.0)
                    .fill_null(0.5)
                    .alias(col_name)
                )

                # 임시 컬럼 정리
                df = df.drop([raw_col, f"_min_{col_name}", f"_max_{col_name}"])

            except Exception as e:
                logger.warning("ScoreEngine 팩터 %s 계산 실패: %s", factor_id, e)
                df = df.with_columns(pl.lit(0.5).alias(col_name))

        return df

    async def _publish_to_redis(self, df: pl.DataFrame) -> int:
        """스코어를 Redis Sorted Set + Hash에 저장 (더블 버퍼)."""
        try:
            from app.core.redis import get_client

            r = get_client()

            # 최신 시점 추출
            latest_dt = df["dt"].max()
            latest = df.filter(pl.col("dt") == latest_dt)

            if latest.is_empty():
                return 0

            # 팩터별 스코어 컬럼
            alpha_cols = [c for c in latest.columns if c.startswith("alpha_")]
            if not alpha_cols:
                return 0

            # 주요 알파 컬럼 (첫 번째 팩터 기준으로 랭킹)
            primary_col = alpha_cols[0]
            rows = latest.to_dicts()

            now_str = datetime.now(_KST).isoformat()

            # 더블 버퍼: tmp 키에 기록 → RENAME으로 원자적 교체
            pipe = r.pipeline()

            # tmp 키 초기화
            pipe.delete("tmp:alpha:buy_ranking")
            pipe.delete("tmp:alpha:sell_ranking")

            scored = 0
            for row in rows:
                sym = row.get("symbol", "")
                if not sym:
                    continue

                score = row.get(primary_col)
                if score is None or math.isnan(score):
                    score = 0.5

                # Sorted Set: 매수 랭킹 (score 높은 순)
                pipe.zadd("tmp:alpha:buy_ranking", {sym: score})
                # Sorted Set: 매도 랭킹 (1-score, 낮은 score = 매도 임박)
                pipe.zadd("tmp:alpha:sell_ranking", {sym: 1.0 - score})

                # Hash: 종목별 상세
                detail = {
                    "score": f"{score:.4f}",
                    "close": str(row.get("close", 0)),
                    "sma_20": f"{row.get('sma_20', 0):.2f}" if row.get("sma_20") else "0",
                    "rsi": f"{row.get('rsi', 0):.1f}" if row.get("rsi") else "0",
                    "volume_ratio": f"{row.get('volume_ratio', 0):.2f}" if row.get("volume_ratio") else "0",
                    "zscore_volume": f"{row.get('zscore_volume', 0):.3f}" if row.get("zscore_volume") else "0",
                    "updated_at": now_str,
                    "name": get_stock_name(sym),
                }
                # 모든 알파 스코어 추가
                for ac in alpha_cols:
                    v = row.get(ac)
                    detail[ac] = f"{v:.4f}" if v is not None and not math.isnan(v) else "0.5"

                pipe.hset(f"tmp:alpha:detail:{sym}", mapping=detail)
                scored += 1

            await pipe.execute()

            # RENAME으로 원자적 교체
            pipe2 = r.pipeline()
            pipe2.rename("tmp:alpha:buy_ranking", "alpha:buy_ranking")
            pipe2.rename("tmp:alpha:sell_ranking", "alpha:sell_ranking")
            # detail은 키가 많으므로 tmp → 실제 키로 개별 RENAME
            for row in rows:
                sym = row.get("symbol", "")
                if sym:
                    pipe2.rename(f"tmp:alpha:detail:{sym}", f"alpha:detail:{sym}")
            # 버전 + TTL
            pipe2.set("alpha:version", str(self._version))
            pipe2.set("alpha:updated_at", now_str)
            pipe2.expire("alpha:buy_ranking", 43200)
            pipe2.expire("alpha:sell_ranking", 43200)
            await pipe2.execute()

            return scored

        except Exception as e:
            logger.warning("ScoreEngine Redis 저장 실패: %s", e)
            return 0

    def get_status(self) -> dict:
        """엔진 상태 반환."""
        return {
            "running": self._running,
            "version": self._version,
            "last_update": self._last_update.isoformat() if self._last_update else None,
            "symbols_count": len(self._symbols),
            "factors_count": len(self._factor_configs),
            "cache_rows": self._cache.height if self._cache is not None else 0,
            "cache_mb": round(self._cache.estimated_size("mb"), 1) if self._cache is not None else 0,
        }


# ── 싱글턴 인스턴스 ──

_engine: AlphaScoreEngine | None = None


def get_score_engine() -> AlphaScoreEngine:
    """싱글턴 스코어 엔진 반환."""
    global _engine
    if _engine is None:
        _engine = AlphaScoreEngine()
    return _engine


async def start_score_engine_loop(
    symbols: list[str],
    factor_configs: list[dict],
    interval_seconds: int = 300,
) -> None:
    """스코어 엔진 시작 + 5분 주기 루프."""
    engine = get_score_engine()

    # Cold start
    await engine.cold_start(symbols, factor_configs)

    # 5분 주기 루프
    while engine._running:
        await asyncio.sleep(interval_seconds)
        try:
            now = datetime.now(_KST)
            # 장 시간만 (09:00~15:30)
            if now.hour < 9 or (now.hour >= 15 and now.minute >= 30):
                continue
            await engine.update_tick()
        except Exception as e:
            logger.error("ScoreEngine loop 에러: %s", e, exc_info=True)
