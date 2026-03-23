"""팩터 포트폴리오 — 복합 팩터 생성 + 상관행렬 + 자동 최적 조합.

다수의 단일 팩터를 IC 가중 또는 동일 가중으로 합성하여
복합 팩터를 만들고, 팩터 간 상관행렬을 계산한다.
자동 최적 조합은 탐욕적 전진선택 + Two-tier Shrinkage로 수행.
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.alpha.ast_converter import (
    ASTConversionError,
    classify_niche,
    ensure_alpha_features,
    parse_expression,
    sympy_to_polars,
)
from app.alpha.evaluator import (
    compute_factor_metrics,
    compute_forward_returns,
    compute_ic_series,
    compute_long_only_returns,
    compute_position_turnover,
    FactorMetrics,
)
from app.alpha.models import AlphaFactor
from app.backtest.data_loader import load_candles

logger = logging.getLogger(__name__)


@dataclass
class CompositeResult:
    """복합 팩터 생성 결과."""

    composite_expression: str
    component_ids: list[str]
    weights: dict[str, float]
    metrics: FactorMetrics
    correlation_matrix: list[list[float]]
    factor_names: list[str]


async def build_composite_factor(
    db: AsyncSession,
    factor_ids: list[str],
    method: str = "ic_weighted",
    name: str = "Composite Alpha",
) -> CompositeResult:
    """복합 팩터 생성.

    1. 팩터 로드 → 캔들 로드 → 피처 생성 → 팩터 컬럼 계산
    2. 상관행렬 계산
    3. 가중 합산 (ic_weighted 또는 equal_weight)
    4. 복합 팩터 IC 평가
    """
    # 팩터 로드
    factor_uuids = [uuid.UUID(fid) for fid in factor_ids]
    result = await db.execute(
        select(AlphaFactor).where(AlphaFactor.id.in_(factor_uuids))
    )
    factors = result.scalars().all()

    if len(factors) < 2:
        raise ValueError(f"최소 2개 팩터가 필요합니다 (로드됨: {len(factors)})")

    # 데이터 로드 — 최근 1년 기본 제한 (전체 DB 로드 방지)
    default_start = date.today() - timedelta(days=365)
    data = await load_candles(start_date=default_start)
    if data.height == 0:
        raise ValueError("캔들 데이터가 없습니다")

    data = ensure_alpha_features(data)
    data = compute_forward_returns(data, periods=1)

    # 팩터 컬럼 계산
    valid_factors: list[AlphaFactor] = []
    col_names: list[str] = []

    for factor in factors:
        col_name = f"factor_{factor.id}"
        try:
            parsed = parse_expression(factor.expression_str)
            polars_expr = sympy_to_polars(parsed)
            data = data.with_columns(polars_expr.alias(col_name))
            valid_factors.append(factor)
            col_names.append(col_name)
        except (ASTConversionError, Exception) as e:
            logger.warning(
                "Factor %s parse failed, skipping: %s", factor.id, e
            )

    if len(valid_factors) < 2:
        raise ValueError(
            f"유효한 팩터가 2개 미만입니다 (유효: {len(valid_factors)})"
        )

    # null 제거
    data = data.drop_nulls(subset=col_names + ["fwd_return"])
    if data.height < 30:
        raise ValueError(f"데이터 행이 부족합니다: {data.height}")

    # 상관행렬 계산
    factor_matrix = data.select(col_names).to_numpy()
    corr_matrix = np.corrcoef(factor_matrix, rowvar=False)
    corr_list = corr_matrix.tolist()

    # 가중치 결정
    weights: dict[str, float] = {}
    if method == "ic_weighted":
        total_ic = sum(
            abs(f.ic_mean) for f in valid_factors if f.ic_mean is not None
        )
        if total_ic == 0:
            # IC 정보 없으면 동일 가중으로 폴백
            for f in valid_factors:
                weights[str(f.id)] = 1.0 / len(valid_factors)
        else:
            for f in valid_factors:
                ic = abs(f.ic_mean) if f.ic_mean is not None else 0.0
                weights[str(f.id)] = ic / total_ic
    else:
        # equal_weight
        for f in valid_factors:
            weights[str(f.id)] = 1.0 / len(valid_factors)

    # 복합 팩터 계산
    composite_expr_parts: list[str] = []
    composite_col = pl.lit(0.0)

    for factor, col_name in zip(valid_factors, col_names):
        w = weights[str(factor.id)]
        composite_col = composite_col + pl.col(col_name) * w
        composite_expr_parts.append(f"{w:.4f} * ({factor.expression_str})")

    composite_expression = " + ".join(composite_expr_parts)

    data = data.with_columns(composite_col.alias("composite_factor"))
    data = data.drop_nulls(subset=["composite_factor", "fwd_return"])

    # 복합 팩터 IC 평가
    ic_series = compute_ic_series(data, factor_col="composite_factor")
    metrics = compute_factor_metrics(ic_series)

    factor_names = [f.name for f in valid_factors]
    component_ids_str = [str(f.id) for f in valid_factors]

    return CompositeResult(
        composite_expression=composite_expression,
        component_ids=component_ids_str,
        weights=weights,
        metrics=metrics,
        correlation_matrix=corr_list,
        factor_names=factor_names,
    )


async def compute_correlation_matrix(
    db: AsyncSession,
    factor_ids: list[str],
) -> dict:
    """팩터 간 Pearson 상관행렬 계산."""
    factor_uuids = [uuid.UUID(fid) for fid in factor_ids]
    result = await db.execute(
        select(AlphaFactor).where(AlphaFactor.id.in_(factor_uuids))
    )
    factors = result.scalars().all()

    default_start = date.today() - timedelta(days=365)
    data = await load_candles(start_date=default_start)
    if data.height == 0:
        raise ValueError("캔들 데이터가 없습니다")

    data = ensure_alpha_features(data)

    valid_factors: list[AlphaFactor] = []
    col_names: list[str] = []

    for factor in factors:
        col_name = f"factor_{factor.id}"
        try:
            parsed = parse_expression(factor.expression_str)
            polars_expr = sympy_to_polars(parsed)
            data = data.with_columns(polars_expr.alias(col_name))
            valid_factors.append(factor)
            col_names.append(col_name)
        except (ASTConversionError, Exception):
            continue

    if len(valid_factors) < 2:
        raise ValueError("유효한 팩터가 2개 미만입니다")

    data = data.drop_nulls(subset=col_names)
    factor_matrix = data.select(col_names).to_numpy()
    corr_matrix = np.corrcoef(factor_matrix, rowvar=False)

    return {
        "factor_ids": [str(f.id) for f in valid_factors],
        "factor_names": [f.name for f in valid_factors],
        "matrix": corr_matrix.tolist(),
    }


# ── 자동 최적 조합 (탐욕적 전진선택 + Two-tier Shrinkage) ──


@dataclass
class SelectionStep:
    """전진선택 한 스텝의 로그."""

    step: int
    selected_id: str
    selected_name: str
    niche: str
    ic: float
    reason: str
    cumulative_ir2: float
    avg_correlation: float


@dataclass
class OptimizationResult:
    """K개 팩터 조합의 결과."""

    k: int
    factor_ids: list[str]
    factor_names: list[str]
    weights: dict[str, float]
    composite_ic: float
    composite_icir: float
    composite_sharpe: float
    avg_correlation: float
    expression_str: str


@dataclass
class AutoOptimizeOutput:
    """자동 최적 조합 전체 결과."""

    best_k: int
    results: list[OptimizationResult]
    selection_log: list[SelectionStep]
    correlation_matrix: dict
    candidate_count: int
    logs: list[str] = field(default_factory=list)


def _greedy_forward_select(
    ic_vector: np.ndarray,
    corr_matrix: np.ndarray,
    max_k: int = 7,
    lambda_decorr: float = 0.5,
    hard_cutoff: float = 0.70,
) -> list[tuple[int, SelectionStep]]:
    """탐욕적 전진선택.

    ic_vector: (N,) 팩터별 IC
    corr_matrix: (N, N) 팩터 간 상관행렬
    max_k: 최대 선택 수
    lambda_decorr: 상관 페널티 강도
    hard_cutoff: 이 이상 상관이면 후보에서 제외

    Returns: [(index, SelectionStep), ...] 선택 순서
    """
    n = len(ic_vector)
    selected_indices: list[int] = []
    steps: list[tuple[int, SelectionStep]] = []
    candidates = set(range(n))

    for step_num in range(1, min(max_k, n) + 1):
        best_score = -np.inf
        best_idx = -1
        best_reason = ""

        for j in candidates:
            # Hard cut-off: 이미 선택된 팩터와 상관 > 0.70이면 스킵
            if selected_indices:
                max_corr_with_selected = max(
                    abs(corr_matrix[j, s]) for s in selected_indices
                )
                if max_corr_with_selected > hard_cutoff:
                    continue
            else:
                max_corr_with_selected = 0.0

            if step_num == 1:
                # 첫 번째: IC 최고
                score = abs(ic_vector[j])
                reason = f"IC 최고 ({ic_vector[j]:.4f})"
            else:
                # 이후: marginal IR² - λ × mean(|ρ|)
                subset = selected_indices + [j]
                ic_s = ic_vector[subset]
                c_s = corr_matrix[np.ix_(subset, subset)]
                try:
                    c_inv = np.linalg.pinv(c_s)
                    ir2 = float(ic_s @ c_inv @ ic_s)
                except Exception:
                    ir2 = sum(ic_vector[i] ** 2 for i in subset)

                mean_corr = float(np.mean([
                    abs(corr_matrix[j, s]) for s in selected_indices
                ]))
                score = ir2 - lambda_decorr * mean_corr
                reason = f"ρ_avg={mean_corr:.3f}, marginal IR²={ir2:.4f}"

            if score > best_score:
                best_score = score
                best_idx = j
                best_reason = reason

        if best_idx < 0:
            break  # 후보 소진

        selected_indices.append(best_idx)
        candidates.discard(best_idx)

        # 누적 IR² 계산
        ic_s = ic_vector[selected_indices]
        c_s = corr_matrix[np.ix_(selected_indices, selected_indices)]
        try:
            c_inv = np.linalg.pinv(c_s)
            cum_ir2 = float(ic_s @ c_inv @ ic_s)
        except Exception:
            cum_ir2 = sum(x ** 2 for x in ic_s)

        avg_corr = 0.0
        if len(selected_indices) > 1:
            pairs = []
            for i, a in enumerate(selected_indices):
                for b in selected_indices[i + 1:]:
                    pairs.append(abs(corr_matrix[a, b]))
            avg_corr = float(np.mean(pairs)) if pairs else 0.0

        step_info = SelectionStep(
            step=step_num,
            selected_id="",  # 나중에 채움
            selected_name="",
            niche="",
            ic=float(ic_vector[best_idx]),
            reason=best_reason,
            cumulative_ir2=round(cum_ir2, 6),
            avg_correlation=round(avg_corr, 4),
        )
        steps.append((best_idx, step_info))

    return steps


def _two_tier_shrinkage_weights(
    ic_vector: np.ndarray,
    corr_matrix: np.ndarray,
    shrinkage_delta: float = 0.5,
    max_weight: float = 0.4,
) -> np.ndarray:
    """Two-tier Shrinkage 가중치.

    ICIR 가중 ↔ 등가중 블렌드 + 상한 클리핑.
    """
    k = len(ic_vector)
    if k <= 1:
        return np.ones(1)

    # 역공분산 가중
    try:
        c_inv = np.linalg.pinv(corr_matrix)
        w_icir = c_inv @ ic_vector
        w_sum = np.sum(np.abs(w_icir))
        if w_sum > 1e-12:
            w_icir = w_icir / w_sum
        else:
            w_icir = np.ones(k) / k
    except Exception:
        w_icir = np.ones(k) / k

    # 등가중
    w_equal = np.ones(k) / k

    # 블렌드
    w_final = (1 - shrinkage_delta) * w_icir + shrinkage_delta * w_equal

    # 음수 가중치 → 0으로 클리핑 (long-only)
    w_final = np.clip(w_final, 0.0, max_weight)
    w_sum = np.sum(w_final)
    if w_sum > 1e-12:
        w_final = w_final / w_sum

    return w_final


async def auto_optimize_composite(
    db: AsyncSession,
    min_ic: float = 0.03,
    min_turnover: float = 0.02,
    max_k: int = 7,
    lambda_decorr: float = 0.5,
    shrinkage_delta: float = 0.5,
    interval: str = "5m",
    causal_only: bool = False,
    job_id: str | None = None,
) -> AutoOptimizeOutput:
    """자동 최적 복합 팩터 조합 (일별 집계 + Welford + EWMA).

    딥리서치 권고 기반 재구현:
    - 5분봉 원시 스태킹 대신 **일별 집계** (자기상관 제거, 메모리 ~0.1MB)
    - 270일 전체 기간 사용 (평가기간과 정렬)
    - EWMA 반감기 60일 (적응성 + 표본 깊이)
    - Ledoit-Wolf 수축 (T/N=5.1 보정)
    - 개장/폐장 봉 제외 (단일가 왜곡 방지)

    job_id가 주어지면 Redis에 실시간 로그/상태를 저장한다.
    """
    import gc
    import json as _json
    import time
    from app.alpha.interval import default_round_trip_cost, is_intraday
    from sklearn.covariance import LedoitWolf

    _t0 = time.time()
    _logs: list[str] = []

    # Redis 키 (job_id가 있을 때만 사용)
    _redis_key = f"alpha:optimize:{job_id}" if job_id else None

    async def _redis_update(mapping: dict) -> None:
        """Redis Hash에 상태 업데이트 (job_id 있을 때만)."""
        if not _redis_key:
            return
        try:
            from app.core.redis import get_client as get_redis
            redis = get_redis()
            serialized = {}
            for k, v in mapping.items():
                if isinstance(v, (dict, list)):
                    serialized[k] = _json.dumps(v, ensure_ascii=False, default=str)
                elif v is None:
                    serialized[k] = ""
                else:
                    serialized[k] = str(v)
            await redis.hset(_redis_key, mapping=serialized)
            await redis.expire(_redis_key, 86400)  # TTL 24시간
        except Exception:
            pass

    async def _log(msg: str) -> None:
        from datetime import datetime as _dt, timezone as _tz, timedelta as _td
        elapsed = time.time() - _t0
        kst_now = _dt.now(_tz(_td(hours=9)))
        entry = f"[{kst_now.strftime('%H:%M:%S')}] [{elapsed:.1f}s] {msg}"
        _logs.append(entry)
        logger.info("Auto-optimize: %s", entry)
        # Redis에 로그 업데이트
        await _redis_update({"logs": _logs})
        try:
            from app.services.ws_manager import manager
            await manager.broadcast("alpha:factory", {
                "type": "optimize_log",
                "log": entry,
                "logs": list(_logs),
            })
        except Exception:
            pass

    # job_id가 있으면 Redis에 RUNNING 상태 기록
    await _redis_update({"status": "running", "logs": []})

    # ── Step 1: 후보 팩터 로드 (DB IC 값 사용) ──
    filters = [
        AlphaFactor.factor_type == "single",
        AlphaFactor.ic_mean.isnot(None),
        AlphaFactor.ic_mean >= min_ic,
        AlphaFactor.turnover.isnot(None),
        AlphaFactor.turnover >= min_turnover,
    ]
    if interval:
        filters.append(AlphaFactor.interval == interval)
    if causal_only:
        filters.append(AlphaFactor.causal_robust == True)  # noqa: E712

    result = await db.execute(
        select(AlphaFactor).where(*filters).order_by(AlphaFactor.ic_mean.desc())
    )
    candidates = list(result.scalars().all())
    if len(candidates) < 3:
        raise ValueError(f"후보 팩터 부족: {len(candidates)}개 (최소 3개)")

    await _log(f"Step 1/4: 후보 팩터 {len(candidates)}개 로드 (interval={interval}, causal_only={causal_only})")

    # ── Step 2: 유효 팩터 필터 (수식 파싱 + Polars 변환 가능) ──
    valid_factors: list[AlphaFactor] = []
    ic_values: list[float] = []
    polars_exprs: list = []  # 캐시

    for factor in candidates:
        try:
            parsed = parse_expression(factor.expression_str)
            pe = sympy_to_polars(parsed)
            valid_factors.append(factor)
            ic_values.append(float(factor.ic_mean))
            polars_exprs.append(pe)
        except Exception:
            continue

    if len(valid_factors) < 3:
        raise ValueError(f"유효 팩터 부족: {len(valid_factors)}개")

    n_factors = len(valid_factors)
    await _log(f"Step 2/4: 유효 팩터 {n_factors}개 (파싱 실패 {len(candidates) - n_factors}개 제외)")

    # ── Step 3: 일별 집계 + EWMA 상관행렬 ──
    # 종목 50개씩 청크로 로딩 → 팩터 수식 적용 → 일별 평균 → 원본 해제
    # 최종: 일별 팩터 평균값으로 상관행렬 (메모리 ~수 MB)
    await _log("Step 3/4: 일별 집계 상관행렬 계산 시작 (종목 청크 방식)")

    from app.backtest.data_loader import load_enriched_candles
    _SYM_CHUNK = 50
    # 커버리지 기반 외부 데이터 포함/제외 (딥리서치 권고)
    _ENRICH_OPTS = dict(
        include_investor=True,       # 100% 커버리지
        include_dart=True,           # 68% 커버리지
        include_sentiment=False,     # 1% 커버리지 → 제외
        include_sector=True,         # 100% 커버리지
        include_margin_short=True,   # 100% 커버리지
        include_program_trading=False,  # 5% 커버리지 → 제외
    )

    # 종목 목록 (최소 1000행 이상 = ~13거래일 데이터가 있는 종목만)
    from sqlalchemy import text as sa_text
    sym_result = await db.execute(sa_text(
        "SELECT symbol FROM stock_candles WHERE interval = :iv "
        "GROUP BY symbol HAVING COUNT(*) >= 1000 ORDER BY symbol"
    ), {"iv": interval})
    all_symbols = [r[0] for r in sym_result.fetchall()]
    await _log(f"Step 3/4: {len(all_symbols)}종목 (5m 1000행+), {_SYM_CHUNK}개씩 청크 로딩")

    # 종목 청크별로 팩터 일별 평균 수집
    # 결과: daily_factor_map[date_str] = {factor_idx: [값들(종목별)]}
    daily_factor_sums: dict[str, np.ndarray] = {}  # date → sum of factor values
    daily_factor_counts: dict[str, np.ndarray] = {}  # date → count

    sym_chunks = [all_symbols[i:i + _SYM_CHUNK] for i in range(0, len(all_symbols), _SYM_CHUNK)]
    failed_exprs: set[int] = set()  # 실패한 수식 인덱스 캐시 (이후 청크 스킵)

    # ── 사전 필터: 소량 데이터(5종목, enriched)로 유효 수식 식별 ──
    # end_date를 어제로 제한: 오늘 데이터 미존재 시 1분봉 폴백 방지
    _enrich_end = date.today() - timedelta(days=1)
    _pre_syms = all_symbols[:5]
    _pre_df = await load_enriched_candles(
        symbols=_pre_syms, interval=interval, end_date=_enrich_end, **_ENRICH_OPTS,
    )
    if not _pre_df.is_empty():
        _pre_df = ensure_alpha_features(_pre_df)
        _pre_sample = _pre_df.head(100)  # 100행이면 충분
        for fi, pe in enumerate(polars_exprs):
            try:
                _pre_sample.with_columns(pe.alias("_test"))
            except Exception:
                failed_exprs.add(fi)
        del _pre_df, _pre_sample
        gc.collect()
    _inc = [k.replace("include_", "") for k, v in _ENRICH_OPTS.items() if v]
    _exc = [k.replace("include_", "") for k, v in _ENRICH_OPTS.items() if not v]
    await _log(f"  외부 데이터: {', '.join(_inc)} ✓ | {', '.join(_exc)} ✗ (커버리지 부족)")
    await _log(f"  사전 필터: {len(failed_exprs)}개 수식 실패 → {n_factors - len(failed_exprs)}개만 평가")

    for ci, chunk_syms in enumerate(sym_chunks):
        # enriched candles 로딩 (50종목분 + 외부 데이터 JOIN)
        chunk_df = await load_enriched_candles(
            symbols=chunk_syms, interval=interval, end_date=_enrich_end, **_ENRICH_OPTS,
        )
        if chunk_df.is_empty():
            continue

        # 피처 생성
        chunk_df = ensure_alpha_features(chunk_df)

        # 날짜 컬럼 추가
        chunk_df = chunk_df.with_columns(
            pl.col("dt").dt.date().cast(pl.Utf8).alias("_date_str")
        )

        # 팩터 수식을 50개씩 배치로 적용 (with_columns 1회 + group_by 1회)
        _FACTOR_BATCH = 500
        active_indices = [fi for fi in range(n_factors) if fi not in failed_exprs]
        for fb_start in range(0, len(active_indices), _FACTOR_BATCH):
            fb_indices = active_indices[fb_start:fb_start + _FACTOR_BATCH]
            # 배치: 유효한 수식만 한번에 적용
            batch_aliases: list[pl.Expr] = []
            batch_fi: list[int] = []
            for fi in fb_indices:
                try:
                    batch_aliases.append(polars_exprs[fi].alias(f"_f{fi}"))
                    batch_fi.append(fi)
                except Exception:
                    failed_exprs.add(fi)
            if not batch_fi:
                continue
            try:
                batch_df = chunk_df.with_columns(batch_aliases)
            except Exception:
                # 배치 전체 실패 → 개별로 폴백
                for fi in batch_fi:
                    try:
                        one = chunk_df.with_columns(polars_exprs[fi].alias(f"_f{fi}"))
                        # 성공한 건 개별 group_by
                        col_name = f"_f{fi}"
                        daily_avg = (
                            one.filter(pl.col(col_name).is_not_null() & pl.col(col_name).is_not_nan())
                            .group_by("_date_str")
                            .agg([pl.col(col_name).sum().alias("_sum"), pl.col(col_name).count().alias("_cnt")])
                        )
                        for row in daily_avg.iter_rows(named=True):
                            d = row["_date_str"]
                            if d not in daily_factor_sums:
                                daily_factor_sums[d] = np.zeros(n_factors)
                                daily_factor_counts[d] = np.zeros(n_factors)
                            daily_factor_sums[d][fi] += row["_sum"]
                            daily_factor_counts[d][fi] += row["_cnt"]
                    except Exception:
                        failed_exprs.add(fi)
                continue
            # 배치 성공 → 단일 group_by로 모든 팩터 집계
            sum_aggs = []
            cnt_aggs = []
            for fi in batch_fi:
                col_name = f"_f{fi}"
                sum_aggs.append(pl.col(col_name).sum().alias(f"_s{fi}"))
                cnt_aggs.append(
                    pl.col(col_name).filter(
                        pl.col(col_name).is_not_null() & pl.col(col_name).is_not_nan()
                    ).count().alias(f"_c{fi}")
                )
            daily_agg = batch_df.group_by("_date_str").agg(sum_aggs + cnt_aggs)
            for row in daily_agg.iter_rows(named=True):
                d = row["_date_str"]
                if d not in daily_factor_sums:
                    daily_factor_sums[d] = np.zeros(n_factors)
                    daily_factor_counts[d] = np.zeros(n_factors)
                for fi in batch_fi:
                    daily_factor_sums[d][fi] += row.get(f"_s{fi}", 0) or 0
                    daily_factor_counts[d][fi] += row.get(f"_c{fi}", 0) or 0

        del chunk_df
        gc.collect()
        msg = f"  종목 청크: {min((ci + 1) * _SYM_CHUNK, len(all_symbols))}/{len(all_symbols)} ({(ci + 1) * 100 // len(sym_chunks)}%)"
        if ci == 0:
            msg += f" — 실패 수식 {len(failed_exprs)}개 캐시 (이후 스킵)"
        await _log(msg)

    # 일별 평균 계산: 전체 청크에서 한 번도 값이 없는 팩터 제외
    all_dates = sorted(daily_factor_sums.keys())
    if not all_dates:
        raise ValueError("팩터 일별 데이터가 전혀 수집되지 않았습니다")

    # 팩터별 전체 count 합산 → 0인 팩터 제거
    total_counts_per_factor = np.zeros(n_factors)
    for d in all_dates:
        total_counts_per_factor += daily_factor_counts[d]

    alive_mask = total_counts_per_factor > 0
    n_alive = int(alive_mask.sum())
    n_dead = n_factors - n_alive
    if n_dead > 0:
        await _log(f"  팩터 {n_dead}개 제외 (전체 기간 데이터 0건): 유효 {n_alive}개 남음")
        if n_alive < 3:
            raise ValueError(f"유효 팩터 부족: {n_alive}개 (최소 3개)")
        # valid_factors, ic_values, polars_exprs 동기화
        alive_indices = [i for i in range(n_factors) if alive_mask[i]]
        valid_factors = [valid_factors[i] for i in alive_indices]
        ic_values = [ic_values[i] for i in alive_indices]
        polars_exprs = [polars_exprs[i] for i in alive_indices]
        # daily sums/counts 리인덱싱
        for d in all_dates:
            daily_factor_sums[d] = daily_factor_sums[d][alive_mask]
            daily_factor_counts[d] = daily_factor_counts[d][alive_mask]
        n_factors = n_alive

    valid_dates = sorted([d for d in daily_factor_sums if np.all(daily_factor_counts[d] > 0)])
    await _log(f"  전체 날짜 {len(all_dates)}일 중 유효(모든 팩터 데이터 있음) {len(valid_dates)}일")
    if len(valid_dates) < 30:
        raise ValueError(f"유효 거래일 부족: {len(valid_dates)}일 (최소 30일)")

    # 최근 270일만
    valid_dates = valid_dates[-270:]
    n_days = len(valid_dates)

    daily_matrix = np.zeros((n_days, n_factors))
    for di, d in enumerate(valid_dates):
        counts = daily_factor_counts[d]
        counts = np.where(counts > 0, counts, 1)  # 0 방지
        daily_matrix[di] = daily_factor_sums[d] / counts

    del daily_factor_sums, daily_factor_counts
    gc.collect()

    # Inf/NaN 클리닝 (0 나누기 등으로 발생 가능)
    inf_count = int(np.isinf(daily_matrix).sum())
    nan_count = int(np.isnan(daily_matrix).sum())
    if inf_count > 0 or nan_count > 0:
        await _log(f"  Inf {inf_count}개, NaN {nan_count}개 → 열 중위값으로 대체")
        for col_i in range(n_factors):
            col = daily_matrix[:, col_i]
            valid_mask = np.isfinite(col)
            if valid_mask.sum() > 0:
                median_val = float(np.median(col[valid_mask]))
                col[~valid_mask] = median_val
            else:
                col[:] = 0.0

    await _log(f"Step 3/4: 일별 팩터 평균 {n_days}일 × {n_factors}팩터 구축")

    # EWMA 가중 공분산
    ewma_half_life = 60
    ewma_lambda = 0.5 ** (1.0 / ewma_half_life)
    ewma_weights = np.array([ewma_lambda ** (n_days - 1 - i) for i in range(n_days)])
    ewma_weights /= ewma_weights.sum()

    # 가중 평균 + 중심화
    weighted_mean = ewma_weights @ daily_matrix
    centered = daily_matrix - weighted_mean
    # 가중 공분산: X^T diag(w) X
    cov_matrix = (centered * ewma_weights[:, None]).T @ centered
    n_obs = n_days

    if n_obs < 30:
        raise ValueError(f"유효 거래일 부족: {n_obs}일 (최소 30일)")

    # Ledoit-Wolf 수축 (실제 daily_matrix에 직접 피팅)
    await _log(f"Step 3/4: Ledoit-Wolf 수축 적용 (T/N={n_obs}/{n_factors}={n_obs/n_factors:.1f})")
    try:
        lw = LedoitWolf()
        lw.fit(daily_matrix)  # 실제 관측 데이터로 수축 강도 추정
        shrinkage = lw.shrinkage_
        # EWMA 공분산에 LW 수축 강도만 차용
        target = np.diag(np.diag(cov_matrix))  # 대각행렬 (타겟)
        shrunk_cov = shrinkage * target + (1 - shrinkage) * cov_matrix
        await _log(f"  수축 강도: {shrinkage:.3f} ({shrinkage*100:.1f}%)")
    except Exception as e:
        logger.warning("Ledoit-Wolf failed, using raw cov: %s", e)
        shrunk_cov = cov_matrix

    # 공분산 → 상관행렬
    std_vec = np.sqrt(np.maximum(np.diag(shrunk_cov), 1e-12))
    corr_matrix = shrunk_cov / np.outer(std_vec, std_vec)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    np.fill_diagonal(corr_matrix, 1.0)

    ic_vector = np.array(ic_values)

    await _log(f"Step 3/4 완료: {n_factors}×{n_factors} 상관행렬 (EWMA 반감기 {ewma_half_life}일, {n_obs}일 관측)")

    # ── Step 4: 탐욕적 전진선택 ──
    await _log(f"Step 4/4: 탐욕적 전진선택 시작 (K=3~{max_k}, λ_decorr={lambda_decorr})")
    raw_steps = _greedy_forward_select(
        ic_vector, corr_matrix, max_k=max_k, lambda_decorr=lambda_decorr
    )
    await _log(f"Step 4/4: 전진선택 완료 — {len(raw_steps)}개 스텝")

    # 스텝 로그에 팩터 정보 채우기
    selection_log: list[SelectionStep] = []
    for idx, step in raw_steps:
        factor = valid_factors[idx]
        try:
            expr = parse_expression(factor.expression_str)
            niche = classify_niche(expr)
        except Exception:
            niche = "unknown"
        step.selected_id = str(factor.id)
        step.selected_name = factor.name or factor.expression_str[:40]
        step.niche = niche
        selection_log.append(step)

    # ── K별 복합 팩터 평가 (DB IC 기반, 재계산 없음) ──
    results: list[OptimizationResult] = []
    best_k = 3
    best_sharpe = -np.inf

    for k in range(3, min(len(raw_steps), max_k) + 1):
        await _log(f"Step 4/4: K={k} 조합 가중치 + 복합 IC 계산")
        selected = [raw_steps[i][0] for i in range(k)]
        sel_factors = [valid_factors[i] for i in selected]
        sel_ic = ic_vector[selected]
        sel_corr = corr_matrix[np.ix_(selected, selected)]

        # 가중치
        weights_arr = _two_tier_shrinkage_weights(
            sel_ic, sel_corr, shrinkage_delta=shrinkage_delta
        )
        weights_dict = {str(sel_factors[i].id): round(float(weights_arr[i]), 4) for i in range(k)}

        # 복합 IC/ICIR 추정 (Grinold의 법칙 기반)
        try:
            sel_cov = corr_matrix[np.ix_(selected, selected)]
            w = weights_arr
            comp_var = float(w @ sel_cov @ w)
            comp_ic = float(w @ sel_ic) / max(math.sqrt(comp_var), 1e-8) if comp_var > 0 else 0.0
            # DB에서 개별 팩터 ICIR로 복합 ICIR 추정
            sel_icir = np.array([float(sel_factors[i].icir or 0) for i in range(k)])
            comp_icir = float(w @ sel_icir) / max(math.sqrt(comp_var), 1e-8) if comp_var > 0 else 0.0
            # Sharpe: DB 개별 Sharpe의 가중 추정
            sel_sharpe = np.array([float(sel_factors[i].sharpe or 0) for i in range(k)])
            comp_sharpe = float(w @ sel_sharpe)
        except Exception:
            comp_ic = float(np.mean(sel_ic))
            comp_icir = 0.0
            comp_sharpe = 0.0

        # 평균 상관
        pairs_corr = []
        for i in range(k):
            for j in range(i + 1, k):
                pairs_corr.append(abs(sel_corr[i, j]))
        avg_corr = float(np.mean(pairs_corr)) if pairs_corr else 0.0

        # 수식 문자열
        expr_parts = []
        for i, factor in enumerate(sel_factors):
            w = float(weights_arr[i])
            expr_parts.append(f"{w:.4f} * ({factor.expression_str})")
        composite_expr = " + ".join(expr_parts)

        opt_result = OptimizationResult(
            k=k,
            factor_ids=[str(f.id) for f in sel_factors],
            factor_names=[f.name or f.expression_str[:40] for f in sel_factors],
            weights=weights_dict,
            composite_ic=round(comp_ic, 6),
            composite_icir=round(comp_icir, 4),
            composite_sharpe=round(comp_sharpe, 3),
            avg_correlation=round(avg_corr, 4),
            expression_str=composite_expr,
        )
        results.append(opt_result)
        await _log(f"  K={k}: IC={comp_ic:.4f}, ICIR={comp_icir:.3f}, Sharpe={comp_sharpe:.2f}, avg_corr={avg_corr:.3f}")

        if comp_sharpe > best_sharpe:
            best_sharpe = comp_sharpe
            best_k = k

    await _log(f"완료: best_k={best_k}, best_sharpe={best_sharpe:.3f}, 후보={len(candidates)}개, 유효={n_factors}개, {n_obs}일 관측, {time.time() - _t0:.1f}초")

    output = AutoOptimizeOutput(
        best_k=best_k,
        results=results,
        selection_log=selection_log,
        correlation_matrix={
            "factor_ids": [str(f.id) for f in valid_factors],
            "factor_names": [f.name or "" for f in valid_factors],
            "matrix": corr_matrix.tolist(),
        },
        candidate_count=len(candidates),
        logs=_logs,
    )

    # Redis에 완료 결과 저장
    await _redis_update({
        "status": "completed",
        "result": _serialize_optimize_output(output),
        "logs": _logs,
    })

    return output


def _serialize_optimize_output(output: AutoOptimizeOutput) -> dict:
    """AutoOptimizeOutput을 JSON-serializable dict로 변환."""
    return {
        "best_k": output.best_k,
        "results": [
            {
                "k": r.k,
                "factor_ids": r.factor_ids,
                "factor_names": r.factor_names,
                "weights": r.weights,
                "composite_ic": r.composite_ic,
                "composite_icir": r.composite_icir,
                "composite_sharpe": r.composite_sharpe,
                "avg_correlation": r.avg_correlation,
                "expression_str": r.expression_str,
            }
            for r in output.results
        ],
        "selection_log": [
            {
                "step": s.step,
                "selected_id": s.selected_id,
                "selected_name": s.selected_name,
                "niche": s.niche,
                "ic": s.ic,
                "reason": s.reason,
                "cumulative_ir2": s.cumulative_ir2,
                "avg_correlation": s.avg_correlation,
            }
            for s in output.selection_log
        ],
        "correlation_matrix": output.correlation_matrix,
        "candidate_count": output.candidate_count,
        "logs": output.logs,
    }
