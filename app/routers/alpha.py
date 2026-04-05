"""알파 마이닝 REST API 라우터."""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import date

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.alpha.models import AlphaExperience, AlphaFactor, AlphaMiningRun
from app.alpha.runner import execute_alpha_mining
from app.alpha.schemas import (
    AlphaFactorBacktestRequest,
    AlphaFactorPageResponse,
    AlphaFactorResponse,
    AlphaFactoryStartRequest,
    AlphaFactoryStatusResponse,
    AlphaMineRequest,
    AlphaMineResponse,
    AlphaMiningRunResponse,
    AlphaMiningRunSummary,
    AutoOptimizeRequest,
    CompositeFactorBuildRequest,
    CompositeFactorResponse,
    CorrelationRequest,
    CorrelationMatrixResponse,
    FactorChatCreateResponse,
    FactorChatMessageRequest,
    FactorChatMessageResponse,
    FactorChatSessionResponse,
    MiningIterationLogs,
    MiningReportResponse,
    MiningReportsRangeResponse,
)
from app.alpha.universe import Universe, get_universe_info
from app.core.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/alpha", tags=["alpha"])


# ── 유니버스 ──


@router.get("/universes")
async def list_universes():
    """사용 가능한 유니버스 목록과 종목 수 반환."""
    return await get_universe_info()


# ── 마이닝 실행 ──

@router.post("/mine", status_code=202, response_model=AlphaMineResponse)
async def start_mining(data: AlphaMineRequest, db: AsyncSession = Depends(get_db)):
    """알파 마이닝 비동기 실행. 즉시 run_id 반환."""
    start = date.fromisoformat(data.start_date)
    end = date.fromisoformat(data.end_date)

    if start >= end:
        raise HTTPException(400, "시작일이 종료일보다 이전이어야 합니다.")

    run = AlphaMiningRun(
        name=data.name,
        context={"text": data.context, "universe": data.universe},
        config={
            "start_date": data.start_date,
            "end_date": data.end_date,
            "universe": data.universe,
            "interval": data.interval,
            "max_iterations": data.max_iterations,
            "ic_threshold": data.ic_threshold,
            "orthogonality_threshold": data.orthogonality_threshold,
            "use_pysr": data.use_pysr,
            "pysr_max_size": data.pysr_max_size,
            "pysr_parsimony": data.pysr_parsimony,
        },
        status="PENDING",
        progress=0,
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)

    asyncio.create_task(
        execute_alpha_mining(
            run_id=run.id,
            name=data.name,
            context=data.context,
            universe=data.universe,
            start_date=start,
            end_date=end,
            max_iterations=data.max_iterations,
            ic_threshold=data.ic_threshold,
            orthogonality_threshold=data.orthogonality_threshold,
            use_pysr=data.use_pysr,
            interval=data.interval,
            seed_factor_ids=data.seed_factor_ids or None,
        )
    )

    return AlphaMineResponse(
        id=str(run.id),
        status=run.status,
        created_at=run.created_at.isoformat(),
    )


@router.get("/mine/{run_id}", response_model=AlphaMiningRunResponse)
async def get_mining_run(run_id: str, db: AsyncSession = Depends(get_db)):
    """마이닝 실행 상태/진행률 조회."""
    result = await db.execute(
        select(AlphaMiningRun).where(AlphaMiningRun.id == uuid.UUID(run_id))
    )
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(404, "Mining run not found")

    return AlphaMiningRunResponse(
        id=str(run.id),
        name=run.name,
        context=run.context,
        config=run.config,
        status=run.status,
        progress=run.progress,
        factors_found=run.factors_found,
        total_evaluated=run.total_evaluated,
        error_message=run.error_message,
        has_logs=run.iteration_logs is not None,
        created_at=run.created_at.isoformat(),
        completed_at=run.completed_at.isoformat() if run.completed_at else None,
    )


@router.get("/mine/{run_id}/logs", response_model=MiningIterationLogs)
async def get_mining_logs(run_id: str, db: AsyncSession = Depends(get_db)):
    """마이닝 실행의 상세 iteration 로그 조회."""
    result = await db.execute(
        select(AlphaMiningRun).where(AlphaMiningRun.id == uuid.UUID(run_id))
    )
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(404, "Mining run not found")

    logs = run.iteration_logs or {}
    return MiningIterationLogs(
        run_id=str(run.id),
        iterations=logs.get("iterations", []),
        summary=logs.get("summary", {}),
    )


@router.get("/mines", response_model=list[AlphaMiningRunSummary])
async def list_mining_runs(db: AsyncSession = Depends(get_db)):
    """마이닝 실행 목록."""
    result = await db.execute(
        select(AlphaMiningRun).order_by(AlphaMiningRun.created_at.desc())
    )
    runs = result.scalars().all()

    return [
        AlphaMiningRunSummary(
            id=str(r.id),
            name=r.name,
            status=r.status,
            progress=r.progress,
            factors_found=r.factors_found,
            total_evaluated=r.total_evaluated,
            created_at=r.created_at.isoformat(),
        )
        for r in runs
    ]


@router.delete("/mine/{run_id}", status_code=204)
async def delete_mining_run(run_id: str, db: AsyncSession = Depends(get_db)):
    """마이닝 실행 삭제 (cascade로 팩터도 삭제)."""
    await db.execute(
        delete(AlphaMiningRun).where(AlphaMiningRun.id == uuid.UUID(run_id))
    )
    await db.commit()


# ── 팩터 조회 ──

_ALLOWED_SORT_COLUMNS = {
    "ic_mean", "icir", "sharpe", "max_drawdown",
    "generation", "fitness_composite", "created_at",
}


@router.get("/factors", response_model=AlphaFactorPageResponse)
async def list_factors(
    status: str | None = None,
    min_ic: float | None = None,
    causal_robust: bool | None = None,
    interval: str | None = None,
    factor_type: str | None = None,
    search: str | None = None,
    sort_by: str = "created_at",
    order: str = "desc",
    offset: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
):
    """팩터 목록 (status/min_ic/causal_robust 필터, 멀티 정렬, 페이지네이션).

    sort_by/order는 쉼표 구분 문자열로 멀티 정렬 지원.
    예: sort_by=ic_mean,sharpe&order=desc,asc
    """
    sort_cols = [s.strip() for s in sort_by.split(",") if s.strip()]
    sort_orders = [s.strip() for s in order.split(",") if s.strip()]

    order_clauses = []
    for i, sc in enumerate(sort_cols):
        if sc not in _ALLOWED_SORT_COLUMNS:
            continue
        col = getattr(AlphaFactor, sc)
        od = sort_orders[i] if i < len(sort_orders) else "desc"
        order_clauses.append(
            col.asc().nulls_last() if od == "asc" else col.desc().nulls_last()
        )
    if not order_clauses:
        order_clauses.append(AlphaFactor.ic_mean.desc().nulls_last())

    # WHERE 조건 구성
    filters = []
    if status:
        filters.append(AlphaFactor.status == status)
    if min_ic is not None:
        filters.append(AlphaFactor.ic_mean >= min_ic)
    if causal_robust is not None:
        filters.append(AlphaFactor.causal_robust == causal_robust)
    if interval:
        filters.append(AlphaFactor.interval == interval)
    if factor_type:
        filters.append(AlphaFactor.factor_type == factor_type)
    if search:
        like_pat = f"%{search}%"
        filters.append(
            AlphaFactor.expression_str.ilike(like_pat)
            | AlphaFactor.name.ilike(like_pat)
        )

    # 전체 개수 (ORDER BY 없이 — 정렬은 COUNT에 불필요)
    count_q = select(func.count()).select_from(AlphaFactor)
    for f in filters:
        count_q = count_q.where(f)
    total = await db.scalar(count_q)

    # 페이지 데이터 (WHERE + ORDER BY + OFFSET/LIMIT)
    data_q = select(AlphaFactor)
    for f in filters:
        data_q = data_q.where(f)
    data_q = data_q.order_by(*order_clauses).offset(offset).limit(limit)

    result = await db.execute(data_q)
    factors = result.scalars().all()

    return AlphaFactorPageResponse(
        items=[_factor_to_response(f) for f in factors],
        total=total or 0,
    )


@router.get("/factor/{factor_id}", response_model=AlphaFactorResponse)
async def get_factor(factor_id: str, db: AsyncSession = Depends(get_db)):
    """팩터 상세."""
    result = await db.execute(
        select(AlphaFactor).where(AlphaFactor.id == uuid.UUID(factor_id))
    )
    factor = result.scalar_one_or_none()
    if not factor:
        raise HTTPException(404, "Factor not found")

    return _factor_to_response(factor)


@router.delete("/factor/{factor_id}", status_code=204)
async def delete_factor(factor_id: str, db: AsyncSession = Depends(get_db)):
    """팩터 삭제."""
    await db.execute(
        delete(AlphaFactor).where(AlphaFactor.id == uuid.UUID(factor_id))
    )
    await db.commit()


@router.post("/factors/delete-batch", status_code=204)
async def delete_factors_batch(
    body: dict,
    db: AsyncSession = Depends(get_db),
):
    """팩터 일괄 삭제."""
    ids = body.get("factor_ids", [])
    if not ids:
        return
    uuids = [uuid.UUID(fid) for fid in ids]
    await db.execute(
        delete(AlphaFactor).where(AlphaFactor.id.in_(uuids))
    )
    await db.commit()


@router.post("/factors/prune")
async def prune_factors(
    max_per_interval: int = 3000,
    dry_run: bool = False,
    db: AsyncSession = Depends(get_db),
):
    """인터벌별 팩터 속아내기(Pruning).

    성능 하위 팩터를 삭제하여 인터벌별 max_per_interval 이하로 유지.

    보존 우선순위:
    1. causal_robust=true -> 무조건 보존
    2. 니치별 최소 비율 보장 (다양성 유지)
    3. fitness_composite 상위 -> 보존 (없으면 ic_mean 폴백)

    dry_run=true면 삭제하지 않고 삭제될 팩터 수만 반환.
    """
    from app.alpha.ast_converter import classify_niche, parse_expression

    if max_per_interval < 1:
        raise HTTPException(400, "max_per_interval은 1 이상이어야 합니다.")

    # 1. 인터벌별 팩터 수 조회
    counts_result = await db.execute(
        select(AlphaFactor.interval, func.count(AlphaFactor.id))
        .group_by(AlphaFactor.interval)
    )
    interval_counts = {row[0]: row[1] for row in counts_result.fetchall()}

    results: dict[str, dict] = {}
    total_pruned = 0

    for interval_key, count in interval_counts.items():
        if count <= max_per_interval:
            results[interval_key] = {
                "before": count,
                "after": count,
                "pruned": 0,
            }
            continue

        # 2. 해당 인터벌의 모든 팩터 로드
        factors_result = await db.execute(
            select(
                AlphaFactor.id,
                AlphaFactor.ic_mean,
                AlphaFactor.fitness_composite,
                AlphaFactor.causal_robust,
                AlphaFactor.expression_str,
            ).where(AlphaFactor.interval == interval_key)
        )
        all_factors = factors_result.fetchall()

        # 3. 3단계 분류: 보존 / 우선삭제 / 경쟁
        causal_keep = [f for f in all_factors if f.causal_robust is True]    # 무조건 보존
        mirages = [f for f in all_factors if f.causal_robust is False]       # 우선 삭제 (가짜)
        unvalidated = [f for f in all_factors if f.causal_robust is None]    # fitness 경쟁

        remaining_slots = max_per_interval - len(causal_keep)
        if remaining_slots <= 0:
            results[interval_key] = {
                "before": count,
                "after": count,
                "pruned": 0,
                "note": "causal_robust 팩터만으로 limit 초과",
            }
            continue

        # 4. 미검증 팩터를 니치+fitness로 경쟁 (미라지 제외)
        def _sort_key(f):
            """fitness_composite 우선, 없으면 ic_mean 폴백."""
            fc = f.fitness_composite
            if fc is not None:
                return fc
            return f.ic_mean if f.ic_mean is not None else -999.0

        niche_map: dict[str, list] = {}
        for f in unvalidated:  # ← 미라지 제외, 미검증만 경쟁
            try:
                expr = parse_expression(f.expression_str or "")
                niche = classify_niche(expr)
            except Exception:
                niche = "unknown"
            niche_map.setdefault(niche, []).append(f)

        num_niches = max(len(niche_map), 1)
        min_per_niche = max(10, remaining_slots // (num_niches * 2))

        niche_guaranteed: list = []
        niche_rest: list = []
        for _niche, factors_in_niche in niche_map.items():
            sorted_niche = sorted(factors_in_niche, key=_sort_key, reverse=True)
            niche_guaranteed.extend(sorted_niche[:min_per_niche])
            niche_rest.extend(sorted_niche[min_per_niche:])

        # 5. 나머지 슬롯을 fitness 순으로 채움
        slots_after_niche = remaining_slots - len(niche_guaranteed)
        if slots_after_niche > 0:
            sorted_rest = sorted(niche_rest, key=_sort_key, reverse=True)
            keep_rest = sorted_rest[:slots_after_niche]
        else:
            keep_rest = []

        # 6. 보존 목록 = causal + 미검증(niche+fitness) + 미라지(남은 슬롯만)
        keep_ids = set()
        keep_ids.update(f.id for f in causal_keep)
        keep_ids.update(f.id for f in niche_guaranteed)
        keep_ids.update(f.id for f in keep_rest)

        # 미라지는 슬롯이 남을 때만 (fitness 순) — 사실상 거의 삭제됨
        mirage_slots = max_per_interval - len(keep_ids)
        if mirage_slots > 0 and mirages:
            sorted_mirages = sorted(mirages, key=_sort_key, reverse=True)
            keep_ids.update(f.id for f in sorted_mirages[:mirage_slots])

        # 7. 삭제 대상
        prune_ids = [f.id for f in all_factors if f.id not in keep_ids]
        mirages_pruned = sum(1 for f in mirages if f.id not in keep_ids)

        if not dry_run and prune_ids:
            # alpha_experiences FK가 SET NULL이므로 명시적 정리
            await db.execute(
                delete(AlphaExperience).where(
                    AlphaExperience.factor_id.in_(prune_ids)
                )
            )
            await db.execute(
                delete(AlphaFactor).where(AlphaFactor.id.in_(prune_ids))
            )

        pruned = len(prune_ids)
        total_pruned += pruned
        results[interval_key] = {
            "before": count,
            "after": count - pruned,
            "pruned": pruned,
            "mirages_pruned": mirages_pruned,
            "causal_kept": len(causal_keep),
            "niche_distribution": {n: len(fs) for n, fs in niche_map.items()},
        }

    if not dry_run:
        await db.commit()

    return {
        "dry_run": dry_run,
        "max_per_interval": max_per_interval,
        "total_pruned": total_pruned,
        "intervals": results,
    }


@router.post("/factor/{factor_id}/backtest", status_code=202)
async def backtest_with_factor(
    factor_id: str,
    data: AlphaFactorBacktestRequest,
    db: AsyncSession = Depends(get_db),
):
    """횡단면 포트폴리오 기반 팩터 백테스트.

    매일 전체 종목을 팩터 값으로 랭킹하여 상위 top_pct% 종목을 매수한다.
    symbols가 비어 있으면 마이닝 유니버스를 기본값으로 사용한다.
    """
    result = await db.execute(
        select(AlphaFactor).where(AlphaFactor.id == uuid.UUID(factor_id))
    )
    factor = result.scalar_one_or_none()
    if not factor:
        raise HTTPException(404, "Factor not found")

    # 마이닝 run config 조회 (symbols/날짜 폴백 공용)
    run_config: dict | None = None
    need_run_config = not data.symbols or not data.start_date or not data.end_date
    if need_run_config and factor.mining_run_id:
        run_result = await db.execute(
            select(AlphaMiningRun.config).where(
                AlphaMiningRun.id == factor.mining_run_id
            )
        )
        run_config = run_result.scalar_one_or_none()

    # symbols가 비어 있으면 마이닝 유니버스에서 가져옴
    symbols = data.symbols if data.symbols else None
    if not symbols:
        universe_code = None
        if run_config and run_config.get("universe"):
            universe_code = run_config["universe"]
        else:
            # mining_run_id 없는 팩터 (진화 모집단 등): 기본 유니버스 폴백
            universe_code = "KOSPI200"
            logger.info("팩터 백테스트: mining_run 없음, 기본 유니버스 KOSPI200 사용")

        from app.alpha.universe import Universe, resolve_universe

        symbols = await resolve_universe(Universe(universe_code))
        if not symbols:
            raise HTTPException(
                500,
                f"유니버스 '{universe_code}' 리졸브 결과가 비어 있습니다.",
            )
        logger.info(
            "팩터 백테스트: 유니버스 '%s' 사용 (%d종목)",
            universe_code,
            len(symbols),
        )

    # 날짜 범위: 요청값 → 마이닝 config 폴백 → 기본값 (최근 1년)
    start_str = data.start_date or (run_config.get("start_date") if run_config else None)
    end_str = data.end_date or (run_config.get("end_date") if run_config else None)
    if not start_str or not end_str:
        from datetime import timedelta
        end_str = end_str or date.today().isoformat()
        start_str = start_str or (date.today() - timedelta(days=365)).isoformat()
        logger.info("팩터 백테스트: 날짜 범위 미지정, 기본값 사용 (%s ~ %s)", start_str, end_str)

    from app.backtest.cost_model import CostConfig, default_cost_config
    from app.backtest.models import BacktestRun
    from app.alpha.factor_backtest import execute_factor_backtest

    start = date.fromisoformat(start_str)
    end = date.fromisoformat(end_str)

    # 인터벌 검증: 팩터의 원래 인터벌과 불일치 방지
    bt_interval = data.interval
    if factor.interval and bt_interval != factor.interval:
        raise HTTPException(
            400,
            f"팩터 인터벌({factor.interval})과 요청 인터벌({bt_interval})이 다릅니다. "
            f"팩터에 맞는 인터벌을 사용하세요.",
        )
    if not factor.interval:
        bt_interval = "1d"  # interval 컬럼 추가 이전 레거시 팩터

    # 분봉 백테스트: OOM 방지를 위해 종목 수 제한
    from app.alpha.interval import max_symbols_for_mining
    max_sym = max_symbols_for_mining(bt_interval)
    if len(symbols) > max_sym:
        symbols = symbols[:max_sym]
        logger.info("팩터 백테스트: %s — 종목 수 %d 제한 (OOM 방지)", bt_interval, max_sym)

    # 거래 비용 설정: 커스텀 값이 지정되면 적용, 아니면 인터벌 기본값
    if data.buy_commission is not None or data.sell_commission is not None or data.slippage_pct is not None:
        base_cfg = default_cost_config(bt_interval)
        cost_cfg = CostConfig(
            buy_commission=data.buy_commission if data.buy_commission is not None else base_cfg.buy_commission,
            sell_commission=data.sell_commission if data.sell_commission is not None else base_cfg.sell_commission,
            slippage_pct=data.slippage_pct if data.slippage_pct is not None else base_cfg.slippage_pct,
            slippage_model=base_cfg.slippage_model,
            vs_price_impact=base_cfg.vs_price_impact,
            vs_volume_limit=base_cfg.vs_volume_limit,
        )
    else:
        cost_cfg = default_cost_config(bt_interval)

    # ── 듀얼 팩터 모드 분기 ──
    intraday_factor = None
    if data.intraday_factor_id:
        intraday_result = await db.execute(
            select(AlphaFactor).where(AlphaFactor.id == uuid.UUID(data.intraday_factor_id))
        )
        intraday_factor = intraday_result.scalar_one_or_none()
        if not intraday_factor:
            raise HTTPException(404, "Intraday factor not found")

    if intraday_factor:
        # 듀얼 팩터: 일봉 팩터로 선별, 분봉 팩터로 진입/퇴출
        run = BacktestRun(
            strategy_name=f"Dual: {factor.name} + {intraday_factor.name}",
            strategy_json={
                "name": f"Dual: {factor.name} + {intraday_factor.name}",
                "daily_expression": factor.expression_str,
                "intraday_expression": intraday_factor.expression_str,
                "mode": "dual_factor",
                "interval": bt_interval,
                "intraday_interval": data.intraday_interval,
                "top_pct": data.top_pct,
                "max_positions": data.max_positions,
                "intraday_entry_threshold": data.intraday_entry_threshold,
                "intraday_exit_threshold": data.intraday_exit_threshold,
            },
            start_date=start,
            end_date=end,
            initial_capital=float(data.initial_capital),
            cost_config=cost_cfg.model_dump(),
            status="PENDING",
            progress=0,
        )
        db.add(run)
        await db.commit()
        await db.refresh(run)

        from app.alpha.factor_backtest import execute_dual_factor_backtest

        asyncio.create_task(
            execute_dual_factor_backtest(
                run_id=run.id,
                daily_expression_str=factor.expression_str,
                intraday_expression_str=intraday_factor.expression_str,
                symbols=symbols,
                start_date=start,
                end_date=end,
                initial_capital=data.initial_capital,
                top_pct=data.top_pct,
                max_positions=data.max_positions,
                intraday_interval=data.intraday_interval,
                intraday_entry_threshold=data.intraday_entry_threshold,
                intraday_exit_threshold=data.intraday_exit_threshold,
                stop_loss_pct=data.stop_loss_pct,
                trailing_stop_pct=data.trailing_stop_pct,
                cost_config=cost_cfg,
                use_limit_orders=data.use_limit_orders,
                strict_fill=data.strict_fill,
                limit_ttl_bars=data.limit_ttl_bars,
                collect_daily_snapshots=data.collect_daily_snapshots,
            )
        )
    else:
        # 단일 팩터 모드 (기존 로직)
        run = BacktestRun(
            strategy_name=f"Alpha: {factor.name}",
            strategy_json={
                "name": f"Alpha: {factor.name}",
                "expression": factor.expression_str,
                "mode": "cross_sectional_portfolio",
                "interval": bt_interval,
                "top_pct": data.top_pct,
                "max_positions": data.max_positions,
                "rebalance_freq": data.rebalance_freq,
                "band_threshold": data.band_threshold,
            },
            start_date=start,
            end_date=end,
            initial_capital=float(data.initial_capital),
            cost_config=cost_cfg.model_dump(),
            status="PENDING",
            progress=0,
        )
        db.add(run)
        await db.commit()
        await db.refresh(run)

        asyncio.create_task(
            execute_factor_backtest(
                run_id=run.id,
                expression_str=factor.expression_str,
                symbols=symbols,
                start_date=start,
                end_date=end,
                initial_capital=data.initial_capital,
                top_pct=data.top_pct,
                max_positions=data.max_positions,
                rebalance_freq=data.rebalance_freq,
                band_threshold=data.band_threshold,
                cost_config=cost_cfg,
                interval=bt_interval,
                stop_loss_pct=data.stop_loss_pct,
                trailing_stop_pct=data.trailing_stop_pct,
                max_drawdown_pct=data.max_drawdown_pct,
                eod_liquidation=data.eod_liquidation,
                skip_opening_minutes=data.skip_opening_minutes,
                engine=data.engine,
                use_limit_orders=data.use_limit_orders,
                strict_fill=data.strict_fill,
                limit_ttl_bars=data.limit_ttl_bars,
                collect_daily_snapshots=data.collect_daily_snapshots,
            )
        )

    return {"backtest_run_id": str(run.id)}


@router.post("/factor/{factor_id}/validate")
async def validate_factor(factor_id: str, db: AsyncSession = Depends(get_db)):
    """인과 검증 트리거 — DoWhy 4단계 인과 검증 실행."""
    from app.alpha.causal_runner import validate_single_factor
    from app.alpha.schemas import CausalValidationResponse

    result = await db.execute(
        select(AlphaFactor).where(AlphaFactor.id == uuid.UUID(factor_id))
    )
    factor = result.scalar_one_or_none()
    if not factor:
        raise HTTPException(404, "Factor not found")

    try:
        causal_result = await validate_single_factor(uuid.UUID(factor_id), db)
    except Exception as e:
        raise HTTPException(500, f"Causal validation failed: {str(e)[:200]}")

    return CausalValidationResponse(
        factor_id=str(factor.id),
        is_causally_robust=causal_result.is_causally_robust,
        causal_effect_size=causal_result.causal_effect_size,
        p_value=causal_result.p_value,
        placebo_passed=causal_result.placebo_passed,
        placebo_effect=causal_result.placebo_effect,
        random_cause_passed=causal_result.random_cause_passed,
        random_cause_delta=causal_result.random_cause_delta,
        regime_shift_passed=getattr(causal_result, "regime_shift_passed", False),
        regime_ate_first_half=getattr(causal_result, "regime_ate_first_half", 0.0),
        regime_ate_second_half=getattr(causal_result, "regime_ate_second_half", 0.0),
        dag_edges=causal_result.dag_edges,
    )


@router.post("/factors/validate-batch", status_code=202)
async def validate_factors_batch_endpoint(
    body: dict,
    db: AsyncSession = Depends(get_db),
):
    """선택된 팩터 일괄 인과 검증 (비동기). job_id를 반환하고 백그라운드에서 실행."""
    from app.alpha.factory_client import get_factory_client

    ids = body.get("factor_ids", [])
    if not ids:
        return {"job_id": None, "total": 0, "skipped": 0}

    uuids = [uuid.UUID(fid) for fid in ids]
    result = await db.execute(
        select(AlphaFactor).where(AlphaFactor.id.in_(uuids))
    )
    factors = result.scalars().all()

    to_validate = []
    skipped = 0

    for factor in factors:
        if factor.causal_robust is not None:
            skipped += 1
        else:
            to_validate.append(factor)

    if not to_validate:
        return {"job_id": None, "total": 0, "skipped": skipped}

    job_id = uuid.uuid4().hex[:12]
    factor_ids = [f.id for f in to_validate]

    client = get_factory_client()
    await client.start_validation_batch(factor_ids, job_id, len(factor_ids))

    return {"job_id": job_id, "total": len(factor_ids), "skipped": skipped}


@router.get("/validate/{job_id}/status")
async def get_validation_status(job_id: str, since_idx: int = 0):
    """인과 검증 잡 진행 상황 조회. since_idx로 새 로그만 반환."""
    from app.alpha.factory_client import get_factory_client

    client = get_factory_client()
    progress = await client.get_validation_progress(job_id)
    if progress is None:
        raise HTTPException(404, "Validation job not found")

    # 딕셔너리 복사 후 로그 슬라이싱 (원본 변경 방지)
    result = {**progress}
    if "logs" in result:
        result["logs"] = result["logs"][since_idx:]

    return result


@router.post("/validate/{job_id}/cancel")
async def cancel_validation(job_id: str):
    """인과 검증 잡 중단 요청."""
    from app.alpha.causal_runner import _validation_jobs

    job = _validation_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Validation job not found")
    job["cancelled"] = True
    return {"cancelled": True}


# ── Causal Sweep (전수 인과검증) ──


@router.post("/causal-sweep")
async def start_causal_sweep(
    interval: str = "1d",
    db: AsyncSession = Depends(get_db),
):
    """마이닝 중단 → 미검증 전수 인과검증 → 자동 재시작."""
    from app.alpha.factory_client import get_factory_client
    from app.core.redis import get_client as get_redis

    # 이미 sweep 진행 중이면 기존 job 반환 (중복 실행 방지)
    # [2026-04-01] Worker 재시작 시 in-memory job 소멸 → Redis 플래그만 남는 좀비 방지
    redis = get_redis()
    existing_job = await redis.get("alpha:causal_sweep:running")
    if existing_job:
        existing_id = existing_job if isinstance(existing_job, str) else existing_job.decode()
        # in-memory job이 실제로 살아있는지 확인
        progress = await client.get_validation_progress(existing_id)
        if progress and progress.get("status") != "completed":
            return {"job_id": existing_id, "total": 0, "interval": interval, "auto_restart": False, "already_running": True}
        # 좀비 플래그 — Worker 재시작 등으로 job이 사라짐 → 정리 후 새로 시작
        await redis.delete("alpha:causal_sweep:running")
        logger.info("좀비 sweep 플래그 정리: %s (in-memory job 부재)", existing_id)

    client = get_factory_client(interval)

    # 1. Factory 상태 확인 + 중지 (이미 중지 상태면 스킵)
    status = await client.get_status()
    was_running = status.get("running", False)
    last_config = status.get("config") or {}

    if was_running:
        try:
            await client.stop(interval=interval)
        except TypeError:
            await client.stop()
        # user_stopped 플래그는 설정하지 않음 — sweep 완료 후 자동 재시작해야 하므로

    # 2a. IC < threshold 팩터를 일괄 MIRAGE 처리 (검증 가치 없음)
    low_ic_result = await db.execute(
        update(AlphaFactor)
        .where(
            AlphaFactor.causal_robust.is_(None),
            AlphaFactor.status.notin_(["validated", "mirage", "rejected"]),
            AlphaFactor.ic_mean < 0.03,
            AlphaFactor.interval == interval,
        )
        .values(causal_robust=False, status="mirage", causal_failure_type="LOW_IC")
    )
    await db.commit()
    low_ic_count = low_ic_result.rowcount
    if low_ic_count > 0:
        logger.info("Causal sweep: %d factors marked as LOW_IC mirage (IC < 0.03)", low_ic_count)

    # 2b. 검증 대상 팩터 수집 (IC >= threshold)
    result = await db.execute(
        select(AlphaFactor.id).where(
            AlphaFactor.causal_robust.is_(None),
            AlphaFactor.status.notin_(["validated", "mirage", "rejected"]),
            AlphaFactor.ic_mean >= 0.03,
            AlphaFactor.interval == interval,
        ).order_by(AlphaFactor.mining_run_id, AlphaFactor.ic_mean.desc())
    )
    factor_ids = [str(row[0]) for row in result.fetchall()]

    if not factor_ids:
        # 미검증 없음 — 팩토리가 돌고 있었으면 재시작
        if was_running and last_config:
            from app.core.redis import get_client as get_redis

            redis = get_redis()
            # user_stopped 존중 — 플래그 삭제하지 않음
            _flag = await redis.get("alpha:factory:user_stopped")
            if not (_flag and str(_flag) == "true"):
                await _restart_factory(client, last_config)
        return {"job_id": None, "total": 0, "interval": interval, "auto_restart": was_running}

    # 3. 배치 검증 시작 (기존 인프라 재사용)
    job_id = uuid.uuid4().hex[:12]

    # Sweep 진행 플래그 설정 (워치독이 팩토리를 재시작하지 않도록)
    from app.core.redis import get_client as get_redis

    redis = get_redis()
    await redis.set("alpha:causal_sweep:running", job_id)

    await client.start_validation_batch(factor_ids, job_id, len(factor_ids))

    # 4. 완료 시 자동 재시작 태스크
    if was_running and last_config:
        asyncio.create_task(
            _auto_restart_after_sweep(client, job_id, interval, last_config)
        )

    return {
        "job_id": job_id,
        "total": len(factor_ids),
        "interval": interval,
        "auto_restart": was_running,
    }


@router.post("/causal-sweep/cancel")
async def cancel_causal_sweep(
    job_id: str,
    interval: str = "1d",
):
    """인과검증 취소 + 마이닝 즉시 재시작."""
    from app.alpha.causal_runner import _validation_jobs
    from app.alpha.factory_client import get_factory_client

    # 1. 검증 취소
    job = _validation_jobs.get(job_id)
    if job and job.get("status") != "completed":
        job["cancelled"] = True

    # 2. Sweep 플래그 삭제
    from app.core.redis import get_client as get_redis

    redis = get_redis()
    await redis.delete("alpha:causal_sweep:running")

    # 3. Factory 재시작
    client = get_factory_client(interval)
    status = await client.get_status()
    last_config = status.get("config") or {}

    factory_restarted = False
    if last_config:
        # user_stopped 존중 — 플래그 삭제하지 않음
        _flag = await redis.get("alpha:factory:user_stopped")
        if not (_flag and str(_flag) == "true"):
            try:
                await _restart_factory(client, last_config)
                factory_restarted = True
            except Exception:
                logger.exception("Causal sweep cancel: factory restart failed")

    return {"cancelled": True, "factory_restarted": factory_restarted}


async def _restart_factory(client, config: dict) -> None:
    """config 딕셔너리로 팩토리 재시작. 내부 헬퍼."""
    # config의 키를 scheduler.start()가 받는 kwargs로 매핑
    kwargs = {k: v for k, v in config.items() if v is not None}
    await client.start(**kwargs)


async def _auto_restart_after_sweep(
    client,
    job_id: str,
    interval: str,
    last_config: dict,
) -> None:
    """검증 완료 대기 후 Factory 자동 재시작. 백그라운드 태스크."""
    try:
        while True:
            await asyncio.sleep(5)
            progress = await client.get_validation_progress(job_id)
            if progress is None:
                break
            if progress.get("status") == "completed" or progress.get("cancelled"):
                break

        # Sweep 완료 플래그 삭제 + Factory 재시작
        from app.core.redis import get_client as get_redis

        redis = get_redis()
        try:
            await redis.delete("alpha:causal_sweep:running")
        except Exception:
            pass

        # user_stopped 존중 — 플래그 삭제하지 않음
        _flag = await redis.get("alpha:factory:user_stopped")
        if not (_flag and str(_flag) == "true"):
            await _restart_factory(client, last_config)
        logger.info("Causal sweep complete — factory restarted (interval=%s)", interval)
    except Exception:
        logger.exception("Failed to restart factory after causal sweep (interval=%s)", interval)


# ── 데이터 가용성 ──


@router.get("/data-availability")
async def get_data_availability(interval: str = "1d"):
    """마이닝에 사용되는 데이터 소스별 가용성 확인."""
    from sqlalchemy import text

    from app.core.database import async_session

    async with async_session() as db:
        results = {}

        # OHLCV (stock_candles)
        r = await db.execute(
            text("SELECT COUNT(*) FROM stock_candles WHERE interval = :iv"),
            {"iv": interval},
        )
        ohlcv_count = r.scalar() or 0
        results["ohlcv"] = {"available": ohlcv_count > 0, "rows": ohlcv_count}

        # technical, cross_section은 OHLCV에서 자동 계산
        results["technical"] = {"available": ohlcv_count > 0, "rows": ohlcv_count}
        results["cross_section"] = {"available": ohlcv_count > 0, "rows": ohlcv_count}

        # investor_trading
        r = await db.execute(text("SELECT COUNT(*) FROM investor_trading"))
        cnt = r.scalar() or 0
        results["investor"] = {"available": cnt > 0, "rows": cnt}

        # news_sentiment_daily
        r = await db.execute(text("SELECT COUNT(*) FROM news_sentiment_daily"))
        cnt = r.scalar() or 0
        results["sentiment"] = {"available": cnt > 0, "rows": cnt}

        # dart_financials
        r = await db.execute(text("SELECT COUNT(*) FROM dart_financials"))
        cnt = r.scalar() or 0
        results["dart"] = {"available": cnt > 0, "rows": cnt}

        # margin_short_daily
        r = await db.execute(text("SELECT COUNT(*) FROM margin_short_daily"))
        cnt = r.scalar() or 0
        results["margin_short"] = {"available": cnt > 0, "rows": cnt}

        # program_trading
        r = await db.execute(text("SELECT COUNT(*) FROM program_trading"))
        cnt = r.scalar() or 0
        results["program"] = {"available": cnt > 0, "rows": cnt}

        # sector (stock_masters with embedding)
        r = await db.execute(
            text("SELECT COUNT(*) FROM stock_masters WHERE embedding IS NOT NULL")
        )
        cnt = r.scalar() or 0
        results["sector"] = {"available": cnt > 0, "rows": cnt}

    return results


# ── Phase 3: 알파 팩토리 ──


@router.post("/factory/start", response_model=AlphaFactoryStatusResponse)
async def start_factory(data: AlphaFactoryStartRequest):
    """알파 팩토리 시작. data_interval별 독립 인스턴스."""
    from app.alpha.factory_client import get_factory_client
    from app.core.redis import get_client as get_redis

    # Redis 플래그 해제 — 와치독 재시작 허용
    try:
        redis = get_redis()
        await redis.delete("alpha:factory:user_stopped")
    except Exception:
        pass

    client = get_factory_client(interval=data.data_interval)
    result = await client.start(
        context=data.context,
        universe=data.universe,
        start_date=data.start_date,
        end_date=data.end_date,
        data_interval=data.data_interval,
        interval_minutes=data.interval_minutes,
        max_iterations=data.max_iterations_per_cycle,
        ic_threshold=data.ic_threshold,
        orthogonality_threshold=data.orthogonality_threshold,
        enable_crossover=data.enable_crossover,
        max_cycles=data.max_cycles,
        seed_factor_ids=data.seed_factor_ids,
    )

    if not result.get("started"):
        raise HTTPException(409, f"팩토리({data.data_interval}) 시작 명령 전송 실패")

    # fire-and-forget: 명령 전송 성공이면 즉시 반환
    # Worker가 처리하면 GET /factory/status 폴링으로 상태 변경 감지
    status = result.get("status", {})
    return AlphaFactoryStatusResponse(
        running=status.get("running", False),
        cycles_completed=status.get("cycles_completed", 0),
        factors_discovered_total=status.get("factors_discovered_total", 0),
        current_cycle_progress=status.get("current_cycle_progress", 0),
        current_cycle_message=status.get("current_cycle_message", ""),
        last_cycle_at=status.get("last_cycle_at"),
        started_at=status.get("started_at"),
        config=status.get("config") or data.model_dump(),
        population_size=status.get("population_size", 0),
        elite_count=status.get("elite_count", 0),
        generation=status.get("generation", 0),
        operator_stats=status.get("operator_stats"),
        last_funnel=status.get("last_funnel"),
        user_stopped=status.get("user_stopped", False),
    )


@router.post("/factory/stop", response_model=AlphaFactoryStatusResponse)
async def stop_factory(interval: str = "1d"):
    """알파 팩토리 중지. interval별. Redis 플래그로 와치독 재시작도 방지."""
    from app.alpha.factory_client import get_factory_client
    from app.core.redis import get_client as get_redis

    # Redis 플래그 설정 — 와치독이 재시작하지 않도록
    try:
        redis = get_redis()
        await redis.set("alpha:factory:user_stopped", "true")
    except Exception:
        pass

    client = get_factory_client(interval=interval)
    try:
        result = await client.stop(interval=interval)
    except TypeError:
        result = await client.stop()

    # ExternalFactoryClient는 DB에 명령만 넣으므로, DB 상태도 직접 업데이트
    try:
        from app.core.database import async_session
        from app.models.base import WorkerState
        from sqlalchemy import update as sa_update

        async with async_session() as db:
            await db.execute(
                sa_update(WorkerState).where(WorkerState.id == 1).values(
                    factory_status={"running": False, "user_stopped": True}
                )
            )
            await db.commit()
    except Exception:
        pass

    return AlphaFactoryStatusResponse(**{**result["status"], "running": False, "user_stopped": True})


@router.put("/factory/auto-restart")
async def set_auto_restart(enabled: bool = True):
    """팩토리 자동 재시작(워크플로우 연동) 활성/비활성.

    enabled=False: Redis 플래그 설정 → 워크플로우가 팩토리를 자동 시작하지 않음.
    enabled=True: Redis 플래그 삭제 → 워크플로우가 MINING 페이즈에서 팩토리를 관리.
    워크플로우 자체는 영향받지 않음 (emergency_stop 호출 안 함).
    """
    from app.core.redis import get_client as get_redis

    try:
        redis = get_redis()
        if enabled:
            await redis.delete("alpha:factory:user_stopped")
        else:
            await redis.set("alpha:factory:user_stopped", "true")
    except Exception as e:
        raise HTTPException(500, f"Redis error: {e}")

    return {"auto_restart": enabled}


@router.get("/factory/status", response_model=AlphaFactoryStatusResponse)
async def get_factory_status(
    interval: str = "1d",
    db: AsyncSession = Depends(get_db),
):
    """알파 팩토리 상태 조회. interval별."""
    from app.alpha.factory_client import get_factory_client

    client = get_factory_client(interval=interval)
    status = await client.get_status()

    # 미검증 팩터 수 추가
    count_result = await db.execute(
        select(func.count(AlphaFactor.id)).where(
            AlphaFactor.causal_robust.is_(None),
            AlphaFactor.status.notin_(["validated", "mirage", "rejected"]),
            AlphaFactor.ic_mean >= 0.03,  # LOW_IC 제외 (sweep과 동일 기준)
            AlphaFactor.interval == interval,
        )
    )
    causal_pending = count_result.scalar() or 0

    # Sweep job ID 조회
    sweep_job_id = None
    try:
        from app.core.redis import get_client as get_redis

        redis = get_redis()
        sweep_val = await redis.get("alpha:causal_sweep:running")
        if sweep_val:
            sweep_job_id = sweep_val if isinstance(sweep_val, str) else sweep_val.decode()
    except Exception:
        pass

    return AlphaFactoryStatusResponse(
        **status,
        causal_pending_count=causal_pending,
        causal_sweep_job_id=sweep_job_id,
    )


@router.get("/factory/status/all")
async def get_all_factory_status():
    """모든 인터벌의 팩토리 상태 조회."""
    from app.alpha.scheduler import get_all_schedulers

    result = {}
    for interval, scheduler in get_all_schedulers().items():
        result[interval] = scheduler.get_status()
    return result


# ── Mega-Alpha (Phase 2 — 딥리서치 R1+R2) ──


@router.post("/mega-alpha/build", status_code=202)
async def build_mega_alpha(
    interval: str = "1d",
    min_icir: float = 0.3,
    db: AsyncSession = Depends(get_db),
):
    """
    # [2026-03-31] 딥리서치 R1+R2 공통 권장 — 메가알파 앙상블 구축
    # 프로세스: /deep-research → 2건 보고서 교차 분석
    # 변경/추가: 인과검증 통과 팩터 중 ICIR > min_icir인 것들을 자동 직교화 + 가중 결합
    """
    import json as _json
    from app.core.redis import get_client as get_redis

    # 후보 팩터 개수 사전 조회 (causal_robust=True AND icir >= min_icir)
    count_result = await db.scalar(
        select(func.count()).select_from(AlphaFactor).where(
            AlphaFactor.causal_robust == True,  # noqa: E712
            AlphaFactor.icir.isnot(None),
            AlphaFactor.icir >= min_icir,
            AlphaFactor.interval == interval,
            AlphaFactor.factor_type == "single",
        )
    )
    total_candidates = count_result or 0
    if total_candidates < 3:
        raise HTTPException(
            400,
            f"인과검증 통과 + ICIR>={min_icir} 팩터가 {total_candidates}개로 부족합니다 (최소 3개)",
        )

    job_id = uuid.uuid4().hex[:16]
    redis_key = "alpha:mega_alpha:status"

    # Redis에 초기 상태 저장
    try:
        redis = get_redis()
        await redis.set(
            redis_key,
            _json.dumps({
                "status": "pending",
                "total_candidates": total_candidates,
                "selected": 0,
                "current_step": "초기화",
                "logs": [],
                "job_id": job_id,
            }, ensure_ascii=False),
        )
        await redis.expire(redis_key, 86400)
    except Exception as e:
        logger.warning("Redis init failed for mega-alpha: %s", e)

    asyncio.create_task(
        _run_mega_alpha_build(job_id, interval, min_icir)
    )

    return {
        "status": "pending",
        "total_candidates": total_candidates,
        "job_id": job_id,
    }


async def _run_mega_alpha_build(
    job_id: str, interval: str, min_icir: float,
) -> None:
    """
    # [2026-03-31] 딥리서치 R1+R2 공통 권장 — 메가알파 백그라운드 빌드
    # 프로세스: /deep-research → 2건 보고서 교차 분석
    # 변경/추가: auto_optimize_composite 호출 + Redis 진행 상태 업데이트
    """
    import json as _json
    from app.alpha.portfolio import auto_optimize_composite
    from app.core.database import async_session
    from app.core.redis import get_client as get_redis

    redis_key = "alpha:mega_alpha:status"

    async def _update_status(data: dict) -> None:
        try:
            redis = get_redis()
            existing_raw = await redis.get(redis_key)
            existing = _json.loads(existing_raw) if existing_raw else {}
            existing.update(data)
            await redis.set(
                redis_key,
                _json.dumps(existing, ensure_ascii=False, default=str),
            )
            await redis.expire(redis_key, 86400)
        except Exception:
            pass

    try:
        await _update_status({
            "status": "running",
            "current_step": "auto_optimize_composite 실행 중",
        })

        async with async_session() as db:
            # [2026-03-31] 딥리서치 R1+R2 공통 권장 — causal_only=True로 인과 통과 팩터만 사용
            # 프로세스: /deep-research → 2건 보고서 교차 분석
            # 변경/추가: min_icir을 내부 ICIR 필터와 맞추기 위해 max(min_icir, 0.3) 적용
            result = await auto_optimize_composite(
                db=db,
                min_ic=0.03,
                min_turnover=0.02,
                max_k=7,
                lambda_decorr=0.5,
                shrinkage_delta=0.5,
                interval=interval,
                causal_only=True,
                job_id=f"mega_{job_id}",
            )

        # 결과에서 best-K 추출
        best_opt = next((r for r in result.results if r.k == result.best_k), None)
        selected_count = result.best_k if best_opt else 0
        logs = result.logs if hasattr(result, "logs") else []

        # best-K 복합 팩터를 DB에 저장
        saved_factor_id = None
        if best_opt:
            try:
                async with async_session() as db2:
                    composite = AlphaFactor(
                        name=f"MegaAlpha K={best_opt.k} (IC={best_opt.composite_ic:.4f})",
                        expression_str=best_opt.expression_str,
                        factor_type="composite",
                        interval=interval,
                        ic_mean=best_opt.composite_ic,
                        icir=best_opt.composite_icir,
                        sharpe=best_opt.composite_sharpe,
                        component_ids=best_opt.factor_ids,
                        causal_robust=True,
                        status="validated",
                    )
                    db2.add(composite)
                    await db2.commit()
                    saved_factor_id = str(composite.id)
                    logger.info(
                        "mega-alpha: composite saved (K=%d, id=%s)",
                        best_opt.k, saved_factor_id,
                    )
            except Exception as e:
                logger.warning("mega-alpha: DB save failed: %s", e)
                logs.append(f"DB 저장 실패: {e}")

        await _update_status({
            "status": "completed",
            "selected": selected_count,
            "current_step": "완료",
            "logs": logs,
            "saved_factor_id": saved_factor_id,
            "best_k": result.best_k,
            "candidate_count": result.candidate_count,
        })

        logger.info("mega-alpha job %s completed: best_k=%d", job_id, result.best_k)

    except ValueError as e:
        logger.error("mega-alpha job %s ValueError: %s", job_id, e)
        await _update_status({
            "status": "failed",
            "current_step": f"실패: {e}",
            "logs": [str(e)],
        })
    except Exception as e:
        logger.exception("mega-alpha job %s unexpected error: %s", job_id, e)
        await _update_status({
            "status": "failed",
            "current_step": f"오류: {str(e)[:200]}",
            "logs": [str(e)[:500]],
        })


@router.get("/mega-alpha/status")
async def get_mega_alpha_status():
    """
    # [2026-03-31] 딥리서치 R1+R2 공통 권장 — 메가알파 구축 진행 상태 조회
    # 프로세스: /deep-research → 2건 보고서 교차 분석
    # 변경/추가: Redis에서 상태 JSON 조회하여 반환
    """
    import json as _json
    from app.core.redis import get_client as get_redis

    try:
        redis = get_redis()
        cached = await redis.get("alpha:mega_alpha:status")
        if cached:
            return _json.loads(cached)
    except Exception as e:
        logger.warning("mega-alpha status Redis read failed: %s", e)

    return {"status": "idle"}


# ── Phase 3: 팩터 포트폴리오 ──


@router.post("/portfolio/build", response_model=CompositeFactorResponse)
async def build_composite(
    data: CompositeFactorBuildRequest,
    db: AsyncSession = Depends(get_db),
):
    """복합 팩터 생성."""
    from app.alpha.portfolio import build_composite_factor

    try:
        result = await build_composite_factor(
            db=db,
            factor_ids=data.factor_ids,
            method=data.method,
            name=data.name,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    # 복합 팩터 DB 저장
    composite = AlphaFactor(
        name=data.name,
        expression_str=result.composite_expression,
        factor_type="composite",
        component_ids=result.component_ids,
        ic_mean=result.metrics.ic_mean,
        ic_std=result.metrics.ic_std,
        icir=result.metrics.icir,
        turnover=result.metrics.turnover,
        sharpe=result.metrics.sharpe,
        max_drawdown=result.metrics.max_drawdown,
        status="discovered",
        interval=data.interval if hasattr(data, "interval") else "1d",
    )
    db.add(composite)
    await db.commit()
    await db.refresh(composite)

    return CompositeFactorResponse(
        id=str(composite.id),
        name=composite.name,
        factor_type=composite.factor_type,
        expression_str=composite.expression_str,
        component_ids=result.component_ids,
        ic_mean=result.metrics.ic_mean,
        created_at=composite.created_at.isoformat(),
    )


@router.post("/portfolio/correlation", response_model=CorrelationMatrixResponse)
async def get_correlation(
    data: CorrelationRequest,
    db: AsyncSession = Depends(get_db),
):
    """팩터 간 상관행렬 조회."""
    from app.alpha.portfolio import compute_correlation_matrix

    try:
        result = await compute_correlation_matrix(db=db, factor_ids=data.factor_ids)
    except ValueError as e:
        raise HTTPException(400, str(e))

    return CorrelationMatrixResponse(**result)


@router.post("/portfolio/auto-optimize", status_code=202)
async def auto_optimize(
    data: AutoOptimizeRequest,
):
    """자동 최적 복합 팩터 조합 (비동기).

    즉시 job_id를 반환하고 백그라운드에서 실행.
    GET /alpha/portfolio/auto-optimize/{job_id} 로 상태 폴링.
    """
    import json as _json
    from app.core.redis import get_client as get_redis

    job_id = uuid.uuid4().hex[:16]
    redis_key = f"alpha:optimize:{job_id}"

    # Redis에 PENDING 상태 초기화
    try:
        redis = get_redis()
        await redis.hset(redis_key, mapping={
            "status": "pending",
            "logs": "[]",
            "result": "",
            "error": "",
        })
        await redis.expire(redis_key, 86400)
    except Exception as e:
        logger.warning("Redis init failed for optimize job %s: %s", job_id, e)

    asyncio.create_task(_run_auto_optimize(job_id, data))

    return {"job_id": job_id}


async def _run_auto_optimize(job_id: str, data: AutoOptimizeRequest) -> None:
    """백그라운드에서 auto_optimize_composite를 실행하고 결과를 Redis에 저장."""
    import json as _json
    from app.alpha.portfolio import auto_optimize_composite
    from app.core.database import async_session
    from app.core.redis import get_client as get_redis

    redis_key = f"alpha:optimize:{job_id}"

    try:
        async with async_session() as db:
            result = await auto_optimize_composite(
                db=db,
                min_ic=data.min_ic,
                min_turnover=data.min_turnover,
                max_k=data.max_k,
                lambda_decorr=data.lambda_decorr,
                shrinkage_delta=data.shrinkage_delta,
                interval=data.interval,
                causal_only=data.causal_only,
                job_id=job_id,
            )
        # 완료 — portfolio.py 내부에서 이미 Redis에 결과 저장됨
        logger.info("auto_optimize job %s completed: best_k=%d", job_id, result.best_k)

        # best-K 복합 팩터를 DB에 영구 저장
        best_opt = next((r for r in result.results if r.k == result.best_k), None)
        if best_opt:
            try:
                async with async_session() as db2:
                    from app.alpha.models import AlphaFactor
                    composite = AlphaFactor(
                        name=f"Auto K={best_opt.k} (IC={best_opt.composite_ic:.4f})",
                        expression_str=best_opt.expression_str,
                        factor_type="composite",
                        interval=data.interval or "5m",
                        ic_mean=best_opt.composite_ic,
                        icir=best_opt.composite_icir,
                        sharpe=best_opt.composite_sharpe,
                        component_ids=best_opt.factor_ids,
                        causal_robust=True,  # 구성 팩터가 이미 인과 통과 → 복합은 자동 통과
                    )
                    db2.add(composite)
                    await db2.commit()
                    logger.info("auto_optimize: best composite saved to DB (K=%d, id=%s)", best_opt.k, composite.id)
            except Exception as e:
                logger.warning("auto_optimize: DB save failed: %s", e)
    except ValueError as e:
        logger.error("auto_optimize job %s ValueError: %s", job_id, e)
        try:
            redis = get_redis()
            await redis.hset(redis_key, mapping={
                "status": "failed",
                "error": str(e),
            })
            await redis.expire(redis_key, 86400)
        except Exception:
            pass
    except Exception as e:
        logger.exception("auto_optimize job %s unexpected error: %s", job_id, e)
        try:
            redis = get_redis()
            await redis.hset(redis_key, mapping={
                "status": "failed",
                "error": str(e)[:500],
            })
            await redis.expire(redis_key, 86400)
        except Exception:
            pass


@router.get("/portfolio/auto-optimize/{job_id}")
async def get_auto_optimize_status(job_id: str):
    """자동 최적 조합 잡 상태 폴링.

    Redis Hash에서 status, result, error, logs를 읽어 반환.
    status: pending | running | completed | failed
    """
    import json as _json
    from app.core.redis import get_client as get_redis

    redis_key = f"alpha:optimize:{job_id}"

    try:
        redis = get_redis()
        raw = await redis.hgetall(redis_key)
    except Exception as e:
        logger.warning("Redis read failed for optimize job %s: %s", job_id, e)
        raise HTTPException(503, "Redis 연결 실패")

    if not raw:
        raise HTTPException(404, "Optimization job not found")

    status = raw.get("status", "pending")
    error = raw.get("error", "") or None
    logs_raw = raw.get("logs", "[]")
    result_raw = raw.get("result", "")

    # JSON 파싱
    try:
        logs = _json.loads(logs_raw) if logs_raw else []
    except Exception:
        logs = []

    result = None
    if result_raw:
        try:
            result = _json.loads(result_raw)
        except Exception:
            pass

    return {
        "status": status,
        "result": result,
        "error": error,
        "logs": logs,
    }


# ── 팩터 AI 채팅 ──


@router.post(
    "/factor/{factor_id}/chat",
    response_model=FactorChatCreateResponse,
)
async def create_factor_chat(
    factor_id: str,
    db: AsyncSession = Depends(get_db),
):
    """팩터 기반 AI 채팅 세션 생성."""
    result = await db.execute(
        select(AlphaFactor).where(AlphaFactor.id == uuid.UUID(factor_id))
    )
    factor = result.scalar_one_or_none()
    if not factor:
        raise HTTPException(404, "Factor not found")

    # 마이닝 run에서 universe/dates/interval 추출
    universe = "KOSPI200"
    start_date = "2025-06-01"
    end_date = "2025-12-31"
    interval = getattr(factor, "interval", "1d") or "1d"

    if factor.mining_run_id:
        run_result = await db.execute(
            select(AlphaMiningRun.config).where(
                AlphaMiningRun.id == factor.mining_run_id
            )
        )
        run_config = run_result.scalar_one_or_none()
        if run_config:
            universe = run_config.get("universe", universe)
            start_date = run_config.get("start_date", start_date)
            end_date = run_config.get("end_date", end_date)
            interval = run_config.get("interval", interval)

    from app.alpha.factor_chat import factor_chat_store

    session = factor_chat_store.create(
        source_factor_id=str(factor.id),
        source_expression=factor.expression_str,
        source_hypothesis=factor.hypothesis or "",
        source_metrics={
            "ic_mean": factor.ic_mean or 0,
            "ic_std": factor.ic_std or 0,
            "icir": factor.icir or 0,
            "turnover": factor.turnover or 0,
            "sharpe": factor.sharpe or 0,
            "max_drawdown": factor.max_drawdown or 0,
        },
        current_expression=factor.expression_str,
        current_metrics={
            "ic_mean": factor.ic_mean or 0,
            "ic_std": factor.ic_std or 0,
            "icir": factor.icir or 0,
            "turnover": factor.turnover or 0,
            "sharpe": factor.sharpe or 0,
            "max_drawdown": factor.max_drawdown or 0,
        },
        universe=universe,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
    )

    return FactorChatCreateResponse(
        session_id=session.id,
        source_factor_id=str(factor.id),
        source_expression=factor.expression_str,
        universe=universe,
        interval=interval,
        status=session.status,
    )


@router.post(
    "/factor/chat/{session_id}/message",
    response_model=FactorChatMessageResponse,
)
async def send_factor_chat_message(
    session_id: str,
    req: FactorChatMessageRequest,
):
    """팩터 채팅 메시지 전송."""
    from app.alpha.factor_chat import factor_chat_store, process_message

    session = factor_chat_store.get(session_id)
    if not session:
        raise HTTPException(404, "Chat session not found or expired")

    try:
        assistant_msg = await process_message(session, req.message)
    except Exception as e:
        logger.exception("Factor chat error for session %s", session_id)
        raise HTTPException(500, f"채팅 처리 실패: {str(e)[:200]}")

    return FactorChatMessageResponse(
        role=assistant_msg.role,
        content=assistant_msg.content,
        timestamp=assistant_msg.timestamp,
        factor_draft=assistant_msg.factor_draft,
        current_expression=session.current_expression,
        current_metrics=session.current_metrics,
    )


@router.get(
    "/factor/chat/{session_id}",
    response_model=FactorChatSessionResponse,
)
async def get_factor_chat_session(session_id: str):
    """팩터 채팅 세션 조회."""
    from app.alpha.factor_chat import factor_chat_store

    session = factor_chat_store.get(session_id)
    if not session:
        raise HTTPException(404, "Chat session not found or expired")

    return FactorChatSessionResponse(**session.to_dict())


@router.post("/factor/chat/{session_id}/save", response_model=AlphaFactorResponse)
async def save_factor_from_chat(
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    """채팅에서 수정한 팩터를 새 AlphaFactor로 DB 저장."""
    from app.alpha.factor_chat import factor_chat_store

    session = factor_chat_store.get(session_id)
    if not session:
        raise HTTPException(404, "Chat session not found or expired")

    if not session.current_expression:
        raise HTTPException(400, "저장할 수식이 없습니다. 먼저 대화로 수식을 수정하세요.")

    # SymPy 문자열 생성
    expression_sympy = None
    polars_code = None
    try:
        from app.alpha.ast_converter import (
            parse_expression,
            sympy_to_code_string,
            sympy_to_polars,
        )
        import sympy

        parsed = parse_expression(session.current_expression)
        expression_sympy = sympy.srepr(parsed)
        polars_code = sympy_to_code_string(parsed)
    except Exception:
        pass

    metrics = session.current_metrics or {}

    new_factor = AlphaFactor(
        mining_run_id=None,
        name=f"Custom: {session.current_expression[:50]}",
        expression_str=session.current_expression,
        expression_sympy=expression_sympy,
        polars_code=polars_code,
        hypothesis=session.source_hypothesis,
        generation=0,
        ic_mean=metrics.get("ic_mean"),
        ic_std=metrics.get("ic_std"),
        icir=metrics.get("icir"),
        turnover=metrics.get("turnover"),
        sharpe=metrics.get("sharpe"),
        max_drawdown=metrics.get("max_drawdown"),
        status="discovered",
        operator_origin="manual",
        parent_ids=[session.source_factor_id],
        interval=session.interval,
    )
    db.add(new_factor)
    await db.commit()
    await db.refresh(new_factor)

    session.status = "saved"
    session.touch()

    return _factor_to_response(new_factor)


@router.delete("/factor/chat/{session_id}", status_code=204)
async def delete_factor_chat_session(session_id: str):
    """팩터 채팅 세션 삭제."""
    from app.alpha.factor_chat import factor_chat_store

    if not factor_chat_store.delete(session_id):
        raise HTTPException(404, "Chat session not found")


# ── 내부 헬퍼 ──

def _factor_to_response(f: AlphaFactor) -> AlphaFactorResponse:
    return AlphaFactorResponse(
        id=str(f.id),
        mining_run_id=str(f.mining_run_id) if f.mining_run_id else None,
        name=f.name,
        expression_str=f.expression_str,
        interval=getattr(f, "interval", "1d"),
        expression_sympy=f.expression_sympy,
        polars_code=f.polars_code,
        hypothesis=f.hypothesis,
        generation=f.generation,
        ic_mean=f.ic_mean,
        ic_std=f.ic_std,
        icir=f.icir,
        turnover=f.turnover,
        sharpe=f.sharpe,
        max_drawdown=f.max_drawdown,
        status=f.status,
        causal_robust=f.causal_robust,
        causal_effect_size=f.causal_effect_size,
        causal_p_value=f.causal_p_value,
        parent_ids=f.parent_ids,
        factor_type=f.factor_type,
        component_ids=f.component_ids,
        fitness_composite=getattr(f, "fitness_composite", None),
        tree_depth=getattr(f, "tree_depth", None),
        tree_size=getattr(f, "tree_size", None),
        expression_hash=getattr(f, "expression_hash", None),
        operator_origin=getattr(f, "operator_origin", None),
        is_elite=getattr(f, "is_elite", None),
        population_active=getattr(f, "population_active", None),
        birth_generation=getattr(f, "birth_generation", None),
        created_at=f.created_at.isoformat(),
        updated_at=f.updated_at.isoformat(),
    )


# ── 마이닝 개선 히스토리 ──


@router.get("/improvement-history")
async def get_improvement_history(interval: str = "5m"):
    """마이닝 개선 히스토리 JSON 반환 (인터벌별 분리)."""
    import json
    from pathlib import Path

    docs_dir = Path(__file__).parent.parent.parent / "docs"
    # 인터벌별 파일: mining_improvements_1d.json, mining_improvements_5m.json
    json_path = docs_dir / f"mining_improvements_{interval}.json"
    if not json_path.exists():
        # fallback: 기존 파일명
        json_path = docs_dir / "mining_improvements.json"
    if not json_path.exists():
        return {"rounds": []}

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        return data
    except Exception:
        return {"rounds": []}


# ── 마이닝 리포트 대시보드 ──


@router.get("/mining-report", response_model=MiningReportResponse)
async def get_mining_report(
    interval: str = "1d",
    db: AsyncSession = Depends(get_db),
):
    """최신 마이닝 리포트 데이터 반환 (Redis 캐시 → DB fallback)."""
    import json
    from app.core.redis import get_client as get_redis

    # 1차: Redis 캐시 (사이클 완료 시 갱신, TTL 600초)
    try:
        redis = get_redis()
        cached = await redis.get(f"alpha:mining_report:{interval}")
        if cached:
            return json.loads(cached)
    except Exception:
        pass

    # 2차: DB fallback — 캐시 만료 시 DB에서 직접 조회
    from sqlalchemy import text
    from app.alpha.report_metrics import compute_ic_trend

    ic_trend = await compute_ic_trend(interval=interval, limit=10)
    if not ic_trend:
        return MiningReportResponse()

    # 최근 세대의 상위 팩터 5개
    latest_gen = ic_trend[-1]["gen"] if ic_trend else 0
    result = await db.execute(
        text("""
            SELECT expression_str, ic_mean, icir, sharpe, max_drawdown, turnover
            FROM alpha_factors
            WHERE interval = :interval AND generation = :gen AND ic_mean > 0
            ORDER BY ic_mean DESC LIMIT 5
        """),
        {"interval": interval, "gen": latest_gen},
    )
    top_factors = [
        {
            "expression": row.expression_str[:70],
            "ic_mean": round(float(row.ic_mean), 4),
            "icir": round(float(row.icir or 0), 2),
            "sharpe": round(float(row.sharpe or 0), 2),
            "max_drawdown": round(float(row.max_drawdown or 0), 3),
            "turnover": round(float(row.turnover or 0), 3),
            "hypothesis": None,
        }
        for row in result.fetchall()
    ]

    # 총 발견 수
    count_result = await db.execute(
        text("SELECT COUNT(*) FROM alpha_factors WHERE interval = :interval AND ic_mean > 0"),
        {"interval": interval},
    )
    total = count_result.scalar() or 0

    return MiningReportResponse(
        generation=latest_gen,
        total_discovered=total,
        data_interval=interval,
        discovered_factors=top_factors,
        ic_trend=ic_trend,
    )


@router.get("/mining-reports", response_model=MiningReportsRangeResponse)
async def get_mining_reports(
    interval: str = "1d",
    gen_from: int | None = None,
    gen_to: int | None = None,
    date_from: str | None = None,  # YYYY-MM-DD
    date_to: str | None = None,  # YYYY-MM-DD
    db: AsyncSession = Depends(get_db),
):
    """세대별 마이닝 리포트 범위 조회."""
    from sqlalchemy import select as sa_select

    from app.alpha.models import AlphaGenerationReport

    query = sa_select(AlphaGenerationReport).where(
        AlphaGenerationReport.data_interval == interval
    )

    if gen_from is not None:
        query = query.where(AlphaGenerationReport.generation >= gen_from)
    if gen_to is not None:
        query = query.where(AlphaGenerationReport.generation <= gen_to)
    if date_from:
        from datetime import datetime as _dt

        query = query.where(
            AlphaGenerationReport.created_at >= _dt.fromisoformat(date_from)
        )
    if date_to:
        from datetime import datetime as _dt
        from datetime import timedelta

        end = _dt.fromisoformat(date_to) + timedelta(days=1)
        query = query.where(AlphaGenerationReport.created_at < end)

    query = query.order_by(AlphaGenerationReport.generation.asc()).limit(200)
    result = await db.execute(query)
    rows = result.scalars().all()

    # 해당 인터벌의 전체 min/max 세대 조회 (슬라이더 범위용)
    from sqlalchemy import func as sa_func

    bounds = await db.execute(
        sa_select(
            sa_func.min(AlphaGenerationReport.generation),
            sa_func.max(AlphaGenerationReport.generation),
        ).where(AlphaGenerationReport.data_interval == interval)
    )
    bounds_row = bounds.one_or_none()
    min_gen = bounds_row[0] or 0 if bounds_row else 0
    max_gen = bounds_row[1] or 0 if bounds_row else 0

    return MiningReportsRangeResponse(
        reports=[row.report_data for row in rows],
        total=len(rows),
        interval=interval,
        min_gen=min_gen,
        max_gen=max_gen,
    )
