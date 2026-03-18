"""AutoSelector — DB에서 매매 가능 최적 팩터를 자동 선택.

복합 점수 6요소 (설계서 §8):
  IC(0.25) + Sharpe(0.20) + ICIR(0.15) + MDD(0.15) + 인과검증(0.15) + 최신도(0.10)
"""

from __future__ import annotations

import logging
import math
import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.alpha.models import AlphaFactor
from app.core.config import settings
from app.workflow.models import TradingContextModel

logger = logging.getLogger(__name__)


def _compute_score(factor: AlphaFactor) -> tuple[float, dict]:
    """팩터에 대한 6요소 복합 점수와 breakdown을 계산한다."""
    ic = factor.ic_mean or 0.0
    sharpe = factor.sharpe or 0.0
    icir = factor.icir or 0.0
    mdd = getattr(factor, "max_drawdown", None) or 0.0  # 음수 (예: -15.0)

    # 정규화 (설계서 §8.3)
    ic_norm = min(1.0, max(0.0, ic / 0.15))
    sharpe_norm = min(1.0, max(0.0, sharpe / 3.0))
    icir_norm = min(1.0, max(0.0, icir / 1.0))
    mdd_norm = min(1.0, max(0.0, 1.0 - abs(mdd) / 30.0))

    # 인과 검증 (통과=1.0, 미검증=0.5, 실패=0.0)
    if factor.causal_robust is None:
        causal_score = 0.5
    elif factor.causal_robust:
        causal_score = 1.0
    else:
        causal_score = 0.0

    # 최신도 — 지수 감쇠 (설계서 §8.3)
    age_days = (
        (datetime.now(timezone.utc) - factor.created_at.replace(tzinfo=timezone.utc)).days
        if factor.created_at
        else 999
    )
    recency_score = math.exp(-age_days / 30.0)

    # 가중치 (config에서 로드)
    w_ic = settings.WORKFLOW_SCORE_W_IC
    w_sharpe = settings.WORKFLOW_SCORE_W_SHARPE
    w_icir = settings.WORKFLOW_SCORE_W_ICIR
    w_mdd = settings.WORKFLOW_SCORE_W_MDD
    w_causal = settings.WORKFLOW_SCORE_W_CAUSAL
    w_recency = settings.WORKFLOW_SCORE_W_RECENCY

    composite = (
        w_ic * ic_norm
        + w_sharpe * sharpe_norm
        + w_icir * icir_norm
        + w_mdd * mdd_norm
        + w_causal * causal_score
        + w_recency * recency_score
    )

    breakdown = {
        "ic_raw": round(ic, 6),
        "ic_norm": round(ic_norm, 4),
        "sharpe_raw": round(sharpe, 4),
        "sharpe_norm": round(sharpe_norm, 4),
        "icir_raw": round(icir, 4),
        "icir_norm": round(icir_norm, 4),
        "mdd_raw": round(mdd, 4),
        "mdd_norm": round(mdd_norm, 4),
        "causal_score": round(causal_score, 4),
        "recency_score": round(recency_score, 4),
        "age_days": age_days,
        "composite": round(composite, 4),
    }
    return composite, breakdown


async def select_best_factors(
    session: AsyncSession,
    *,
    limit: int = 5,
    min_ic: float | None = None,
    min_sharpe: float | None = None,
    require_causal: bool | None = None,
    interval: str | None = None,
) -> list[dict]:
    """매매 가능한 최적 팩터를 점수 기반으로 선택한다.

    Returns:
        [{"factor": AlphaFactor, "score": float, "breakdown": dict}, ...]
    """
    min_ic = min_ic if min_ic is not None else settings.WORKFLOW_MIN_FACTOR_IC
    min_sharpe = min_sharpe if min_sharpe is not None else settings.WORKFLOW_MIN_FACTOR_SHARPE
    if require_causal is None:
        require_causal = settings.WORKFLOW_REQUIRE_CAUSAL
    interval = interval or settings.WORKFLOW_DATA_INTERVAL

    stmt = (
        select(AlphaFactor)
        .where(
            AlphaFactor.status.in_(["discovered", "validated"]),
            AlphaFactor.ic_mean.isnot(None),
            AlphaFactor.sharpe.isnot(None),
            AlphaFactor.ic_mean >= min_ic,
            AlphaFactor.sharpe >= min_sharpe,
        )
    )
    if require_causal:
        stmt = stmt.where(AlphaFactor.causal_robust.is_(True))
    if interval:
        stmt = stmt.where(AlphaFactor.interval == interval)

    result = await session.execute(stmt)
    factors = list(result.scalars().all())

    if not factors:
        logger.info("AutoSelector: 기준 미달 — 매매 가능 팩터 없음")
        return []

    scored = []
    for f in factors:
        score, breakdown = _compute_score(f)
        scored.append({"factor": f, "score": score, "breakdown": breakdown})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:limit]


async def _resolve_symbols(
    session: AsyncSession, factor: AlphaFactor
) -> list[str]:
    """팩터의 mining run config에서 유니버스를 리졸브."""
    from app.alpha.models import AlphaMiningRun

    if factor.mining_run_id:
        stmt = select(AlphaMiningRun).where(AlphaMiningRun.id == factor.mining_run_id)
        result = await session.execute(stmt)
        mining_run = result.scalar_one_or_none()
        if mining_run and mining_run.config:
            universe_str = mining_run.config.get("universe", "KOSPI200")
            try:
                from app.alpha.universe import Universe, resolve_universe
                universe = Universe(universe_str)
                return await resolve_universe(universe)
            except Exception as e:
                logger.warning("유니버스 리졸브 실패 (%s): %s", universe_str, e)

    # Fallback: KOSPI200
    try:
        from app.alpha.universe import Universe, resolve_universe
        return await resolve_universe(Universe.KOSPI200)
    except Exception as e:
        logger.warning("KOSPI200 폴백 리졸브 실패: %s", e)
        return []


async def build_context_from_factor(
    session: AsyncSession,
    factor: AlphaFactor,
    *,
    mode: str = "paper",
    param_overrides: dict | None = None,
    initial_capital: float | None = None,
) -> TradingContextModel:
    """팩터에서 TradingContext DB 모델을 생성하여 저장한다.

    Args:
        param_overrides: 전일 피드백 기반 파라미터 조정값.
            키: max_positions, stop_loss_pct, trailing_stop_pct,
                max_drawdown_pct, position_size_pct
        initial_capital: 배정 자본금. None이면 WORKFLOW_INITIAL_CAPITAL 사용.
    """
    initial_capital = initial_capital or settings.WORKFLOW_INITIAL_CAPITAL
    max_positions = settings.WORKFLOW_MAX_POSITIONS
    stop_loss = settings.WORKFLOW_STOP_LOSS_PCT
    max_dd = settings.WORKFLOW_MAX_DRAWDOWN_PCT
    trailing_stop = 3.0

    # 전일 피드백 기반 파라미터 오버라이드 적용
    if param_overrides:
        max_positions = int(param_overrides.get("max_positions", max_positions))
        stop_loss = param_overrides.get("stop_loss_pct", stop_loss)
        trailing_stop = param_overrides.get("trailing_stop_pct", trailing_stop)
        max_dd = param_overrides.get("max_drawdown_pct", max_dd)
        logger.info(
            "AutoSelector: 피드백 파라미터 적용 — max_pos=%d, stop=%.1f%%, trail=%.1f%%, mdd=%.1f%%",
            max_positions, stop_loss, trailing_stop, max_dd,
        )

    # conviction 기반 포지션 사이징 (설계서 §7.2)
    ic_mean = factor.ic_mean or 0.0
    conviction = min(1.0, ic_mean / 0.1)

    # 유니버스 리졸브 → symbols
    symbols = await _resolve_symbols(session, factor)

    # 팩터 수식 → buy/sell 조건 자동 생성 (퍼센타일 랭크 0.7/0.3 기준)
    short_id = str(factor.id)[:8]
    indicator_name = f"alpha_{short_id}"

    position_size_pct = 1.0 / max_positions
    if param_overrides and "position_size_pct" in param_overrides:
        position_size_pct = param_overrides["position_size_pct"]

    ctx = TradingContextModel(
        id=uuid.uuid4(),
        mode=mode,
        status="active",
        strategy={
            "factor_id": str(factor.id),
            "factor_name": factor.name,
            "expression_str": factor.expression_str,
            "interval": factor.interval,
            "buy_conditions": [
                {"indicator": indicator_name, "params": {}, "op": ">", "value": 0.7},
            ],
            "sell_conditions": [
                {"indicator": indicator_name, "params": {}, "op": "<", "value": 0.3},
            ],
            "buy_logic": "AND",
            "sell_logic": "AND",
        },
        strategy_name=f"auto:{factor.name}",
        position_sizing={
            "mode": "conviction",
            "conviction": round(conviction, 4),
        },
        risk_management={
            "stop_loss_pct": stop_loss,
            "trailing_stop_pct": trailing_stop,
            "max_drawdown_pct": max_dd,
        },
        cost_config={
            "buy_commission": 0.00015,
            "sell_commission": 0.00215,
            "slippage_pct": 0.001,
        },
        initial_capital=initial_capital,
        position_size_pct=position_size_pct,
        max_positions=max_positions,
        symbols=symbols,
        source_factor_id=factor.id,
        auto_created=True,
    )
    session.add(ctx)
    await session.flush()
    logger.info(
        "AutoSelector: 컨텍스트 생성 — factor=%s, ctx=%s, symbols=%d개",
        factor.name, ctx.id, len(symbols),
    )
    return ctx
