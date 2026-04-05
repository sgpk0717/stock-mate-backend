"""DailyWorkflowOrchestrator — FSM 상태 머신 + APScheduler 크론잡.

상태 전이:
  IDLE → PRE_MARKET → TRADING → MARKET_CLOSE → REVIEW → MINING → IDLE

시간 정확 실행은 APScheduler가, 지능적 판단은 OpenClaw(MCP)이 담당.
Phase Watchdog가 5분마다 실제 FSM 상태를 검증하여 누락된 페이즈를 자동 catch-up.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import date, datetime, time as time_type, timedelta, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from app.core.config import settings
from app.core.database import async_session
from app.core.timezone import KST, now_kst, to_kst, today_kst
from app.workflow.auto_selector import build_context_from_factor, select_best_factors
from app.workflow.mining_config import get_mining_config
from app.workflow.models import TradingContextModel, WorkflowEvent, WorkflowRun

logger = logging.getLogger(__name__)

# 허용 상태 전이
_TRANSITIONS: dict[str, list[str]] = {
    "IDLE": ["PRE_MARKET", "MINING"],
    "PRE_MARKET": ["TRADING", "IDLE", "MINING"],  # +MINING: 비거래일 복귀
    "TRADING": ["MARKET_CLOSE"],
    "MARKET_CLOSE": ["REVIEW"],
    "REVIEW": ["MINING"],
    "MINING": ["IDLE", "PRE_MARKET"],  # +PRE_MARKET: 거래일 아침
    "EMERGENCY_STOP": ["IDLE", "MINING"],  # +MINING: 긴급 해제 후 마이닝 재개
}

# OpenClaw 헬스체크 상태
_openclaw_fail_count: int = 0
_independent_mode: bool = False

# 외부 피드백 마커 (이 접두사로 시작하는 줄은 외부 소스 피드백)
_EXTERNAL_FEEDBACK_MARKERS = ("[OpenClaw", "[외부 피드백]")


def _extract_external_feedback(mining_context: str) -> str:
    """mining_context에서 외부 소스 피드백(OpenClaw 등)만 추출.

    내부 생성 피드백(구조화된 피드백, generate_context 결과)은 매 REVIEW마다
    새로 생성되므로 누적하지 않는다. 외부 피드백만 보존하여 토큰 낭비를 방지.
    """
    if not mining_context:
        return ""
    lines: list[str] = []
    capturing = False
    for line in mining_context.split("\n"):
        stripped = line.strip()
        if any(stripped.startswith(m) for m in _EXTERNAL_FEEDBACK_MARKERS):
            capturing = True
        elif capturing and stripped == "":
            # 외부 피드백 블록의 빈 줄은 유지
            lines.append(line)
            continue
        elif capturing and not stripped.startswith(("개선 제안:", "시장 체제:")):
            # 다음 외부 피드백이 아닌 새 블록이 시작되면 캡처 중지
            capturing = False
        if capturing:
            lines.append(line)
    return "\n".join(lines).strip()


class DailyWorkflowOrchestrator:
    """일일 워크플로우 FSM.

    APScheduler가 각 페이즈를 시간에 맞춰 트리거한다.
    """

    def __init__(self) -> None:
        self._scheduler = None
        self._session_unhealthy_counts: dict[str, int] = {}
        self._session_restart_failures: dict[str, int] = {}

    async def _get_or_create_today(self, session: AsyncSession) -> WorkflowRun:
        """오늘 날짜의 WorkflowRun을 가져오거나 생성.

        팩토리가 실행 중이면 MINING 상태로 시작 (마이닝 상시가동).
        """
        today = today_kst()
        stmt = select(WorkflowRun).where(WorkflowRun.date == today)
        result = await session.execute(stmt)
        run = result.scalar_one_or_none()
        if run is None:
            initial_phase = "IDLE"
            try:
                from app.alpha.factory_client import get_factory_client
                factory = get_factory_client()
                if (await factory.get_status())["running"]:
                    initial_phase = "MINING"
            except Exception:
                pass
            run = WorkflowRun(
                id=uuid.uuid4(), date=today, phase=initial_phase, status="PENDING",
            )
            session.add(run)
            await session.flush()
            logger.info("WorkflowRun 생성: date=%s, id=%s, phase=%s", today, run.id, initial_phase)
        return run

    async def _log_event(
        self,
        session: AsyncSession,
        run: WorkflowRun,
        event_type: str,
        message: str,
        data: dict | None = None,
        level: str = "info",
    ) -> None:
        event = WorkflowEvent(
            id=uuid.uuid4(),
            workflow_run_id=run.id,
            phase=run.phase,
            event_type=event_type,
            message=message,
            data={**(data or {}), "level": level},
        )
        session.add(event)

    async def _transition(
        self, session: AsyncSession, run: WorkflowRun, new_phase: str,
        *, force: bool = False,
    ) -> bool:
        """상태 전이를 시도한다.

        force=True 면 허용 목록을 무시하고 강제 전이 (폴백 용도).
        """
        allowed = _TRANSITIONS.get(run.phase, [])
        if not force and new_phase not in allowed:
            logger.warning(
                "워크플로우 전이 거부: %s → %s (허용: %s)", run.phase, new_phase, allowed
            )
            return False
        old_phase = run.phase
        run.phase = new_phase
        forced = " [forced]" if force and new_phase not in allowed else ""
        await self._log_event(
            session, run, "phase_transition", f"{old_phase} → {new_phase}{forced}"
        )
        logger.info("워크플로우 전이: %s → %s%s", old_phase, new_phase, forced)

        # Redis 캐시 즉시 갱신 (API가 stale phase를 반환하지 않도록)
        try:
            from app.core.redis import hset
            await hset("workflow:status", {"phase": new_phase, "status": run.status or ""})
        except Exception as e:
            logger.warning("Redis 워크플로우 상태 캐시 실패: %s", e)

        return True

    # ── step_status 헬퍼 ──

    def _get_step(self, run: WorkflowRun, step_name: str) -> dict:
        """step_status에서 특정 단계 조회."""
        steps = run.step_status or {}
        return steps.get(step_name, {})

    async def _set_step(
        self,
        session: AsyncSession,
        run: WorkflowRun,
        step_name: str,
        status: str,
        **extra,
    ) -> None:
        """step_status에서 특정 단계 갱신 + Redis 즉시 동기화."""
        steps = dict(run.step_status or {})
        steps[step_name] = {
            "status": status,
            "at": now_kst().isoformat(),
            **extra,
        }
        run.step_status = steps
        flag_modified(run, "step_status")

        # Redis 즉시 동기화 (lazy sync 문제 해결)
        try:
            import json as _json
            from app.core.redis import hset
            await hset("workflow:status", {
                "step_status": _json.dumps(steps, ensure_ascii=False, default=str),
            }, ttl=300)
        except Exception as e:
            logger.warning("Redis step_status 동기화 실패: %s", e)

    # ── 거래일 확인 ──

    async def _is_trading_day(self) -> bool:
        """pykrx로 오늘이 거래일인지 확인.

        pykrx 데이터가 없거나 예외 시 평일(월-금)이면 거래일로 간주.
        """
        today = today_kst()
        is_weekday = today.weekday() < 5  # 0=월 ~ 4=금

        try:
            from pykrx import stock
            today_str = today.strftime("%Y%m%d")
            trading_days = stock.get_previous_business_days(
                fromdate=today_str, todate=today_str
            )
            if len(trading_days) > 0:
                return True
            # pykrx가 빈 리스트 반환 → 미래 날짜이거나 공휴일
            # 주말이면 확실히 비거래일, 평일이면 데이터 부재일 수 있으므로 거래일로 간주
            if is_weekday:
                logger.info("pykrx 거래일 데이터 없음 (평일) — 거래일로 간주")
                return True
            return False
        except Exception as e:
            logger.warning("거래일 확인 실패 (기본: 평일=True): %s", e)
            return is_weekday

    # ── 페이즈 핸들러 ──

    async def handle_pre_market(self, *, force: bool = False) -> dict:
        """08:30 — 거래일 확인, 팩토리 중지, 최적 팩터 확인."""
        # user_stopped 플래그는 사용자가 명시적으로 해제할 때만 삭제.
        # pre_market에서 자동 초기화하지 않음 (사용자 의도 존중).

        async with async_session() as session:
            run = await self._get_or_create_today(session)

            # step_status 중복/완료 체크 (force=True면 우회)
            if not force:
                step_info = self._get_step(run, "pre_market")
                if step_info.get("status") == "completed":
                    logger.info("pre_market 이미 완료 — 스킵")
                    await session.commit()
                    return {"success": True, "phase": "PRE_MARKET", "message": "이미 완료"}
                if step_info.get("status") == "running":
                    logger.info("pre_market 진행 중 — 스킵")
                    await session.commit()
                    return {"success": True, "phase": "PRE_MARKET", "message": "진행 중"}

            # 비거래일 체크 — 전이 전에 수행 (마이닝 중단 방지)
            if not await self._is_trading_day():
                await self._log_event(
                    session, run, "non_trading_day",
                    "비거래일 — 매매 스킵, 마이닝 계속 가동",
                )
                # 마이닝이 안 돌고 있으면 시작
                await self._ensure_mining_running()
                await session.commit()
                return {"success": True, "message": "비거래일 — 마이닝 계속 가동"}

            if not await self._transition(session, run, "PRE_MARKET"):
                # 전이 실패 시 강제 전이 (폴백)
                if not await self._transition(session, run, "PRE_MARKET", force=True):
                    await session.commit()
                    return {"success": False, "message": f"전이 불가: {run.phase} → PRE_MARKET"}

            await self._set_step(session, run, "pre_market", "running")
            run.started_at = datetime.now(timezone.utc)

            try:
                # 알파 팩토리 중지
                try:
                    from app.alpha.factory_client import get_factory_client
                    factory = get_factory_client()
                    if (await factory.get_status())["running"]:
                        await factory.stop()
                        await self._log_event(session, run, "factory_stop", "알파 팩토리 중지")
                except Exception as e:
                    logger.warning("팩토리 중지 실패: %s", e)

                # 유니버스 프리페치 (하루 1회 — pykrx → Redis 캐싱)
                try:
                    from app.alpha.universe import Universe, prefetch_universe
                    _univ_code = getattr(settings, "WORKFLOW_DATA_UNIVERSE", "KOSPI200")
                    prefetched = await prefetch_universe(Universe(_univ_code))
                    logger.info("유니버스 프리페치: %d종목", len(prefetched))
                except Exception as e:
                    logger.warning("유니버스 프리페치 실패 (DB 폴백 사용): %s", e)

                # PRE_MARKET 다이버전스 체크 (전일 팩터 사후 검증)
                try:
                    from app.workflow.divergence_detector import check_all_active_factors
                    div_actions = await check_all_active_factors(session)
                    if div_actions:
                        logger.info(
                            "PRE_MARKET 다이버전스: %d건 감지 — %s",
                            len(div_actions), div_actions,
                        )
                except Exception as e:
                    logger.warning("PRE_MARKET 다이버전스 체크 실패: %s", e)

                # 듀얼 타임프레임 팩터 선택 (5분봉 + 일봉)
                from app.workflow.auto_selector import select_dual_factors
                dual = await select_dual_factors(
                    session,
                    intraday_count=settings.WORKFLOW_INTRADAY_FACTOR_COUNT,
                    daily_count=settings.WORKFLOW_DAILY_FACTOR_COUNT,
                    min_ic=settings.WORKFLOW_MIN_FACTOR_IC,
                    min_sharpe=settings.WORKFLOW_MIN_FACTOR_SHARPE,
                    require_causal=settings.WORKFLOW_REQUIRE_CAUSAL,
                )
                best = dual["intraday"] + dual["daily"]  # 하위호환: 합쳐서 best로 사용
                if best:
                    # 첫 번째 팩터를 selected_factor_id에 설정 (기존 호환)
                    factor = best[0]["factor"]
                    run.selected_factor_id = factor.id

                    # interval별로 구분하여 config에 저장
                    run.config = run.config or {}
                    run.config["selected_factors"] = [
                        {
                            "factor_id": str(b["factor"].id),
                            "name": b["factor"].name,
                            "score": round(b["score"], 4),
                            "interval": b["factor"].interval,
                        }
                        for b in best
                    ]
                    run.config["selected_factors_intraday"] = [
                        {"factor_id": str(b["factor"].id), "name": b["factor"].name}
                        for b in dual["intraday"]
                    ]
                    run.config["selected_factors_daily"] = [
                        {"factor_id": str(b["factor"].id), "name": b["factor"].name}
                        for b in dual["daily"]
                    ]
                    flag_modified(run, "config")

                    factors_desc = ", ".join(
                        f"{b['factor'].name}[{b['factor'].interval}]({b['score']:.4f})" for b in best
                    )
                    await self._log_event(
                        session, run, "factor_selected",
                        f"팩터 {len(best)}개 선택: {factors_desc}",
                        data={
                            "count": len(best),
                            "factors": [
                                {"id": str(b["factor"].id), "name": b["factor"].name, "score": round(b["score"], 4)}
                                for b in best
                            ],
                        },
                    )
                else:
                    await self._log_event(session, run, "no_factor", "매매 가능 팩터 없음")

                factor_id = str(factor.id) if best else None
                await self._set_step(
                    session, run, "pre_market", "completed",
                    detail=f"factor={factor_id}, total={len(best) if best else 0}",
                )
            except Exception as e:
                await self._set_step(
                    session, run, "pre_market", "error",
                    error=str(e)[:200],
                )
                raise

            await session.commit()
            return {
                "success": True,
                "phase": "PRE_MARKET",
                "factor": factor.name if best else None,
                "score": best[0]["score"] if best else None,
                "factor_count": len(best) if best else 0,
                "factors": [
                    {"name": b["factor"].name, "score": round(b["score"], 4)}
                    for b in best
                ] if best else [],
            }

    async def handle_market_open(self, *, force: bool = False) -> dict:
        """09:00 — 최적 팩터 → TradingContext → LiveSession 시작."""
        async with async_session() as session:
            run = await self._get_or_create_today(session)

            # 자동매매 수동 중단 플래그 체크 — TRADING 페이즈는 정상 전환, 세션만 안 시작
            _trading_user_stopped = False
            try:
                from app.core.redis import get_client as _get_redis
                _r = _get_redis()
                _flag = await _r.get("trading:user_stopped")
                if _flag and str(_flag) == "true":
                    _trading_user_stopped = True
            except Exception:
                pass

            # 비거래일 체크
            if not await self._is_trading_day():
                logger.info("market_open: 비거래일 — 스킵")
                await session.commit()
                return {"success": True, "message": "비거래일 — 매매 스킵"}

            # step_status 중복/완료 체크 (force=True면 우회)
            if not force:
                step_info = self._get_step(run, "market_open")
                if step_info.get("status") == "completed":
                    logger.info("market_open 이미 완료 — 스킵")
                    await session.commit()
                    return {"success": True, "phase": "TRADING", "message": "이미 완료"}
                if step_info.get("status") == "running":
                    logger.info("market_open 진행 중 — 스킵")
                    await session.commit()
                    return {"success": True, "phase": "TRADING", "message": "진행 중"}

            # 기존 실행 중인 세션 확인 (중복 생성 방지)
            try:
                from app.trading.live_runner import _sessions
                if _sessions:
                    running = [s for s in _sessions.values() if s.status == "running"]
                    if running:
                        # 세션은 이미 돌고 있지만 phase가 PRE_MARKET이면 TRADING으로 전이
                        if run.phase != "TRADING":
                            await self._transition(session, run, "TRADING")
                            await self._set_step(session, run, "market_open", "completed",
                                                 detail=f"sessions={len(running)}, reused=True")
                            await session.commit()
                        logger.info("market_open: 이미 %d개 세션 실행 중 — phase=%s", len(running), run.phase)
                        return {"success": True, "message": f"기존 {len(running)}개 세션 운영 중"}
            except Exception:
                pass

            # pre_market이 미실행된 경우 여기서 직접 팩터 선택 (catch-up)
            if run.selected_factor_id is None and run.phase == "IDLE":
                logger.info("market_open: pre_market 미실행 — 팩터 직접 선택 시도")
                _mcfg = await get_mining_config()
                multi_count = settings.WORKFLOW_MULTI_FACTOR_COUNT
                best = await select_best_factors(
                    session,
                    limit=multi_count,
                    min_ic=settings.WORKFLOW_MIN_FACTOR_IC,
                    min_sharpe=settings.WORKFLOW_MIN_FACTOR_SHARPE,
                    require_causal=settings.WORKFLOW_REQUIRE_CAUSAL,
                    interval=_mcfg["interval"],
                )
                if best:
                    factor = best[0]["factor"]
                    run.selected_factor_id = factor.id
                    # 멀티 팩터 목록 저장
                    run.config = run.config or {}
                    run.config["selected_factors"] = [
                        {
                            "factor_id": str(b["factor"].id),
                            "name": b["factor"].name,
                            "score": round(b["score"], 4),
                        }
                        for b in best
                    ]
                    flag_modified(run, "config")
                    # IDLE → PRE_MARKET → (바로) TRADING 진행 위해 PRE_MARKET 전이
                    await self._transition(session, run, "PRE_MARKET")
                    factors_desc = ", ".join(
                        f"{b['factor'].name}({b['score']:.4f})" for b in best
                    )
                    await self._log_event(
                        session, run, "factor_selected",
                        f"catch-up 팩터 {len(best)}개 선택: {factors_desc}",
                        data={
                            "count": len(best),
                            "factors": [
                                {"id": str(b["factor"].id), "name": b["factor"].name, "score": round(b["score"], 4)}
                                for b in best
                            ],
                        },
                    )
                    await session.commit()
                    # 새 세션으로 재진입 (커밋된 상태에서 TRADING 진행)
                    return await self.handle_market_open()

            if run.selected_factor_id is None:
                # 팩터 미달 → 매매 스킵
                await self._transition(session, run, "IDLE")
                run.status = "SKIPPED"
                run.error_message = "매매 가능 팩터 없음"
                await self._log_event(session, run, "trading_skip", "팩터 미달로 매매 스킵")
                await session.commit()
                return {"success": False, "message": "팩터 미달 — 매매 스킵"}

            if run.phase != "TRADING":
                if not await self._transition(session, run, "TRADING"):
                    await session.commit()
                    return {"success": False, "message": f"전이 불가: {run.phase} → TRADING"}

            # 자동매매 수동 중단 — TRADING 페이즈로는 전환했지만 세션 시작 안 함
            if _trading_user_stopped:
                logger.info("market_open: 자동매매 수동 중단 — TRADING 전환 완료, 세션 시작 스킵")
                await self._set_step(session, run, "market_open", "completed",
                                     detail="trading_user_stopped")
                await self._log_event(
                    session, run, "trading_skipped",
                    "자동매매 수동 중단 상태 — 세션 시작 스킵",
                )
                await session.commit()
                return {"success": True, "phase": "TRADING", "detail": "trading_user_stopped"}

            await self._set_step(session, run, "market_open", "running")

            try:
                from app.alpha.models import AlphaFactor

                # 선택된 팩터 목록 로드 (멀티 팩터)
                selected_factors = (run.config or {}).get("selected_factors", [])
                if not selected_factors and run.selected_factor_id:
                    # 폴백: 단일 팩터 (기존 호환 — WORKFLOW_MULTI_FACTOR_COUNT=1 또는 레거시)
                    selected_factors = [{"factor_id": str(run.selected_factor_id)}]

                # 전일 피드백 기반 파라미터 오버라이드 로드
                param_overrides = None
                if settings.WORKFLOW_PARAM_EVAL_ENABLED:
                    try:
                        yesterday = run.date - timedelta(days=1)
                        prev_stmt = select(WorkflowRun).where(WorkflowRun.date == yesterday)
                        prev_result = await session.execute(prev_stmt)
                        prev_run = prev_result.scalar_one_or_none()
                        if prev_run and prev_run.config:
                            param_overrides = prev_run.config.get("param_adjustments")
                            if param_overrides:
                                logger.info("전일 파라미터 조정 적용: %s", param_overrides)
                    except Exception as e:
                        logger.warning("전일 파라미터 로드 실패: %s", e)

                # 각 세션이 독립 자본금 보유 (모의투자이므로 세션별 전액 배정)
                capital_per_factor = settings.WORKFLOW_INITIAL_CAPITAL

                mode = settings.WORKFLOW_TRADING_MODE
                started_sessions: list[dict] = []
                context_ids: list[str] = []

                # ── 일봉 세션 보존 처리 ──
                # 이미 실행 중인 일봉 세션이 있으면 그대로 유지 (포지션 보존)
                from app.trading.live_runner import list_sessions as _list_live
                existing_daily_live = [
                    s for s in _list_live()
                    if s.status == "running"
                    and s.context.strategy.get("interval") == "1d"
                ]
                # 이미 활성인 일봉 팩터 ID 추출
                active_daily_factor_ids = {
                    s.context.strategy.get("factor_id", "")
                    for s in existing_daily_live
                }
                # selected_factors에서 이미 활성인 일봉 팩터는 제외 (세션 중복 방지)
                selected_factors = [
                    f for f in selected_factors
                    if f.get("interval") != "1d" or f["factor_id"] not in active_daily_factor_ids
                ]
                if existing_daily_live:
                    logger.info(
                        "market_open: 일봉 세션 %d개 이미 활성 — 포지션 유지 (%s)",
                        len(existing_daily_live),
                        ", ".join(s.context.strategy_name for s in existing_daily_live),
                    )
                    # 일봉 세션의 daily_scores 재계산 + 리밸런스 매도
                    for ds in existing_daily_live:
                        try:
                            from app.trading.live_runner import _compute_daily_factor_scores, _daily_rebalance_sell
                            from app.backtest.cost_model import CostConfig as _CostConfig
                            await _compute_daily_factor_scores(ds)
                            sell_count = await _daily_rebalance_sell(ds, _CostConfig(
                                buy_commission=ds.context.cost_config.buy_commission,
                                sell_commission=ds.context.cost_config.sell_commission,
                                slippage_pct=ds.context.cost_config.slippage_pct,
                            ))
                            if sell_count > 0:
                                await self._log_event(
                                    session, run, "daily_rebalance",
                                    f"일봉 세션 {ds.context.strategy_name}: {sell_count}종목 리밸런스 매도",
                                )
                        except Exception as e:
                            logger.warning("일봉 세션 리밸런스 실패 (%s): %s", ds.id[:8], e)
                    # 이미 실행 중인 세션은 started_sessions에 추가
                    for ds in existing_daily_live:
                        started_sessions.append({
                            "factor_id": ds.context.strategy.get("factor_id", ""),
                            "factor_name": ds.context.strategy_name,
                            "context_id": ds.context.id,
                            "session_id": ds.id,
                            "capital": round(ds._cash + sum(p.avg_price * p.qty for p in ds.positions.values()), 2),
                            "interval": "1d",
                            "reused": True,
                        })

                # ── 기존 context 재사용 체크 (서버 재시작 시 중복 생성 방지) ──
                # 오늘 이미 생성된 context를 factor_id별로 매핑
                from sqlalchemy import func as sa_func
                existing_today_stmt = (
                    select(TradingContextModel)
                    .where(
                        sa_func.date(TradingContextModel.created_at) == run.date,
                    )
                    .order_by(TradingContextModel.created_at.desc())
                )
                existing_result = await session.execute(existing_today_stmt)
                existing_contexts = list(existing_result.scalars().all())

                # factor_id → 최신 context (archived 포함 — market_close 후 재시작 대응)
                existing_by_factor: dict[str, TradingContextModel] = {}
                for ec in existing_contexts:
                    fid = (ec.strategy or {}).get("factor_id", "")
                    if fid and fid not in existing_by_factor:
                        existing_by_factor[fid] = ec

                # 선택된 팩터 전부에 기존 context가 있으면 재사용
                reuse_all = (
                    selected_factors
                    and len(existing_by_factor) > 0
                    and all(
                        f["factor_id"] in existing_by_factor
                        for f in selected_factors
                    )
                )

                if reuse_all:
                    from app.trading.context import TradingContext, _contexts
                    from app.alpha.backtest_bridge import register_alpha_factor
                    from app.trading.live_runner import start_session

                    for factor_info in selected_factors:
                        fid = factor_info["factor_id"]
                        ec = existing_by_factor[fid]
                        context_ids.append(str(ec.id))

                        # archived → active 복원
                        if ec.status != "active":
                            ec.status = "active"

                        if run.trading_context_id is None:
                            run.trading_context_id = ec.id

                        # 인메모리 등록 + 세션 시작
                        session_id = None
                        try:
                            ctx = TradingContext.from_db_model(ec)
                            _contexts[ctx.id] = ctx

                            factor_stmt = select(AlphaFactor).where(
                                AlphaFactor.id == uuid.UUID(fid)
                            )
                            factor_r = await session.execute(factor_stmt)
                            factor = factor_r.scalar_one_or_none()
                            if factor:
                                register_alpha_factor(str(factor.id), factor.expression_str)

                            live_session = await start_session(ctx)
                            session_id = live_session.id
                        except Exception as e:
                            logger.error("기존 context 세션 시작 실패 — %s: %s", fid[:8], e)

                        started_sessions.append({
                            "factor_id": fid,
                            "factor_name": factor_info.get("factor_name", fid[:8]),
                            "context_id": str(ec.id),
                            "session_id": session_id,
                            "capital": round(capital_per_factor, 2),
                        })

                    logger.info(
                        "market_open: 기존 %d개 context 재사용 (%s)",
                        len(context_ids),
                        ", ".join(c[:8] for c in context_ids),
                    )
                else:
                    # ── 새 context 생성 (기존 코드) ──
                    for factor_info in selected_factors:
                        factor_id_str = factor_info["factor_id"]
                        try:
                            factor_uuid = uuid.UUID(factor_id_str)
                        except (ValueError, TypeError):
                            logger.warning("잘못된 factor_id 형식: %s", factor_id_str)
                            continue

                        factor_stmt = select(AlphaFactor).where(AlphaFactor.id == factor_uuid)
                        factor_result = await session.execute(factor_stmt)
                        factor = factor_result.scalar_one_or_none()
                        if factor is None:
                            logger.warning("팩터 조회 실패 — factor_id=%s, 스킵", factor_id_str)
                            continue

                        ctx_model = await build_context_from_factor(
                            session, factor, mode=mode,
                            param_overrides=param_overrides,
                            initial_capital=capital_per_factor,
                        )
                        context_ids.append(str(ctx_model.id))

                        if run.trading_context_id is None:
                            run.trading_context_id = ctx_model.id

                        session_id = None
                        try:
                            from app.trading.context import TradingContext, _contexts
                            ctx = TradingContext.from_db_model(ctx_model)
                            _contexts[ctx.id] = ctx

                            from app.alpha.backtest_bridge import register_alpha_factor
                            register_alpha_factor(str(factor.id), factor.expression_str)

                            from app.trading.live_runner import start_session
                            live_session = await start_session(ctx)
                            session_id = live_session.id
                        except Exception as e:
                            logger.error(
                                "LiveSession 시작 실패 — factor=%s: %s", factor.name, e,
                            )
                            await self._log_event(
                                session, run, "error",
                                f"LiveSession 시작 실패 — factor={factor.name}: {e}",
                            )

                        started_sessions.append({
                            "factor_id": str(factor.id),
                            "factor_name": factor.name,
                            "context_id": str(ctx_model.id),
                            "session_id": session_id,
                            "capital": round(capital_per_factor, 2),
                        })

                        await self._log_event(
                            session, run, "trading_start",
                            f"매매 시작: factor={factor.name}, ctx={ctx_model.id}, "
                            f"capital={capital_per_factor:,.0f}, mode={mode}",
                        )

                if not started_sessions:
                    run.error_message = "모든 팩터의 세션 시작 실패"
                    await self._set_step(
                        session, run, "market_open", "error",
                        error="세션 0개 시작",
                    )
                    await session.commit()
                    return {"success": False, "message": "모든 팩터 세션 시작 실패"}

                # 멀티 컨텍스트 ID 목록 저장 (market_close에서 전체 아카이브용)
                run.config = run.config or {}
                run.config["trading_context_ids"] = context_ids
                flag_modified(run, "config")

                run.status = "RUNNING"
                await self._set_step(
                    session, run, "market_open", "completed",
                    detail=f"sessions={len(started_sessions)}, contexts={context_ids}",
                )
                await session.commit()

                # 장중 분봉 수집기 시작 (live_runner와 독립)
                try:
                    from app.trading.live_runner import start_intraday_candle_collector
                    scan_symbols = []
                    for sf in selected_factors:
                        scan_symbols.extend(sf.get("symbols", []))
                    if not scan_symbols:
                        # 유니버스 폴백
                        from app.alpha.universe import Universe, resolve_universe
                        try:
                            scan_symbols = await resolve_universe(Universe("KOSPI200"))
                        except Exception:
                            from sqlalchemy import text as _t
                            _r = await session.execute(
                                _t("SELECT symbol FROM stock_masters WHERE market = 'KOSPI' ORDER BY symbol LIMIT 200")
                            )
                            scan_symbols = [r[0] for r in _r.fetchall()]
                    asyncio.create_task(start_intraday_candle_collector(scan_symbols))
                except Exception as _e:
                    logger.warning("장중 분봉 수집기 시작 실패: %s", _e)

                # 알파 스코어 엔진 자동 시작 (팩터별 실시간 랭킹)
                try:
                    from app.trading.alpha_score_engine import get_score_engine, start_score_engine_loop
                    engine = get_score_engine()
                    if not engine._running and started_sessions:
                        factor_configs = []
                        for ss in started_sessions:
                            fid = ss.get("factor_id", "")
                            fname = ss.get("factor_name", "")
                            # DB에서 팩터 수식 조회
                            try:
                                f_stmt = select(AlphaFactor).where(AlphaFactor.id == uuid.UUID(fid))
                                f_result = await session.execute(f_stmt)
                                f_obj = f_result.scalar_one_or_none()
                                if f_obj and f_obj.expression_str:
                                    factor_configs.append({
                                        "id": fid,
                                        "name": fname,
                                        "expression_str": f_obj.expression_str,
                                        "interval": f_obj.interval or "5m",
                                    })
                            except Exception:
                                pass
                        if factor_configs:
                            from app.alpha.universe import resolve_universe, Universe
                            try:
                                univ_enum = Universe(settings.WORKFLOW_UNIVERSE)
                            except (ValueError, KeyError):
                                univ_enum = Universe.KOSPI200
                            univ_symbols = await resolve_universe(univ_enum)
                            # KOSPI200 전체 종목 사용 (Redis pipeline 배치로 타임아웃 해소)
                            asyncio.create_task(start_score_engine_loop(
                                univ_symbols, factor_configs, interval_seconds=300,
                            ))
                            logger.info("알파 스코어 엔진 시작: %d팩터, %d종목", len(factor_configs), len(univ_symbols))
                except Exception as e:
                    logger.warning("알파 스코어 엔진 시작 실패: %s", e)

                return {
                    "success": True,
                    "phase": "TRADING",
                    "sessions": started_sessions,
                    "session_count": len(started_sessions),
                    "capital_per_factor": round(capital_per_factor, 2),
                    "mode": mode,
                    # 기존 호환 필드
                    "context_id": context_ids[0] if context_ids else None,
                    "session_id": started_sessions[0]["session_id"] if started_sessions else None,
                    "factor_name": started_sessions[0]["factor_name"] if started_sessions else None,
                }
            except Exception as e:
                await self._set_step(
                    session, run, "market_open", "error",
                    error=str(e)[:200],
                )
                await session.commit()
                raise

    async def handle_market_close(self, *, force: bool = False) -> dict:
        """15:30 — 전량 청산 + LiveSession 중지 + PnL 스냅샷."""
        async with async_session() as session:
            run = await self._get_or_create_today(session)

            # 비거래일 체크
            if not await self._is_trading_day():
                logger.info("market_close: 비거래일 — 스킵")
                await session.commit()
                return {"success": True, "message": "비거래일 — 매매 스킵"}

            # step_status 중복/완료 체크 (force=True면 우회)
            if not force:
                step_info = self._get_step(run, "market_close")
                if step_info.get("status") == "completed":
                    logger.info("market_close 이미 완료 — 스킵")
                    await session.commit()
                    return {"success": True, "phase": "MARKET_CLOSE", "message": "이미 완료"}
                if step_info.get("status") == "running":
                    logger.info("market_close 진행 중 — 스킵")
                    await session.commit()
                    return {"success": True, "phase": "MARKET_CLOSE", "message": "진행 중"}

            if not await self._transition(session, run, "MARKET_CLOSE"):
                await session.commit()
                return {"success": False, "message": f"전이 불가: {run.phase} → MARKET_CLOSE"}

            await self._set_step(session, run, "market_close", "running")

            # 미체결 주문 전량 취소 (세션 중지 전 안전장치)
            try:
                from app.trading.order_manager import get_order_manager
                om = get_order_manager()
                cancelled = await om.cancel_all(reason="market_close")
                if cancelled:
                    await self._log_event(
                        session, run, "orders_cancelled",
                        f"장마감 미체결 {len(cancelled)}건 취소",
                    )
                    await asyncio.sleep(2)
                    await om.check_orders()
            except Exception as e:
                logger.warning("장마감 미체결 취소 실패: %s", e)

            # LiveSession 중지: 5분봉은 전량 청산, 일봉은 포지션 유지
            total_trades = 0
            try:
                from app.trading.live_runner import list_sessions, stop_session, _save_session_state, _sync_session_to_redis
                for live_session in list_sessions():
                    if live_session.status != "running":
                        continue

                    interval = live_session.context.strategy.get("interval", "5m")

                    if interval == "1d":
                        # ── 일봉 세션: 포지션 유지, 세션 상태 저장만 ──
                        await _save_session_state(live_session)
                        await _sync_session_to_redis(live_session)
                        pos_count = len(live_session.positions)
                        logger.info(
                            "일봉 세션 %s (%s) 장 마감 — 포지션 %d종목 유지 (오버나잇)",
                            live_session.id[:8], live_session.context.strategy_name, pos_count,
                        )
                        await self._log_event(
                            session, run, "daily_session_overnight",
                            f"일봉 세션 {live_session.context.strategy_name} — {pos_count}종목 오버나잇 보유",
                        )
                        total_trades += len(live_session.trade_log)
                        continue

                    # ── 5분봉 세션: 현재 로직 그대로 (전량 청산 + 세션 종료) ──
                    await self._close_all_positions(live_session)

                    # 로그 파일 저장 (stop 전에 데이터 접근)
                    try:
                        from app.trading.log_writer import save_session_log
                        ctx = live_session.context
                        log_path = save_session_log(
                            session_id=live_session.id,
                            mode=ctx.mode,
                            strategy_name=ctx.strategy_name,
                            initial_capital=ctx.initial_capital,
                            position_size_pct=ctx.position_size_pct,
                            max_positions=ctx.max_positions,
                            trade_log=live_session.trade_log,
                            decision_log=live_session.decision_log,
                            cost_config=ctx.cost_config,
                        )
                        if log_path:
                            logger.info("매매 로그 파일: %s", log_path)
                    except Exception as e:
                        logger.error("매매 로그 저장 실패: %s", e)

                    stopped = await stop_session(live_session.id)
                    if stopped:
                        total_trades += len(stopped.trade_log)
                        await self._log_event(
                            session, run, "session_stopped",
                            f"5분봉 세션 {live_session.id} 중지 (거래 {len(stopped.trade_log)}건)",
                        )
            except Exception as e:
                logger.error("LiveSession 중지 실패: %s", e)

            # DB에서 오늘 실제 매매 건수 집계 (인메모리 세션이 없어도 정확)
            try:
                from sqlalchemy import text as _text
                _db_count = await session.execute(_text(
                    "SELECT COUNT(*) FROM live_trades lt "
                    "JOIN trading_contexts tc ON lt.context_id = tc.id "
                    "WHERE tc.created_at::date = :today"
                ), {"today": str(run.date)})
                db_trade_count = _db_count.scalar() or 0
                run.trade_count = max(total_trades, db_trade_count)
            except Exception:
                run.trade_count = total_trades

            # 예비 PnL 계산 (REVIEW 전에 OpenClaw이 조회할 수 있도록)
            try:
                from app.workflow.models import LiveTrade
                from sqlalchemy import func

                # 멀티 팩터: 모든 컨텍스트의 거래를 합산
                pnl_context_ids = (run.config or {}).get("trading_context_ids", [])
                if not pnl_context_ids and run.trading_context_id:
                    pnl_context_ids = [str(run.trading_context_id)]

                # 오늘 전체 SELL 매매 조회 (멀티 context 합산)
                sell_stmt = select(LiveTrade).where(
                    func.date(LiveTrade.executed_at) == run.date,
                    LiveTrade.side == "SELL",
                )
                sell_result = await session.execute(sell_stmt)
                sell_trades = list(sell_result.scalars().all())

                if sell_trades:
                    pnls = [t.pnl_pct or 0 for t in sell_trades]
                    pnl_amounts = [
                        float(t.pnl_amount) if t.pnl_amount else 0
                        for t in sell_trades
                    ]
                    wins = [p for p in pnls if p > 0]
                    run.pnl_pct = sum(pnls)
                    run.pnl_amount = sum(pnl_amounts)
                    run.review_summary = {
                        "preliminary": True,
                        "trade_count": run.trade_count,
                        "total_pnl_pct": round(sum(pnls), 4),
                        "total_pnl": round(sum(pnl_amounts), 2),
                        "win_rate": round(
                            len(wins) / len(sell_trades) * 100, 2
                        ) if sell_trades else 0,
                        "context_count": len(pnl_context_ids),
                    }
            except Exception as e:
                logger.warning("MARKET_CLOSE 예비 PnL 계산 실패: %s", e)

            # 다이버전스 체크 (팩터 실매매 성능 감시 — 멀티 팩터 전체)
            div_factor_ids = [
                f["factor_id"]
                for f in (run.config or {}).get("selected_factors", [])
            ]
            if not div_factor_ids and run.selected_factor_id:
                div_factor_ids = [str(run.selected_factor_id)]
            for div_fid in div_factor_ids:
                try:
                    from app.workflow.divergence_detector import check_divergence
                    div_result = await check_divergence(session, div_fid)
                    if div_result.get("action") in ("halt", "warn"):
                        logger.info(
                            "다이버전스 감지 — factor=%s: %s — %s",
                            div_fid, div_result["action"], div_result,
                        )
                except Exception as e:
                    logger.warning("다이버전스 체크 실패 — factor=%s: %s", div_fid, e)

            # TradingContext archived (멀티 팩터: 전체 컨텍스트 아카이브)
            context_ids_to_archive = (run.config or {}).get("trading_context_ids", [])
            if not context_ids_to_archive and run.trading_context_id:
                # 폴백: 단일 컨텍스트 (기존 호환)
                context_ids_to_archive = [str(run.trading_context_id)]
            for ctx_id_str in context_ids_to_archive:
                try:
                    ctx_uuid = uuid.UUID(ctx_id_str)
                    stmt = (
                        update(TradingContextModel)
                        .where(TradingContextModel.id == ctx_uuid)
                        .values(status="archived")
                    )
                    await session.execute(stmt)
                except (ValueError, TypeError) as e:
                    logger.warning("컨텍스트 아카이브 실패 — id=%s: %s", ctx_id_str, e)

            await self._set_step(
                session, run, "market_close", "completed",
                detail=f"trades={total_trades}",
            )
            await self._log_event(
                session, run, "market_close",
                f"전량 청산 + 장 마감 완료 (거래 {total_trades}건)",
            )
            await session.commit()
            return {"success": True, "phase": "MARKET_CLOSE", "total_trades": total_trades}

    async def _close_all_positions(self, live_session) -> None:
        """LiveSession의 모든 포지션 전량 청산 (분봉 단타 원칙).

        paper 모드: 내부 포지션 정리 + 현금 회수 (KIS API 미사용).
        real 모드: KIS API 시장가 매도.
        """
        ctx = live_session.context
        is_paper = ctx.mode != "real"

        # Paper 모드: pending orders 취소 + 예약 현금 반환
        if is_paper and hasattr(live_session, 'pending_orders') and live_session.pending_orders:
            for order in live_session.pending_orders:
                if order.get("side") == "BUY":
                    live_session._cash += order.get("reserved_cash", 0)
            cancelled_count = len(live_session.pending_orders)
            live_session.pending_orders.clear()
            logger.info("장마감 pending orders %d건 취소 [paper]", cancelled_count)

        for symbol, pos in list(live_session.positions.items()):
            if pos.qty <= 0:
                continue
            try:
                if is_paper:
                    # paper: 마지막 종가 기준 매도 처리
                    from app.backtest.cost_model import CostConfig, effective_sell_price
                    cost = CostConfig(
                        buy_commission=ctx.cost_config.buy_commission,
                        sell_commission=ctx.cost_config.sell_commission,
                        slippage_pct=ctx.cost_config.slippage_pct,
                    )
                    # 최신 캔들에서 종가 + dt 가져오기
                    close_price = pos.highest_price if pos.highest_price > 0 else pos.avg_price
                    close_candle_dt = None
                    try:
                        from datetime import date as date_cls, timedelta
                        from app.backtest.data_loader import load_candles
                        interval = ctx.strategy.get("interval", "1d")
                        end_d = date_cls.today()
                        start_d = end_d - timedelta(days=30)
                        _df = await load_candles([symbol], start_d, end_d, interval)
                        import polars as _pl
                        _sym_df = _df.filter(_pl.col("symbol") == symbol).sort("dt")
                        if _sym_df.height > 0:
                            _last = _sym_df.tail(1).to_dicts()[0]
                            close_price = _last.get("close", close_price)
                            _dt = _last.get("dt")
                            close_candle_dt = _dt.isoformat() if hasattr(_dt, "isoformat") else str(_dt)
                    except Exception:
                        pass  # 캔들 로딩 실패 시 기존 로직 유지

                    sell_price = effective_sell_price(close_price, cost)
                    from app.trading.live_runner import _paper_sell
                    _paper_sell(
                        live_session, symbol, pos.qty, close_price, sell_price, cost,
                        step="S-CLOSE", reason="장 마감 전량 청산",
                        candle_dt=close_candle_dt,
                    )
                    logger.info("전량 청산 [paper]: %s %d주 @ %s", symbol, pos.qty, sell_price)
                else:
                    from app.trading.kis_client import get_kis_client
                    from app.trading.kis_order import KISOrderExecutor

                    client = get_kis_client(is_mock=False)
                    executor = KISOrderExecutor(client)
                    result = await executor.sell(
                        symbol=symbol, qty=pos.qty, order_type="MARKET",
                    )
                    logger.info("전량 청산 [real]: %s %d주 → %s", symbol, pos.qty, result)
            except Exception as e:
                logger.error("전량 청산 실패: %s %d주 — %s", symbol, pos.qty, e)

    async def handle_review(self, *, force: bool = False) -> dict:
        """16:30 — 매매 요약 + 성과 지표 + live_feedback 기록."""
        async with async_session() as session:
            run = await self._get_or_create_today(session)

            # 비거래일 체크
            if not await self._is_trading_day():
                logger.info("review: 비거래일 — 스킵")
                await session.commit()
                return {"success": True, "message": "비거래일 — 리뷰 스킵"}

            # step_status 중복/완료 체크 (force=True면 우회)
            if not force:
                step_info = self._get_step(run, "review")
                if step_info.get("status") == "completed":
                    logger.info("review 이미 완료 — 스킵")
                    await session.commit()
                    return {"success": True, "phase": "REVIEW", "message": "이미 완료"}
                if step_info.get("status") == "running":
                    logger.info("review 진행 중 — 스킵")
                    await session.commit()
                    return {"success": True, "phase": "REVIEW", "message": "진행 중"}

            if not await self._transition(session, run, "REVIEW", force=force):
                await session.commit()
                return {"success": False, "message": f"전이 불가: {run.phase} → REVIEW"}

            await self._set_step(session, run, "review", "running")

            # TradeReviewer로 리뷰 생성
            try:
                from app.workflow.trade_reviewer import TradeReviewer
                reviewer = TradeReviewer()
                review = await reviewer.generate_review(session, run)
                run.review_summary = review.to_dict()
                run.trade_count = review.trade_count
                run.pnl_pct = review.total_pnl_pct
                run.pnl_amount = review.total_pnl
            except Exception as e:
                logger.warning("TradeReviewer 실행 실패: %s", e)
                run.review_summary = {"status": "error", "error": str(e)}

            # FeedbackEngine으로 mining_context 생성 (기존 외부 피드백 보존)
            adj_result = None  # 텔레그램 보고용
            try:
                from app.workflow.feedback_engine import FeedbackEngine
                engine = FeedbackEngine()

                # v2: 구조화된 피드백 생성 (파라미터 평가 + 마이닝 힌트)
                structured_feedback = await engine.generate_structured_feedback(
                    session, run
                )

                # 파라미터 자동 튜닝 (활성화 시에만)
                if settings.WORKFLOW_PARAM_EVAL_ENABLED:
                    try:
                        from app.workflow.param_adjuster import ParameterAdjuster
                        adjuster = ParameterAdjuster()
                        adj_result = await adjuster.apply_adjustments(
                            session, run, structured_feedback.param_assessments
                        )
                        logger.info(
                            "파라미터 조정: applied=%d, skipped=%d",
                            len(adj_result["applied"]), len(adj_result["skipped"]),
                        )
                    except Exception as e:
                        logger.warning("ParameterAdjuster 실행 실패: %s", e)
                        adj_result = {"applied": [], "skipped": [], "error": str(e)}

                # 구조화된 마이닝 프롬프트 생성
                structured_context = engine.format_mining_prompt(structured_feedback)

                # 기존 generate_context() 결과도 포함 (하위 호환)
                basic_context = await engine.generate_context(session, run)

                # 외부 피드백(OpenClaw 등)만 선별 보존 — 내부 생성 피드백은 매회 새로 생성
                external_lines = _extract_external_feedback(run.mining_context or "")
                parts = [structured_context, basic_context]
                if external_lines:
                    parts.append(external_lines)
                run.mining_context = "\n".join(parts)

                # 구조화된 피드백 전문 저장
                if run.review_summary and isinstance(run.review_summary, dict):
                    run.review_summary["structured_feedback"] = structured_feedback.to_dict()
                    flag_modified(run, "review_summary")
            except Exception as e:
                logger.warning("FeedbackEngine 실행 실패: %s", e)
                if not run.mining_context:
                    run.mining_context = "자동 워크플로우 — 피드백 생성 실패"

            # H3: 팩터 스탈니스 체크
            try:
                from app.workflow.staleness_checker import check_staleness
                staleness = await check_staleness(session)
                if staleness["warned"] or staleness["stale"] or staleness["retired"]:
                    logger.info(
                        "스탈니스 결과: warned=%d, stale=%d, retired=%d",
                        staleness["warned"], staleness["stale"], staleness["retired"],
                    )
            except Exception as e:
                logger.warning("스탈니스 체크 실패: %s", e)

            # 다이버전스 재확인 (정밀 리뷰 후 — 멀티 팩터 전체)
            review_factor_ids = [
                f["factor_id"]
                for f in (run.config or {}).get("selected_factors", [])
            ]
            if not review_factor_ids and run.selected_factor_id:
                review_factor_ids = [str(run.selected_factor_id)]
            for rev_fid in review_factor_ids:
                try:
                    from app.workflow.divergence_detector import check_divergence
                    div_result = await check_divergence(session, rev_fid)
                    if div_result.get("action") in ("halt", "warn"):
                        logger.info(
                            "REVIEW 다이버전스 — factor=%s: %s — %s",
                            rev_fid, div_result["action"], div_result,
                        )
                except Exception as e:
                    logger.warning("REVIEW 다이버전스 체크 실패 — factor=%s: %s", rev_fid, e)

            run.completed_at = datetime.now(timezone.utc)
            await self._set_step(session, run, "review", "completed")
            await self._log_event(session, run, "review_complete", "리뷰 + 피드백 완료")
            await session.commit()

        # 텔레그램: Claude LLM 일일 리뷰 보고 (1일 1회)
        try:
            import json as _json
            from app.telegram.bot import send_once

            review_summary = run.review_summary or {}

            # Claude에게 전달할 구조화된 데이터
            review_data = {
                "date": str(run.date),
                "initial_capital": int(settings.WORKFLOW_INITIAL_CAPITAL),
                "num_sessions": len((run.config or {}).get("trading_context_ids", [])),
                "total_trades": run.trade_count or 0,
                "total_pnl_amount": float(run.pnl_amount) if run.pnl_amount else 0.0,
                "total_pnl_pct": round(float(run.pnl_pct or 0), 2),
                "win_rate": review_summary.get("win_rate", 0),
                "per_session": review_summary.get("per_session", []),
                "improvements": review_summary.get("improvements", []),
                "time_breakdown": review_summary.get("time_breakdown", {}),
                "param_adjustment": adj_result if adj_result else None,
            }

            # Claude LLM으로 의미 있는 리뷰 생성
            try:
                from app.core.llm._anthropic import chat_simple
                llm_response = await chat_simple(
                    system=(
                        "당신은 퀀트 트레이딩 일일 리뷰어입니다. "
                        "세션별(팩터별) 매매 결과를 분석하여 텔레그램 리포트를 작성하세요.\n"
                        "- 각 팩터 수식이 무엇을 의미하는지 1줄 설명\n"
                        "- 세션별 성과 비교 (거래수, 승률, 손익)\n"
                        "- 전체 포트폴리오 수익률 (자본금 기준)\n"
                        "- 어떤 팩터가 가장 잘/못했는지 분석\n"
                        "- 개선 방향 1-2줄\n"
                        "HTML 태그(<b>, <i>, <code>)를 사용하세요. "
                        "이모지 적절히 사용. 800자 이내. 한국어로 작성."
                    ),
                    messages=[{
                        "role": "user",
                        "content": _json.dumps(review_data, ensure_ascii=False, default=str),
                    }],
                    max_tokens=1200,
                    caller="workflow.review",
                )
                tg_msg = llm_response.text
            except Exception as llm_err:
                logger.warning("Claude 리뷰 생성 실패, 정적 폴백: %s", llm_err)
                # 정적 폴백
                pnl_pct = run.pnl_pct or 0.0
                pnl_amount = float(run.pnl_amount) if run.pnl_amount else 0.0
                tg_msg = (
                    f"\U0001f4cb <b>일일 리뷰</b> ({run.date})\n"
                    f"거래 {run.trade_count or 0}건 | "
                    f"손익 {pnl_amount:+,.0f}원 ({pnl_pct:+.2f}%)\n"
                    f"승률 {review_summary.get('win_rate', 0):.1f}%"
                )

            await send_once(
                "daily_review", tg_msg,
                category="review_report", caller="workflow.orchestrator",
            )
        except Exception as e:
            logger.debug("리뷰 텔레그램 보고 실패: %s", e)

        # 리뷰 완료 → 즉시 마이닝 시작 (18:00까지 기다리지 않음)
        try:
            mining_result = await self.handle_mining()
            logger.info("REVIEW 완료 후 마이닝 자동 시작: %s", mining_result)
        except Exception as e:
            logger.warning("REVIEW 후 마이닝 자동 시작 실패: %s", e)

        return {"success": True, "phase": "REVIEW"}

    async def handle_mining(self, *, force: bool = False) -> dict:
        """마이닝 시작 — 상시 가동, 장중에만 비가동."""
        async with async_session() as session:
            run = await self._get_or_create_today(session)

            # step_status: mining은 반복 작업이므로 completed여도 재실행 OK
            # running이면 중복 실행 방지
            step_info = self._get_step(run, "mining")
            if step_info.get("status") == "running":
                logger.info("mining 진행 중 — 스킵")
                await session.commit()
                await self._ensure_mining_running()
                return {"success": True, "phase": "MINING", "message": "진행 중"}

            # 이미 MINING이면 팩토리만 확인
            if run.phase == "MINING":
                await session.commit()
                await self._ensure_mining_running()
                return {"success": True, "phase": "MINING", "message": "이미 MINING 상태"}

            # Redis 플래그 체크: 사용자가 수동 중지한 경우 MINING 전이하지 않음
            try:
                from app.core.redis import get_client as get_redis
                _r = get_redis()
                _flag = await _r.get("alpha:factory:user_stopped")
                if _flag and str(_flag) == "true":
                    logger.info("마이닝 사용자 중지 플래그 감지 — MINING 전이 건너뜀")
                    await self._log_event(
                        session, run, "mining_skipped",
                        "팩토리 수동 중단 상태 — 마이닝 스킵",
                    )
                    await session.commit()
                    return {"success": True, "phase": run.phase, "detail": "factory_user_stopped"}
            except Exception:
                pass

            if not await self._transition(session, run, "MINING"):
                # 전이 실패 시 강제 전이 (마이닝은 항상 가동 가능해야 함)
                logger.warning("MINING 전이 실패 — force 폴백: %s → MINING", run.phase)
                if not await self._transition(session, run, "MINING", force=True):
                    await session.commit()
                    return {"success": False, "message": f"전이 불가: {run.phase} → MINING"}

            await self._set_step(session, run, "mining", "running")

            mining_context = run.mining_context or "자동 워크플로우 상시 마이닝"
            try:

                from app.alpha.factory_client import get_factory_client
                _mcfg = await get_mining_config()
                _data_interval = _mcfg["interval"]
                factory = get_factory_client(interval=_data_interval)
                if not (await factory.get_status())["running"]:
                    # 인터벌별 디폴트 날짜 (메모리 효율 + 통계적 유의성)
                    _today = today_kst().isoformat()
                    _start_date = "2014-01-01" if _data_interval == "1d" else ""
                    _end_date = _today if _data_interval == "1d" else ""

                    await factory.start(
                        context=mining_context,
                        start_date=_start_date,
                        end_date=_end_date,
                        interval_minutes=0,
                        max_iterations=_mcfg["max_iterations"],
                        enable_crossover=settings.ALPHA_FACTORY_CROSSOVER_ENABLED,
                        max_cycles=0,  # 무제한 (PRE_MARKET에 의해 중지)
                        data_interval=_data_interval,
                        population_size=_mcfg["population_size"],
                        cpcv_n_groups=_mcfg["cpcv_n_groups"],
                        cpcv_n_test=_mcfg["cpcv_n_test"],
                        cpcv_embargo_days=_mcfg["cpcv_embargo_days"],
                    )
                    await self._log_event(
                        session, run, "mining_start",
                        f"상시 마이닝 시작: context={mining_context[:80]}",
                    )
            except Exception as e:
                logger.error("알파 팩토리 시작 실패: %s", e)
                run.error_message = f"마이닝 시작 실패: {e}"
                await self._set_step(
                    session, run, "mining", "error",
                    error=str(e)[:200],
                )
                await session.commit()
                return {"success": False, "message": str(e)}

            run.status = "MINING"
            await self._set_step(
                session, run, "mining", "completed",
                detail="factory_started",
            )
            await session.commit()
            return {"success": True, "phase": "MINING"}

    async def handle_stop_mining(self) -> dict:
        """06:00 — 야간 마이닝 중지 + IDLE 복귀."""
        try:
            from app.alpha.factory_client import get_factory_client
            factory = get_factory_client()
            if (await factory.get_status())["running"]:
                await factory.stop()
                logger.info("06:00 야간 마이닝 자동 중지")
        except Exception as e:
            logger.warning("마이닝 중지 실패: %s", e)

        async with async_session() as session:
            run = await self._get_or_create_today(session)
            if run.phase == "MINING":
                run.phase = "IDLE"
                run.status = "COMPLETED"
                await self._log_event(
                    session, run, "mining_stop",
                    f"야간 마이닝 완료: {self._state_summary(factory)}",
                )
                await session.commit()
        return {"success": True, "message": "야간 마이닝 중지 완료"}

    @staticmethod
    def _state_summary(factory) -> str:
        try:
            st = factory.get_status()
            return f"cycles={st.get('cycles_completed',0)}, factors={st.get('factors_discovered_total',0)}"
        except Exception:
            return "상태 조회 실패"

    # ── Phase Watchdog (누락된 페이즈 자동 catch-up) ──

    # 시간대별 기대 페이즈 (KST 기준)
    # (시작시각, 끝시각, 기대페이즈, 핸들러이름)
    # 마이닝 상시가동: 장중(08:30~16:30)에만 매매, 나머지는 마이닝
    _PHASE_SCHEDULE: list[tuple[time_type, time_type, str, str]] = [
        (time_type(0, 0), time_type(8, 30), "MINING", "handle_mining"),
        (time_type(8, 30), time_type(9, 0), "PRE_MARKET", "handle_pre_market"),
        (time_type(9, 0), time_type(15, 30), "TRADING", "handle_market_open"),
        (time_type(15, 30), time_type(16, 30), "MARKET_CLOSE", "handle_market_close"),
        (time_type(16, 30), time_type(18, 0), "REVIEW", "handle_review"),
        (time_type(18, 0), time_type(23, 59, 59), "MINING", "handle_mining"),
    ]

    # 페이즈 실행 순서 (catch-up 시 순차 실행)
    _PHASE_ORDER: list[str] = [
        "PRE_MARKET", "TRADING", "MARKET_CLOSE", "REVIEW", "MINING",
    ]

    def _expected_phase_now(self) -> tuple[str, str]:
        """현재 KST 시간 기준 기대 페이즈 + 핸들러 이름.

        비거래일(주말)이면 항상 MINING 반환.
        공휴일은 pykrx 비동기 호출이 필요해서 여기서는 주말만 체크.
        (공휴일은 각 핸들러의 _is_trading_day()에서 걸러짐)
        """
        now_kst_dt = now_kst()

        # 주말이면 항상 마이닝
        if now_kst_dt.weekday() >= 5:  # 5=토, 6=일
            return "MINING", "handle_mining"

        for start, end, phase, handler in self._PHASE_SCHEDULE:
            if start <= now_kst_dt.time() < end:
                return phase, handler
        # 23:59:59 이후 (사실상 없지만 안전장치)
        return "MINING", "handle_mining"

    # step_status 키와 기대 페이즈의 매핑
    _PHASE_TO_STEP: dict[str, str] = {
        "PRE_MARKET": "pre_market",
        "TRADING": "market_open",
        "MARKET_CLOSE": "market_close",
        "REVIEW": "review",
        "MINING": "mining",
    }

    async def _phase_watchdog(self) -> None:
        """5분마다 실행: 실제 FSM 페이즈와 기대 페이즈를 비교하여 누락 catch-up.

        - SKIPPED/EMERGENCY_STOP: catch-up 하지 않음
        - MINING 페이즈인데 팩토리가 안 돌고 있으면 재시작
        - step_status 완료/진행 중인 단계는 catch-up 스킵
        - 그 외 누락된 페이즈를 순차적으로 실행
        """
        try:
            async with async_session() as session:
                run = await self._get_or_create_today(session)
                actual_phase = run.phase
                actual_status = run.status

                # EMERGENCY_STOP은 건드리지 않음
                if actual_phase == "EMERGENCY_STOP":
                    return

                # 장중 시간대(PRE_MARKET~MARKET_CLOSE)인데 팩토리가 돌고 있으면 즉시 중지
                # (크론잡 누락 방지 — APScheduler가 오늘 08:30을 건너뛸 수 있음)
                expected_phase, _ = self._expected_phase_now()
                if expected_phase in ("PRE_MARKET", "TRADING", "MARKET_CLOSE"):
                    try:
                        from app.alpha.factory_client import get_factory_client
                        factory = get_factory_client()
                        if (await factory.get_status())["running"]:
                            await factory.stop()
                            logger.warning("장중 팩토리 실행 감지 → 강제 중지 (expected=%s)", expected_phase)
                    except Exception as e:
                        logger.error("장중 팩토리 강제 중지 실패: %s", e)

                # SKIPPED/STOPPED 상태에서도 마이닝은 허용 (비거래일 상시가동)
                if actual_status in ("SKIPPED", "STOPPED"):
                    expected_phase, _ = self._expected_phase_now()
                    if expected_phase == "MINING":
                        if actual_phase != "MINING":
                            await self._transition(
                                session, run, "MINING", force=True,
                            )
                            await session.commit()
                        await self._ensure_mining_running()
                    return

            expected_phase, handler_name = self._expected_phase_now()

            # step_status 기반 추가 체크
            step_key = self._PHASE_TO_STEP.get(expected_phase)
            if step_key:
                async with async_session() as session:
                    run = await self._get_or_create_today(session)
                    step_info = self._get_step(run, step_key)

                    if step_info.get("status") == "completed" and step_key != "mining":
                        # 시간대 검증: completed 시각이 해당 단계의 정상 시간대 안인지 확인
                        # 예: market_open은 09:00 이후에 완료되어야 유효
                        _step_valid = True
                        _at = step_info.get("at", "")
                        if _at:
                            try:
                                _dt = datetime.fromisoformat(_at)
                                _completed_hour = to_kst(_dt).hour
                                _STEP_MIN_HOUR = {
                                    "pre_market": 8, "market_open": 9,
                                    "market_close": 15, "review": 16,
                                }
                                _min_h = _STEP_MIN_HOUR.get(step_key)
                                if _min_h is not None and _completed_hour < _min_h:
                                    _step_valid = False
                                    logger.info(
                                        "step_status %s completed at %s (KST %d시 < %d시) — 무효, 재실행",
                                        step_key, _at[:19], _completed_hour, _min_h,
                                    )
                            except (ValueError, IndexError):
                                pass
                        if _step_valid:
                            # 정상 완료이지만, TRADING 중이면 세션 생존 확인 필요
                            if expected_phase == "TRADING" and step_key == "market_open":
                                await self._session_health_check()
                            return

                    # 진행 중인 단계는 중복 실행 금지
                    if step_info.get("status") == "running":
                        return

            # 이미 기대 페이즈에 있으면 OK — 단, step 미완료면 재실행
            if actual_phase == expected_phase:
                if expected_phase == "MINING":
                    await self._ensure_mining_running()
                elif expected_phase == "TRADING":
                    # market_open step이 미완료면 세션 시작 필요
                    async with async_session() as _s:
                        _r = await self._get_or_create_today(_s)
                        _mo = self._get_step(_r, "market_open")
                    if _mo.get("status") != "completed":
                        logger.info("Phase Watchdog: TRADING이지만 market_open 미완료 — handle_market_open 실행")
                        await self.handle_market_open()
                    else:
                        await self._session_health_check()
                return

            # 기대 페이즈가 IDLE이면 catch-up 불필요
            if expected_phase == "IDLE":
                return

            # 페이즈가 뒤처진 경우 catch-up 실행
            logger.warning(
                "Phase Watchdog: 실제=%s, 기대=%s — catch-up 시작",
                actual_phase, expected_phase,
            )

            await self._catchup_to_phase(actual_phase, expected_phase)

        except Exception as e:
            logger.error("Phase Watchdog 오류: %s", e)

    # catch-up 시 각 페이즈가 실행 가능한 시간 윈도우 (KST)
    _PHASE_VALID_WINDOW: dict[str, tuple[time_type, time_type]] = {
        "PRE_MARKET": (time_type(8, 0), time_type(9, 30)),
        "TRADING": (time_type(8, 50), time_type(15, 40)),
        "MARKET_CLOSE": (time_type(15, 20), time_type(17, 0)),
        "REVIEW": (time_type(15, 20), time_type(20, 0)),
        # MINING: 항상 실행 가능 (윈도우 없음)
    }

    def _should_skip_catchup_phase(self, phase_name: str) -> bool:
        """catch-up 시 현재 시간이 해당 페이즈의 유효 윈도우 밖이면 True."""
        window = self._PHASE_VALID_WINDOW.get(phase_name)
        if window is None:
            return False  # MINING 등은 항상 실행
        now_t = now_kst().time()
        start, end = window
        return not (start <= now_t < end)

    async def _catchup_to_phase(self, actual: str, target: str) -> None:
        """actual → target까지 누락된 페이즈 핸들러를 순차 실행."""
        order = self._PHASE_ORDER

        # actual이 IDLE이면 처음부터 시작
        if actual == "IDLE" or actual not in order:
            start_idx = 0
        else:
            start_idx = order.index(actual) + 1

        # target 인덱스
        if target not in order:
            return
        target_idx = order.index(target)

        # target까지 순차 실행
        handlers = {
            "PRE_MARKET": self.handle_pre_market,
            "TRADING": self.handle_market_open,
            "MARKET_CLOSE": self.handle_market_close,
            "REVIEW": self.handle_review,
            "MINING": self.handle_mining,
        }

        for i in range(start_idx, target_idx + 1):
            phase_name = order[i]
            handler = handlers.get(phase_name)
            if handler is None:
                continue
            # 현재 시간이 해당 페이즈의 유효 시간대가 아니면 스킵
            if self._should_skip_catchup_phase(phase_name):
                logger.info(
                    "Phase Watchdog catch-up: %s 스킵 (현재 시간대 불일치)",
                    phase_name,
                )
                continue
            try:
                logger.info("Phase Watchdog catch-up: %s 실행", phase_name)
                result = await handler()
                logger.info("Phase Watchdog catch-up: %s 결과=%s", phase_name, result)

                # 핸들러가 실패하면 마이닝이 목표이면 직접 전환, 아니면 중단
                if not result.get("success", False):
                    logger.warning(
                        "Phase Watchdog: %s 실패 — %s",
                        phase_name, result.get("message", ""),
                    )
                    if target == "MINING":
                        # 중간 단계 실패해도 마이닝은 시작해야 함
                        logger.info("Phase Watchdog: 중간 실패 무시, MINING 직접 전환")
                        await self.handle_mining()
                    break
            except Exception as e:
                logger.error("Phase Watchdog catch-up 실패 (%s): %s", phase_name, e)
                if target == "MINING":
                    try:
                        await self.handle_mining()
                    except Exception as me:
                        logger.error("Phase Watchdog: MINING 폴백도 실패: %s", me)
                break

    # ── Session Health Check (TRADING 페이즈 세션 감시) ──

    async def _session_health_check(self) -> None:
        """TRADING 페이즈 중 LiveSession 건강 상태를 점검하고 비정상 시 재시작."""
        # 자동매매 수동 중단 플래그 → 건강 체크 비활성화
        try:
            from app.core.redis import get_client as _get_redis
            _r = _get_redis()
            _flag = await _r.get("trading:user_stopped")
            if _flag and str(_flag) == "true":
                return
        except Exception:
            pass

        try:
            from app.trading.live_runner import list_sessions

            sessions = list_sessions()
            running = [s for s in sessions if s.status == "running"]
            if not running:
                # TRADING 중인데 running 세션이 0개 → 컨테이너 재시작 등으로 유실
                logger.warning(
                    "Session Health: TRADING 중 running 세션 0개 — handle_market_open 재실행"
                )
                # step_status의 market_open을 제거하여 handle_market_open이 재실행되도록
                async with async_session() as _s:
                    _r = await self._get_or_create_today(_s)
                    steps = dict(_r.step_status or {})
                    steps.pop("market_open", None)
                    _r.step_status = steps
                    flag_modified(_r, "step_status")
                    await _s.commit()
                await self.handle_market_open()
                return

            for live_sess in sessions:
                if live_sess.status == "stopped":
                    self._session_unhealthy_counts.pop(live_sess.id, None)
                    self._session_restart_failures.pop(live_sess.id, None)
                    continue

                sid = live_sess.id
                issues: list[tuple[str, str]] = []  # (severity, message)

                # Task 크래시 체크 (유일한 재시작 조건)
                # 시간 기반 "멈춤" 판단은 오판 위험이 너무 높아 제거
                # (950종목 처리에 15분+ 소요 — 정상 동작을 죽이는 역효과)
                task = getattr(live_sess, "_task", None)
                if task is not None and task.done() and live_sess.status == "running":
                    exc = task.exception() if not task.cancelled() else None
                    issues.append((
                        "CRITICAL",
                        f"background task 크래시: {exc}" if exc else "background task 종료됨",
                    ))

                elif live_sess.status == "error":
                    issues.append(("CRITICAL", f"세션 에러: {live_sess.error_message}"))

                # 건강하면 카운터 리셋
                if not issues:
                    if sid in self._session_unhealthy_counts:
                        self._session_unhealthy_counts.pop(sid, None)
                    continue

                # 비정상 카운터 증가
                count = self._session_unhealthy_counts.get(sid, 0) + 1
                self._session_unhealthy_counts[sid] = count

                severity_summary = ", ".join(f"[{s}] {m}" for s, m in issues)
                logger.warning(
                    "Session Health: %s 비정상 (%d연속): %s",
                    sid[:8], count, severity_summary,
                )

                # 재시작 판단: CRITICAL → 즉시, WARNING → 3연속(15분)
                needs_restart = any(s == "CRITICAL" for s, _ in issues)
                if not needs_restart and count >= 3:
                    needs_restart = True

                if needs_restart:
                    await self._restart_trading_session(sid, issues)

            # 장중 분봉 수집기 생존 체크 (Worker 재시작 시 Task 유실 대응)
            try:
                _now_t = now_kst().time()
                # 09:00~15:30 장중 시간대에서만 수집기 재시작
                if time_type(9, 0) <= _now_t < time_type(15, 30):
                    from app.trading.live_runner import _intraday_collector_task, start_intraday_candle_collector
                    if _intraday_collector_task is None or _intraday_collector_task.done():
                        # 수집 대상 종목 결정
                        scan_symbols: list[str] = []
                        for s in sessions:
                            scan_symbols.extend(getattr(s, "scan_symbols", []))
                        if not scan_symbols:
                            from sqlalchemy import text as _t
                            async with async_session() as _db:
                                _r = await _db.execute(
                                    _t("SELECT symbol FROM stock_masters WHERE market = 'KOSPI' ORDER BY symbol LIMIT 200")
                                )
                                scan_symbols = [r[0] for r in _r.fetchall()]
                        if scan_symbols:
                            asyncio.create_task(start_intraday_candle_collector(scan_symbols))
                            logger.warning("장중 분봉 수집기 재시작 (%d종목) — Task 유실 감지", len(scan_symbols))
            except Exception as _e:
                logger.debug("분봉 수집기 체크 실패: %s", _e)

        except Exception as e:
            logger.error("Session Health Check 오류: %s", e)

    async def _restart_trading_session(
        self, session_id: str, issues: list[tuple[str, str]]
    ) -> None:
        """비정상 세션을 중지하고 새 세션으로 교체."""
        # 재시작 실패 횟수 확인 (3회 초과 시 포기)
        fail_count = self._session_restart_failures.get(session_id, 0)
        if fail_count >= 3:
            logger.error(
                "Session Restart: %s 재시작 3회 실패 — 추가 시도 중단", session_id[:8]
            )
            from app.telegram.bot import send_message as tg_send
            await tg_send(
                f"[CRITICAL] 매매 세션 {session_id[:8]} 재시작 3회 실패. 수동 확인 필요.",
                category="workflow_alert",
                caller="orchestrator.restart_fail",
            )
            return

        try:
            from app.alpha.backtest_bridge import register_alpha_factor
            from app.alpha.models import AlphaFactor
            from app.trading.context import TradingContext, _contexts
            from app.trading.live_runner import start_session, stop_session

            # 1. 기존 세션 중지
            logger.info("Session Restart: %s 중지 시작", session_id[:8])
            await stop_session(session_id)

            # 2. 기존 TradingContext 아카이브
            async with async_session() as db:
                run = await self._get_or_create_today(db)

                stmt = (
                    update(TradingContextModel)
                    .where(TradingContextModel.id == uuid.UUID(session_id))
                    .values(status="replaced")
                )
                await db.execute(stmt)

                # 3. 팩터 로드
                if not run.selected_factor_id:
                    logger.error("Session Restart: selected_factor_id 없음")
                    self._session_restart_failures[session_id] = fail_count + 1
                    return

                factor_stmt = select(AlphaFactor).where(
                    AlphaFactor.id == run.selected_factor_id
                )
                factor_result = await db.execute(factor_stmt)
                factor = factor_result.scalar_one_or_none()
                if factor is None:
                    logger.error("Session Restart: 팩터 조회 실패")
                    self._session_restart_failures[session_id] = fail_count + 1
                    return

                # 4. 새 TradingContext 생성
                mode = settings.WORKFLOW_TRADING_MODE
                ctx_model = await build_context_from_factor(db, factor, mode=mode)

                # 5. interval 오버라이드 (현재 config 반영)
                current_interval = (await get_mining_config())["interval"]
                if ctx_model.strategy.get("interval") != current_interval:
                    strategy = dict(ctx_model.strategy)
                    strategy["interval"] = current_interval
                    ctx_model.strategy = strategy
                    stmt_upd = (
                        update(TradingContextModel)
                        .where(TradingContextModel.id == ctx_model.id)
                        .values(strategy=strategy)
                    )
                    await db.execute(stmt_upd)

                # 6. WorkflowRun 업데이트
                run.trading_context_id = ctx_model.id

                issue_summary = "; ".join(f"[{s}]{m}" for s, m in issues)
                await self._log_event(
                    db, run, "session_restart",
                    f"세션 재시작: {session_id[:8]} → {str(ctx_model.id)[:8]} "
                    f"(interval={current_interval}, 원인: {issue_summary[:120]})",
                )
                await db.commit()

            # 7. 인메모리 등록 + 새 세션 시작 (DB 세션 밖에서)
            ctx = TradingContext.from_db_model(ctx_model)
            _contexts[ctx.id] = ctx
            register_alpha_factor(str(factor.id), factor.expression_str)

            new_session = await start_session(ctx)
            logger.info(
                "Session Restart 완료: %s → %s (interval=%s)",
                session_id[:8], new_session.id[:8], current_interval,
            )

            # 카운터 리셋
            self._session_unhealthy_counts.pop(session_id, None)
            self._session_restart_failures.pop(session_id, None)

            from app.telegram.bot import send_message as tg_send
            await tg_send(
                f"매매 세션 자동 재시작: {session_id[:8]} → {new_session.id[:8]} "
                f"(interval={current_interval})",
                category="workflow_alert",
                caller="orchestrator.restart_ok",
            )

        except Exception as e:
            logger.error("Session Restart 실패 (%s): %s", session_id[:8], e)
            self._session_restart_failures[session_id] = fail_count + 1

    async def _ensure_mining_running(self) -> None:
        """팩토리가 안 돌고 있으면 재시작 (마이닝 상시가동 보장)."""
        try:
            from app.alpha.scheduler import get_all_schedulers, get_scheduler

            # Redis 플래그 체크: 프론트/API에서 user_stopped 설정 시 와치독 비활성화
            try:
                from app.core.redis import get_client as get_redis
                _redis = get_redis()
                _user_stopped_flag = await _redis.get("alpha:factory:user_stopped")
                if _user_stopped_flag and str(_user_stopped_flag) == "true":
                    return  # 와치독 비활성화 — 재시작 안 함
            except Exception:
                pass

            # Causal sweep 진행 중이면 재시작 건너뜀
            try:
                _sweep_flag = await _redis.get("alpha:causal_sweep:running")
                if _sweep_flag:
                    logger.info("인과검증 전수 진행 중 — watchdog 재시작 건너뜀")
                    return
            except Exception:
                pass

            # ★ 모든 interval 스케줄러 중 하나라도 실행 중이면 재시작 불필요
            for _iv, _sched in get_all_schedulers().items():
                if _sched._task and not _sched._task.done():
                    return

            # 설정된 interval의 스케줄러 가져오기 (5m 고정 아님)
            _mcfg = await get_mining_config()
            scheduler = get_scheduler(_mcfg["interval"])

            status = scheduler.get_status()
            if not status["running"]:
                # 사용자가 의도적으로 중지한 경우 watchdog 재시작 안 함
                if status.get("user_stopped"):
                    logger.info("팩토리 수동 중지 상태 — watchdog 재시작 건너뜀")
                    return

                # 쿨다운: 최근 10분 이내 시작된 팩토리는 재시작하지 않음
                # (데이터 로딩에 수 분 소요 → 크래시 직후 재시작 루프 방지)
                started_at = status.get("started_at")
                if started_at:
                    from datetime import datetime
                    try:
                        started = datetime.fromisoformat(started_at)
                        if started.tzinfo is None:
                            started = started.replace(tzinfo=timezone.utc)
                        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
                        if elapsed < 600:
                            logger.info(
                                "마이닝 팩토리 미실행이나 최근 시작(%d초 전) — 쿨다운 대기",
                                int(elapsed),
                            )
                            return
                    except Exception:
                        pass

                logger.warning("마이닝 팩토리 미실행 감지 — 재시작")
                async with async_session() as session:
                    run = await self._get_or_create_today(session)
                    mining_context = run.mining_context or "자동 워크플로우 상시 마이닝 (watchdog 복구)"
                    _today = today_kst().isoformat()
                    _ws = "2014-01-01" if _mcfg["interval"] == "1d" else ""
                    _we = _today if _mcfg["interval"] == "1d" else ""
                    await scheduler.start(
                        context=mining_context,
                        start_date=_ws,
                        end_date=_we,
                        interval_minutes=0,
                        max_iterations=_mcfg["max_iterations"],
                        enable_crossover=settings.ALPHA_FACTORY_CROSSOVER_ENABLED,
                        max_cycles=0,
                        data_interval=_mcfg["interval"],
                        population_size=_mcfg["population_size"],
                        cpcv_n_groups=_mcfg["cpcv_n_groups"],
                        cpcv_n_test=_mcfg["cpcv_n_test"],
                        cpcv_embargo_days=_mcfg["cpcv_embargo_days"],
                    )
                    await self._log_event(
                        session, run, "watchdog_mining_restart",
                        "마이닝 팩토리 자동 재시작 (watchdog)",
                    )
                    await session.commit()
                logger.info("마이닝 팩토리 재시작 완료")
        except Exception as e:
            logger.error("마이닝 재시작 실패: %s", e)

    async def handle_emergency_stop(self) -> dict:
        """긴급 정지 — 모든 세션 즉시 중지 + 전량 청산."""
        async with async_session() as session:
            run = await self._get_or_create_today(session)
            old_phase = run.phase
            run.phase = "EMERGENCY_STOP"
            run.status = "STOPPED"

            # 모든 LiveSession 즉시 중지
            try:
                from app.trading.live_runner import list_sessions, stop_session
                for live_session in list_sessions():
                    if live_session.status == "running":
                        await self._close_all_positions(live_session)
                        await stop_session(live_session.id)
            except Exception as e:
                logger.error("긴급 세션 중지 실패: %s", e)

            # TradingContext 비활성화
            if run.trading_context_id:
                stmt = (
                    update(TradingContextModel)
                    .where(TradingContextModel.id == run.trading_context_id)
                    .values(status="stopped")
                )
                await session.execute(stmt)

            # 알파 팩토리 즉시 중지
            try:
                from app.alpha.factory_client import get_factory_client
                factory = get_factory_client()
                if (await factory.get_status())["running"]:
                    await factory.stop()
                    logger.info("긴급 정지: 알파 팩토리 중지")
            except Exception as e:
                logger.error("긴급 정지 시 팩토리 중지 실패: %s", e)

            await self._log_event(
                session, run, "emergency_stop",
                f"긴급 정지: {old_phase} → EMERGENCY_STOP (전량 청산 + 팩토리 중지)",
            )
            await session.commit()
            return {"success": True, "phase": "EMERGENCY_STOP", "previous": old_phase}

    async def handle_resume(self) -> dict:
        """긴급 정지 해제 → IDLE → (MINING 시간대면) 즉시 팩토리 시작."""
        async with async_session() as session:
            run = await self._get_or_create_today(session)
            if run.phase != "EMERGENCY_STOP":
                return {"success": False, "message": f"현재 {run.phase} — EMERGENCY_STOP이 아님"}
            await self._transition(session, run, "IDLE")
            run.status = "PENDING"
            await self._log_event(session, run, "resume", "긴급 정지 해제 → IDLE")
            await session.commit()

        # 현재 시간이 MINING 시간대면 즉시 팩토리 시작 (watchdog 5분 대기 불필요)
        try:
            expected_phase, _ = self._expected_phase_now()
            if expected_phase == "MINING":
                await self.handle_mining()
        except Exception as e:
            logger.warning("resume 후 즉시 mining 시작 실패: %s", e)

        return {"success": True, "phase": "IDLE"}

    async def handle_reset(self) -> dict:
        """어떤 상태에서든 IDLE로 강제 리셋 (다음 사이클 준비)."""
        async with async_session() as session:
            run = await self._get_or_create_today(session)
            old_phase = run.phase
            if old_phase == "IDLE":
                return {"success": True, "phase": "IDLE", "message": "이미 IDLE"}
            run.phase = "IDLE"
            run.status = "COMPLETED" if old_phase == "MINING" else "RESET"
            await self._log_event(
                session, run, "reset",
                f"수동 리셋: {old_phase} → IDLE",
            )
            await session.commit()
            return {"success": True, "phase": "IDLE", "previous": old_phase}

    async def get_status(self) -> dict:
        """현재 워크플로우 상태를 반환."""
        async with async_session() as session:
            today = today_kst()
            stmt = select(WorkflowRun).where(WorkflowRun.date == today)
            result = await session.execute(stmt)
            run = result.scalar_one_or_none()

            if run is None:
                return {
                    "phase": "IDLE",
                    "date": str(today),
                    "status": "NO_RUN",
                    "message": "오늘 워크플로우 미시작",
                }

            # 선택된 팩터 이름/IC 조회
            factor_name = None
            factor_ic = None
            selected_factors_info = (run.config or {}).get("selected_factors", [])
            if selected_factors_info:
                # 멀티 팩터: 첫 번째 팩터 이름 + 전체 개수
                names = [f.get("name", "?") for f in selected_factors_info]
                factor_name = ", ".join(names[:3])
                if len(names) > 3:
                    factor_name += f" 외 {len(names) - 3}개"
            elif run.selected_factor_id:
                try:
                    from app.alpha.models import AlphaFactor
                    f_stmt = select(AlphaFactor).where(AlphaFactor.id == run.selected_factor_id)
                    f_result = await session.execute(f_stmt)
                    factor = f_result.scalar_one_or_none()
                    if factor:
                        factor_name = factor.name
                        factor_ic = factor.ic_mean
                except Exception:
                    pass

            data = {
                "phase": run.phase,
                "date": str(run.date),
                "status": run.status,
                "step_status": run.step_status,
                "selected_factor_id": str(run.selected_factor_id) if run.selected_factor_id else None,
                "selected_factor_name": factor_name,
                "selected_factor_ic": factor_ic,
                "trading_context_id": str(run.trading_context_id) if run.trading_context_id else None,
                "trade_count": run.trade_count,
                "pnl_pct": run.pnl_pct,
                "pnl_amount": float(run.pnl_amount) if run.pnl_amount is not None else None,
                "error_message": run.error_message,
                "started_at": run.started_at.isoformat() if run.started_at else None,
            }

            # 마이닝 상태 포함
            try:
                from app.alpha.factory_client import get_factory_client
                factory_status = await get_factory_client().get_status()
                data["mining_running"] = factory_status["running"]
                data["mining_cycles"] = factory_status["cycles_completed"]
                data["mining_factors"] = factory_status["factors_discovered_total"]
            except Exception:
                data["mining_running"] = False

            # Redis 캐싱 (Phase 1: 이중 기록)
            try:
                import json as _json
                from app.core.redis import hset
                await hset("workflow:status", {
                    k: _json.dumps(v, ensure_ascii=False, default=str) if isinstance(v, (dict, list)) else str(v) if v is not None else ""
                    for k, v in data.items()
                })
                # 12시간 TTL — 날짜 변경 시 stale 캐시 자동 만료
                from app.core.redis import get_client
                r = get_client()
                await r.expire("workflow:status", 43200)
            except Exception:
                pass

            return data

    async def get_history(
        self, session: AsyncSession, *, limit: int = 30
    ) -> list[WorkflowRun]:
        """최근 워크플로우 실행 히스토리."""
        stmt = (
            select(WorkflowRun)
            .order_by(WorkflowRun.date.desc())
            .limit(limit)
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())

    async def get_events(
        self, session: AsyncSession, run_id: uuid.UUID, *, limit: int = 100
    ) -> list[WorkflowEvent]:
        """특정 워크플로우 실행의 이벤트 로그."""
        stmt = (
            select(WorkflowEvent)
            .where(WorkflowEvent.workflow_run_id == run_id)
            .order_by(WorkflowEvent.created_at.asc())
            .limit(limit)
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())

    # ── OpenClaw 헬스체크 (설계서 §9) ──

    async def check_openclaw_health(self) -> None:
        """5분마다 OpenClaw 프로세스 상태 확인."""
        global _openclaw_fail_count, _independent_mode

        try:
            import httpx
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(settings.OPENCLAW_HEALTH_URL)
                if resp.status_code == 200:
                    _openclaw_fail_count = 0
                    if _independent_mode:
                        _independent_mode = False
                        logger.info("OpenClaw 복구 감지 — 정상 모드 복귀")
                    return
        except Exception:
            pass

        _openclaw_fail_count += 1
        logger.warning("OpenClaw 헬스체크 실패 (%d회 연속)", _openclaw_fail_count)

        if _openclaw_fail_count == 3:
            # subprocess.Popen은 Docker(Linux) 내에서 openclaw 바이너리가 없으므로 사용 불가
            # → 호스트에서 실행 중인 restart_agent.py(18790)에 HTTP POST로 재시작 요청
            from app.telegram.bot import send_message as tg_send
            ok = await self._try_restart_openclaw()
            if ok:
                logger.info("OpenClaw 재시작 요청 성공 — 60초 후 재확인")
                import asyncio as _asyncio
                import httpx as _httpx
                await _asyncio.sleep(60)
                try:
                    async with _httpx.AsyncClient(timeout=10) as _c:
                        _r = await _c.get(settings.OPENCLAW_HEALTH_URL)
                        if _r.status_code == 200:
                            _openclaw_fail_count = 0
                            await tg_send(
                                "OpenClaw 자동 복구 완료. 정상 모드 복귀.",
                                category="workflow_alert",
                                caller="orchestrator.openclaw_recovered",
                            )
                            return
                except Exception:
                    pass
                await tg_send(
                    "⚠️ OpenClaw 재시작 후에도 미응답. 수동 확인 필요.\n`openclaw gateway start`",
                    category="workflow_alert",
                    caller="orchestrator.openclaw_restart_failed",
                )
            else:
                logger.warning("OpenClaw restart agent(18790) 미응답 — 수동 재시작 필요")
                await tg_send(
                    "⚠️ OpenClaw restart agent 미응답. 수동 재시작 필요.\n`openclaw gateway start`",
                    category="workflow_alert",
                    caller="orchestrator.openclaw_restart_agent_down",
                )

        if _openclaw_fail_count >= 6 and not _independent_mode:
            _independent_mode = True
            logger.error("OpenClaw 30분 무응답 — APScheduler 독립 모드 전환")
            from app.telegram.bot import send_message as tg_send
            await tg_send(
                "OpenClaw 30분간 무응답. APScheduler 독립 모드 전환. 수동 확인 필요.",
                category="workflow_alert",
                caller="orchestrator.openclaw_timeout",
            )

    async def _try_restart_openclaw(self) -> bool:
        """restart_agent.py(18790)에 HTTP POST → openclaw gateway start 실행."""
        import httpx
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(settings.OPENCLAW_RESTART_URL)
                return resp.status_code in (200, 202, 204)
        except Exception as e:
            logger.warning("OpenClaw restart agent 호출 실패: %s", e)
            return False

    async def _restart_openclaw(self) -> None:
        """03:00 KST — OpenClaw 데몬 일일 재시작 (메모리 누수 방지, 설계서 §9.0).

        정상 동작 중이면 재시작하지 않음 (헬스체크 통과 시 스킵).
        """
        global _openclaw_fail_count
        import httpx

        # 헬스체크: 정상이면 재시작 스킵
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(settings.OPENCLAW_HEALTH_URL)
                if resp.status_code == 200:
                    _openclaw_fail_count = 0
                    logger.info("OpenClaw 03:00 — 정상 동작 중, 재시작 스킵")
                    return
        except Exception:
            logger.info("OpenClaw 03:00 — 헬스체크 실패, 재시작 시도")

        from app.telegram.bot import send_message as tg_send
        ok = await self._try_restart_openclaw()
        if ok:
            _openclaw_fail_count = 0
            logger.info("OpenClaw 03:00 일일 재시작 요청 완료")
        else:
            logger.error("OpenClaw 03:00 재시작 실패 — restart agent 미응답")
            await tg_send(
                "OpenClaw 03:00 자동 재시작 실패. 수동 확인 필요.\n`openclaw gateway start`",
                category="workflow_alert",
                caller="orchestrator.openclaw_restart_fail",
            )

    # ── APScheduler 연동 ──

    async def setup_scheduler(self) -> None:
        """APScheduler 크론잡 등록.

        APScheduler v4는 AsyncScheduler를 async context manager로 초기화 후
        start_in_background()로 백그라운드 실행해야 한다.
        """
        try:
            from apscheduler import AsyncScheduler
            from apscheduler.triggers.cron import CronTrigger
            from apscheduler.triggers.interval import IntervalTrigger

            scheduler = AsyncScheduler()
            # context manager 진입 → 내부 서비스 초기화
            await scheduler.__aenter__()
            self._scheduler = scheduler

            # 08:30 KST — PRE_MARKET
            # 매매 관련: 평일(월~금)만 실행 — 공휴일은 핸들러 내 _is_trading_day()에서 차단
            await scheduler.add_schedule(
                self.handle_pre_market,
                CronTrigger(hour=8, minute=30, day_of_week="mon-fri", timezone="Asia/Seoul"),
                id="workflow_pre_market",
            )
            # 09:00 KST — MARKET_OPEN (평일만)
            await scheduler.add_schedule(
                self.handle_market_open,
                CronTrigger(hour=9, minute=0, day_of_week="mon-fri", timezone="Asia/Seoul"),
                id="workflow_market_open",
            )
            # 15:30 KST — MARKET_CLOSE (평일만)
            await scheduler.add_schedule(
                self.handle_market_close,
                CronTrigger(hour=15, minute=30, day_of_week="mon-fri", timezone="Asia/Seoul"),
                id="workflow_market_close",
            )
            # 16:30 KST — REVIEW (평일만)
            await scheduler.add_schedule(
                self.handle_review,
                CronTrigger(hour=16, minute=30, day_of_week="mon-fri", timezone="Asia/Seoul"),
                id="workflow_review",
            )
            # 18:00 KST — MINING (매일 실행 — 주말/공휴일에도 마이닝)
            await scheduler.add_schedule(
                self.handle_mining,
                CronTrigger(hour=18, minute=0, timezone="Asia/Seoul"),
                id="workflow_mining",
            )
            # 06:00 stop_mining 제거: 마이닝 상시가동, PRE_MARKET(08:30)에서만 중지
            # 5분마다 — OpenClaw 헬스체크
            await scheduler.add_schedule(
                self.check_openclaw_health,
                IntervalTrigger(minutes=5),
                id="openclaw_health_check",
            )
            # 03:00 KST — OpenClaw 데몬 일일 재시작 (메모리 누수 방지)
            await scheduler.add_schedule(
                self._restart_openclaw,
                CronTrigger(hour=3, minute=0, timezone="Asia/Seoul"),
                id="openclaw_daily_restart",
            )
            # 5분마다 — Phase Watchdog (누락 페이즈 catch-up)
            await scheduler.add_schedule(
                self._phase_watchdog,
                IntervalTrigger(minutes=5),
                id="phase_watchdog",
            )

            await scheduler.start_in_background()
            logger.info("워크플로우 APScheduler 크론잡 8개 등록 완료")
        except ImportError:
            logger.warning(
                "apscheduler 미설치 — 워크플로우 자동 스케줄링 비활성. "
                "REST API 수동 트리거만 가능."
            )
        except Exception as e:
            logger.error("APScheduler 설정 실패: %s", e)

    async def start_command_consumer(self) -> None:
        """Redis Stream에서 워크플로우 명령을 소비하는 루프 (Phase 4)."""
        try:
            from app.core.redis import get_client
            import json as _json

            r = get_client()
            last_id = "0-0"
            logger.info("워크플로우 명령 소비자 시작 (Redis Stream)")

            while True:
                try:
                    results = await r.xread(
                        {"commands:workflow": last_id},
                        count=5,
                        block=5000,  # 5초 대기
                    )
                    if not results:
                        continue

                    for stream_name, messages in results:
                        for msg_id, fields in messages:
                            last_id = msg_id
                            action = fields.get("action", "")
                            payload = fields.get("payload", "{}")

                            try:
                                if action == "trigger_phase":
                                    phase = fields.get("phase", "")
                                    handlers = {
                                        "pre_market": self.handle_pre_market,
                                        "market_open": self.handle_market_open,
                                        "market_close": self.handle_market_close,
                                        "review": self.handle_review,
                                        "mining": self.handle_mining,
                                        "emergency_stop": self.handle_emergency_stop,
                                        "resume": self.handle_resume,
                                    }
                                    handler = handlers.get(phase)
                                    if handler:
                                        result = await handler()
                                        # 결과를 Redis에 기록 (MCP가 읽을 수 있도록)
                                        await r.set(
                                            f"commands:result:{msg_id}",
                                            _json.dumps(result, ensure_ascii=False, default=str),
                                        )
                                        await r.expire(f"commands:result:{msg_id}", 60)
                                        logger.info("명령 실행 완료: %s %s → %s", action, phase, result.get("success"))
                                else:
                                    logger.warning("알 수 없는 명령: %s", action)

                            except Exception as cmd_err:
                                logger.error("명령 실행 실패 (%s): %s", action, cmd_err)
                                await r.set(
                                    f"commands:result:{msg_id}",
                                    _json.dumps({"success": False, "error": str(cmd_err)[:200]}),
                                )
                                await r.expire(f"commands:result:{msg_id}", 60)

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.warning("명령 소비자 오류: %s", e)
                    await asyncio.sleep(5)

        except ImportError:
            logger.info("Redis 미설치 — 명령 소비자 비활성")
        except Exception as e:
            logger.error("명령 소비자 시작 실패: %s", e)

    async def shutdown_scheduler(self) -> None:
        """APScheduler 종료."""
        if self._scheduler:
            try:
                await self._scheduler.__aexit__(None, None, None)
            except Exception as e:
                logger.warning("APScheduler 종료 실패: %s", e)
            self._scheduler = None


# 싱글톤
_orchestrator: DailyWorkflowOrchestrator | None = None


def get_orchestrator() -> DailyWorkflowOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = DailyWorkflowOrchestrator()
    return _orchestrator
