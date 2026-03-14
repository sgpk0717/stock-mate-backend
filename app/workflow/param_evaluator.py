"""ParameterEvaluator — 규칙 기반 매매 파라미터 적정성 평가.

기본 결과는 "adequate" (변경 불필요).
충분한 데이터(≥MIN_TRADES) + confidence ≥ 0.6일 때만 조정 추천.
Claude API 미사용 — 순수 규칙 기반.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.workflow.feedback_types import ParameterAssessment
from app.workflow.models import LiveFeedback, LiveTrade, WorkflowRun

logger = logging.getLogger(__name__)


class ParameterEvaluator:
    """매매 파라미터 적정성을 규칙 기반으로 평가한다."""

    def __init__(self) -> None:
        self.lookback_days = settings.WORKFLOW_PARAM_EVAL_LOOKBACK_DAYS
        self.min_trades = settings.WORKFLOW_PARAM_EVAL_MIN_TRADES

    async def evaluate(
        self, session: AsyncSession, run: WorkflowRun
    ) -> list[ParameterAssessment]:
        """최근 N일 매매 데이터를 분석하여 파라미터 평가 목록을 반환."""
        assessments: list[ParameterAssessment] = []

        # 최근 N일 매매 로드
        since = date.today() - timedelta(days=self.lookback_days)
        stmt = select(LiveTrade).where(
            func.date(LiveTrade.executed_at) >= since,
        )
        result = await session.execute(stmt)
        trades = list(result.scalars().all())

        sell_trades = [t for t in trades if t.side == "SELL"]
        total_sells = len(sell_trades)

        # 데이터 부족 → 전부 "adequate"
        if total_sells < self.min_trades:
            return self._all_adequate(total_sells)

        # 현재 파라미터 값 로드
        current_stop = settings.WORKFLOW_STOP_LOSS_PCT
        current_trail = 3.0  # build_context_from_factor 하드코딩 기본값
        current_max_pos = settings.WORKFLOW_MAX_POSITIONS
        current_max_dd = settings.WORKFLOW_MAX_DRAWDOWN_PCT
        current_pos_size = 1.0 / current_max_pos

        # 전일 조정이 있었으면 해당 값 사용
        if run.config and run.config.get("param_adjustments"):
            adj = run.config["param_adjustments"]
            current_stop = adj.get("stop_loss_pct", current_stop)
            current_trail = adj.get("trailing_stop_pct", current_trail)
            current_max_pos = adj.get("max_positions", current_max_pos)
            current_max_dd = adj.get("max_drawdown_pct", current_max_dd)
            current_pos_size = adj.get("position_size_pct", current_pos_size)

        # 1. 손절 효과성
        assessments.append(
            self._eval_stop_loss(sell_trades, total_sells, current_stop)
        )

        # 2. 트레일링 스탑 효과성
        assessments.append(
            self._eval_trailing_stop(sell_trades, total_sells, current_trail)
        )

        # 3. 포지션 집중도 (max_positions)
        assessments.append(
            await self._eval_position_concentration(
                session, since, current_max_pos
            )
        )

        # 4. 포지션 사이즈 효율
        assessments.append(
            self._eval_position_size(trades, current_pos_size)
        )

        # 5. MDD 헤드룸
        assessments.append(
            await self._eval_mdd_headroom(session, since, current_max_dd)
        )

        return assessments

    def _eval_stop_loss(
        self,
        sell_trades: list[LiveTrade],
        total_sells: int,
        current: float,
    ) -> ParameterAssessment:
        """손절 비율 평가."""
        stop_exits = [t for t in sell_trades if t.step == "S-STOP"]
        stop_rate = len(stop_exits) / total_sells if total_sells else 0

        # 최대 단일 손실
        losses = [abs(t.pnl_pct or 0) for t in sell_trades if (t.pnl_pct or 0) < 0]
        max_loss = max(losses) if losses else 0

        if stop_rate > 0.6:
            return ParameterAssessment(
                param_name="stop_loss_pct",
                current_value=current,
                assessment="too_tight",
                evidence=f"SELL {total_sells}건 중 S-STOP {len(stop_exits)}건 ({stop_rate:.0%}). 손절 과다 발동.",
                recommended_value=current + 0.5,
                confidence=min(1.0, stop_rate),
            )
        elif stop_rate < 0.05 and max_loss > current * 2:
            return ParameterAssessment(
                param_name="stop_loss_pct",
                current_value=current,
                assessment="too_loose",
                evidence=f"S-STOP 거의 미발동({stop_rate:.0%}), 최대 단일 손실 {max_loss:.1f}% (손절 {current}%의 {max_loss / current:.1f}배).",
                recommended_value=current - 0.5,
                confidence=0.7,
            )
        else:
            return ParameterAssessment(
                param_name="stop_loss_pct",
                current_value=current,
                assessment="adequate",
                evidence=f"S-STOP 비율 {stop_rate:.0%}, 적정 범위.",
                confidence=0.5,
            )

    def _eval_trailing_stop(
        self,
        sell_trades: list[LiveTrade],
        total_sells: int,
        current: float,
    ) -> ParameterAssessment:
        """트레일링 스탑 평가."""
        trail_exits = [t for t in sell_trades if t.step == "S-TRAIL"]
        trail_rate = len(trail_exits) / total_sells if total_sells else 0

        # 트레일링 발동 건의 평균 수익
        trail_pnls = [t.pnl_pct or 0 for t in trail_exits]
        avg_trail_pnl = sum(trail_pnls) / len(trail_pnls) if trail_pnls else 0

        if trail_exits and avg_trail_pnl < 1.0:
            return ParameterAssessment(
                param_name="trailing_stop_pct",
                current_value=current,
                assessment="too_tight",
                evidence=f"S-TRAIL {len(trail_exits)}건 평균 수익 {avg_trail_pnl:.2f}%. 수익을 너무 일찍 차단.",
                recommended_value=current + 0.5,
                confidence=min(1.0, trail_rate + 0.3),
            )
        elif trail_rate < 0.05 and total_sells > 20:
            # 트레일링이 거의 발동 안 됨 → 현재 값이 너무 느슨할 수 있음 (정보 부족, adequate)
            return ParameterAssessment(
                param_name="trailing_stop_pct",
                current_value=current,
                assessment="adequate",
                evidence=f"S-TRAIL 거의 미발동 ({trail_rate:.0%}). 데이터 부족으로 판단 보류.",
                confidence=0.3,
            )
        else:
            return ParameterAssessment(
                param_name="trailing_stop_pct",
                current_value=current,
                assessment="adequate",
                evidence=f"S-TRAIL 비율 {trail_rate:.0%}, 적정 범위.",
                confidence=0.5,
            )

    async def _eval_position_concentration(
        self,
        session: AsyncSession,
        since: date,
        current_max: int,
    ) -> ParameterAssessment:
        """포지션 집중도 평가 — decision_log의 SKIP_MAX_POS 빈도 기반."""
        # decision_log는 메모리/세션 기반이라 DB에서 직접 추정
        # BUY 거래 수 vs 기간으로 평균 동시 보유 추정
        buy_stmt = select(func.count(LiveTrade.id)).where(
            func.date(LiveTrade.executed_at) >= since,
            LiveTrade.side == "BUY",
        )
        sell_stmt = select(func.count(LiveTrade.id)).where(
            func.date(LiveTrade.executed_at) >= since,
            LiveTrade.side == "SELL",
        )
        buy_count = (await session.execute(buy_stmt)).scalar() or 0
        sell_count = (await session.execute(sell_stmt)).scalar() or 0

        # 평균 보유 = 누적 매수 - 누적 매도 (단순 추정)
        avg_held = max(0, buy_count - sell_count)

        if avg_held < current_max * 0.3 and buy_count > 10:
            return ParameterAssessment(
                param_name="max_positions",
                current_value=float(current_max),
                assessment="too_high",
                evidence=f"평균 보유 ~{avg_held}개 (최대 {current_max}개의 {avg_held / current_max:.0%}). 슬롯 미활용.",
                recommended_value=float(max(current_max - 1, 3)),
                confidence=0.6,
            )
        else:
            return ParameterAssessment(
                param_name="max_positions",
                current_value=float(current_max),
                assessment="adequate",
                evidence=f"포지션 활용률 적정.",
                confidence=0.5,
            )

    def _eval_position_size(
        self,
        trades: list[LiveTrade],
        current: float,
    ) -> ParameterAssessment:
        """포지션 사이즈 효율 평가."""
        # sizing 정보가 있는 매수건에서 현금 부족 여부 추정
        buy_trades = [t for t in trades if t.side == "BUY" and t.sizing]
        if not buy_trades:
            return ParameterAssessment(
                param_name="position_size_pct",
                current_value=current,
                assessment="adequate",
                evidence="매수 sizing 데이터 부족.",
                confidence=0.2,
            )

        # sizing에 cash_limited 플래그가 있는 경우 카운트
        cash_limited = sum(
            1 for t in buy_trades if (t.sizing or {}).get("cash_limited", False)
        )
        limited_rate = cash_limited / len(buy_trades) if buy_trades else 0

        if limited_rate > 0.3:
            return ParameterAssessment(
                param_name="position_size_pct",
                current_value=current,
                assessment="too_high",
                evidence=f"매수 {len(buy_trades)}건 중 {cash_limited}건({limited_rate:.0%}) 현금 부족 축소.",
                recommended_value=max(current - 0.02, 0.03),
                confidence=min(1.0, limited_rate),
            )
        else:
            return ParameterAssessment(
                param_name="position_size_pct",
                current_value=current,
                assessment="adequate",
                evidence=f"포지션 사이즈 적정.",
                confidence=0.5,
            )

    async def _eval_mdd_headroom(
        self,
        session: AsyncSession,
        since: date,
        current_max_dd: float,
    ) -> ParameterAssessment:
        """MDD 여유분 평가 — 안전 우선, 증가 추천 안 함."""
        # 최근 피드백에서 realized_pnl_pct 기반 추정
        stmt = (
            select(LiveFeedback.realized_pnl_pct)
            .where(LiveFeedback.date >= since)
            .order_by(LiveFeedback.date)
        )
        result = await session.execute(stmt)
        daily_pnls = [r[0] or 0 for r in result.all()]

        if not daily_pnls:
            return ParameterAssessment(
                param_name="max_drawdown_pct",
                current_value=current_max_dd,
                assessment="adequate",
                evidence="일일 PnL 데이터 부족.",
                confidence=0.2,
            )

        # 누적 수익 곡선에서 MDD 계산
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for pnl in daily_pnls:
            cumulative += pnl
            peak = max(peak, cumulative)
            dd = peak - cumulative
            max_dd = max(max_dd, dd)

        dd_ratio = max_dd / current_max_dd if current_max_dd > 0 else 0

        if dd_ratio > 0.8:
            return ParameterAssessment(
                param_name="max_drawdown_pct",
                current_value=current_max_dd,
                assessment="too_tight",
                evidence=f"실현 DD {max_dd:.1f}% (한도 {current_max_dd}%의 {dd_ratio:.0%}). 서킷브레이커 근접.",
                # 안전 우선: MDD 한도를 증가시키지 않음
                recommended_value=None,
                confidence=0.8,
            )
        else:
            return ParameterAssessment(
                param_name="max_drawdown_pct",
                current_value=current_max_dd,
                assessment="adequate",
                evidence=f"실현 DD {max_dd:.1f}% (한도의 {dd_ratio:.0%}). 여유 충분.",
                confidence=0.5,
            )

    def _all_adequate(self, trade_count: int) -> list[ParameterAssessment]:
        """데이터 부족 시 전부 'adequate' 반환."""
        reason = f"최근 매도 {trade_count}건 (최소 {self.min_trades}건 필요). 데이터 부족."
        params = [
            ("stop_loss_pct", settings.WORKFLOW_STOP_LOSS_PCT),
            ("trailing_stop_pct", 3.0),
            ("max_positions", float(settings.WORKFLOW_MAX_POSITIONS)),
            ("position_size_pct", 1.0 / settings.WORKFLOW_MAX_POSITIONS),
            ("max_drawdown_pct", settings.WORKFLOW_MAX_DRAWDOWN_PCT),
        ]
        return [
            ParameterAssessment(
                param_name=name,
                current_value=val,
                assessment="adequate",
                evidence=reason,
                confidence=0.1,
            )
            for name, val in params
        ]
