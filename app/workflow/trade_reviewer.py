"""TradeReviewer — 규칙 기반 매매 리뷰 생성 (Claude API 미사용).

설계서 §12: 매매 실적 → 성과 분석 → 개선 방향 도출.
분봉 단타 특화: 건별 분석 + 시간대별 집계 (09-10, 10-12, 12-15:30).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, time

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.workflow.models import LiveFeedback, LiveTrade, WorkflowRun

logger = logging.getLogger(__name__)


@dataclass
class TradeReviewResult:
    """리뷰 결과."""

    trade_count: int = 0
    win_count: int = 0
    loss_count: int = 0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    win_rate: float = 0.0
    avg_holding_minutes: float = 0.0
    max_single_loss_pct: float = 0.0
    max_single_gain_pct: float = 0.0

    # 시간대별 성과
    morning_pnl_pct: float = 0.0  # 09:00-10:00
    midday_pnl_pct: float = 0.0  # 10:00-12:00
    afternoon_pnl_pct: float = 0.0  # 12:00-15:30

    # 개선 방향 (규칙 기반)
    improvements: list[str] = field(default_factory=list)

    # 파라미터 분석 메트릭 (통합 피드백 시스템)
    stop_loss_trigger_rate: float = 0.0  # SELL 중 S-STOP 비율
    trailing_trigger_rate: float = 0.0  # SELL 중 S-TRAIL 비율

    def to_dict(self) -> dict:
        return {
            "trade_count": self.trade_count,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "total_pnl": round(self.total_pnl, 2),
            "total_pnl_pct": round(self.total_pnl_pct, 4),
            "win_rate": round(self.win_rate, 2),
            "avg_holding_minutes": round(self.avg_holding_minutes, 1),
            "max_single_loss_pct": round(self.max_single_loss_pct, 4),
            "max_single_gain_pct": round(self.max_single_gain_pct, 4),
            "time_breakdown": {
                "morning": round(self.morning_pnl_pct, 4),
                "midday": round(self.midday_pnl_pct, 4),
                "afternoon": round(self.afternoon_pnl_pct, 4),
            },
            "improvements": self.improvements,
            "stop_loss_trigger_rate": round(self.stop_loss_trigger_rate, 4),
            "trailing_trigger_rate": round(self.trailing_trigger_rate, 4),
        }


class TradeReviewer:
    """규칙 기반 매매 리뷰어.

    LiveSession의 trade_log를 분석하여 성과 지표 + 개선 방향을 도출한다.
    """

    async def generate_review(
        self, session: AsyncSession, run: WorkflowRun
    ) -> TradeReviewResult:
        """WorkflowRun에 연결된 세션의 매매 로그를 분석."""
        result = TradeReviewResult()

        # 1차: DB에서 당일 매매 조회 (C3 영속화된 데이터)
        trade_logs: list[dict] = []
        try:
            stmt = select(LiveTrade).where(
                func.date(LiveTrade.executed_at) == run.date,
            )
            if run.trading_context_id:
                stmt = stmt.where(LiveTrade.context_id == run.trading_context_id)
            db_result = await session.execute(stmt)
            db_trades = db_result.scalars().all()
            if db_trades:
                trade_logs = [self._trade_model_to_dict(t) for t in db_trades]
        except Exception as e:
            logger.warning("DB trade_log 조회 실패: %s", e)

        # 2차 fallback: 메모리 (하위 호환)
        if not trade_logs:
            try:
                from app.trading.live_runner import list_sessions as list_live
                for live_session in list_live():
                    if hasattr(live_session, "trade_log"):
                        trade_logs.extend(live_session.trade_log)
            except Exception as e:
                logger.warning("메모리 trade_log 수집 실패: %s", e)

        if not trade_logs:
            result.improvements.append("매매 기록 없음 — 팩터 시그널 강도 확인 필요")
            return result

        # 성과 계산 (SELL 거래만 PnL 보유)
        sell_logs = [t for t in trade_logs if t.get("side") == "SELL"]
        result.trade_count = len(trade_logs)
        wins = [t for t in sell_logs if (t.get("pnl_pct") or 0) > 0]
        losses = [t for t in sell_logs if (t.get("pnl_pct") or 0) < 0]

        result.win_count = len(wins)
        result.loss_count = len(losses)
        result.win_rate = (len(wins) / len(sell_logs) * 100) if sell_logs else 0

        pnls = [t.get("pnl_pct") or 0 for t in sell_logs]
        result.total_pnl_pct = sum(pnls)
        result.max_single_loss_pct = min(pnls) if pnls else 0
        result.max_single_gain_pct = max(pnls) if pnls else 0

        # 총 PnL 금액
        pnl_amounts = [t.get("pnl_amount") or 0 for t in sell_logs]
        result.total_pnl = sum(pnl_amounts)

        # 보유 시간 계산
        holdings = [
            t.get("holding_minutes") or 0
            for t in trade_logs
            if t.get("holding_minutes") is not None
        ]
        result.avg_holding_minutes = (sum(holdings) / len(holdings)) if holdings else 0

        # 시간대별 집계
        result.morning_pnl_pct = self._time_range_pnl(
            trade_logs, time(9, 0), time(10, 0)
        )
        result.midday_pnl_pct = self._time_range_pnl(
            trade_logs, time(10, 0), time(12, 0)
        )
        result.afternoon_pnl_pct = self._time_range_pnl(
            trade_logs, time(12, 0), time(15, 30)
        )

        # 파라미터 분석 메트릭
        if sell_logs:
            stop_exits = [t for t in sell_logs if t.get("step") == "S-STOP"]
            trail_exits = [t for t in sell_logs if t.get("step") == "S-TRAIL"]
            result.stop_loss_trigger_rate = len(stop_exits) / len(sell_logs)
            result.trailing_trigger_rate = len(trail_exits) / len(sell_logs)

        # 규칙 기반 개선 방향 도출 (설계서 §12.2)
        result.improvements = self._derive_improvements(result)

        # live_feedback 테이블에 기록
        if run.selected_factor_id:
            await self._save_feedback(session, run, result)

        return result

    def _time_range_pnl(
        self, logs: list[dict], start: time, end: time
    ) -> float:
        """특정 시간대의 PnL 합계."""
        total = 0.0
        for t in logs:
            ts = t.get("timestamp")
            if ts and isinstance(ts, str):
                try:
                    dt = datetime.fromisoformat(ts)
                    if start <= dt.time() < end:
                        total += t.get("pnl_pct") or 0
                except (ValueError, TypeError):
                    pass
        return total

    def _derive_improvements(self, result: TradeReviewResult) -> list[str]:
        """규칙 기반 개선 방향 도출 (설계서 §12.2)."""
        improvements: list[str] = []

        if result.trade_count == 0:
            return improvements

        # 승률 < 40%
        if result.win_rate < 40:
            improvements.append(
                "진입 조건이 너무 느슨함. IC > 0.05 이상 팩터 탐색 권장"
            )

        # 평균 보유 < 3분 → 틱 노이즈
        if 0 < result.avg_holding_minutes < 3:
            improvements.append(
                "틱 노이즈에 민감. 5m봉 기반 팩터 탐색 권장"
            )

        # 최대 단일 손실 > 5%
        if abs(result.max_single_loss_pct) > 5:
            improvements.append(
                f"단일 손실 {result.max_single_loss_pct:.1f}%. ATR 필터 추가 검토"
            )

        # 장 초반 손실 > 전체 손실의 60%
        if result.total_pnl_pct < 0 and result.morning_pnl_pct < 0:
            total_loss = abs(result.total_pnl_pct)
            if total_loss > 0:
                morning_ratio = abs(result.morning_pnl_pct) / total_loss
                if morning_ratio > 0.6:
                    improvements.append(
                        "09:00-10:00 장 초반 손실 집중. 시간대 가중 팩터 또는 09:30 이후 매매 개시 검토"
                    )

        # 오후 수익 > 오전 손실
        if result.afternoon_pnl_pct > 0 and result.morning_pnl_pct < 0:
            improvements.append(
                "오후 시간대 강세 패턴. 오전 진입 기준 강화 + 오후 비중 확대 검토"
            )

        # 과매매 (100건 이상/일)
        if result.trade_count > 100:
            improvements.append(
                f"일 {result.trade_count}건 과매매. 시그널 임계값 상향 (0.7→0.8) 검토"
            )

        # 과소매매 (5건 미만/일)
        if 0 < result.trade_count < 5:
            improvements.append(
                f"일 {result.trade_count}건 과소매매. 시그널 임계값 하향 (0.7→0.6) 검토"
            )

        return improvements

    @staticmethod
    def _trade_model_to_dict(t: LiveTrade) -> dict:
        """LiveTrade DB 모델 → trade_log dict 변환."""
        return {
            "symbol": t.symbol,
            "name": t.name,
            "side": t.side,
            "step": t.step,
            "qty": t.qty,
            "price": float(t.price),
            "pnl_pct": t.pnl_pct,
            "pnl_amount": float(t.pnl_amount) if t.pnl_amount is not None else None,
            "holding_minutes": t.holding_minutes,
            "success": t.success,
            "order_id": t.order_id,
            "reason": t.reason,
            "timestamp": t.executed_at.isoformat() if t.executed_at else "",
            "snapshot": t.snapshot,
            "conditions": t.conditions,
            "sizing": t.sizing,
            "position_context": t.position_context,
        }

    async def _save_feedback(
        self, session: AsyncSession, run: WorkflowRun, result: TradeReviewResult
    ) -> None:
        """live_feedback 테이블에 실적 기록."""
        feedback = LiveFeedback(
            id=uuid.uuid4(),
            factor_id=run.selected_factor_id,
            workflow_run_id=run.id,
            date=run.date,
            realized_pnl=result.total_pnl,
            realized_pnl_pct=result.total_pnl_pct,
            trade_count=result.trade_count,
            win_rate=result.win_rate,
            avg_holding_minutes=result.avg_holding_minutes,
            max_single_loss_pct=result.max_single_loss_pct,
            feedback_context="; ".join(result.improvements) if result.improvements else None,
        )
        session.add(feedback)
