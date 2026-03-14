"""FeedbackEngine — 실매매 피드백 → 마이닝 컨텍스트 자동 생성.

설계서 §12: live_feedback + review_summary → mining_context 문자열 생성.
이 문자열이 AlphaFactoryScheduler.start(context=)에 전달되어
Claude 가설 생성 프롬프트에 포함됨.

v2: 통합 피드백 시스템 (UnifiedFeedback) 지원.
- generate_structured_feedback(): 파라미터 평가 + 패턴 인사이트 + 마이닝 힌트
- format_mining_prompt(): 구조화된 마이닝 프롬프트 (직교성 보존 + 소프트 힌트)
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.workflow.feedback_types import (
    TradingPatternInsight,
    UnifiedFeedback,
)
from app.workflow.models import LiveFeedback, WorkflowRun

logger = logging.getLogger(__name__)


class FeedbackEngine:
    """규칙 기반 마이닝 컨텍스트 생성기."""

    async def generate_context(
        self, session: AsyncSession, run: WorkflowRun
    ) -> str:
        """오늘의 리뷰 결과 + 최근 피드백 히스토리 → 마이닝 컨텍스트."""
        parts: list[str] = []

        # 1. 오늘 리뷰 요약
        if run.review_summary and isinstance(run.review_summary, dict):
            summary = run.review_summary
            pnl = summary.get("total_pnl_pct", 0)
            trades = summary.get("trade_count", 0)
            win_rate = summary.get("win_rate", 0)
            parts.append(
                f"전일 매매 결과: {pnl:+.2f}% ({trades}건, 승률 {win_rate:.0f}%)"
            )

            # 시간대별 패턴
            time_data = summary.get("time_breakdown", {})
            if time_data:
                morning = time_data.get("morning", 0)
                afternoon = time_data.get("afternoon", 0)
                if morning < -1:
                    parts.append(
                        f"장 초반(09-10시) 약세 ({morning:+.2f}%). 시간대 필터링 검토."
                    )
                if afternoon > 1:
                    parts.append(
                        f"오후(12-15:30) 강세 ({afternoon:+.2f}%). 오후 가중 팩터 탐색."
                    )

            # 개선 방향
            improvements = summary.get("improvements", [])
            if improvements:
                parts.append("개선 방향: " + "; ".join(improvements[:3]))

        # 2. 최근 7일 피드백 히스토리 집계
        week_ago = date.today() - timedelta(days=7)
        stmt = (
            select(LiveFeedback)
            .where(LiveFeedback.date >= week_ago)
            .order_by(LiveFeedback.date.desc())
        )
        result = await session.execute(stmt)
        feedbacks = list(result.scalars().all())

        if feedbacks:
            avg_pnl = sum(f.realized_pnl_pct or 0 for f in feedbacks) / len(feedbacks)
            avg_win = sum(f.win_rate or 0 for f in feedbacks) / len(feedbacks)
            parts.append(
                f"최근 7일 평균: PnL {avg_pnl:+.2f}%, 승률 {avg_win:.0f}%"
            )

            # 반복 패턴 감지
            negative_days = sum(
                1 for f in feedbacks if (f.realized_pnl_pct or 0) < 0
            )
            if negative_days >= 4:
                parts.append(
                    "최근 7일 중 4일 이상 손실. 현재 팩터 방향 재검토 필요."
                )

            # 피드백 컨텍스트 병합
            contexts = [
                f.feedback_context for f in feedbacks if f.feedback_context
            ]
            if contexts:
                parts.append("최근 피드백: " + " | ".join(contexts[:3]))

        # 3. H4: 최근 인과 검증 실패 유형 집계
        try:
            from app.alpha.models import AlphaFactor
            from sqlalchemy import func as sa_func
            failure_stmt = (
                select(
                    AlphaFactor.causal_failure_type,
                    sa_func.count(AlphaFactor.id).label("cnt"),
                )
                .where(
                    AlphaFactor.causal_failure_type.isnot(None),
                    AlphaFactor.causal_failure_type != "PASSED",
                    AlphaFactor.updated_at >= week_ago,
                )
                .group_by(AlphaFactor.causal_failure_type)
            )
            failure_result = await session.execute(failure_stmt)
            failure_rows = failure_result.all()
            if failure_rows:
                failure_hints = {
                    "CONFOUNDED": "허위 상관 팩터 다수. 독립적 피처 조합 탐색 필요",
                    "FRAGILE": "교란 변수에 민감한 팩터 다수. 강건한 팩터 탐색 필요",
                    "REGIME_SHIFT": "체제 변화에 취약한 팩터 다수. 적응형 전략 검토",
                    "LOW_IC": "예측력 부족 팩터 다수. IC 기준 상향 검토",
                }
                for ftype, cnt in failure_rows:
                    hint = failure_hints.get(ftype, "")
                    if hint:
                        parts.append(f"인과검증 실패 [{ftype}] {cnt}건: {hint}")
        except Exception as e:
            logger.warning("인과 실패 집계 실패: %s", e)

        # 4. 사용된 팩터 정보
        if run.selected_factor_id:
            try:
                from app.alpha.models import AlphaFactor
                factor_stmt = select(AlphaFactor).where(
                    AlphaFactor.id == run.selected_factor_id
                )
                factor_result = await session.execute(factor_stmt)
                factor = factor_result.scalar_one_or_none()
                if factor:
                    ic_str = f"{factor.ic_mean:.4f}" if factor.ic_mean is not None else "N/A"
                    sharpe_str = f"{factor.sharpe:.2f}" if factor.sharpe is not None else "N/A"
                    parts.append(
                        f"사용 팩터: {factor.name} ({factor.expression_str[:80]}), "
                        f"IC={ic_str}, Sharpe={sharpe_str}"
                    )
            except Exception as e:
                logger.warning("팩터 정보 로드 실패: %s", e)

        # 기본 컨텍스트
        if not parts:
            parts.append("자동 워크플로우 야간 마이닝. 시장 일반 팩터 탐색.")

        return "\n".join(parts)

    # ─── v2: 통합 피드백 시스템 ───────────────────────────────

    async def generate_structured_feedback(
        self, session: AsyncSession, run: WorkflowRun
    ) -> UnifiedFeedback:
        """통합 피드백 생성 — 파라미터 평가 + 패턴 인사이트 + 마이닝 힌트."""
        from app.workflow.param_evaluator import ParameterEvaluator

        feedback = UnifiedFeedback()
        feedback.analysis_window_days = 7

        # 1. 파라미터 평가
        try:
            evaluator = ParameterEvaluator()
            feedback.param_assessments = await evaluator.evaluate(session, run)
            has_adjustment = any(
                a.recommended_value is not None and a.confidence >= 0.6
                for a in feedback.param_assessments
            )
            feedback.overall_param_verdict = (
                "minor_adjustment" if has_adjustment else "no_change"
            )
        except Exception as e:
            logger.warning("파라미터 평가 실패: %s", e)

        # 2. 패턴 인사이트 + 마이닝 힌트 도출
        week_ago = date.today() - timedelta(days=7)
        stmt = (
            select(LiveFeedback)
            .where(LiveFeedback.date >= week_ago)
            .order_by(LiveFeedback.date.desc())
        )
        result = await session.execute(stmt)
        feedbacks = list(result.scalars().all())

        trade_count = sum(f.trade_count or 0 for f in feedbacks)
        feedback.trade_count_in_window = trade_count

        if trade_count < 20:
            feedback.confidence_level = "low"
        elif trade_count < 100:
            feedback.confidence_level = "medium"
        else:
            feedback.confidence_level = "high"

        # 리뷰 요약에서 패턴 추출
        if run.review_summary and isinstance(run.review_summary, dict):
            self._extract_pattern_insights(run.review_summary, feedback)

        # 최근 피드백에서 반복 패턴 추출
        self._extract_history_insights(feedbacks, feedback)

        # 인과 검증 실패 패턴 → 회피 패턴
        await self._extract_causal_avoidance(session, week_ago, feedback)

        # 탐색 힌트 최대 3개 제한
        feedback.exploration_hints = feedback.exploration_hints[:3]

        return feedback

    def _extract_pattern_insights(
        self, summary: dict, feedback: UnifiedFeedback
    ) -> None:
        """리뷰 요약에서 패턴 인사이트와 마이닝 힌트를 추출."""
        time_data = summary.get("time_breakdown", {})
        morning = time_data.get("morning", 0)
        afternoon = time_data.get("afternoon", 0)

        # 시간대 패턴
        if morning < -1:
            feedback.pattern_insights.append(TradingPatternInsight(
                category="time_pattern",
                observation=f"장 초반(09-10시) 손실 집중 ({morning:+.2f}%)",
                direction_hint="변동성 안정화 후 진입하는 팩터 또는 시간대 가중 요소 탐색 가능",
                strength=min(1.0, abs(morning) / 3),
            ))
        if afternoon > 1:
            feedback.pattern_insights.append(TradingPatternInsight(
                category="time_pattern",
                observation=f"오후(12-15:30) 강세 패턴 ({afternoon:+.2f}%)",
                direction_hint="오후 모멘텀 포착 팩터 또는 장 후반 리버설 팩터 탐색 가능",
                strength=min(1.0, afternoon / 3),
            ))

        # 승률 패턴
        win_rate = summary.get("win_rate", 0)
        if win_rate < 40:
            feedback.pattern_insights.append(TradingPatternInsight(
                category="signal_quality",
                observation=f"승률 {win_rate:.0f}% — 진입 시그널 정확도 부족",
                direction_hint="노이즈 필터링이 강한 팩터 (ATR 정규화, 거래량 확인 등) 탐색 가능",
                strength=min(1.0, (50 - win_rate) / 30),
            ))

        # 보유 시간 패턴
        avg_hold = summary.get("avg_holding_minutes", 0)
        if 0 < avg_hold < 3:
            feedback.pattern_insights.append(TradingPatternInsight(
                category="signal_quality",
                observation=f"평균 보유 {avg_hold:.1f}분 — 틱 노이즈에 민감",
                direction_hint="더 느린 주기(5m+)에 적합한 평균 회귀 또는 추세 팩터 탐색 가능",
                strength=0.7,
            ))

        # 개선 방향 → 탐색 힌트 (최대 2개, 부드럽게 변환)
        improvements = summary.get("improvements", [])
        for imp in improvements[:2]:
            # "IC > 0.05 이상 팩터 탐색 권장" 같은 지시형을 부드럽게 변환
            feedback.exploration_hints.append(
                imp.replace("권장", "가능").replace("검토", "고려 가능")
            )

    def _extract_history_insights(
        self, feedbacks: list[LiveFeedback], feedback: UnifiedFeedback
    ) -> None:
        """최근 피드백 히스토리에서 반복 패턴 추출."""
        if not feedbacks:
            return

        negative_days = sum(
            1 for f in feedbacks if (f.realized_pnl_pct or 0) < 0
        )
        total_days = len(feedbacks)

        if negative_days >= 4 and total_days >= 5:
            feedback.pattern_insights.append(TradingPatternInsight(
                category="market_regime",
                observation=f"최근 {total_days}일 중 {negative_days}일 손실",
                direction_hint="현재 시장 체제에서 방어적 팩터(저변동성, 가치) 탐색 가능",
                strength=min(1.0, negative_days / total_days),
            ))

        # 연속 양수 → 모멘텀 체제
        positive_streak = 0
        for f in feedbacks:
            if (f.realized_pnl_pct or 0) > 0:
                positive_streak += 1
            else:
                break
        if positive_streak >= 4:
            feedback.pattern_insights.append(TradingPatternInsight(
                category="market_regime",
                observation=f"연속 {positive_streak}일 수익 — 강세 체제 가능",
                direction_hint="모멘텀 팩터 또는 추세 추종 팩터가 유리할 수 있음",
                strength=min(1.0, positive_streak / 6),
            ))

    async def _extract_causal_avoidance(
        self, session: AsyncSession, since: date, feedback: UnifiedFeedback
    ) -> None:
        """인과 검증 실패 패턴 → 회피 패턴."""
        try:
            from app.alpha.models import AlphaFactor
            from sqlalchemy import func as sa_func

            failure_stmt = (
                select(
                    AlphaFactor.causal_failure_type,
                    sa_func.count(AlphaFactor.id).label("cnt"),
                )
                .where(
                    AlphaFactor.causal_failure_type.isnot(None),
                    AlphaFactor.causal_failure_type != "PASSED",
                    AlphaFactor.updated_at >= since,
                )
                .group_by(AlphaFactor.causal_failure_type)
            )
            failure_result = await session.execute(failure_stmt)
            failure_rows = failure_result.all()

            avoidance_map = {
                "CONFOUNDED": "단순 상관 기반 팩터 (인과 검증 실패 — 허위 상관)",
                "FRAGILE": "교란 변수에 민감한 팩터 구조",
                "LOW_IC": "예측력 부족 팩터 (IC < 0.02 반복)",
            }
            for ftype, cnt in failure_rows:
                if cnt >= 3 and ftype in avoidance_map:
                    feedback.avoid_patterns.append(
                        f"{avoidance_map[ftype]} ({ftype} {cnt}건)"
                    )
        except Exception as e:
            logger.warning("인과 실패 회피 패턴 추출 실패: %s", e)

    def format_mining_prompt(self, feedback: UnifiedFeedback) -> str:
        """UnifiedFeedback → 마이닝 프롬프트용 구조화된 문자열.

        핵심: 직교성/무작위성 보존. 힌트는 선택적이며 탐색 자유를 보장.
        """
        lines: list[str] = []
        lines.append("=== 실매매 피드백 (참고용 — 탐색 방향을 제한하지 않음) ===")
        lines.append("")

        # 관찰된 패턴
        if feedback.pattern_insights:
            lines.append("[관찰된 패턴]")
            for insight in feedback.pattern_insights[:5]:
                lines.append(f"- {insight.observation}")
            lines.append("")

        # 탐색 방향 힌트 (선택적)
        if feedback.exploration_hints:
            lines.append("[탐색 방향 힌트 (선택적 — 무시해도 됨)]")
            for hint in feedback.exploration_hints[:3]:
                lines.append(f"- {hint}")
            lines.append("")

        # 회피 권장 패턴
        if feedback.avoid_patterns:
            lines.append("[회피 권장 패턴]")
            for pattern in feedback.avoid_patterns[:3]:
                lines.append(f"- {pattern}")
            lines.append("")

        # 탐색 자유 보장 문구
        lines.append(
            "중요: 위 내용은 참고 사항입니다. 새로운, 직교적인 팩터 발견이 최우선 목표입니다."
        )
        lines.append("자유로운 가설 탐색을 유지하세요.")
        lines.append("===")

        return "\n".join(lines)
