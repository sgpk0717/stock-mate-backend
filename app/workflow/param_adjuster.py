"""ParameterAdjuster — 보수적·점진적 매매 파라미터 조정.

핵심 원칙:
- 변경은 선택적이다. 기본 결과는 "변경 없음".
- 변경 시 반드시 바운드 내, 사이클당 최대 변경폭 이내.
- max_drawdown_pct는 절대 증가시키지 않는다 (안전 우선).
- 모든 조정은 WorkflowEvent로 감사 로깅.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from app.core.config import settings
from app.workflow.feedback_types import ParameterAssessment
from app.workflow.models import WorkflowEvent, WorkflowRun

logger = logging.getLogger(__name__)


@dataclass
class ParamBounds:
    """파라미터 안전 범위."""

    min_val: float
    max_val: float
    max_delta: float  # 사이클당 최대 변경폭


# 하드 바운드 — 이 범위를 절대 벗어나지 않는다
PARAM_BOUNDS: dict[str, ParamBounds] = {
    "max_positions": ParamBounds(min_val=3, max_val=20, max_delta=1),
    "stop_loss_pct": ParamBounds(min_val=2.0, max_val=10.0, max_delta=0.5),
    "trailing_stop_pct": ParamBounds(min_val=1.0, max_val=8.0, max_delta=0.5),
    "max_drawdown_pct": ParamBounds(min_val=5.0, max_val=20.0, max_delta=1.0),
    "position_size_pct": ParamBounds(min_val=0.03, max_val=0.25, max_delta=0.02),
}


class ParameterAdjuster:
    """파라미터 평가 결과를 바운드 내로 클램프하여 저장."""

    async def apply_adjustments(
        self,
        session: AsyncSession,
        run: WorkflowRun,
        assessments: list[ParameterAssessment],
    ) -> dict:
        """추천값을 검증·클램프하여 WorkflowRun.config에 저장.

        Returns:
            {"applied": [...], "skipped": [...]}
        """
        min_confidence = settings.WORKFLOW_PARAM_EVAL_MIN_CONFIDENCE
        applied: list[dict] = []
        skipped: list[dict] = []

        for assessment in assessments:
            name = assessment.param_name

            # 추천 없으면 스킵
            if assessment.recommended_value is None:
                skipped.append({
                    "param": name,
                    "reason": "no_recommendation",
                    "assessment": assessment.assessment,
                })
                continue

            # 신뢰도 부족하면 스킵
            if assessment.confidence < min_confidence:
                skipped.append({
                    "param": name,
                    "reason": f"low_confidence ({assessment.confidence:.2f} < {min_confidence})",
                    "assessment": assessment.assessment,
                })
                continue

            # 바운드 확인
            bounds = PARAM_BOUNDS.get(name)
            if not bounds:
                skipped.append({"param": name, "reason": "unknown_param"})
                continue

            # max_drawdown_pct 안전 규칙: 절대 증가시키지 않음
            if name == "max_drawdown_pct" and assessment.recommended_value > assessment.current_value:
                skipped.append({
                    "param": name,
                    "reason": "safety_rule: max_drawdown never increases",
                    "assessment": assessment.assessment,
                })
                continue

            # 변경폭 클램프
            delta = assessment.recommended_value - assessment.current_value
            clamped_delta = max(-bounds.max_delta, min(bounds.max_delta, delta))
            new_value = assessment.current_value + clamped_delta

            # 바운드 클램프
            new_value = max(bounds.min_val, min(bounds.max_val, new_value))

            # max_positions는 정수
            if name == "max_positions":
                new_value = float(round(new_value))

            # 변경이 없으면 스킵 (float 비교)
            if abs(new_value - assessment.current_value) < 1e-6:
                skipped.append({
                    "param": name,
                    "reason": "no_effective_change",
                    "assessment": assessment.assessment,
                })
                continue

            applied.append({
                "param": name,
                "old_value": assessment.current_value,
                "new_value": round(new_value, 4),
                "delta": round(new_value - assessment.current_value, 4),
                "assessment": assessment.assessment,
                "evidence": assessment.evidence,
                "confidence": round(assessment.confidence, 3),
            })

        # WorkflowRun.config에 저장
        if applied:
            config = run.config or {}
            param_adj = {}
            for a in applied:
                param_adj[a["param"]] = a["new_value"]
            config["param_adjustments"] = param_adj
            run.config = config
            flag_modified(run, "config")  # SQLAlchemy JSON 필드 변경 감지

            # 감사 로그
            event = WorkflowEvent(
                id=uuid.uuid4(),
                workflow_run_id=run.id,
                phase=run.phase,
                event_type="param_adjustment",
                message=f"파라미터 조정 {len(applied)}건 적용",
                data={"applied": applied, "skipped": skipped},
            )
            session.add(event)

            logger.info(
                "ParameterAdjuster: %d건 조정 적용, %d건 스킵",
                len(applied), len(skipped),
            )
            for a in applied:
                logger.info(
                    "  %s: %.4f → %.4f (Δ%+.4f, %s, conf=%.2f)",
                    a["param"], a["old_value"], a["new_value"],
                    a["delta"], a["assessment"], a["confidence"],
                )
        else:
            logger.info("ParameterAdjuster: 조정 불필요 — 전 항목 adequate 또는 신뢰도 부족")

        return {"applied": applied, "skipped": skipped}
