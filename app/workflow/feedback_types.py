"""통합 피드백 데이터 구조 — 마이닝 방향성 + 매매 파라미터 평가.

마이닝 원칙: 직교성/무작위성 보존 + 부드러운 방향 힌트 (30% 방향성, 70% 자유 탐색).
파라미터 원칙: 보수적·점진적·안전하게. 개선은 선택적이며, 반드시 필요하지 않다.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ParameterAssessment:
    """단일 매매 파라미터 평가 결과."""

    param_name: str  # "stop_loss_pct", "max_positions", etc.
    current_value: float
    assessment: str  # "adequate" | "too_tight" | "too_loose" | "too_high" | "too_low"
    evidence: str  # 근거 (예: "8건 중 5건이 손절 직전 반등")
    recommended_value: float | None = None  # None = 변경 불필요 (기본값)
    confidence: float = 0.0  # 0.0~1.0

    def to_dict(self) -> dict:
        return {
            "param_name": self.param_name,
            "current_value": self.current_value,
            "assessment": self.assessment,
            "evidence": self.evidence,
            "recommended_value": self.recommended_value,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class TradingPatternInsight:
    """관찰된 매매 패턴 — 마이닝 방향 힌트 소스."""

    category: str  # "time_pattern" | "risk_pattern" | "signal_quality" | "market_regime"
    observation: str  # 관찰 내용
    direction_hint: str  # 마이닝 힌트 (부드러운 제안)
    strength: float = 0.0  # 0.0~1.0

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "observation": self.observation,
            "direction_hint": self.direction_hint,
            "strength": round(self.strength, 3),
        }


@dataclass
class UnifiedFeedback:
    """통합 피드백 — 파라미터 평가 + 마이닝 소프트 힌트."""

    # 파라미터 평가
    param_assessments: list[ParameterAssessment] = field(default_factory=list)
    overall_param_verdict: str = "no_change"  # "no_change" | "minor_adjustment"

    # 마이닝 소프트 힌트
    pattern_insights: list[TradingPatternInsight] = field(default_factory=list)
    exploration_hints: list[str] = field(default_factory=list)  # 최대 3개
    avoid_patterns: list[str] = field(default_factory=list)

    # 메타
    analysis_window_days: int = 7
    trade_count_in_window: int = 0
    confidence_level: str = "low"  # "low" (<20 trades) | "medium" | "high"

    def to_dict(self) -> dict:
        return {
            "param_assessments": [a.to_dict() for a in self.param_assessments],
            "overall_param_verdict": self.overall_param_verdict,
            "pattern_insights": [p.to_dict() for p in self.pattern_insights],
            "exploration_hints": self.exploration_hints,
            "avoid_patterns": self.avoid_patterns,
            "analysis_window_days": self.analysis_window_days,
            "trade_count_in_window": self.trade_count_in_window,
            "confidence_level": self.confidence_level,
        }
