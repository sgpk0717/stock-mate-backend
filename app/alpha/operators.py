"""UCB1 기반 적응형 연산자 레지스트리.

AST 연산자(저비용 ~92%)와 LLM 연산자(고비용 ~8%)를 관리하며,
UCB1 알고리즘으로 적합도 개선이 큰 연산자를 우선 선택한다.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class OperatorStats:
    """연산자별 통계."""

    name: str
    calls: int = 0
    fitness_improvements: float = 0.0
    is_llm: bool = False


# 등록할 연산자 이름 목록
_AST_OPERATORS = [
    "ast_mutate_operator",
    "ast_mutate_constant",
    "ast_mutate_feature",
    "ast_mutate_function",
    "ast_hoist",
    "ast_ephemeral_constant",
    "ast_crossover",
]

_LLM_OPERATORS = [
    "llm_seed",
    "llm_mutate",
]


class OperatorRegistry:
    """UCB1 기반 적응형 연산자 선택.

    Parameters
    ----------
    llm_ratio : LLM 연산자 최대 비율 (0.0~1.0). 0이면 LLM 비활성화.
    exploration_c : UCB1 탐색 계수. 높을수록 미시도 연산자 선호.
    """

    def __init__(
        self,
        llm_ratio: float = 0.08,
        exploration_c: float = 1.41,
    ) -> None:
        self._llm_ratio = llm_ratio
        self._exploration_c = exploration_c
        self._total_calls = 0
        self._llm_calls_in_window = 0
        self._window_size = 0
        self._llm_consecutive_failures = 0
        self._llm_disabled = False

        self._operators: dict[str, OperatorStats] = {}
        for name in _AST_OPERATORS:
            self._operators[name] = OperatorStats(name=name, is_llm=False)
        for name in _LLM_OPERATORS:
            self._operators[name] = OperatorStats(name=name, is_llm=True)

    def select(self) -> str:
        """UCB1 스코어로 연산자 선택.

        LLM 연산자는 llm_ratio를 초과하지 않도록 제한.
        LLM 비활성화 상태면 AST만 선택.
        """
        self._window_size += 1

        # LLM 비율 제한 체크
        llm_allowed = (
            not self._llm_disabled
            and self._llm_ratio > 0
            and (
                self._window_size < 10
                or self._llm_calls_in_window / self._window_size < self._llm_ratio * 2
            )
        )

        candidates = []
        for name, stats in self._operators.items():
            if stats.is_llm and not llm_allowed:
                continue
            candidates.append((name, stats))

        if not candidates:
            # fallback: AST만
            candidates = [
                (n, s) for n, s in self._operators.items() if not s.is_llm
            ]

        # UCB1 스코어 계산
        best_name = None
        best_score = -float("inf")

        for name, stats in candidates:
            if stats.calls == 0:
                # 미시도 연산자 최우선 탐색 (+ 약간의 랜덤)
                score = float("inf") - random.random()
            else:
                avg_reward = stats.fitness_improvements / stats.calls
                exploration = self._exploration_c * math.sqrt(
                    math.log(max(self._total_calls, 1)) / stats.calls
                )
                score = avg_reward + exploration

            if score > best_score:
                best_score = score
                best_name = name

        return best_name  # type: ignore

    def update(self, operator_name: str, delta_fitness: float) -> None:
        """실행 결과로 통계 업데이트."""
        if operator_name not in self._operators:
            return

        stats = self._operators[operator_name]
        stats.calls += 1
        stats.fitness_improvements += max(delta_fitness, 0.0)
        self._total_calls += 1

        if stats.is_llm:
            self._llm_calls_in_window += 1

    def record_llm_failure(self) -> None:
        """LLM 연산자 실패 기록. 연속 3회 시 비활성화."""
        self._llm_consecutive_failures += 1
        if self._llm_consecutive_failures >= 3:
            self._llm_disabled = True

    def reset_llm_failures(self) -> None:
        """LLM 장애 카운터 리셋 (새 사이클 시작 시)."""
        self._llm_consecutive_failures = 0
        self._llm_disabled = False
        self._llm_calls_in_window = 0
        self._window_size = 0

    def is_llm_disabled(self) -> bool:
        """LLM 비활성화 상태 반환."""
        return self._llm_disabled

    def get_stats(self) -> list[OperatorStats]:
        """모든 연산자 통계 반환."""
        return list(self._operators.values())

    def to_dict(self) -> dict:
        """직렬화 가능한 dict 반환."""
        return {
            "operators": {
                name: {
                    "calls": s.calls,
                    "fitness_improvements": round(s.fitness_improvements, 6),
                    "is_llm": s.is_llm,
                }
                for name, s in self._operators.items()
            },
            "total_calls": self._total_calls,
            "llm_disabled": self._llm_disabled,
            "llm_ratio": self._llm_ratio,
        }
