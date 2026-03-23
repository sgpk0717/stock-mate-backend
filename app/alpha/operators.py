"""Sliding-Window UCB 기반 적응형 연산자 레지스트리.

AST 연산자(저비용 ~92%)와 LLM 연산자(고비용 ~8%)를 관리하며,
Sliding-Window UCB 알고리즘으로 최근 성과 기반 연산자를 우선 선택한다.

기존 UCB1은 누적 통계 → 레짐 변화 시 과거 성공 연산자를 과대평가.
SW-UCB는 최근 W개 reward만 참조하여 비정상성(non-stationarity) 대응.
(Garivier & Moulines, 2011)
"""

from __future__ import annotations

import logging
import math
import random
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Sliding window 크기 기본값
_DEFAULT_WINDOW_SIZE = 50


@dataclass
class OperatorStats:
    """연산자별 통계."""

    name: str
    calls: int = 0
    fitness_improvements: float = 0.0
    is_llm: bool = False
    # Sliding-Window: 최근 W개 reward 기록
    recent_rewards: deque = field(
        default_factory=lambda: deque(maxlen=_DEFAULT_WINDOW_SIZE)
    )


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
    "llm_crossover",
]


class OperatorRegistry:
    """Sliding-Window UCB 기반 적응형 연산자 선택.

    Parameters
    ----------
    llm_ratio : LLM 연산자 최대 비율 (0.0~1.0). 0이면 LLM 비활성화.
    exploration_c : UCB 탐색 계수. 높을수록 미시도 연산자 선호.
    window_size : Sliding window 크기. 최근 W개 reward만 참조.
    """

    def __init__(
        self,
        llm_ratio: float = 0.08,
        exploration_c: float = 1.41,
        window_size: int = _DEFAULT_WINDOW_SIZE,
    ) -> None:
        self._llm_ratio = llm_ratio
        self._exploration_c = exploration_c
        self._window_size = window_size
        self._total_calls = 0
        self._llm_calls_in_window = 0
        self._call_window_size = 0
        self._llm_consecutive_failures = 0
        self._llm_disabled = False

        self._operators: dict[str, OperatorStats] = {}
        for name in _AST_OPERATORS:
            self._operators[name] = OperatorStats(
                name=name, is_llm=False,
                recent_rewards=deque(maxlen=window_size),
            )
        for name in _LLM_OPERATORS:
            self._operators[name] = OperatorStats(
                name=name, is_llm=True,
                recent_rewards=deque(maxlen=window_size),
            )

    def select(self) -> str:
        """Sliding-Window UCB 스코어로 연산자 선택.

        LLM 연산자는 llm_ratio를 초과하지 않도록 제한.
        LLM 비활성화 상태면 AST만 선택.
        """
        self._call_window_size += 1

        # LLM 비율 제한 체크
        llm_allowed = (
            not self._llm_disabled
            and self._llm_ratio > 0
            and (
                self._call_window_size < 10
                or self._llm_calls_in_window / self._call_window_size < self._llm_ratio * 2
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

        # Sliding-Window UCB 스코어 계산
        # 미시도 연산자가 있으면 그 중 랜덤 선택 (편향 방지)
        untried = [name for name, stats in candidates if stats.calls == 0]
        if untried:
            best_name = random.choice(untried)
        else:
            best_name = None
            best_score = -float("inf")

            # 윈도우 내 총 호출 수 (탐색항 분모)
            window_total = sum(
                len(stats.recent_rewards) for _, stats in candidates
            )

            for name, stats in candidates:
                n_recent = len(stats.recent_rewards)
                if n_recent == 0:
                    # recent_rewards가 비었지만 calls > 0 → 누적 통계 폴백
                    avg_reward = stats.fitness_improvements / stats.calls
                    n_recent = stats.calls
                else:
                    # Sliding window: 최근 W개 reward의 평균
                    avg_reward = sum(stats.recent_rewards) / n_recent

                exploration = self._exploration_c * math.sqrt(
                    math.log(max(window_total, 1)) / max(n_recent, 1)
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
        reward = max(delta_fitness, 0.0)
        stats.fitness_improvements += reward
        stats.recent_rewards.append(reward)  # Sliding window에 추가
        self._total_calls += 1

        if stats.is_llm:
            self._llm_calls_in_window += 1

    def record_llm_failure(self) -> None:
        """LLM 연산자 실패 기록. 연속 3회 시 비활성화."""
        self._llm_consecutive_failures += 1
        will_disable = self._llm_consecutive_failures >= 3
        logger.warning(
            "LLM failure #%d%s",
            self._llm_consecutive_failures,
            " → LLM DISABLED" if will_disable else "",
        )
        if will_disable:
            self._llm_disabled = True

    def reset_llm_failures(self) -> None:
        """LLM 장애 카운터 리셋 (새 사이클 시작 시)."""
        self._llm_consecutive_failures = 0
        self._llm_disabled = False
        self._llm_calls_in_window = 0
        self._call_window_size = 0

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
                    "recent_window_size": len(s.recent_rewards),
                    "recent_avg_reward": round(
                        sum(s.recent_rewards) / len(s.recent_rewards), 6
                    ) if s.recent_rewards else 0.0,
                }
                for name, s in self._operators.items()
            },
            "total_calls": self._total_calls,
            "llm_disabled": self._llm_disabled,
            "llm_ratio": self._llm_ratio,
            "window_size": self._window_size,
        }
