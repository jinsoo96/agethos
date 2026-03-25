"""Hebbian 학습 엔진 — 사회적 패턴의 강화/약화.

"함께 발화하는 뉴런은 함께 연결된다" (Hebb, 1949)
synaptic-memory 프로젝트의 비대칭 학습 모델 참고.

핵심 메커니즘:
- 성공 시 confidence +0.1 (reinforcement)
- 실패 시 confidence -0.15 (weakening, 비대칭 — 실패에서 더 강하게 배움)
- Adaptive rate: delta / (1 + 0.02 × maturity) → 오래된 패턴일수록 변화 저항
- Anti-resonance: confidence가 0 이하면 "이 전략은 역효과" 의미
- 가중치 범위: -0.5 ~ 1.0
"""

from __future__ import annotations

import time

from agethos.models import SocialPattern


# Constants
REINFORCE_DELTA = 0.1
WEAKEN_DELTA = 0.15  # Asymmetric — failure learning is stronger
MIN_CONFIDENCE = -0.5  # Anti-resonance: negative = actively avoid
MAX_CONFIDENCE = 1.0
MATURITY_DECAY = 0.02  # Adaptive rate decay factor


class HebbianEngine:
    """Hebbian 학습 엔진 — SocialPattern의 confidence를 경험 기반으로 조정.

    Usage::

        engine = HebbianEngine()

        # 전략이 성공했을 때
        engine.reinforce(pattern)
        # → confidence += adaptive_delta(+0.1)

        # 전략이 실패했을 때
        engine.weaken(pattern)
        # → confidence -= adaptive_delta(0.15)

        # 배치 업데이트
        engine.update_batch(patterns, outcomes)
    """

    def _adaptive_delta(self, base_delta: float, pattern: SocialPattern) -> float:
        """Maturity 기반 적응형 학습률.

        오래된 패턴일수록 변화 저항 — 성격 안정화.
        maturity = evidence_count (관찰 횟수가 많을수록 성숙)
        """
        maturity = pattern.evidence_count
        return base_delta / (1 + MATURITY_DECAY * maturity)

    def reinforce(self, pattern: SocialPattern) -> SocialPattern:
        """성공 강화 — 전략이 효과적이었을 때."""
        delta = self._adaptive_delta(REINFORCE_DELTA, pattern)
        pattern.confidence = min(MAX_CONFIDENCE, pattern.confidence + delta)
        pattern.evidence_count += 1
        return pattern

    def weaken(self, pattern: SocialPattern) -> SocialPattern:
        """실패 약화 — 전략이 역효과였을 때.

        비대칭: 실패에서 더 강하게 배움 (WEAKEN_DELTA > REINFORCE_DELTA).
        confidence가 0 이하로 떨어지면 anti-resonance (적극 회피).
        """
        delta = self._adaptive_delta(WEAKEN_DELTA, pattern)
        pattern.confidence = max(MIN_CONFIDENCE, pattern.confidence - delta)
        pattern.evidence_count += 1
        return pattern

    def update_batch(
        self,
        patterns: list[SocialPattern],
        outcomes: list[bool],
    ) -> list[SocialPattern]:
        """배치 업데이트 — 여러 패턴의 성공/실패를 한번에 처리.

        Args:
            patterns: SocialPattern 목록.
            outcomes: 각 패턴의 성공 여부 (True=성공, False=실패).
        """
        for pattern, success in zip(patterns, outcomes):
            if success:
                self.reinforce(pattern)
            else:
                self.weaken(pattern)
        return patterns

    def is_anti_resonance(self, pattern: SocialPattern) -> bool:
        """패턴이 anti-resonance 상태인지 (적극 회피해야 하는 전략)."""
        return pattern.confidence < 0

    def should_use(self, pattern: SocialPattern, threshold: float = 0.3) -> bool:
        """이 패턴을 사용해야 하는지 판단.

        Args:
            threshold: 이 이상이면 사용 권장.
        """
        return pattern.confidence >= threshold

    def filter_effective(
        self,
        patterns: list[SocialPattern],
        threshold: float = 0.3,
    ) -> list[SocialPattern]:
        """효과적인 패턴만 필터링."""
        return [p for p in patterns if self.should_use(p, threshold)]

    def filter_avoid(self, patterns: list[SocialPattern]) -> list[SocialPattern]:
        """회피해야 할 패턴 (anti-resonance) 필터링."""
        return [p for p in patterns if self.is_anti_resonance(p)]
