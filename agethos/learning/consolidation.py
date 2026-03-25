"""메모리 통합 — 4단계 수명 관리.

생물학적 기억 통합(memory consolidation) 모델 기반.
synaptic-memory 프로젝트의 ConsolidationCascade 참고.

4단계 수명:
- L0 (Raw):       72시간, 자동 생성. 접근 안 되면 삭제.
- L1 (Sprint):    90일, 3회 이상 접근 시 승격.
- L2 (Monthly):   365일, 10회 이상 접근 시 승격.
- L3 (Permanent): 영구, 성공률 80% 이상 + 10회 이상.

강등 메커니즘: L3도 성공률 60% 미만이면 L2로 강등.
"""

from __future__ import annotations

import time
from enum import IntEnum

from agethos.models import SocialPattern


class ConsolidationLevel(IntEnum):
    """메모리 통합 단계."""
    L0_RAW = 0
    L1_SPRINT = 1
    L2_MONTHLY = 2
    L3_PERMANENT = 3


# TTL in seconds
_TTL = {
    ConsolidationLevel.L0_RAW: 72 * 3600,         # 72 hours
    ConsolidationLevel.L1_SPRINT: 90 * 86400,      # 90 days
    ConsolidationLevel.L2_MONTHLY: 365 * 86400,    # 365 days
    ConsolidationLevel.L3_PERMANENT: float("inf"),  # forever
}

# Promotion thresholds (evidence_count)
_PROMOTE_THRESHOLD = {
    ConsolidationLevel.L0_RAW: 3,       # 3+ accesses → L1
    ConsolidationLevel.L1_SPRINT: 10,   # 10+ accesses → L2
    ConsolidationLevel.L2_MONTHLY: 10,  # 10+ accesses + 80% confidence → L3
}

# L3 demotion threshold
_L3_DEMOTE_CONFIDENCE = 0.48  # 60% of 0.8 threshold


class ConsolidationEngine:
    """메모리 통합 엔진 — SocialPattern의 수명 관리.

    Usage::

        engine = ConsolidationEngine()

        # 패턴 수명 단계 판단
        level = engine.get_level(pattern)

        # 전체 패턴 목록 정리 (삭제/승격/강등)
        active, expired = engine.consolidate(patterns)
    """

    def get_level(self, pattern: SocialPattern) -> ConsolidationLevel:
        """패턴의 현재 통합 단계 판단."""
        ec = pattern.evidence_count
        conf = pattern.confidence

        # L3: 10+ evidence, 80%+ confidence
        if ec >= 10 and conf >= 0.8:
            return ConsolidationLevel.L3_PERMANENT

        # L2: 10+ evidence
        if ec >= 10:
            return ConsolidationLevel.L2_MONTHLY

        # L1: 3+ evidence
        if ec >= 3:
            return ConsolidationLevel.L1_SPRINT

        return ConsolidationLevel.L0_RAW

    def is_expired(self, pattern: SocialPattern) -> bool:
        """패턴이 TTL을 초과했는지 확인."""
        level = self.get_level(pattern)
        ttl = _TTL[level]
        age = time.time() - pattern.created_at
        return age > ttl

    def should_promote(self, pattern: SocialPattern) -> bool:
        """승격 조건을 만족하는지 확인."""
        level = self.get_level(pattern)
        if level >= ConsolidationLevel.L3_PERMANENT:
            return False

        threshold = _PROMOTE_THRESHOLD.get(level, float("inf"))
        if pattern.evidence_count >= threshold:
            # L2 → L3 requires high confidence
            if level == ConsolidationLevel.L2_MONTHLY:
                return pattern.confidence >= 0.8
            return True
        return False

    def should_demote(self, pattern: SocialPattern) -> bool:
        """L3 강등 조건 — 성공률(confidence)이 60% 미만."""
        level = self.get_level(pattern)
        if level != ConsolidationLevel.L3_PERMANENT:
            return False
        return pattern.confidence < _L3_DEMOTE_CONFIDENCE

    def consolidate(
        self,
        patterns: list[SocialPattern],
    ) -> tuple[list[SocialPattern], list[SocialPattern]]:
        """전체 패턴 목록 정리.

        Args:
            patterns: 전체 SocialPattern 목록.

        Returns:
            (active, expired): 유지할 패턴과 만료된 패턴.
        """
        active: list[SocialPattern] = []
        expired: list[SocialPattern] = []

        for p in patterns:
            if self.is_expired(p):
                expired.append(p)
            elif self.should_demote(p):
                # L3 → L2 강등: confidence를 리셋하지 않고 유지
                active.append(p)
            else:
                active.append(p)

        return active, expired

    def summary(self, patterns: list[SocialPattern]) -> dict[str, int]:
        """단계별 패턴 수 요약."""
        counts = {level.name: 0 for level in ConsolidationLevel}
        for p in patterns:
            level = self.get_level(p)
            counts[level.name] += 1
        return counts
