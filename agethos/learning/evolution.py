"""L1 Auto-Evolution — 학습된 패턴을 행동 규칙으로 내면화.

Generative Agents의 revise_identity() 패턴 + Leaked System Prompts의
"성격=행동 제약" 패턴을 결합.

핵심 원칙:
- L0(Innate)은 절대 불변 — Meta-cognitive defense
- L1(Learned)의 behavioral_rules만 진화
- OCEAN 자체는 변경하지 않음
- 충분히 검증된 패턴(L2+, confidence >= 0.7)만 규칙화
"""

from __future__ import annotations

from agethos.learning.consolidation import ConsolidationEngine, ConsolidationLevel
from agethos.models import PersonaSpec, SocialPattern


class PersonaEvolver:
    """L1 자동 진화 엔진 — SocialPattern → behavioral_rules.

    Usage::

        evolver = PersonaEvolver()
        new_rules = evolver.evolve(persona, patterns)
        # persona.behavioral_rules에 검증된 패턴 기반 규칙 추가
    """

    def __init__(self, min_confidence: float = 0.7, min_level: ConsolidationLevel = ConsolidationLevel.L1_SPRINT):
        self._min_confidence = min_confidence
        self._min_level = min_level
        self._consolidation = ConsolidationEngine()

    def evolve(
        self,
        persona: PersonaSpec,
        patterns: list[SocialPattern],
        max_new_rules: int = 5,
    ) -> list[str]:
        """검증된 패턴을 behavioral_rules로 변환 및 추가.

        Args:
            persona: 업데이트할 PersonaSpec.
            patterns: 축적된 SocialPattern 목록.
            max_new_rules: 한 번에 추가할 최대 규칙 수.

        Returns:
            새로 추가된 규칙 목록.
        """
        # Filter: confidence + consolidation level
        candidates = self._select_candidates(patterns)

        # Convert to rules
        new_rules: list[str] = []
        existing_rules_lower = {r.lower() for r in persona.behavioral_rules}

        for pattern in candidates[:max_new_rules]:
            rule = self._pattern_to_rule(pattern)
            if rule.lower() not in existing_rules_lower:
                new_rules.append(rule)
                existing_rules_lower.add(rule.lower())

        # Apply to persona L1
        persona.behavioral_rules.extend(new_rules)

        # Also update L1 learned traits
        if new_rules:
            learned_count = len(persona.l1_learned.traits.get("learned_rules", "").split(",")) if persona.l1_learned.traits.get("learned_rules") else 0
            persona.l1_learned.traits["learned_rules_count"] = str(learned_count + len(new_rules))

        return new_rules

    def _select_candidates(self, patterns: list[SocialPattern]) -> list[SocialPattern]:
        """규칙화 가능한 패턴 선별."""
        candidates = []
        for p in patterns:
            level = self._consolidation.get_level(p)
            if level >= self._min_level and p.confidence >= self._min_confidence:
                candidates.append(p)

        # Sort by confidence × evidence_count (가장 검증된 패턴 우선)
        candidates.sort(key=lambda p: p.confidence * p.evidence_count, reverse=True)
        return candidates

    def _pattern_to_rule(self, pattern: SocialPattern) -> str:
        """SocialPattern → 구체적 행동 규칙 텍스트.

        Leaked System Prompts 분석 결과:
        "friendly"보다 "반론 시 질문 형태 사용, 직접 부정 금지" 같은 구체적 제약이 효과적.
        """
        parts = []

        # Context prefix
        if pattern.community:
            parts.append(f"In {pattern.community}")
        if pattern.context:
            parts.append(f"when {pattern.context}")

        # Effective strategy (positive rule)
        parts.append(f": {pattern.effective_strategy}")

        rule = " ".join(parts) if len(parts) > 1 else pattern.effective_strategy

        # Add counterexample as negative constraint
        if pattern.counterexample:
            rule += f" (avoid: {pattern.counterexample})"

        return rule

    def suggest_rules(
        self,
        patterns: list[SocialPattern],
        max_suggestions: int = 5,
    ) -> list[dict[str, str]]:
        """규칙화 후보를 제안 (적용하지 않고 반환만).

        Returns:
            [{"rule": "...", "source": "...", "confidence": "...", "evidence": "..."}]
        """
        candidates = self._select_candidates(patterns)
        suggestions = []
        for p in candidates[:max_suggestions]:
            suggestions.append({
                "rule": self._pattern_to_rule(p),
                "source": p.community or "observed",
                "confidence": f"{p.confidence:.0%}",
                "evidence": f"{p.evidence_count} observations",
            })
        return suggestions
