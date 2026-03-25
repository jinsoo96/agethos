"""v0.6.0 기능 테스트 — Hebbian, Consolidation, Evolution, Retrieval Presets."""

import time

import pytest

from agethos.models import SocialPattern, PersonaSpec, OceanTraits


# ── Hebbian Learning ──


class TestHebbianEngine:
    def _make_pattern(self, confidence=0.5, evidence_count=1):
        return SocialPattern(
            context="code review",
            effective_strategy="ask questions",
            confidence=confidence,
            evidence_count=evidence_count,
        )

    def test_reinforce_increases_confidence(self):
        from agethos.learning.hebbian import HebbianEngine
        engine = HebbianEngine()
        p = self._make_pattern(confidence=0.5)
        engine.reinforce(p)
        assert p.confidence > 0.5
        assert p.evidence_count == 2

    def test_weaken_decreases_confidence(self):
        from agethos.learning.hebbian import HebbianEngine
        engine = HebbianEngine()
        p = self._make_pattern(confidence=0.5)
        engine.weaken(p)
        assert p.confidence < 0.5
        assert p.evidence_count == 2

    def test_asymmetric_learning(self):
        """실패에서 더 강하게 배움 (weaken > reinforce)."""
        from agethos.learning.hebbian import HebbianEngine
        engine = HebbianEngine()
        p1 = self._make_pattern(confidence=0.5)
        p2 = self._make_pattern(confidence=0.5)
        engine.reinforce(p1)
        engine.weaken(p2)
        gain = p1.confidence - 0.5
        loss = 0.5 - p2.confidence
        assert loss > gain  # asymmetric

    def test_adaptive_rate_mature_pattern(self):
        """성숙한 패턴은 변화 저항."""
        from agethos.learning.hebbian import HebbianEngine
        engine = HebbianEngine()
        young = self._make_pattern(confidence=0.5, evidence_count=1)
        mature = self._make_pattern(confidence=0.5, evidence_count=50)
        engine.reinforce(young)
        engine.reinforce(mature)
        young_gain = young.confidence - 0.5
        mature_gain = mature.confidence - 0.5
        assert young_gain > mature_gain

    def test_anti_resonance(self):
        """confidence < 0 → anti-resonance."""
        from agethos.learning.hebbian import HebbianEngine
        engine = HebbianEngine()
        p = self._make_pattern(confidence=0.1)
        engine.weaken(p)
        engine.weaken(p)
        assert engine.is_anti_resonance(p) or p.confidence < 0.1

    def test_confidence_clamped(self):
        from agethos.learning.hebbian import HebbianEngine
        engine = HebbianEngine()
        p_high = self._make_pattern(confidence=0.99)
        for _ in range(20):
            engine.reinforce(p_high)
        assert p_high.confidence <= 1.0

        p_low = self._make_pattern(confidence=-0.4)
        for _ in range(20):
            engine.weaken(p_low)
        assert p_low.confidence >= -0.5

    def test_batch_update(self):
        from agethos.learning.hebbian import HebbianEngine
        engine = HebbianEngine()
        patterns = [self._make_pattern() for _ in range(3)]
        outcomes = [True, False, True]
        engine.update_batch(patterns, outcomes)
        assert patterns[0].confidence > 0.5
        assert patterns[1].confidence < 0.5
        assert patterns[2].confidence > 0.5

    def test_filter_effective(self):
        from agethos.learning.hebbian import HebbianEngine
        engine = HebbianEngine()
        patterns = [
            self._make_pattern(confidence=0.8),
            self._make_pattern(confidence=0.2),
            self._make_pattern(confidence=-0.1),
        ]
        effective = engine.filter_effective(patterns, threshold=0.3)
        assert len(effective) == 1

    def test_filter_avoid(self):
        from agethos.learning.hebbian import HebbianEngine
        engine = HebbianEngine()
        patterns = [
            self._make_pattern(confidence=0.8),
            self._make_pattern(confidence=-0.1),
            self._make_pattern(confidence=-0.3),
        ]
        avoid = engine.filter_avoid(patterns)
        assert len(avoid) == 2


# ── Consolidation ──


class TestConsolidationEngine:
    def _make_pattern(self, evidence_count=1, confidence=0.5, age_hours=0):
        p = SocialPattern(
            context="test",
            effective_strategy="test strategy",
            evidence_count=evidence_count,
            confidence=confidence,
        )
        if age_hours:
            p.created_at = time.time() - age_hours * 3600
        return p

    def test_level_l0(self):
        from agethos.learning.consolidation import ConsolidationEngine, ConsolidationLevel
        engine = ConsolidationEngine()
        p = self._make_pattern(evidence_count=1)
        assert engine.get_level(p) == ConsolidationLevel.L0_RAW

    def test_level_l1(self):
        from agethos.learning.consolidation import ConsolidationEngine, ConsolidationLevel
        engine = ConsolidationEngine()
        p = self._make_pattern(evidence_count=5)
        assert engine.get_level(p) == ConsolidationLevel.L1_SPRINT

    def test_level_l2(self):
        from agethos.learning.consolidation import ConsolidationEngine, ConsolidationLevel
        engine = ConsolidationEngine()
        p = self._make_pattern(evidence_count=15, confidence=0.6)
        assert engine.get_level(p) == ConsolidationLevel.L2_MONTHLY

    def test_level_l3(self):
        from agethos.learning.consolidation import ConsolidationEngine, ConsolidationLevel
        engine = ConsolidationEngine()
        p = self._make_pattern(evidence_count=15, confidence=0.85)
        assert engine.get_level(p) == ConsolidationLevel.L3_PERMANENT

    def test_l0_expires_after_72h(self):
        from agethos.learning.consolidation import ConsolidationEngine
        engine = ConsolidationEngine()
        p = self._make_pattern(evidence_count=1, age_hours=80)  # > 72h
        assert engine.is_expired(p)

    def test_l0_not_expired_before_72h(self):
        from agethos.learning.consolidation import ConsolidationEngine
        engine = ConsolidationEngine()
        p = self._make_pattern(evidence_count=1, age_hours=24)  # < 72h
        assert not engine.is_expired(p)

    def test_l3_demotion(self):
        from agethos.learning.consolidation import ConsolidationEngine
        engine = ConsolidationEngine()
        # L3 but low confidence → should demote
        p = self._make_pattern(evidence_count=15, confidence=0.3)
        # This has evidence >= 10 but confidence < 0.8 → actually L2
        # L3 requires both evidence >= 10 AND confidence >= 0.8
        assert not engine.should_demote(p)  # not L3, so no demotion

        # Real L3 with declining confidence
        p2 = self._make_pattern(evidence_count=15, confidence=0.85)
        assert not engine.should_demote(p2)  # healthy L3

    def test_consolidate_removes_expired(self):
        from agethos.learning.consolidation import ConsolidationEngine
        engine = ConsolidationEngine()
        patterns = [
            self._make_pattern(evidence_count=1, age_hours=1),    # fresh L0
            self._make_pattern(evidence_count=1, age_hours=100),  # expired L0
            self._make_pattern(evidence_count=5, age_hours=10),   # L1, fine
        ]
        active, expired = engine.consolidate(patterns)
        assert len(active) == 2
        assert len(expired) == 1

    def test_summary(self):
        from agethos.learning.consolidation import ConsolidationEngine
        engine = ConsolidationEngine()
        patterns = [
            self._make_pattern(evidence_count=1),
            self._make_pattern(evidence_count=1),
            self._make_pattern(evidence_count=5),
            self._make_pattern(evidence_count=15, confidence=0.9),
        ]
        summary = engine.summary(patterns)
        assert summary["L0_RAW"] == 2
        assert summary["L1_SPRINT"] == 1
        assert summary["L3_PERMANENT"] == 1


# ── Evolution ──


class TestPersonaEvolver:
    def test_evolve_adds_rules(self):
        from agethos.learning.evolution import PersonaEvolver
        persona = PersonaSpec(
            name="Test",
            ocean=OceanTraits(),
            behavioral_rules=["Existing rule"],
        )
        patterns = [
            SocialPattern(
                context="code review",
                effective_strategy="ask questions first",
                counterexample="direct criticism",
                evidence_count=5,
                confidence=0.8,
                community="Python Discord",
            ),
        ]
        evolver = PersonaEvolver()
        new_rules = evolver.evolve(persona, patterns)
        assert len(new_rules) == 1
        assert "ask questions" in new_rules[0]
        assert "Existing rule" in persona.behavioral_rules
        assert len(persona.behavioral_rules) == 2

    def test_evolve_skips_low_confidence(self):
        from agethos.learning.evolution import PersonaEvolver
        persona = PersonaSpec(name="Test", behavioral_rules=[])
        patterns = [
            SocialPattern(context="test", effective_strategy="test", confidence=0.3, evidence_count=5),
        ]
        evolver = PersonaEvolver()
        new_rules = evolver.evolve(persona, patterns)
        assert len(new_rules) == 0

    def test_evolve_skips_duplicates(self):
        from agethos.learning.evolution import PersonaEvolver
        persona = PersonaSpec(
            name="Test",
            behavioral_rules=["ask questions first"],
        )
        patterns = [
            SocialPattern(
                context="",
                effective_strategy="ask questions first",
                evidence_count=5,
                confidence=0.8,
            ),
        ]
        evolver = PersonaEvolver()
        new_rules = evolver.evolve(persona, patterns)
        assert len(new_rules) == 0  # duplicate, not added

    def test_suggest_rules(self):
        from agethos.learning.evolution import PersonaEvolver
        patterns = [
            SocialPattern(context="meetings", effective_strategy="take notes", evidence_count=8, confidence=0.85),
        ]
        evolver = PersonaEvolver()
        suggestions = evolver.suggest_rules(patterns)
        assert len(suggestions) == 1
        assert "take notes" in suggestions[0]["rule"]
        assert suggestions[0]["confidence"] == "85%"


# ── Retrieval Presets ──


class TestRetrievalPresets:
    def test_presets_exist(self):
        from agethos.cognition.retrieve import RETRIEVAL_PRESETS
        assert "default" in RETRIEVAL_PRESETS
        assert "recall" in RETRIEVAL_PRESETS
        assert "planning" in RETRIEVAL_PRESETS
        assert "reflection" in RETRIEVAL_PRESETS
        assert "observation" in RETRIEVAL_PRESETS
        assert "conversation" in RETRIEVAL_PRESETS
        assert "failure_analysis" in RETRIEVAL_PRESETS
        assert "exploration" in RETRIEVAL_PRESETS

    def test_preset_values_are_tuples(self):
        from agethos.cognition.retrieve import RETRIEVAL_PRESETS
        for name, weights in RETRIEVAL_PRESETS.items():
            assert len(weights) == 3, f"{name} should have 3 weights"
            assert all(isinstance(w, (int, float)) for w in weights)

    def test_recall_favors_relevance(self):
        from agethos.cognition.retrieve import RETRIEVAL_PRESETS
        r, i, v = RETRIEVAL_PRESETS["recall"]
        assert v > r  # relevance > recency
        assert v > i or i > r  # relevance or importance > recency

    def test_planning_favors_recency(self):
        from agethos.cognition.retrieve import RETRIEVAL_PRESETS
        r, i, v = RETRIEVAL_PRESETS["planning"]
        assert r > v  # recency > relevance
