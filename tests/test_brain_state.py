"""BrainState 저장/복원 + Export 어댑터 테스트."""

import json
import os
import tempfile

import pytest

from agethos.models import (
    BrainState,
    CommunityProfile,
    EmotionalState,
    MemoryNode,
    NodeType,
    OceanTraits,
    PersonaLayer,
    PersonaSpec,
    SocialPattern,
)
from agethos.persona.renderer import PersonaRenderer


# ── SocialPattern ──


class TestSocialPattern:
    def test_create(self):
        p = SocialPattern(
            context="기술 토론에서 반론 제시 시",
            effective_strategy="질문 형태로 우회",
            counterexample="직접 반박 시 분위기 경직",
            confidence=0.7,
            community="Python Discord",
        )
        assert p.context == "기술 토론에서 반론 제시 시"
        assert p.evidence_count == 1
        assert p.confidence == 0.7
        assert p.id  # auto-generated

    def test_defaults(self):
        p = SocialPattern(
            context="test",
            effective_strategy="test strategy",
        )
        assert p.counterexample is None
        assert p.evidence_count == 1
        assert p.confidence == 0.5
        assert p.community == ""


# ── CommunityProfile ──


class TestCommunityProfile:
    def test_create(self):
        patterns = [
            SocialPattern(context="c1", effective_strategy="s1"),
            SocialPattern(context="c2", effective_strategy="s2"),
        ]
        cp = CommunityProfile(
            name="Python Discord",
            norms=patterns,
            tone_baseline="casual-technical",
            conflict_style="indirect-questioning",
            observed_count=100,
        )
        assert cp.name == "Python Discord"
        assert len(cp.norms) == 2
        assert cp.observed_count == 100

    def test_empty_defaults(self):
        cp = CommunityProfile(name="test")
        assert cp.norms == []
        assert cp.observed_count == 0


# ── BrainState ──


class TestBrainState:
    def _make_state(self) -> BrainState:
        persona = PersonaSpec(
            name="TestAgent",
            ocean=OceanTraits(openness=0.8, extraversion=0.3),
            l0_innate=PersonaLayer(traits={"age": "28"}),
            tone="Warm",
            values=["Quality"],
            behavioral_rules=["Think first"],
        )
        persona.init_emotion_from_ocean()

        memories = [
            MemoryNode(description="Had coffee", node_type=NodeType.EVENT, importance=3.0),
            MemoryNode(description="Meeting at 3pm", node_type=NodeType.PLAN, importance=5.0),
        ]

        patterns = [
            SocialPattern(context="code review", effective_strategy="ask questions", confidence=0.8),
        ]

        return BrainState(
            persona=persona,
            memories=memories,
            social_patterns=patterns,
            community_profiles=[CommunityProfile(name="test-community", norms=patterns)],
            history=[
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
            ],
            total_interactions=1,
        )

    def test_create(self):
        state = self._make_state()
        assert state.persona.name == "TestAgent"
        assert len(state.memories) == 2
        assert len(state.social_patterns) == 1
        assert state.total_interactions == 1

    def test_json_roundtrip(self):
        """BrainState → JSON → BrainState 무손실 변환."""
        state = self._make_state()

        # Serialize
        data = state.model_dump(mode="json")
        json_str = json.dumps(data, ensure_ascii=False)

        # Deserialize
        restored = BrainState.model_validate(json.loads(json_str))

        assert restored.persona.name == state.persona.name
        assert restored.persona.ocean.openness == state.persona.ocean.openness
        assert len(restored.memories) == len(state.memories)
        assert restored.memories[0].description == state.memories[0].description
        assert len(restored.social_patterns) == len(state.social_patterns)
        assert restored.social_patterns[0].context == state.social_patterns[0].context
        assert len(restored.history) == len(state.history)
        assert restored.total_interactions == state.total_interactions

    def test_file_roundtrip(self):
        """파일 저장/복원 테스트."""
        state = self._make_state()

        with tempfile.NamedTemporaryFile(suffix=".brain.json", delete=False, mode="w") as f:
            json.dump(state.model_dump(mode="json"), f, ensure_ascii=False, indent=2)
            path = f.name

        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            restored = BrainState.model_validate(data)
            assert restored.persona.name == "TestAgent"
            assert len(restored.memories) == 2
        finally:
            os.unlink(path)


# ── Export Adapters ──


class TestExportAdapters:
    def _make_brain_mock(self):
        """Brain-like 객체 생성 (LLM 없이)."""

        class FakeBrain:
            def __init__(self):
                self.persona = PersonaSpec(
                    name="Minsoo",
                    identity="A senior backend engineer",
                    ocean=OceanTraits(openness=0.8, conscientiousness=0.9, extraversion=0.3, agreeableness=0.7, neuroticism=0.2),
                    l0_innate=PersonaLayer(traits={"age": "28", "occupation": "Backend Engineer"}),
                    tone="Precise but warm",
                    values=["Code quality", "Knowledge sharing"],
                    behavioral_rules=["Include examples", "Say I don't know when uncertain"],
                )
                self.persona.init_emotion_from_ocean()
                self._social_patterns = [
                    SocialPattern(
                        context="code review",
                        effective_strategy="ask questions instead of direct criticism",
                        counterexample="blunt negative feedback",
                        confidence=0.8,
                        community="Python Discord",
                    ),
                ]
                self._community_profiles = []

            @property
            def social_patterns(self):
                return list(self._social_patterns)

            @property
            def community_profiles(self):
                return list(self._community_profiles)

        return FakeBrain()

    def test_export_system_prompt(self):
        from agethos.export.adapters import export_brain
        brain = self._make_brain_mock()
        result = export_brain(brain, "system_prompt")
        assert isinstance(result, str)
        assert "senior backend engineer" in result
        assert "OCEAN" in result
        assert "Learned Social Patterns" in result
        assert "ask questions" in result

    def test_export_anthropic(self):
        from agethos.export.adapters import export_brain
        brain = self._make_brain_mock()
        result = export_brain(brain, "anthropic")
        assert isinstance(result, str)
        assert "senior backend engineer" in result

    def test_export_openai_assistant(self):
        from agethos.export.adapters import export_brain
        brain = self._make_brain_mock()
        result = export_brain(brain, "openai_assistant")
        assert isinstance(result, dict)
        assert result["name"] == "Minsoo"
        assert "instructions" in result
        assert len(result["instructions"]) <= 256000
        assert result["model"] == "gpt-4o"

    def test_export_crewai(self):
        from agethos.export.adapters import export_brain
        brain = self._make_brain_mock()
        result = export_brain(brain, "crewai")
        assert isinstance(result, dict)
        assert "role" in result
        assert "goal" in result
        assert "backstory" in result
        assert "Code quality" in result["goal"]

    def test_export_bedrock(self):
        from agethos.export.adapters import export_brain
        brain = self._make_brain_mock()
        result = export_brain(brain, "bedrock_agent")
        assert isinstance(result, dict)
        assert len(result["instruction"]) <= 4000
        assert result["agentName"] == "Minsoo"

    def test_export_a2a_card(self):
        from agethos.export.adapters import export_brain
        brain = self._make_brain_mock()
        result = export_brain(brain, "a2a_card")
        assert isinstance(result, dict)
        assert result["name"] == "Minsoo"
        assert "skills" in result
        assert result["provider"]["name"] == "agethos"

    def test_export_unknown_format_raises(self):
        from agethos.export.adapters import export_brain
        brain = self._make_brain_mock()
        with pytest.raises(ValueError, match="Unknown export format"):
            export_brain(brain, "nonexistent")
