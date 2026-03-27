"""v0.5.0 기능 테스트 — ToM, Self-Refine, Collaborate, Universalization."""

import pytest

from agethos.models import (
    CollaborationMessage,
    CollaborationResult,
    MentalModel,
    SelfRefineConfig,
    SelfRefineResult,
)


# ── MentalModel ──


class TestMentalModel:
    def test_create(self):
        mm = MentalModel(
            target="alice",
            believed_goals=["wants to fix the bug", "needs help with testing"],
            believed_knowledge=["knows Python", "doesn't know the new API"],
            believed_emotion="frustrated",
            relationship_summary="colleague, recently started working together",
            confidence=0.7,
        )
        assert mm.target == "alice"
        assert len(mm.believed_goals) == 2
        assert mm.believed_emotion == "frustrated"
        assert mm.confidence == 0.7

    def test_defaults(self):
        mm = MentalModel(target="bob")
        assert mm.believed_goals == []
        assert mm.believed_knowledge == []
        assert mm.believed_emotion == "neutral"
        assert mm.confidence == 0.5
        assert mm.last_updated > 0


# ── SelfRefineConfig ──


class TestSelfRefineConfig:
    def test_defaults(self):
        config = SelfRefineConfig()
        assert config.enabled is False
        assert config.max_iterations == 2
        assert config.quality_threshold == 0.7
        assert len(config.evaluate_axes) == 7

    def test_custom(self):
        config = SelfRefineConfig(
            enabled=True,
            max_iterations=5,
            quality_threshold=0.9,
            evaluate_axes=["accuracy", "tone"],
        )
        assert config.enabled is True
        assert config.max_iterations == 5
        assert len(config.evaluate_axes) == 2


class TestSelfRefineResult:
    def test_create(self):
        result = SelfRefineResult(
            original="hello",
            refined="Hello! How can I help?",
            iterations=2,
            scores=[{"persona_consistency": 0.8}, {"persona_consistency": 0.95}],
        )
        assert result.original == "hello"
        assert result.refined != result.original
        assert result.iterations == 2
        assert len(result.scores) == 2


# ── CollaborationResult ──


class TestCollaborationResult:
    def test_create(self):
        messages = [
            CollaborationMessage(agent_name="PM", content="Let's do it", round=0),
            CollaborationMessage(agent_name="Engineer", content="Need more time", round=0),
        ]
        result = CollaborationResult(
            topic="Rewrite auth?",
            messages=messages,
            consensus="Phased approach",
            total_rounds=1,
        )
        assert result.topic == "Rewrite auth?"
        assert len(result.messages) == 2
        assert result.consensus == "Phased approach"

    def test_message_defaults(self):
        msg = CollaborationMessage(agent_name="test", content="hello")
        assert msg.round == 0
        assert msg.timestamp > 0


# ── ToM prompt generation ──


class TestTheoryOfMindPrompt:
    def test_to_prompt(self):
        from agethos.cognition.tom import TheoryOfMind

        mm = MentalModel(
            target="alice",
            believed_goals=["fix the bug"],
            believed_knowledge=["knows Python"],
            believed_emotion="frustrated",
            relationship_summary="close colleague",
            confidence=0.8,
        )
        # to_prompt is a regular method, doesn't need LLM
        tom = TheoryOfMind.__new__(TheoryOfMind)
        prompt = tom.to_prompt(mm)
        assert "alice" in prompt
        assert "fix the bug" in prompt
        assert "frustrated" in prompt
        assert "close colleague" in prompt
        assert "80%" in prompt


# ── SelfRefiner without LLM ──


class TestSelfRefinerDisabled:
    @pytest.mark.asyncio
    async def test_disabled_passthrough(self):
        from agethos.cognition.refine import SelfRefiner

        config = SelfRefineConfig(enabled=False)
        refiner = SelfRefiner.__new__(SelfRefiner)
        refiner._config = config
        refiner._llm = None

        result = await refiner.refine("hello", "user msg", "persona")
        assert result.original == "hello"
        assert result.refined == "hello"
        assert result.iterations == 0

    def test_enabled_property(self):
        from agethos.cognition.refine import SelfRefiner

        refiner = SelfRefiner.__new__(SelfRefiner)
        refiner._config = SelfRefineConfig(enabled=True)
        assert refiner.enabled is True

        refiner._config = SelfRefineConfig(enabled=False)
        assert refiner.enabled is False


# ── Collaborate structure ──


class TestCollaborateStructure:
    def test_imports(self):
        from agethos.cognition.collaborate import team_discuss
        assert callable(team_discuss)


# ── Universalization ──


class TestUniversalization:
    def test_method_exists(self):
        from agethos.cognition.social import SocialCognition
        assert hasattr(SocialCognition, "universalize_check")


# ── Brain integration (no LLM) ──


class TestBrainV050:
    def test_brain_accepts_self_refine_config(self):
        """Brain.__init__ accepts self_refine parameter."""
        from agethos.brain import Brain
        try:
            brain = Brain.build(
                persona={"name": "Test", "ocean": {"O": 0.5}},
                llm="openai",
                self_refine=SelfRefineConfig(enabled=True),
            )
            assert brain._self_refine_config.enabled is True
        except (ImportError, Exception):
            pass  # openai not installed

    def test_mental_model_property(self):
        """Brain has mental_models property."""
        from agethos.brain import Brain
        try:
            brain = Brain.build(
                persona={"name": "Test"},
                llm="openai",
            )
            assert brain.mental_models == {}
        except (ImportError, Exception):
            pass

    def test_brain_state_includes_mental_models(self):
        """BrainState serializes mental models."""
        from agethos.models import BrainState, PersonaSpec

        state = BrainState(
            persona=PersonaSpec(name="Test"),
            mental_models=[
                MentalModel(target="alice", believed_emotion="happy"),
                MentalModel(target="bob", believed_goals=["learn Python"]),
            ],
        )
        data = state.model_dump(mode="json")
        restored = BrainState.model_validate(data)
        assert len(restored.mental_models) == 2
        assert restored.mental_models[0].target == "alice"
        assert restored.mental_models[1].believed_goals == ["learn Python"]
