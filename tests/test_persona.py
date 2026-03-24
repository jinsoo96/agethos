"""인격 주입 테스트 — LLM 없이 모델 + 렌더링 검증."""

import pytest

from agethos.models import (
    CharacterCard,
    EmotionalState,
    OceanTraits,
    PersonaLayer,
    PersonaSpec,
)
from agethos.persona.renderer import PersonaRenderer


# ── OceanTraits ──


class TestOceanTraits:
    def test_default_values(self):
        ocean = OceanTraits()
        assert ocean.openness == 0.5
        assert ocean.neuroticism == 0.5

    def test_to_prompt_contains_all_traits(self):
        ocean = OceanTraits(openness=0.9, conscientiousness=0.8, extraversion=0.2, agreeableness=0.7, neuroticism=0.1)
        prompt = ocean.to_prompt()
        assert "HIGH" in prompt  # openness, conscientiousness, agreeableness
        assert "LOW" in prompt   # extraversion, neuroticism
        assert "creative" in prompt.lower()
        assert "reserved" in prompt.lower()
        assert "stable" in prompt.lower() or "calm" in prompt.lower()

    def test_boundary_values(self):
        ocean = OceanTraits(openness=0.0, conscientiousness=0.0, extraversion=0.0, agreeableness=0.0, neuroticism=0.0)
        prompt = ocean.to_prompt()
        assert prompt.count("LOW") == 5

        ocean = OceanTraits(openness=1.0, conscientiousness=1.0, extraversion=1.0, agreeableness=1.0, neuroticism=1.0)
        prompt = ocean.to_prompt()
        assert prompt.count("HIGH") == 5


# ── EmotionalState ──


class TestEmotionalState:
    def test_from_ocean(self):
        ocean = OceanTraits(openness=0.8, conscientiousness=0.7, extraversion=0.3, agreeableness=0.9, neuroticism=0.2)
        emotion = EmotionalState.from_ocean(ocean)
        # P = 0.21*0.3 + 0.59*0.9 - 0.19*0.2 = 0.063 + 0.531 - 0.038 = 0.556
        assert emotion.pleasure > 0
        assert -1 <= emotion.pleasure <= 1
        assert -1 <= emotion.arousal <= 1
        assert -1 <= emotion.dominance <= 1

    def test_closest_emotion_joy(self):
        emotion = EmotionalState(pleasure=0.7, arousal=0.5, dominance=0.3)
        assert emotion.closest_emotion() == "joy"

    def test_closest_emotion_sadness(self):
        emotion = EmotionalState(pleasure=-0.6, arousal=-0.3, dominance=-0.3)
        assert emotion.closest_emotion() == "sadness"

    def test_apply_stimulus_shifts_state(self):
        base = EmotionalState(pleasure=0.5, arousal=0.0, dominance=0.0)
        shifted = base.apply_stimulus((-0.8, 0.6, -0.4), sensitivity=0.5)
        assert shifted.pleasure < base.pleasure
        assert shifted.arousal > base.arousal

    def test_decay_returns_toward_baseline(self):
        current = EmotionalState(pleasure=-0.5, arousal=0.5, dominance=-0.5)
        baseline = EmotionalState(pleasure=0.5, arousal=0.0, dominance=0.3)
        decayed = current.decay(baseline=baseline, rate=0.5)
        # Should move halfway toward baseline
        assert decayed.pleasure > current.pleasure
        assert decayed.arousal < current.arousal

    def test_to_prompt_format(self):
        emotion = EmotionalState(pleasure=0.7, arousal=0.5, dominance=0.3)
        prompt = emotion.to_prompt()
        assert "Current emotional state:" in prompt
        assert "P=" in prompt
        assert "A=" in prompt
        assert "D=" in prompt

    def test_clamp_values(self):
        """Stimulus should not push PAD beyond [-1, 1]."""
        extreme = EmotionalState(pleasure=0.9, arousal=0.9, dominance=0.9)
        shifted = extreme.apply_stimulus((1.0, 1.0, 1.0), sensitivity=1.0)
        assert shifted.pleasure <= 1.0
        assert shifted.arousal <= 1.0
        assert shifted.dominance <= 1.0


# ── PersonaSpec ──


class TestPersonaSpec:
    def test_init_emotion_from_ocean(self):
        spec = PersonaSpec(
            name="TestAgent",
            ocean=OceanTraits(openness=0.8, extraversion=0.3, agreeableness=0.9, neuroticism=0.2),
        )
        assert spec.emotion is None
        spec.init_emotion_from_ocean()
        assert spec.emotion is not None
        assert spec.emotion.pleasure > 0  # high agreeableness → positive pleasure

    def test_apply_event_auto_sensitivity(self):
        """High neuroticism → higher sensitivity → larger emotion shift."""
        spec_stable = PersonaSpec(
            name="Stable",
            ocean=OceanTraits(neuroticism=0.1),
        )
        spec_stable.init_emotion_from_ocean()
        p_before_stable = spec_stable.emotion.pleasure

        spec_neurotic = PersonaSpec(
            name="Neurotic",
            ocean=OceanTraits(neuroticism=0.9),
        )
        spec_neurotic.init_emotion_from_ocean()
        p_before_neurotic = spec_neurotic.emotion.pleasure

        event = (-0.8, 0.6, -0.4)  # negative event
        spec_stable.apply_event(event)
        spec_neurotic.apply_event(event)

        shift_stable = abs(spec_stable.emotion.pleasure - p_before_stable)
        shift_neurotic = abs(spec_neurotic.emotion.pleasure - p_before_neurotic)
        assert shift_neurotic > shift_stable  # neurotic reacts more

    def test_decay_emotion(self):
        spec = PersonaSpec(
            name="Test",
            ocean=OceanTraits(openness=0.5, extraversion=0.5, agreeableness=0.5, neuroticism=0.5),
        )
        spec.init_emotion_from_ocean()
        baseline_p = spec.emotion.pleasure
        spec.apply_event((-0.8, 0.6, -0.4))
        assert spec.emotion.pleasure != baseline_p
        # Decay 10 times
        for _ in range(10):
            spec.decay_emotion(rate=0.3)
        # Should be close to baseline
        assert abs(spec.emotion.pleasure - baseline_p) < 0.1


# ── PersonaRenderer ──


class TestPersonaRenderer:
    def test_render_iss_includes_name(self):
        spec = PersonaSpec(name="Luna")
        renderer = PersonaRenderer(spec)
        iss = renderer.render_iss()
        assert "Luna" in iss

    def test_render_iss_includes_ocean(self):
        spec = PersonaSpec(
            name="Luna",
            ocean=OceanTraits(openness=0.9),
        )
        renderer = PersonaRenderer(spec)
        iss = renderer.render_iss()
        assert "OCEAN" in iss
        assert "HIGH" in iss

    def test_render_iss_includes_all_facets(self):
        spec = PersonaSpec(
            name="Luna",
            tone="Warm and precise",
            values=["Honesty", "Curiosity"],
            boundaries=["No medical advice"],
            behavioral_rules=["Always include examples"],
            conversation_style="Socratic questioning",
            transparency="I am an AI assistant",
        )
        renderer = PersonaRenderer(spec)
        iss = renderer.render_iss()
        assert "Warm and precise" in iss
        assert "Honesty" in iss
        assert "No medical advice" in iss
        assert "Always include examples" in iss
        assert "Socratic" in iss
        assert "AI assistant" in iss

    def test_render_full_includes_emotion(self):
        spec = PersonaSpec(
            name="Luna",
            ocean=OceanTraits(openness=0.8, agreeableness=0.9, neuroticism=0.2),
        )
        spec.init_emotion_from_ocean()
        renderer = PersonaRenderer(spec)
        full = renderer.render_full()
        assert "Emotional State" in full
        assert "P=" in full

    def test_render_full_includes_memories(self):
        from agethos.models import MemoryNode, NodeType

        spec = PersonaSpec(name="Luna")
        renderer = PersonaRenderer(spec)
        memories = [
            MemoryNode(description="Had coffee with friend", node_type=NodeType.EVENT),
            MemoryNode(description="Deadline is tomorrow", node_type=NodeType.EVENT),
        ]
        full = renderer.render_full(context_memories=memories)
        assert "coffee" in full
        assert "Deadline" in full

    def test_render_full_includes_layers(self):
        spec = PersonaSpec(
            name="Luna",
            l0_innate=PersonaLayer(traits={"age": "25", "role": "researcher"}),
            l1_learned=PersonaLayer(traits={"skill": "Python", "relationship": "close with team"}),
            l2_situation=PersonaLayer(traits={"location": "office", "task": "debugging"}),
        )
        renderer = PersonaRenderer(spec)
        full = renderer.render_full()
        assert "25" in full
        assert "Python" in full
        assert "debugging" in full


# ── CharacterCard ──


class TestCharacterCard:
    def test_from_wpp(self):
        card = CharacterCard.from_wpp('''
        [character("Luna")
        {
          Personality("analytical" + "curious" + "dry humor")
          Age("25")
          Occupation("AI Researcher")
        }]
        ''')
        assert card.name == "Luna"
        assert "analytical" in card.personality
        assert "curious" in card.personality
        assert "25" in card.description

    def test_from_sbf(self):
        card = CharacterCard.from_sbf('''
        [character: Marcus;
        personality: stoic, loyal, pragmatic;
        occupation: mercenary captain;
        speech: blunt, few words]
        ''')
        assert card.name == "Marcus"
        assert "stoic" in card.personality
        assert "mercenary" in card.description.lower()

    def test_to_persona_spec(self):
        card = CharacterCard(
            name="Luna",
            description="A brilliant AI researcher",
            personality="Analytical, warm",
            system_prompt="Always be helpful\nNever give medical advice",
        )
        spec = card.to_persona_spec()
        assert spec.name == "Luna"
        assert spec.identity == "A brilliant AI researcher"
        assert spec.tone == "Analytical, warm"
        assert len(spec.behavioral_rules) == 2


# ── Full Pipeline (no LLM) ──


class TestFullPipeline:
    def test_persona_to_system_prompt_pipeline(self):
        """PersonaSpec 생성 → OCEAN → 감정 초기화 → 이벤트 → 렌더링 전체 파이프라인."""
        # 1. Create persona
        spec = PersonaSpec(
            name="Minsoo",
            ocean=OceanTraits(
                openness=0.8,
                conscientiousness=0.7,
                extraversion=0.3,
                agreeableness=0.9,
                neuroticism=0.2,
            ),
            l0_innate=PersonaLayer(traits={"age": "28", "occupation": "Software Engineer"}),
            tone="Precise but warm",
            values=["Code quality", "Knowledge sharing"],
            behavioral_rules=["Include code examples", "Say 'I don't know' when uncertain"],
        )

        # 2. Init emotion from OCEAN
        spec.init_emotion_from_ocean()
        assert spec.emotion is not None
        initial_emotion = spec.emotion.closest_emotion()

        # 3. Apply negative event
        spec.apply_event((-0.5, 0.4, -0.3))
        post_event_emotion = spec.emotion.closest_emotion()

        # 4. Render system prompt
        renderer = PersonaRenderer(spec)
        prompt = renderer.render_full()

        # Verify all components present
        assert "Minsoo" in prompt
        assert "Software Engineer" in prompt
        assert "OCEAN" in prompt
        assert "Precise but warm" in prompt
        assert "Code quality" in prompt
        assert "Include code examples" in prompt
        assert "Emotional State" in prompt
        assert post_event_emotion in prompt

        # 5. Decay emotion
        for _ in range(20):
            spec.decay_emotion(rate=0.2)
        recovered_emotion = spec.emotion.closest_emotion()

        # 6. Re-render — emotion should have changed
        prompt2 = renderer.render_full()
        assert prompt2 != prompt  # emotion shifted

    def test_character_card_to_brain_pipeline(self):
        """W++ Card → PersonaSpec → Renderer 파이프라인."""
        card = CharacterCard.from_wpp('''
        [character("Nova")
        {
          Personality("logical" + "empathetic" + "witty")
          Age("30")
          Occupation("Data Scientist")
          Speech("uses analogies" + "concise")
        }]
        ''')
        spec = card.to_persona_spec()
        spec.ocean = OceanTraits(openness=0.85, conscientiousness=0.6, extraversion=0.5, agreeableness=0.75, neuroticism=0.3)
        spec.init_emotion_from_ocean()

        renderer = PersonaRenderer(spec)
        prompt = renderer.render_full()

        assert spec.name == "Nova"
        assert "OCEAN" in prompt
        assert "Emotional State" in prompt
