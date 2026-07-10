"""v0.13.0 — persona forge: description → typed config, judge loop, universal mount."""
from __future__ import annotations

from agethos.brain import Brain
from agethos.forge import (
    ForgeReport,
    deterministic_judge,
    draft_spec,
    estimate_ocean,
    forge,
    judge_spec,
    plan_vectors,
)
from agethos.llm.base import LLMAdapter
from agethos.models import DecisionStyle, MoralFoundation, OceanTraits, PersonaSpec
from agethos.steering import MockSteeringBackend

KO_DESC = "까칠하지만 속정 깊은 30대 시니어 백엔드 개발자. 꼼꼼하고 직설적이며, 리뷰는 냉소적이지만 후배는 챙긴다."
EN_DESC = "An outgoing, talkative barista who is anxious about the future but endlessly curious."


class StubLLM(LLMAdapter):
    """Canned JSON responses consumed in order (the last one repeats)."""

    def __init__(self, json_responses: list[dict] | None = None, text: str = "ok"):
        self._json = list(json_responses or [])
        self._text = text
        self.json_calls: list[tuple[str, str]] = []

    async def generate(self, system_prompt: str = "", user_prompt: str = "", temperature: float = 0.7) -> str:
        return self._text

    async def generate_json(self, system_prompt: str = "", user_prompt: str = "", temperature: float = 0.2) -> dict:
        self.json_calls.append((system_prompt, user_prompt))
        if len(self._json) > 1:
            return self._json.pop(0)
        return self._json[0] if self._json else {}


# ── (A) lexicon + deterministic compiler ──

def test_estimate_ocean_korean():
    scores, evidence = estimate_ocean(KO_DESC)
    assert scores["conscientiousness"] > 0.5          # 꼼꼼
    assert scores["agreeableness"] < 0.5              # 까칠+직설+냉소 > 속정
    assert "까칠" in evidence["agreeableness"]


def test_estimate_ocean_english():
    scores, _ = estimate_ocean(EN_DESC)
    assert scores["extraversion"] > 0.5               # outgoing, talkative
    assert scores["neuroticism"] > 0.5                # anxious
    assert scores["openness"] > 0.5                   # curious


async def test_deterministic_draft_spec():
    spec = await draft_spec(KO_DESC, llm=None, name="Minsoo")
    assert spec.name == "Minsoo"
    assert spec.ocean is not None and spec.ocean.agreeableness < 0.5
    assert spec.seed_memory == KO_DESC
    assert spec.l0_innate.traits.get("age") == "30"
    assert spec.l0_innate.traits.get("occupation") == "개발자"


async def test_pin_survives_forge():
    result = await forge(KO_DESC, pin={"tone": "warm and slow", "ocean": {"E": 0.9}})
    assert result.spec.tone == "warm and slow"
    assert result.spec.ocean.extraversion == 0.9


# ── (B) LLM compiler coercion ──

async def test_llm_draft_coerces_invalid_values():
    stub = StubLLM([{
        "name": "Luna",
        "identity": "You are Luna, a blunt senior engineer.",
        "tone": "dry, sarcastic, secretly caring",
        "ocean": {"openness": 1.5, "agreeableness": -0.2, "conscientiousness": 0.9},
        "moral_values": ["care", "empathy"],           # "empathy" is not a MoralFoundation
        "schwartz_values": ["benevolence", "growth"],  # "growth" is invalid
        "decision_style": "vibes",                     # invalid → dropped
        "innate": {"age": 34, "occupation": "engineer"},
        "emotion_label": "calm",
        "seed_memory": "Ten years of backend work.",
    }])
    spec = await draft_spec(EN_DESC, llm=stub)
    assert spec.name == "Luna"
    assert spec.ocean.openness == 1.0 and spec.ocean.agreeableness == 0.0   # clamped
    assert spec.moral_values == [MoralFoundation.CARE]
    assert [v.value for v in spec.schwartz_values] == ["benevolence"]
    assert spec.decision_style is None
    assert spec.l0_innate.traits["age"] == "34"
    assert spec.emotion is not None and spec.emotion.closest_emotion() == "calm"


async def test_llm_draft_valid_decision_style():
    stub = StubLLM([{"name": "A", "decision_style": "analytical"}])
    spec = await draft_spec("analytical person", llm=stub)
    assert spec.decision_style == DecisionStyle.ANALYTICAL


# ── (C) judge ──

def test_deterministic_judge_empty_vs_full():
    empty = PersonaSpec(name="X")
    full = PersonaSpec(
        name="X", identity=KO_DESC, tone="까칠, 직설", seed_memory=KO_DESC,
        ocean=OceanTraits(agreeableness=0.2, conscientiousness=0.8),
    )
    r_empty = deterministic_judge(KO_DESC, empty)
    r_full = deterministic_judge(KO_DESC, full)
    assert r_full.overall > r_empty.overall
    assert set(r_empty.weak()) >= {"identity", "ocean", "tone", "background"}
    assert "identity" not in r_full.weak()


async def test_llm_judge_parses_report():
    stub = StubLLM([{
        "facets": {"identity": {"score": 0.9}, "ocean": {"score": 0.4, "issue": "A too high"}},
        "overall": 0.65,
    }])
    report = await judge_spec(KO_DESC, PersonaSpec(name="X"), llm=stub)
    assert isinstance(report, ForgeReport) and report.overall == 0.65
    assert "ocean" in report.weak() and report.issues()["ocean"] == "A too high"


# ── (D) the forge loop ──

async def test_forge_offline_runs_and_scores():
    result = await forge(KO_DESC, name="Minsoo")
    assert result.rounds >= 1 and result.trace
    assert result.report.overall > 0.5
    assert result.spec.ocean is not None
    assert "Minsoo" in result.render()


async def test_forge_converges_first_round():
    compiler = StubLLM([{"name": "A", "identity": "x", "tone": "y", "ocean": {"openness": 0.9},
                         "seed_memory": "z"}])
    judge = StubLLM([{"facets": {f: {"score": 0.9} for f in
                                 ("identity", "ocean", "tone", "background", "values",
                                  "rules", "constraints", "style")},
                      "overall": 0.9}])
    result = await forge(EN_DESC, llm=compiler, judge_llm=judge)
    assert result.converged and result.rounds == 1


async def test_forge_repairs_weak_facets():
    compiler = StubLLM([
        {"name": "A", "identity": "an engineer", "ocean": {"openness": 0.8}, "seed_memory": "s"},
        {"tone": "gruff but caring"},                  # repair round returns only weak field
    ])
    judge = StubLLM([
        {"facets": {"tone": {"score": 0.1, "issue": "missing"},
                    "identity": {"score": 0.9}, "ocean": {"score": 0.9},
                    "background": {"score": 0.9}, "values": {"score": 0.9},
                    "rules": {"score": 0.9}, "constraints": {"score": 0.9},
                    "style": {"score": 0.9}}, "overall": 0.6},
        {"facets": {f: {"score": 0.95} for f in
                    ("identity", "ocean", "tone", "background", "values",
                     "rules", "constraints", "style")}, "overall": 0.95},
    ])
    result = await forge(EN_DESC, llm=compiler, judge_llm=judge)
    assert result.converged and result.rounds == 2
    assert result.spec.tone == "gruff but caring"
    assert result.spec.identity == "an engineer"       # untouched facet preserved
    assert result.trace[0].weak == ["tone"]


# ── (E) steering plan → activation layer ──

async def test_steering_plan_and_vectors():
    result = await forge("creative and curious but very blunt", name="V")
    plan = result.steering_plan(threshold=0.1)
    traits = {p.trait: p for p in plan}
    assert traits["openness"].direction == 1
    assert traits["agreeableness"].direction == -1
    vecs = plan_vectors(plan, MockSteeringBackend())
    assert len(vecs) == len(plan)
    assert all(any(abs(x) > 0 for x in v.vector) for v in vecs)


# ── (F) universal mount ──

async def test_brain_from_description():
    stub = StubLLM([{
        "name": "Hana", "identity": "You are Hana.", "tone": "bright",
        "ocean": {"extraversion": 0.9}, "seed_memory": "Grew up in Busan.",
    }, {"facets": {f: {"score": 0.9} for f in
                   ("identity", "ocean", "tone", "background", "values",
                    "rules", "constraints", "style")}, "overall": 0.9}])
    brain = await Brain.from_description(EN_DESC, llm=stub)
    assert brain.forge_result is not None and brain.forge_result.converged
    assert brain._persona.name == "Hana"
    assert "Hana" in brain._renderer.render_iss()
