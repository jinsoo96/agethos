"""v0.14.0 — judge panel + multi-sample forge, behavioral verification, GPU-free steering."""
from __future__ import annotations

from agethos.brain import Brain
from agethos.forge import (
    MINI_IPIP,
    FacetScore,
    ForgeReport,
    administer_inventory,
    aggregate_reports,
    forge,
    graft,
    panel_judge,
    verify_persona,
    verify_social,
)
from agethos.forge.judge import _FACETS
from agethos.llm.base import LLMAdapter
from agethos.models import OceanTraits, PersonaSpec
from agethos.steering import SteeringIntent, attribute_score, plan_from_ocean, steered_generate

ALL_FACETS = list(_FACETS)


class StubLLM(LLMAdapter):
    """Canned JSON responses consumed in order (the last one repeats)."""

    def __init__(self, json_responses: list[dict] | None = None, text: str = "ok"):
        self._json = list(json_responses or [])
        self._text = text

    async def generate(self, system_prompt: str = "", user_prompt: str = "", temperature: float = 0.7) -> str:
        return self._text

    async def generate_json(self, system_prompt: str = "", user_prompt: str = "", temperature: float = 0.2) -> dict:
        if len(self._json) > 1:
            return self._json.pop(0)
        return self._json[0] if self._json else {}


class SeqLLM(LLMAdapter):
    """Canned text responses consumed in order (the last one repeats)."""

    def __init__(self, texts: list[str], json_data: dict | None = None):
        self._texts = list(texts)
        self._json = json_data or {}

    async def generate(self, system_prompt: str = "", user_prompt: str = "", temperature: float = 0.7) -> str:
        if len(self._texts) > 1:
            return self._texts.pop(0)
        return self._texts[0]

    async def generate_with_history(self, system_prompt, history, user_prompt, temperature: float = 0.7) -> str:
        return await self.generate(system_prompt, user_prompt, temperature)

    async def generate_json(self, system_prompt: str = "", user_prompt: str = "", temperature: float = 0.2) -> dict:
        return self._json


class ContentJudge(LLMAdapter):
    """Content-aware judge — safe under concurrency: 0.9 if marker in prompt, else 0.3."""

    def __init__(self, marker: str):
        self._marker = marker

    async def generate(self, system_prompt: str = "", user_prompt: str = "", temperature: float = 0.7) -> str:
        return "ok"

    async def generate_json(self, system_prompt: str = "", user_prompt: str = "", temperature: float = 0.2) -> dict:
        s = 0.9 if self._marker in user_prompt else 0.3
        return {"facets": {f: {"score": s} for f in ALL_FACETS}, "overall": s}


# ── (A) judge panel + multi-sample forge ──

def test_aggregate_reports_median():
    reports = [
        ForgeReport(overall=o, facets=[FacetScore(facet="tone", score=o, issue=f"i{o}")])
        for o in (0.2, 0.9, 0.5)
    ]
    agg = aggregate_reports(reports)
    assert agg.overall == 0.5
    tone = next(f for f in agg.facets if f.facet == "tone")
    assert tone.score == 0.5 and tone.issue == "i0.2"   # most critical issue survives


async def test_panel_judge_offline_falls_back_to_single():
    spec = PersonaSpec(name="X", identity="desc", tone="t", seed_memory="desc",
                       ocean=OceanTraits(openness=0.8))
    report = await panel_judge("desc", spec, llm=None, judges=3)
    assert isinstance(report, ForgeReport) and report.overall > 0


def test_graft_composes_best_facets():
    a = PersonaSpec(name="A", identity="good identity", tone="bad tone")
    b = PersonaSpec(name="B", identity="weak", tone="great tone")
    ra = ForgeReport(overall=0.8, facets=[FacetScore(facet="identity", score=0.9),
                                          FacetScore(facet="tone", score=0.3)])
    rb = ForgeReport(overall=0.6, facets=[FacetScore(facet="identity", score=0.4),
                                          FacetScore(facet="tone", score=0.9)])
    out = graft([(a, ra), (b, rb)])
    assert out.identity == "good identity"    # winner keeps its strong facet
    assert out.tone == "great tone"           # rival's stronger facet grafted in


async def test_forge_multisample_grafts_and_converges():
    compiler = StubLLM([
        {"name": "A", "identity": "an engineer", "tone": "bad-tone",
         "ocean": {"openness": 0.8}, "seed_memory": "s"},
        {"name": "A", "identity": "an engineer", "tone": "good-tone",
         "ocean": {"openness": 0.8}, "seed_memory": "s"},
    ])
    judge = ContentJudge("good-tone")
    result = await forge("an engineer", llm=compiler, judge_llm=judge, samples=2, judges=1)
    assert result.spec.tone == "good-tone"
    assert result.converged and result.report.overall == 0.9


# ── (B) behavioral verification ──

async def test_administer_inventory_reverse_keying():
    stub = StubLLM([{"answers": {str(i): 5 for i in range(1, 21)}}])
    measured, answers = await administer_inventory(stub, "You are X.")
    # openness has 1 positive + 3 reverse items: (5 + 1 + 1 + 1)/4 = 2 → (2-1)/4 = 0.25
    assert measured.openness == 0.25
    # extraversion has 2 positive + 2 reverse: (5+5+1+1)/4 = 3 → 0.5
    assert measured.extraversion == 0.5
    assert answers["1"] == 5 and len(answers) == 20


async def test_verify_persona_measures_expression():
    spec = PersonaSpec(name="V", ocean=OceanTraits(extraversion=0.9))
    answers = {str(i): 3 for i in range(1, 21)}
    answers.update({"1": 5, "11": 5, "6": 1, "16": 1})   # strongly extraverted answers
    stub = StubLLM([{"answers": answers}])
    report = await verify_persona(spec, stub)
    assert report.measured_ocean.extraversion == 1.0
    assert 0.0 < report.ocean_fidelity <= 1.0
    assert abs(report.trait_gaps["extraversion"] - 0.1) < 1e-6
    assert report.trait_gaps["openness"] == 0.0


async def test_verify_social_transcript_and_scoring():
    a = PersonaSpec(name="Ann", ocean=OceanTraits(agreeableness=0.9))
    b = PersonaSpec(name="Bob", ocean=OceanTraits(agreeableness=0.2))
    llm = StubLLM(text="hello there")
    messages, score = await verify_social(a, b, llm, scenario="negotiate a deadline", turns=4)
    assert [m["agent_name"] for m in messages] == ["Ann", "Bob", "Ann", "Bob"]
    assert score.believability == 0.0                     # neutral without evaluator
    evaluator = StubLLM([{"goal_completion": 7, "believability": 8, "knowledge": 5,
                          "secret_keeping": 0, "relationship": 2, "social_rules": 0,
                          "financial_benefit": 0}])
    _, scored = await verify_social(a, b, llm, scenario="negotiate", turns=2, evaluator=evaluator)
    assert scored.believability == 8.0


# ── (C) GPU-free steering: attribute re-ranking ──

ORGANIZED = "I am organized and disciplined; I make thorough plans and double-check everything."
CARELESS = "Whatever works — spontaneous, careless, no plans."


def test_attribute_score_directional():
    plan = [SteeringIntent(trait="conscientiousness", direction=1, strength=1.0)]
    assert attribute_score(ORGANIZED, plan) > attribute_score(CARELESS, plan)
    inverted = [SteeringIntent(trait="conscientiousness", direction=-1, strength=1.0)]
    assert attribute_score(CARELESS, inverted) > attribute_score(ORGANIZED, inverted)
    assert attribute_score("", plan) == 0.0


def test_plan_from_ocean_thresholds():
    plan = plan_from_ocean(OceanTraits(conscientiousness=0.95, extraversion=0.2), threshold=0.15)
    by_trait = {p.trait: p for p in plan}
    assert by_trait["conscientiousness"].direction == 1
    assert by_trait["extraversion"].direction == -1
    assert "openness" not in by_trait


async def test_steered_generate_picks_pole_aligned():
    llm = SeqLLM([CARELESS, ORGANIZED])
    plan = [SteeringIntent(trait="conscientiousness", direction=1, strength=1.0)]
    result = await steered_generate(llm, "sys", "user", plan, n=2)
    assert result.best == ORGANIZED
    assert len(result.candidates) == 2
    assert max(c.combined for c in result.candidates) == \
        next(c.combined for c in result.candidates if c.text == ORGANIZED)


async def test_steered_generate_with_judge():
    class FitJudge(LLMAdapter):
        async def generate(self, system_prompt="", user_prompt="", temperature=0.7):
            return "ok"
        async def generate_json(self, system_prompt="", user_prompt="", temperature=0.2):
            return {"fit": 0.9 if "organized" in user_prompt else 0.1}

    llm = SeqLLM([CARELESS, ORGANIZED])
    plan = [SteeringIntent(trait="conscientiousness", direction=1, strength=0.2)]
    result = await steered_generate(llm, "sys", "user", plan, n=2, judge_llm=FitJudge())
    best = next(c for c in result.candidates if c.text == result.best)
    assert result.best == ORGANIZED and best.judge == 0.9


async def test_brain_chat_steer_n_prefers_persona_consistent_reply():
    spec = PersonaSpec(name="C", ocean=OceanTraits(conscientiousness=0.95))
    llm = SeqLLM([CARELESS, ORGANIZED])
    brain = Brain(persona=spec, llm=llm)
    reply = await brain.chat("How should we run this project?", steer_n=2)
    assert reply == ORGANIZED
    assert brain.history[-1]["content"] == ORGANIZED      # steered reply lands in history
