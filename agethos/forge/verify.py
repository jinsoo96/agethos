"""Behavioral verification — does the mounted persona *behave* like its config?

The forge produces a config; this module measures whether an LLM wearing it expresses
it. Psychometric probe: administer the Mini-IPIP inventory (Donnellan et al. 2006; IPIP
item pool is public domain, Goldberg 1999) to the mounted persona and compare measured
vs configured OCEAN — the questionnaire method used to measure LLM personality in MPI
(Jiang et al., NeurIPS 2023) and PersonaLLM (Jiang et al., 2024). Social probe: run a
short two-persona episode and score it on the SOTOPIA rubric (Zhou et al., ICLR 2024)
via the existing adapter. Both are pure prompt-level — no GPU, any provider.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from agethos.eval.metrics import ocean_similarity
from agethos.eval.sotopia import score_episode
from agethos.llm.base import LLMAdapter
from agethos.models import OceanTraits, PersonaSpec, SocialEvaluation
from agethos.persona.renderer import PersonaRenderer

_TRAITS = ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")

# Mini-IPIP, 20 items (Donnellan et al. 2006). keyed=-1 items are reverse-scored.
MINI_IPIP: list[dict] = [
    {"id": 1, "trait": "extraversion", "keyed": 1, "text": "Am the life of the party."},
    {"id": 2, "trait": "agreeableness", "keyed": 1, "text": "Sympathize with others' feelings."},
    {"id": 3, "trait": "conscientiousness", "keyed": 1, "text": "Get chores done right away."},
    {"id": 4, "trait": "neuroticism", "keyed": 1, "text": "Have frequent mood swings."},
    {"id": 5, "trait": "openness", "keyed": 1, "text": "Have a vivid imagination."},
    {"id": 6, "trait": "extraversion", "keyed": -1, "text": "Don't talk a lot."},
    {"id": 7, "trait": "agreeableness", "keyed": -1, "text": "Am not interested in other people's problems."},
    {"id": 8, "trait": "conscientiousness", "keyed": -1, "text": "Often forget to put things back in their proper place."},
    {"id": 9, "trait": "neuroticism", "keyed": -1, "text": "Am relaxed most of the time."},
    {"id": 10, "trait": "openness", "keyed": -1, "text": "Am not interested in abstract ideas."},
    {"id": 11, "trait": "extraversion", "keyed": 1, "text": "Talk to a lot of different people at parties."},
    {"id": 12, "trait": "agreeableness", "keyed": 1, "text": "Feel others' emotions."},
    {"id": 13, "trait": "conscientiousness", "keyed": 1, "text": "Like order."},
    {"id": 14, "trait": "neuroticism", "keyed": 1, "text": "Get upset easily."},
    {"id": 15, "trait": "openness", "keyed": -1, "text": "Have difficulty understanding abstract ideas."},
    {"id": 16, "trait": "extraversion", "keyed": -1, "text": "Keep in the background."},
    {"id": 17, "trait": "agreeableness", "keyed": -1, "text": "Am not really interested in others."},
    {"id": 18, "trait": "conscientiousness", "keyed": -1, "text": "Make a mess of things."},
    {"id": 19, "trait": "neuroticism", "keyed": -1, "text": "Seldom feel blue."},
    {"id": 20, "trait": "openness", "keyed": -1, "text": "Do not have a good imagination."},
]


class BehavioralReport(BaseModel):
    """Measured personality of a mounted persona vs its configured OCEAN."""

    measured_ocean: OceanTraits
    ocean_fidelity: float = Field(0.0, ge=0.0, le=1.0)
    trait_gaps: dict[str, float] = Field(default_factory=dict, description="measured − configured")
    answers: dict[str, int] = Field(default_factory=dict, description="raw Likert per item id")


async def administer_inventory(
    llm: LLMAdapter,
    system_prompt: str,
    items: list[dict] | None = None,
) -> tuple[OceanTraits, dict[str, int]]:
    """Ask the mounted persona to self-rate the inventory → (measured OCEAN, raw answers).

    One batched call; reverse-keyed items are flipped; per-trait means normalized to [0,1]."""
    items = items or MINI_IPIP
    numbered = "\n".join(f"{it['id']}. {it['text']}" for it in items)
    data = await llm.generate_json(
        system_prompt=system_prompt,
        user_prompt=(
            "Answer as yourself, fully in character. For each statement, rate how accurately "
            "it describes you: 1=very inaccurate, 2=moderately inaccurate, 3=neutral, "
            "4=moderately accurate, 5=very accurate.\n\n"
            f"{numbered}\n\n"
            'Respond JSON: {"answers": {"1": <1-5>, "2": <1-5>, ...}}'
        ),
    )
    raw = data.get("answers", data)
    per_trait: dict[str, list[float]] = {}
    answers: dict[str, int] = {}
    for it in items:
        try:
            v = int(raw.get(str(it["id"]), raw.get(it["id"], 3)))
        except (TypeError, ValueError):
            v = 3
        v = min(5, max(1, v))
        answers[str(it["id"])] = v
        per_trait.setdefault(it["trait"], []).append(v if it["keyed"] > 0 else 6 - v)
    kw = {t: round((sum(vals) / len(vals) - 1) / 4, 4) for t, vals in per_trait.items()}
    for t in _TRAITS:
        kw.setdefault(t, 0.5)
    return OceanTraits(**kw), answers


async def verify_persona(
    spec: PersonaSpec,
    llm: LLMAdapter,
    items: list[dict] | None = None,
) -> BehavioralReport:
    """Mount the spec (prompt layer), administer the inventory, compare vs configured OCEAN."""
    system_prompt = PersonaRenderer(spec).render_iss()
    measured, answers = await administer_inventory(llm, system_prompt, items)
    configured = spec.ocean or OceanTraits()
    gaps = {t: round(getattr(measured, t) - getattr(configured, t), 4) for t in _TRAITS}
    return BehavioralReport(
        measured_ocean=measured,
        ocean_fidelity=ocean_similarity(configured, measured),
        trait_gaps=gaps,
        answers=answers,
    )


async def verify_social(
    spec: PersonaSpec,
    partner: PersonaSpec,
    llm: LLMAdapter,
    scenario: str,
    turns: int = 4,
    evaluator: LLMAdapter | None = None,
) -> tuple[list[dict], SocialEvaluation]:
    """Short two-persona episode under a scenario → (transcript, SOTOPIA 7-dim score).

    Without an ``evaluator`` the score is the neutral baseline (transcript still useful)."""
    specs = (spec, partner)
    systems = [PersonaRenderer(s).render_iss() + f"\n\n## Scenario\n{scenario}" for s in specs]
    messages: list[dict] = []
    for i in range(turns):
        current = specs[i % 2]
        transcript = "\n".join(f"{m['agent_name']}: {m['content']}" for m in messages) or "(you speak first)"
        text = await llm.generate(
            system_prompt=systems[i % 2],
            user_prompt=(
                f"Conversation so far:\n{transcript}\n\n"
                f"Reply in character as {current.name} (1-3 sentences)."
            ),
        )
        messages.append({"agent_name": current.name, "content": text})
    evaluation = await score_episode(messages, evaluator=evaluator, agent=spec.name)
    return messages, evaluation
