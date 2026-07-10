"""GPU-free steering for black-box LLMs — best-of-n attribute re-ranking.

Activation steering needs weight access; API models expose none. The GPU-free
counterpart: sample n candidate responses and re-rank them with an attribute model —
the PPLM/FUDGE idea (attribute-model-guided generation: Dathathri et al. 2020,
Yang & Klein 2021) applied at the sequence level as best-of-n selection (rejection
sampling: Stiennon et al. 2020; Nakano et al. 2021). The deterministic attribute model
scores trait-pole vocabulary aligned with the steering plan; an optional LLM judge adds
a semantic persona-fit score. Works against any provider, pure Python.
"""
from __future__ import annotations

import re

from pydantic import BaseModel, Field

from agethos.concurrency import gather_bounded
from agethos.llm.base import LLMAdapter
from agethos.steering.contrastive import _TRAIT_POLES
from agethos.steering.plan import SteeringIntent

_WORD = re.compile(r"[a-z']+")

# trait → (high-pole words, low-pole words), split from the contrastive pole phrases
_POLE_WORDS: dict[str, tuple[set[str], set[str]]] = {
    trait: (
        {w for w in _WORD.findall(hi.lower()) if len(w) > 3},
        {w for w in _WORD.findall(lo.lower()) if len(w) > 3},
    )
    for trait, (hi, lo) in _TRAIT_POLES.items()
}


def attribute_score(text: str, plan: list[SteeringIntent]) -> float:
    """Deterministic attribute-model score: pole-word occurrences aligned with the plan.

    For each intent, count trait-high vs trait-low vocabulary in the text; the margin,
    signed by the intent's direction and weighted by its strength, accumulates."""
    words = _WORD.findall(text.lower())
    if not words or not plan:
        return 0.0
    counts: dict[str, int] = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    score = 0.0
    for intent in plan:
        hi, lo = _POLE_WORDS.get(intent.trait, (set(), set()))
        hi_hits = sum(counts.get(w, 0) for w in hi)
        lo_hits = sum(counts.get(w, 0) for w in lo)
        score += intent.strength * intent.direction * (hi_hits - lo_hits)
    return round(score, 4)


class RankedCandidate(BaseModel):
    text: str
    attribute: float = 0.0
    judge: float | None = None
    combined: float = 0.0


class RerankResult(BaseModel):
    best: str
    candidates: list[RankedCandidate] = Field(default_factory=list)


_JUDGE_SYSTEM = """You score how strongly a response expresses a target personality.
Respond JSON: {"fit": 0.0-1.0}"""


async def steered_generate(
    llm: LLMAdapter,
    system_prompt: str,
    user_prompt: str,
    plan: list[SteeringIntent],
    n: int = 4,
    temperature: float = 0.9,
    history: list[dict[str, str]] | None = None,
    judge_llm: LLMAdapter | None = None,
    judge_weight: float = 3.0,
) -> RerankResult:
    """Sample n candidates, re-rank by attribute score (+ optional judge), return the best.

    ``history`` routes through ``generate_with_history`` so it drops into a chat loop."""
    if history is not None:
        coros = [llm.generate_with_history(system_prompt, list(history), user_prompt, temperature)
                 for _ in range(n)]
    else:
        coros = [llm.generate(system_prompt, user_prompt, temperature) for _ in range(n)]
    texts = await gather_bounded(coros)

    candidates = [RankedCandidate(text=t, attribute=attribute_score(t, plan)) for t in texts]

    if judge_llm is not None:
        traits = ", ".join(
            f"{p.trait} {'HIGH' if p.direction > 0 else 'LOW'} (strength {p.strength})" for p in plan
        )
        async def _judge(c: RankedCandidate) -> float:
            data = await judge_llm.generate_json(
                _JUDGE_SYSTEM,
                f"Target personality: {traits}\n\nResponse:\n{c.text}",
            )
            try:
                return max(0.0, min(1.0, float(data.get("fit", 0.0))))
            except (TypeError, ValueError):
                return 0.0
        fits = await gather_bounded([_judge(c) for c in candidates])
        for c, fit in zip(candidates, fits):
            c.judge = fit

    for c in candidates:
        c.combined = round(c.attribute + judge_weight * (c.judge or 0.0), 4)

    best = max(candidates, key=lambda c: c.combined)
    return RerankResult(best=best.text, candidates=candidates)
