"""Forge judge — score how faithfully a PersonaSpec covers its source description.

Per-facet fidelity in [0,1] plus an issue note; the loop repairs only weak facets.
LLM judge is primary; the deterministic fallback scores coverage (facet filled +
lexical overlap with the description) so the loop runs and converges offline.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from agethos.eval.metrics import text_similarity
from agethos.llm.base import LLMAdapter
from agethos.models import PersonaSpec

# facet → (required, extractor). Optional facets score 0.5 when empty (absence in the
# description is a valid reason for absence in the config).
_FACETS: dict[str, bool] = {
    "identity": True,
    "ocean": True,
    "tone": True,
    "background": True,   # seed_memory + innate layer
    "values": False,      # values + moral/schwartz
    "rules": False,       # behavioral_rules
    "constraints": False, # boundaries + hard/soft
    "style": False,       # conversation_style + decision_style
}

_JUDGE_SYSTEM = """You judge how faithfully a typed persona config captures a personality
description. Score each facet 0.0-1.0 for fidelity AND coverage (does the config express
what the description says — no more, no less?). Facets:
identity, ocean, tone, background, values, rules, constraints, style.

Respond with a single JSON object:
{"facets": {"<facet>": {"score": 0.0-1.0, "issue": "<short issue or empty>"}, ...},
 "overall": 0.0-1.0}"""


class FacetScore(BaseModel):
    facet: str
    score: float = Field(0.0, ge=0.0, le=1.0)
    issue: str = ""


class ForgeReport(BaseModel):
    """Fidelity report for one judge pass."""

    overall: float = Field(0.0, ge=0.0, le=1.0)
    facets: list[FacetScore] = Field(default_factory=list)

    def weak(self, threshold: float = 0.7) -> list[str]:
        return [f.facet for f in self.facets if f.score < threshold]

    def issues(self) -> dict[str, str]:
        return {f.facet: f.issue for f in self.facets if f.issue}


def _facet_text(spec: PersonaSpec, facet: str) -> str:
    if facet == "identity":
        return spec.identity
    if facet == "ocean":
        return spec.ocean.to_prompt() if spec.ocean else ""
    if facet == "tone":
        return spec.tone
    if facet == "background":
        innate = " ".join(f"{k} {v}" for k, v in spec.l0_innate.traits.items())
        return f"{spec.seed_memory} {innate}".strip()
    if facet == "values":
        extra = [v.value for v in spec.moral_values] + [v.value for v in spec.schwartz_values]
        return " ".join(spec.values + extra)
    if facet == "rules":
        return " ".join(spec.behavioral_rules)
    if facet == "constraints":
        return " ".join(spec.boundaries + spec.hard_constraints + spec.soft_preferences)
    if facet == "style":
        ds = spec.decision_style.value if spec.decision_style else ""
        return f"{spec.conversation_style} {ds}".strip()
    return ""


def deterministic_judge(description: str, spec: PersonaSpec) -> ForgeReport:
    """Coverage heuristic: empty required facet → 0; filled → 0.6 + overlap bonus."""
    facets: list[FacetScore] = []
    for facet, required in _FACETS.items():
        text = _facet_text(spec, facet)
        if facet == "ocean" and spec.ocean is not None:
            # any deviation from the 0.5 prior = extracted signal
            dims = (spec.ocean.openness, spec.ocean.conscientiousness, spec.ocean.extraversion,
                    spec.ocean.agreeableness, spec.ocean.neuroticism)
            signal = max(abs(v - 0.5) for v in dims)
            facets.append(FacetScore(facet=facet, score=round(min(1.0, 0.6 + signal), 4),
                                     issue="" if signal > 0 else "no trait signal"))
        elif text:
            bonus = min(1.0, 5.0 * text_similarity(description, text))
            facets.append(FacetScore(facet=facet, score=round(0.6 + 0.4 * bonus, 4)))
        else:
            facets.append(FacetScore(
                facet=facet,
                score=0.0 if required else 0.5,
                issue="missing" if required else "empty (optional)",
            ))
    overall = round(sum(f.score for f in facets) / len(facets), 4)
    return ForgeReport(overall=overall, facets=facets)


async def judge_spec(
    description: str,
    spec: PersonaSpec,
    llm: LLMAdapter | None = None,
) -> ForgeReport:
    """Judge spec fidelity to the description (LLM-driven; deterministic fallback)."""
    if llm is None:
        return deterministic_judge(description, spec)

    user = (
        f"Personality description:\n{description}\n\n"
        f"Persona config:\n{spec.model_dump_json(exclude_none=True)}"
    )
    data = await llm.generate_json(_JUDGE_SYSTEM, user)
    facets: list[FacetScore] = []
    raw = data.get("facets", {})
    for facet in _FACETS:
        entry = raw.get(facet, {})
        try:
            score = max(0.0, min(1.0, float(entry.get("score", 0.0))))
        except (TypeError, ValueError):
            score = 0.0
        facets.append(FacetScore(facet=facet, score=score, issue=str(entry.get("issue", ""))))
    try:
        overall = max(0.0, min(1.0, float(data.get("overall"))))
    except (TypeError, ValueError):
        overall = round(sum(f.score for f in facets) / len(facets), 4) if facets else 0.0
    return ForgeReport(overall=overall, facets=facets)
