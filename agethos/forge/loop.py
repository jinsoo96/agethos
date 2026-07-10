"""The forge loop — draft → judge → targeted repair, until the config converges.

CONFIG-FORGE pattern applied to personality: the deliverable is a typed config
(``PersonaSpec``), the judge measures fidelity to the source description, and each
round regenerates only the facets that scored weak. The result mounts on any LLM
(prompt layer via the renderer, activation layer via a steering plan, framework layer
via transplant adapters).
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from agethos.forge.compiler import draft_spec, repair_spec
from agethos.forge.judge import ForgeReport, judge_spec
from agethos.llm.base import LLMAdapter
from agethos.models import PersonaSpec

_TRAIT_DIMS = ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")


class SteeringIntent(BaseModel):
    """One trait the forged persona wants steered at the activation level."""

    trait: str
    direction: int = Field(1, description="+1 = toward trait-high pole, -1 = trait-low")
    strength: float = Field(0.0, ge=0.0, le=1.0, description="|trait − 0.5| × 2")


class ForgeRound(BaseModel):
    round: int
    overall: float
    weak: list[str] = Field(default_factory=list)


class ForgeResult(BaseModel):
    """Forged persona + fidelity report + convergence trace."""

    spec: PersonaSpec
    report: ForgeReport
    rounds: int = 1
    converged: bool = False
    trace: list[ForgeRound] = Field(default_factory=list)

    def render(self) -> str:
        """Mount at the prompt layer — full system prompt for any chat LLM."""
        from agethos.persona.renderer import PersonaRenderer
        return PersonaRenderer(self.spec).render_full()

    def steering_plan(self, threshold: float = 0.15) -> list[SteeringIntent]:
        """Traits deviating from the 0.5 prior by ≥ threshold → activation-steering targets."""
        if self.spec.ocean is None:
            return []
        plan: list[SteeringIntent] = []
        for dim in _TRAIT_DIMS:
            v = getattr(self.spec.ocean, dim)
            if abs(v - 0.5) >= threshold:
                plan.append(SteeringIntent(
                    trait=dim,
                    direction=1 if v > 0.5 else -1,
                    strength=round(min(1.0, abs(v - 0.5) * 2), 4),
                ))
        return plan


def plan_vectors(plan: list[SteeringIntent], backend, layer: int = -1):
    """Steering plan → direction/strength-scaled PersonaVectors for open-weight models."""
    from agethos.steering import PersonaVector, extract_persona_vectors
    vecs = extract_persona_vectors(backend, traits=[p.trait for p in plan], layer=layer)
    by_trait = {v.trait: v for v in vecs}
    return [
        PersonaVector(
            trait=p.trait,
            vector=[x * p.direction * p.strength for x in by_trait[p.trait].vector],
            layer=layer,
        )
        for p in plan
    ]


async def forge(
    description: str,
    llm: LLMAdapter | None = None,
    judge_llm: LLMAdapter | None = None,
    name: str | None = None,
    base: PersonaSpec | None = None,
    pin: dict | None = None,
    max_rounds: int = 3,
    target: float = 0.85,
    weak_threshold: float = 0.7,
) -> ForgeResult:
    """Forge a persona config from a free-text description.

    Args:
        description: The personality, rough or detailed, any language.
        llm: Compiler LLM (None → deterministic lexicon path).
        judge_llm: Fidelity judge (defaults to ``llm``).
        name: Pin the persona's name.
        base: Existing spec to layer the forge onto.
        pin: Fields the forge must never overwrite (highest config layer).
        max_rounds: Judge/repair rounds before returning the best spec seen.
        target: Overall fidelity that counts as converged.
        weak_threshold: Facet score below this gets repaired next round.
    """
    judge_llm = judge_llm or llm
    spec = await draft_spec(description, llm=llm, name=name, base=base, pin=pin)

    trace: list[ForgeRound] = []
    best_spec, best_report = spec, None
    converged = False

    for rnd in range(1, max_rounds + 1):
        report = await judge_spec(description, spec, llm=judge_llm)
        weak = report.weak(weak_threshold)
        trace.append(ForgeRound(round=rnd, overall=report.overall, weak=weak))

        if best_report is None or report.overall > best_report.overall:
            best_spec, best_report = spec, report

        if report.overall >= target:
            converged = True
            break
        if rnd < max_rounds and weak:
            repaired = await repair_spec(
                description, spec, weak, issues=report.issues(), llm=llm, pin=pin,
            )
            if repaired.model_dump() == spec.model_dump():
                break  # repair changed nothing — further rounds are no-ops
            spec = repaired
        elif not weak:
            break  # nothing left to repair

    return ForgeResult(
        spec=best_spec,
        report=best_report,
        rounds=len(trace),
        converged=converged,
        trace=trace,
    )
