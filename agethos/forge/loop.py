"""The forge loop — draft → judge → targeted repair, until the config converges.

CONFIG-FORGE pattern applied to personality: the deliverable is a typed config
(``PersonaSpec``), the judge measures fidelity to the source description, and each
round regenerates only the facets that scored weak. Variance is reduced by sampling
multiple drafts and judging with a lensed panel (``samples`` / ``judges``, see
``agethos.forge.panel``). The result mounts on any LLM: prompt layer (renderer),
activation layer (steering plan → vectors) or black-box re-ranking
(``agethos.steering.rerank``), framework layer (transplant adapters) — and can be
behaviorally verified afterwards (``agethos.forge.verify``).
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from agethos.concurrency import gather_bounded
from agethos.forge.compiler import draft_spec, repair_spec
from agethos.forge.judge import ForgeReport
from agethos.forge.panel import DRAFT_VARIANTS, graft, panel_judge
from agethos.llm.base import LLMAdapter
from agethos.models import PersonaSpec
from agethos.steering.plan import SteeringIntent, plan_from_ocean, plan_vectors  # noqa: F401 (re-export)


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
        """Traits deviating from the 0.5 prior by ≥ threshold → steering targets."""
        if self.spec.ocean is None:
            return []
        return plan_from_ocean(self.spec.ocean, threshold=threshold)

    async def verify(self, llm: LLMAdapter, items: list[dict] | None = None):
        """Behavioral check: mount the spec and psychometrically probe it (Mini-IPIP)."""
        from agethos.forge.verify import verify_persona
        return await verify_persona(self.spec, llm, items=items)


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
    samples: int = 1,
    judges: int = 1,
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
        samples: Independent drafts to sample; the best facets are grafted together
            (self-consistency). Needs an LLM; 1 = single draft.
        judges: Panel size per judge pass (lensed judges, median-aggregated); 1 = single.
    """
    judge_llm = judge_llm or llm

    async def _judge(s: PersonaSpec) -> ForgeReport:
        return await panel_judge(description, s, llm=judge_llm, judges=judges)

    if llm is not None and samples > 1:
        drafts = await gather_bounded([
            draft_spec(description, llm=llm, name=name, base=base, pin=pin,
                       variant=DRAFT_VARIANTS[i % len(DRAFT_VARIANTS)])
            for i in range(samples)
        ])
        reports = await gather_bounded([_judge(d) for d in drafts])
        spec = graft(list(zip(drafts, reports)))
    else:
        spec = await draft_spec(description, llm=llm, name=name, base=base, pin=pin)

    trace: list[ForgeRound] = []
    best_spec, best_report = spec, None
    converged = False

    for rnd in range(1, max_rounds + 1):
        report = await _judge(spec)
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
