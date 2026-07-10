"""Judge panel + multi-sample forging — variance reduction for the forge.

One LLM pass is a single draw from a noisy distribution: a draft can miss facets and a
single judge can be miscalibrated. Sampling N independent drafts and scoring each with a
panel of differently-lensed judges, median-aggregated, reduces both variances
(self-consistency: Wang et al. 2022; panel of LLM judges: Verga et al. 2024). Facet-level
grafting then composes the best-scoring parts of the candidates into one config.
"""
from __future__ import annotations

from statistics import median

from agethos.concurrency import gather_bounded
from agethos.forge.judge import _FACETS, FacetScore, ForgeReport, judge_spec
from agethos.llm.base import LLMAdapter
from agethos.models import PersonaSpec

# each panel member judges through a different failure-mode lens
JUDGE_LENSES = [
    "FIDELITY — penalize config values that contradict or exaggerate the description.",
    "COVERAGE — penalize description content that never made it into any config field.",
    "OVERREACH — penalize invented details the description neither states nor implies.",
]

# multi-sample drafts vary the reading, not the schema
DRAFT_VARIANTS = [
    "",
    "Alternative reading: prioritize what the description implies over what it literally states.",
    "Alternative reading: stay strictly literal; extract only explicit statements.",
]

# judge facet → PersonaSpec fields it covers (graft copies whole facets)
FACET_FIELDS: dict[str, tuple[str, ...]] = {
    "identity": ("identity",),
    "ocean": ("ocean",),
    "tone": ("tone",),
    "background": ("seed_memory", "l0_innate"),
    "values": ("values", "moral_values", "schwartz_values"),
    "rules": ("behavioral_rules",),
    "constraints": ("boundaries", "hard_constraints", "soft_preferences"),
    "style": ("conversation_style", "decision_style"),
}


def aggregate_reports(reports: list[ForgeReport]) -> ForgeReport:
    """Median-aggregate a panel's reports (facet-wise + overall); keeps the most critical issue."""
    if not reports:
        return ForgeReport()
    if len(reports) == 1:
        return reports[0]
    facets: list[FacetScore] = []
    for facet in _FACETS:
        entries = [f for r in reports for f in r.facets if f.facet == facet]
        if not entries:
            continue
        worst = min(entries, key=lambda f: f.score)
        facets.append(FacetScore(
            facet=facet,
            score=round(median(f.score for f in entries), 4),
            issue=worst.issue,
        ))
    return ForgeReport(
        overall=round(median(r.overall for r in reports), 4),
        facets=facets,
    )


async def panel_judge(
    description: str,
    spec: PersonaSpec,
    llm: LLMAdapter | None = None,
    judges: int = 1,
) -> ForgeReport:
    """Judge with a panel of differently-lensed judges; single judge when judges<=1 or no LLM."""
    if llm is None or judges <= 1:
        return await judge_spec(description, spec, llm=llm)
    reports = await gather_bounded([
        judge_spec(description, spec, llm=llm, lens=JUDGE_LENSES[i % len(JUDGE_LENSES)])
        for i in range(judges)
    ])
    return aggregate_reports(reports)


def graft(
    candidates: list[tuple[PersonaSpec, ForgeReport]],
    margin: float = 0.05,
) -> PersonaSpec:
    """Compose the winner with any facet another candidate scores meaningfully higher on.

    Starts from the best-overall spec; for each facet, if a rival beats the winner's facet
    score by more than ``margin``, the rival's fields for that facet replace the winner's."""
    winner_spec, winner_report = max(candidates, key=lambda c: c[1].overall)
    out = winner_spec.model_copy(deep=True)
    win_scores = {f.facet: f.score for f in winner_report.facets}

    for facet, fields in FACET_FIELDS.items():
        donor, best_score = None, win_scores.get(facet, 0.0)
        for spec, report in candidates:
            if spec is winner_spec:
                continue
            score = next((f.score for f in report.facets if f.facet == facet), 0.0)
            if score > best_score + margin:
                donor, best_score = spec, score
        if donor is not None:
            frozen = donor.model_copy(deep=True)
            for field in fields:
                setattr(out, field, getattr(frozen, field))
    return out
