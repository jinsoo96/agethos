"""agethos.forge — free-text description → typed persona config, converged by a judge loop.

"성격을 시스템 프롬프트로 때우지 않는다": the forge compiles a description into
``PersonaSpec`` *config values* (OCEAN, policy inputs, constraints, seed memory), a judge
scores fidelity per facet, and weak facets are re-forged until the config converges —
then the one config mounts on any LLM at three layers: prompt (renderer), activation
(steering plan), framework (transplant adapters).
"""
from agethos.forge.compiler import coerce_draft, deterministic_draft, draft_spec, repair_spec
from agethos.forge.judge import FacetScore, ForgeReport, deterministic_judge, judge_spec
from agethos.forge.lexicon import estimate_ocean, extract_innate
from agethos.forge.loop import ForgeResult, ForgeRound, SteeringIntent, forge, plan_vectors

__all__ = [
    "FacetScore",
    "ForgeReport",
    "ForgeResult",
    "ForgeRound",
    "SteeringIntent",
    "coerce_draft",
    "deterministic_draft",
    "deterministic_judge",
    "draft_spec",
    "estimate_ocean",
    "extract_innate",
    "forge",
    "judge_spec",
    "plan_vectors",
    "repair_spec",
]
