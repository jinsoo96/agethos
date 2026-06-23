"""Sotopia social-intelligence adapter.

Sotopia (Zhou et al. ICLR 2024) evaluates social agents on a 7-dimension rubric.
agethos personas map natively onto Sotopia profiles (OCEAN, decision style, values,
secrets), and outcomes are scored with the built-in ``SocialEvaluation`` (the same
7 dimensions). Provide an LLM ``evaluator`` for real scoring; without one a neutral
baseline is returned.
"""
from __future__ import annotations

from agethos.models import PersonaSpec, SocialEvaluation


def persona_to_sotopia_profile(spec: PersonaSpec) -> dict:
    """Map a PersonaSpec to a Sotopia-style agent profile dict."""
    profile: dict = {"name": spec.name}
    if spec.ocean:
        o = spec.ocean
        profile["personality"] = (
            f"Openness {o.openness:.2f}, Conscientiousness {o.conscientiousness:.2f}, "
            f"Extraversion {o.extraversion:.2f}, Agreeableness {o.agreeableness:.2f}, "
            f"Neuroticism {o.neuroticism:.2f}"
        )
    if spec.decision_style:
        profile["decision_making_style"] = spec.decision_style.value
    if spec.values:
        profile["values"] = list(spec.values)
    if spec.moral_values:
        profile["moral_values"] = [v.value for v in spec.moral_values]
    # hard constraints read naturally as "secrets"/non-negotiables in a social scenario
    if spec.hard_constraints:
        profile["secret"] = "; ".join(spec.hard_constraints)
    if spec.seed_memory:
        profile["background"] = spec.seed_memory
    return profile


async def score_episode(messages: list, evaluator=None, agent: str = "") -> SocialEvaluation:
    """Score a social episode on the SOTOPIA 7 dimensions.

    ``messages``: list of {"agent_name","content"} (or strings). ``evaluator`` is an LLM
    adapter returning the 7 scores as JSON; without it a neutral baseline is returned."""
    if evaluator is None:
        return SocialEvaluation()
    transcript = "\n".join(
        (m.get("agent_name", "") + ": " + m.get("content", "")) if isinstance(m, dict) else str(m)
        for m in messages
    )
    try:
        data = await evaluator.generate_json(
            system_prompt="You are a social-intelligence evaluator using the SOTOPIA 7-dimension rubric.",
            user_prompt=(
                f"Evaluate {agent or 'the agent'} in this episode:\n{transcript}\n\n"
                'Respond JSON with keys goal_completion(0..10), believability(0..10), knowledge(0..10), '
                'secret_keeping(-10..0), relationship(-5..5), social_rules(-10..0), financial_benefit(-5..5).'
            ),
        )
        return SocialEvaluation(
            goal_completion=float(data.get("goal_completion", 0)),
            believability=float(data.get("believability", 0)),
            knowledge=float(data.get("knowledge", 0)),
            secret_keeping=float(data.get("secret_keeping", 0)),
            relationship=float(data.get("relationship", 0)),
            social_rules=float(data.get("social_rules", 0)),
            financial_benefit=float(data.get("financial_benefit", 0)),
        )
    except Exception:
        return SocialEvaluation()
