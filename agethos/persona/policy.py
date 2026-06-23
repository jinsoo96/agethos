"""Trait → Cognition policy — make OCEAN traits *causally* drive reasoning.

Most persona systems render Big Five traits as prose tone ("you are conscientious")
which only changes surface style. Research (BIG5-CHAT, Li et al. 2024; risk-taking via
Cumulative Prospect Theory, 2025) shows traits should change the *reasoning process*
itself — planning depth, verification, risk threshold, hedging, initiative, concession.

``CognitivePolicy`` maps an ``OceanTraits`` vector into concrete control parameters and
renders them as directives injected into plan/decide prompts, so the persona is causal
rather than cosmetic.

Grounding:
- Conscientiousness → planning depth + verification steps (high-C agents plan and self-check).
- Openness → risk tolerance / exploration (Openness is the strongest driver of risk-taking).
- Neuroticism → caution / hedging / help-seeking.
- Extraversion → initiative / verbosity.
- Agreeableness → cooperativeness / concession in negotiation.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from agethos.models import OceanTraits


class CognitivePolicy(BaseModel):
    """Concrete cognitive control parameters derived from OCEAN traits."""

    planning_depth: int = Field(4, ge=1, description="number of plan steps to produce (C-driven)")
    verification_steps: int = Field(1, ge=0, description="self-checks before acting (C-driven)")
    risk_tolerance: float = Field(0.5, ge=0.0, le=1.0, description="willingness to take risk (O-driven)")
    exploration: float = Field(0.5, ge=0.0, le=1.0, description="explore vs exploit (O-driven)")
    caution: float = Field(0.5, ge=0.0, le=1.0, description="hedging / uncertainty flagging (N-driven)")
    initiative: float = Field(0.5, ge=0.0, le=1.0, description="proactivity / verbosity (E-driven)")
    cooperativeness: float = Field(0.5, ge=0.0, le=1.0, description="concession / empathy (A-driven)")
    structure: float = Field(0.5, ge=0.0, le=1.0, description="organized, structured output (C-driven)")

    @classmethod
    def from_ocean(cls, ocean: OceanTraits) -> "CognitivePolicy":
        o, c, e, a, n = (ocean.openness, ocean.conscientiousness, ocean.extraversion,
                         ocean.agreeableness, ocean.neuroticism)
        return cls(
            planning_depth=int(round(2 + 5 * c)),       # 2..7
            verification_steps=int(round(3 * c)),        # 0..3
            risk_tolerance=o,
            exploration=o,
            caution=n,
            initiative=e,
            cooperativeness=a,
            structure=c,
        )

    @property
    def risk_margin(self) -> float:
        """Expected-value margin required to prefer a riskier option (low O → larger margin)."""
        return round(0.25 * (1.0 - self.risk_tolerance), 2)

    def to_directives(self) -> list[str]:
        """Concrete, causal instructions to inject into planning/decision prompts."""
        d: list[str] = []

        if self.structure > 0.66:
            d.append(f"Conscientiousness is HIGH: produce an explicit {self.planning_depth}-step plan and "
                     f"verify each step ({self.verification_steps} self-check pass(es)) before acting.")
        elif self.structure < 0.33:
            d.append("Conscientiousness is LOW: act decisively with minimal planning; keep it brief and flexible.")
        else:
            d.append(f"Conscientiousness is MODERATE: sketch a {self.planning_depth}-step plan; verify key steps.")

        if self.risk_tolerance > 0.66:
            d.append("Openness is HIGH: explore novel, higher-variance options and unconventional angles.")
        elif self.risk_tolerance < 0.33:
            d.append(f"Openness is LOW: prefer conventional, proven options; when options are within "
                     f"{self.risk_margin:.0%} expected value, choose the conservative one.")

        if self.caution > 0.66:
            d.append("Neuroticism is HIGH: hedge uncertain claims, flag risks, and seek confirmation when unsure.")
        elif self.caution < 0.33:
            d.append("Neuroticism is LOW: state conclusions confidently without excessive hedging.")

        if self.initiative > 0.66:
            d.append("Extraversion is HIGH: take initiative, be proactive, and elaborate.")
        elif self.initiative < 0.33:
            d.append("Extraversion is LOW: be concise; let others lead the exchange.")

        if self.cooperativeness > 0.66:
            d.append("Agreeableness is HIGH: prioritize the other party's goals, concede minor points, keep harmony.")
        elif self.cooperativeness < 0.33:
            d.append("Agreeableness is LOW: push back directly, defend your position, concede only when warranted.")

        return d

    def to_prompt(self) -> str:
        return "## Cognitive Policy (how your traits shape your reasoning)\n" + "\n".join(
            f"- {line}" for line in self.to_directives()
        )
