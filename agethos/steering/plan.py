"""Steering plans — which traits to steer, in which direction, how hard.

A plan is derived from an OCEAN config (traits deviating from the 0.5 prior) and is the
shared input of both steering mounts: activation vectors for open-weight models
(``plan_vectors``) and GPU-free best-of-n re-ranking for black-box models
(``agethos.steering.rerank``).
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from agethos.models import OceanTraits
from agethos.steering.contrastive import OCEAN_TRAITS

class SteeringIntent(BaseModel):
    """One trait the persona wants steered at inference."""

    trait: str
    direction: int = Field(1, description="+1 = toward trait-high pole, -1 = trait-low")
    strength: float = Field(0.0, ge=0.0, le=1.0, description="|trait − 0.5| × 2")


def plan_from_ocean(ocean: OceanTraits, threshold: float = 0.15) -> list[SteeringIntent]:
    """Traits deviating from the 0.5 prior by ≥ threshold → steering targets."""
    plan: list[SteeringIntent] = []
    for dim in OCEAN_TRAITS:
        v = getattr(ocean, dim)
        if abs(v - 0.5) >= threshold:
            plan.append(SteeringIntent(
                trait=dim,
                direction=1 if v > 0.5 else -1,
                strength=round(min(1.0, abs(v - 0.5) * 2), 4),
            ))
    return plan


def plan_vectors(plan: list[SteeringIntent], backend, layer: int = -1):
    """Steering plan → direction/strength-scaled PersonaVectors for open-weight models."""
    from agethos.steering.backend import extract_persona_vectors
    from agethos.steering.vectors import PersonaVector
    if not plan:
        return []
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
