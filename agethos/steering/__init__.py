"""Activation steering / persona vectors (open-weight models).

Prompting gives shallow, drift-prone traits; activation steering integrates them
deeper. This package extracts an OCEAN "persona vector" from contrastive prompt pairs
(the difference of mean activations — trait-present minus trait-absent, Anthropic
persona vectors) and steers it at inference. Multi-trait vectors are orthogonalized to
avoid manifold interference (PERSONA). The vector math + prompt generation are pure
Python (testable offline); the actual hidden-state hooks live in an optional
``TransformersSteeringBackend`` (``pip install agethos[steering]``).
"""
from agethos.steering.backend import (
    MockSteeringBackend,
    SteeringBackend,
    extract_persona_vectors,
)
from agethos.steering.contrastive import ocean_contrastive_prompts, trait_contrastive_prompts
from agethos.steering.plan import SteeringIntent, plan_from_ocean, plan_vectors
from agethos.steering.rerank import (
    RankedCandidate,
    RerankResult,
    attribute_score,
    steered_generate,
)
from agethos.steering.vectors import PersonaVector, combine, mean_diff, orthogonalize, steer

__all__ = [
    "PersonaVector", "mean_diff", "orthogonalize", "combine", "steer",
    "trait_contrastive_prompts", "ocean_contrastive_prompts",
    "SteeringBackend", "MockSteeringBackend", "extract_persona_vectors",
    "SteeringIntent", "plan_from_ocean", "plan_vectors",
    "RankedCandidate", "RerankResult", "attribute_score", "steered_generate",
]
