"""Persona-vector algebra — pure Python, no torch.

A persona vector is the mean-difference of activations between trait-present and
trait-absent prompts. Multiple trait vectors are orthogonalized (Gram-Schmidt) before
combining so steering several OCEAN dims at once doesn't interfere (PERSONA finding).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class PersonaVector:
    """A steering direction for one trait at a given layer."""

    trait: str
    vector: list[float] = field(default_factory=list)
    layer: int = -1


def _sub(a, b):
    return [x - y for x, y in zip(a, b)]


def _add(a, b):
    return [x + y for x, y in zip(a, b)]


def _scale(a, k):
    return [x * k for x in a]


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _norm(a):
    return math.sqrt(sum(x * x for x in a))


def _mean(vs: list[list[float]]) -> list[float]:
    if not vs:
        return []
    n = len(vs)
    dim = len(vs[0])
    out = [0.0] * dim
    for v in vs:
        for i in range(dim):
            out[i] += v[i]
    return [x / n for x in out]


def mean_diff(pos: list[list[float]], neg: list[list[float]]) -> list[float]:
    """Persona direction = mean(trait-present activations) − mean(trait-absent)."""
    mp, mn = _mean(pos), _mean(neg)
    if not mp or not mn:
        return mp or [(-x) for x in mn] if mn else []
    return _sub(mp, mn)


def orthogonalize(vectors: list[list[float]]) -> list[list[float]]:
    """Gram-Schmidt — return orthogonal basis (zero vectors dropped to []), order preserved."""
    out: list[list[float]] = []
    for v in vectors:
        w = list(v)
        for u in out:
            nu = _dot(u, u)
            if nu > 1e-12:
                w = _sub(w, _scale(u, _dot(w, u) / nu))
        out.append(w if _norm(w) > 1e-9 else [0.0] * len(v))
    return out


def combine(persona_vectors: list[PersonaVector], weights: list[float] | None = None,
            orthogonal: bool = True) -> list[float]:
    """Weighted sum of persona vectors into one steering vector (orthogonalized by default)."""
    vecs = [pv.vector for pv in persona_vectors if pv.vector]
    if not vecs:
        return []
    if weights is None:
        weights = [1.0] * len(vecs)
    if orthogonal:
        vecs = orthogonalize(vecs)
    out = [0.0] * len(vecs[0])
    for v, w in zip(vecs, weights):
        out = _add(out, _scale(v, w))
    return out


def steer(hidden: list[float], vector: list[float], alpha: float = 4.0) -> list[float]:
    """Add ``alpha`` × unit(vector) to a hidden state (inference-time steering)."""
    if not vector:
        return list(hidden)
    n = _norm(vector) or 1.0
    unit = _scale(vector, 1.0 / n)
    return _add(hidden, _scale(unit, alpha))


def cosine(a: list[float], b: list[float]) -> float:
    na, nb = _norm(a), _norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return _dot(a, b) / (na * nb)
