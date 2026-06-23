"""Steering backends — extract activations + apply steering.

``SteeringBackend`` is the seam. ``MockSteeringBackend`` is a deterministic, dependency-free
stand-in (lets the whole extraction pipeline run + be tested offline).
``TransformersSteeringBackend`` (optional ``[steering]`` extra) hooks a real open-weight model.
"""
from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

from agethos.steering.contrastive import ocean_contrastive_prompts
from agethos.steering.vectors import PersonaVector, mean_diff


@runtime_checkable
class SteeringBackend(Protocol):
    """Return a hidden-state vector per prompt at ``layer``."""

    def activations(self, prompts: list[str], layer: int = -1) -> list[list[float]]:
        ...


class MockSteeringBackend:
    """Deterministic pseudo-activations from prompt text — for offline tests / dry runs.

    Word-presence features over the trait pole vocabulary, so trait-high and trait-low
    prompts land in different directions and ``mean_diff`` yields a real signal."""

    _VOCAB = ["imaginative", "curious", "unconventional", "novel", "practical", "conventional",
              "routine", "organized", "disciplined", "thorough", "spontaneous", "careless",
              "outgoing", "energetic", "talkative", "reserved", "quiet", "withdrawn",
              "warm", "cooperative", "empathetic", "blunt", "competitive", "confrontational",
              "anxious", "reactive", "worried", "calm", "stable", "unshaken"]

    def __init__(self, dim: int = 32):
        self.dim = dim

    def activations(self, prompts: list[str], layer: int = -1) -> list[list[float]]:
        out = []
        for p in prompts:
            low = p.lower()
            vec = [1.0 if w in low else 0.0 for w in self._VOCAB]
            # pad/seed remaining dims deterministically from the text hash
            h = hashlib.sha1(p.encode()).digest()
            while len(vec) < self.dim:
                vec.append((h[len(vec) % len(h)] / 255.0) - 0.5)
            out.append(vec[: self.dim])
        return out


class TransformersSteeringBackend:
    """Hook a real open-weight model (optional ``[steering]`` extra: torch + transformers).

    Captures the residual-stream activation at ``layer`` (mean over tokens). Steering at
    inference is applied by adding the persona vector via a forward pre-hook (left to the
    caller's generation loop)."""

    def __init__(self, model_name: str, device: str = "cpu"):
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("TransformersSteeringBackend needs: pip install 'agethos[steering]'") from e
        self._torch = __import__("torch")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True).to(device)
        self.device = device

    def activations(self, prompts: list[str], layer: int = -1) -> list[list[float]]:  # pragma: no cover
        torch = self._torch
        out = []
        for p in prompts:
            ids = self.tok(p, return_tensors="pt").to(self.device)
            with torch.no_grad():
                hs = self.model(**ids).hidden_states[layer][0]  # (tokens, dim)
            out.append(hs.mean(dim=0).tolist())
        return out


def extract_persona_vectors(
    backend: SteeringBackend,
    traits: list[str] | None = None,
    layer: int = -1,
    stems: list[str] | None = None,
) -> list[PersonaVector]:
    """Extract one PersonaVector per OCEAN trait via contrastive activation differences."""
    pairs_by_trait = ocean_contrastive_prompts(traits, stems)
    vectors: list[PersonaVector] = []
    for trait, pairs in pairs_by_trait.items():
        pos = backend.activations([p for p, _ in pairs], layer)
        neg = backend.activations([n for _, n in pairs], layer)
        vectors.append(PersonaVector(trait=trait, vector=mean_diff(pos, neg), layer=layer))
    return vectors
