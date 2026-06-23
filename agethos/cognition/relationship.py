"""Relationship dynamics — typed bonds whose strength (0-100) evolves with interaction.

AgentSociety-style: each interaction carries a valence (use the SOTOPIA relationship
dimension, −5..+5); positive raises the bond, negative lowers it, and idle time pulls it
back toward neutral. Strength gates communication frequency / tone / ToM depth, and is
*justified* over time rather than a static label.
"""
from __future__ import annotations

import time

from agethos.models import Relationship, RelationshipType


class RelationshipBook:
    """Manage per-target relationships. Wraps a list[Relationship] (e.g. BrainState.relationships)."""

    def __init__(self, relationships: list[Relationship] | None = None):
        self._rels: dict[str, Relationship] = {r.target: r for r in (relationships or [])}

    def get(self, target: str) -> Relationship:
        if target not in self._rels:
            self._rels[target] = Relationship(target=target)
        return self._rels[target]

    def record(
        self,
        target: str,
        valence: float,
        *,
        gain: float = 5.0,
        relationship_type: RelationshipType | None = None,
        now: float | None = None,
    ) -> Relationship:
        """Apply an interaction. ``valence`` in [-5, +5] (SOTOPIA). strength += gain·valence."""
        r = self.get(target)
        r.strength = max(0.0, min(100.0, r.strength + gain * valence))
        r.interactions += 1
        if relationship_type is not None:
            r.type = relationship_type
        r.last_updated = now if now is not None else time.time()
        return r

    def decay(self, rate: float = 1.0, *, neutral: float = 50.0, now: float | None = None,
              idle_hours: float = 24.0) -> None:
        """Pull idle relationships toward neutral (slow forgetting of affect)."""
        now = now if now is not None else time.time()
        for r in self._rels.values():
            if (now - r.last_updated) / 3600.0 >= idle_hours:
                if r.strength > neutral:
                    r.strength = max(neutral, r.strength - rate)
                elif r.strength < neutral:
                    r.strength = min(neutral, r.strength + rate)

    def all(self) -> list[Relationship]:
        return list(self._rels.values())

    def tier(self, target: str) -> str:
        """Coarse bond tier for gating behavior."""
        s = self.get(target).strength
        return "close" if s >= 75 else "warm" if s >= 55 else "neutral" if s >= 40 else "strained"
