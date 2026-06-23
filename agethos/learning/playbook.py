"""ACE-style self-improving playbook — incremental delta lessons with counters.

Self-improvement loops collapse when context is monolithically rewritten (brevity bias,
context collapse). ACE (Zhang et al. 2025) instead accumulates *delta* lessons with
helpful/harmful counters and merges them by deterministic dedup (a duplicate increments
a counter, it isn't re-summarized). ``net = helpful - harmful`` drives ranking and pruning.
This unifies with the library's Hebbian idea — the counter *is* an edge weight.
"""
from __future__ import annotations

import re

from agethos.models import Lesson

_NORM = re.compile(r"[^a-z0-9가-힣 ]+")


def _norm(text: str) -> str:
    return _NORM.sub("", (text or "").lower()).strip()


class Playbook:
    """A growing, self-curating set of lessons. Wraps a list[Lesson] (e.g. BrainState.lessons)."""

    def __init__(self, lessons: list[Lesson] | None = None):
        self._lessons: list[Lesson] = list(lessons or [])
        self._by_norm: dict[str, Lesson] = {_norm(le.text): le for le in self._lessons}

    def add(self, text: str, *, tags: list[str] | None = None, provenance: str = "",
            helpful: int = 1) -> Lesson:
        """Add a lesson. Deterministic dedup: a duplicate just increments its helpful counter."""
        key = _norm(text)
        if key in self._by_norm:
            le = self._by_norm[key]
            le.helpful += helpful
            if tags:
                le.tags = list(dict.fromkeys([*le.tags, *tags]))
            return le
        le = Lesson(text=text, tags=tags or [], provenance=provenance, helpful=helpful)
        self._lessons.append(le)
        self._by_norm[key] = le
        return le

    def record_outcome(self, text_or_id: str, helpful: bool = True) -> Lesson | None:
        """Tally a lesson's outcome by its id or text."""
        le = next((x for x in self._lessons if x.id == text_or_id), None) or \
            self._by_norm.get(_norm(text_or_id))
        if le is None:
            return None
        if helpful:
            le.helpful += 1
        else:
            le.harmful += 1
        return le

    def refine(self, *, min_net: int = 0, max_size: int | None = None) -> int:
        """Grow-and-refine: drop net-harmful lessons; cap to the top ``max_size`` by net. Returns dropped."""
        before = len(self._lessons)
        kept = [le for le in self._lessons if le.net >= min_net]
        kept.sort(key=lambda le: (-le.net, le.created_at))
        if max_size is not None:
            kept = kept[:max_size]
        self._lessons = kept
        self._by_norm = {_norm(le.text): le for le in kept}
        return before - len(self._lessons)

    def top(self, n: int = 10) -> list[Lesson]:
        return sorted(self._lessons, key=lambda le: (-le.net, le.created_at))[:n]

    def render(self, n: int = 10) -> str:
        """Render the top lessons as a prompt-injectable playbook block."""
        top = self.top(n)
        if not top:
            return ""
        return "## Playbook (learned lessons)\n" + "\n".join(
            f"- {le.text}  (+{le.helpful}/-{le.harmful})" for le in top
        )

    def all(self) -> list[Lesson]:
        return list(self._lessons)
