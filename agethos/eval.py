"""agethos.eval — measure what the brain claims: persona consistency, retrieval
quality, and brain-transplant fidelity.

No mature agent-cognition library ships an in-box eval harness — memory tools publish
benchmark papers, social-agent stacks are separate environments. These are pure,
offline metrics (embedding-optional, token-Jaccard fallback) so you can regression-test
believability, retrieval, and the unique "transplant" claim with one import.

Grounding: persona-drift measurement (Li et al. 2024), CharacterEval persona
consistency, LoCoMo-style retrieval metrics, SOTOPIA believability.
"""
from __future__ import annotations

import math
import re

from agethos.models import BrainState, OceanTraits, PersonaSpec
from agethos.persona.renderer import PersonaRenderer

_WORD = re.compile(r"[\w']+", re.UNICODE)


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _WORD.findall(text or "") if len(t) > 1}


def text_similarity(a: str, b: str) -> float:
    """Token-Jaccard similarity in [0,1] (lexical, deterministic, no embeddings)."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ───────────────────────── persona consistency / drift ─────────────────────────


def ocean_similarity(a: OceanTraits, b: OceanTraits) -> float:
    """1 − normalized Euclidean distance over the 5 OCEAN axes → [0,1]."""
    dims = ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")
    dist = math.sqrt(sum((getattr(a, d) - getattr(b, d)) ** 2 for d in dims))
    return round(1.0 - dist / math.sqrt(len(dims)), 4)


def persona_consistency(a, b) -> float:
    """How consistent two personas are in [0,1] (1 = identical).

    Accepts PersonaSpec (rendered to its identity block) or raw prompt strings."""
    ta = PersonaRenderer(a).render_iss() if isinstance(a, PersonaSpec) else str(a)
    tb = PersonaRenderer(b).render_iss() if isinstance(b, PersonaSpec) else str(b)
    return round(text_similarity(ta, tb), 4)


def persona_drift_curve(snapshots: list) -> list[float]:
    """Consecutive persona consistency across a sequence of specs/prompts.

    Each value is consistency(t, t+1); a falling curve signals drift (the documented
    failure mode where an agent abandons its persona over long conversations)."""
    return [persona_consistency(snapshots[i], snapshots[i + 1]) for i in range(len(snapshots) - 1)]


# ───────────────────────── retrieval quality ─────────────────────────


def retrieval_metrics(ranked_ids: list[str], relevant_ids, k: int | None = None) -> dict:
    """Precision@k, Recall@k, MRR, NDCG@k for a ranked id list vs a relevant set."""
    relevant = set(relevant_ids)
    if not relevant:
        return {"precision": 0.0, "recall": 0.0, "mrr": 0.0, "ndcg": 0.0}
    k = k or len(ranked_ids)
    topk = ranked_ids[:k]

    hits = sum(1 for i in topk if i in relevant)
    precision = hits / len(topk) if topk else 0.0
    recall = hits / len(relevant)

    mrr = 0.0
    for rank, i in enumerate(ranked_ids, start=1):
        if i in relevant:
            mrr = 1.0 / rank
            break

    dcg = sum(1.0 / math.log2(rank + 1) for rank, i in enumerate(topk, start=1) if i in relevant)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    ndcg = dcg / ideal if ideal else 0.0

    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "mrr": round(mrr, 4), "ndcg": round(ndcg, 4)}


# ───────────────────────── transplant fidelity (unique) ─────────────────────────


def transplant_fidelity(before: BrainState, after: BrainState) -> dict:
    """How faithfully a brain survives an export → re-import (the 'brain transplant').

    Compares persona identity, OCEAN, and retained memories. Returns per-axis scores +
    a weighted overall in [0,1]. No external benchmark exists for this — it directly
    proves agethos's portable-cognitive-identity claim."""
    persona = persona_consistency(before.persona, after.persona)
    ocean = (ocean_similarity(before.persona.ocean, after.persona.ocean)
             if before.persona.ocean and after.persona.ocean else 1.0)

    ids_before = {m.id for m in before.memories}
    ids_after = {m.id for m in after.memories}
    if ids_before:
        memory_retention = len(ids_before & ids_after) / len(ids_before)
    else:
        memory_retention = 1.0

    overall = round(0.4 * persona + 0.3 * ocean + 0.3 * memory_retention, 4)
    return {"persona": persona, "ocean": round(ocean, 4),
            "memory_retention": round(memory_retention, 4), "overall": overall}
