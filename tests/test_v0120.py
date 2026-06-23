"""v0.12.0 — persona-vector steering, LoCoMo/Sotopia eval adapters, async concurrency."""
from __future__ import annotations

from agethos import eval
from agethos.concurrency import amap, gather_bounded
from agethos.memory.stream import MemoryStream
from agethos.models import OceanTraits, PersonaSpec
from agethos.steering import (
    MockSteeringBackend,
    PersonaVector,
    combine,
    extract_persona_vectors,
    mean_diff,
    ocean_contrastive_prompts,
    orthogonalize,
    steer,
    trait_contrastive_prompts,
)
from agethos.steering.vectors import cosine
from agethos.storage.memory_store import InMemoryStore


# ── (A) persona-vector steering ──

def test_contrastive_pairs_per_trait():
    pairs = trait_contrastive_prompts("openness")
    assert pairs and all(len(p) == 2 and p[0] != p[1] for p in pairs)
    allp = ocean_contrastive_prompts()
    assert set(allp) == {"openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"}


def test_mean_diff_and_steer():
    pos = [[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
    neg = [[0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]
    v = mean_diff(pos, neg)            # ~[1, 0, -1]
    assert v[0] > 0 and v[2] < 0
    h = [0.0, 0.0, 0.0]
    steered = steer(h, v, alpha=2.0)
    assert cosine(steered, v) > 0.99   # moved along the persona direction


def test_orthogonalize_makes_orthogonal():
    a, b = [1.0, 0.0], [1.0, 1.0]
    oa, ob = orthogonalize([a, b])
    assert abs(sum(x * y for x, y in zip(oa, ob))) < 1e-9


def test_combine_uses_orthogonalization():
    pvs = [PersonaVector("o", [1.0, 0.0]), PersonaVector("c", [1.0, 1.0])]
    out = combine(pvs, orthogonal=True)
    assert len(out) == 2


def test_extract_persona_vectors_with_mock_backend():
    vecs = extract_persona_vectors(MockSteeringBackend(), traits=["openness", "neuroticism"])
    assert len(vecs) == 2
    o = next(v for v in vecs if v.trait == "openness")
    assert any(abs(x) > 0 for x in o.vector)        # trait-high vs trait-low differ → real direction


# ── (B) eval adapters ──

async def test_locomo_ingest_and_recall():
    stream = MemoryStream(InMemoryStore())
    turns = [
        {"speaker": "Alice", "text": "I adopted a golden retriever named Max last June."},
        {"speaker": "Bob", "text": "We discussed the quarterly budget."},
        {"speaker": "Alice", "text": "Max loves the dog park on weekends."},
    ]
    nodes = await eval.ingest_conversation(stream, turns)
    assert len(nodes) == 3
    res = await eval.evaluate_recall(
        stream,
        [{"question": "What pet does Alice have?", "evidence": ["golden retriever"]}],
        top_k=3,
    )
    assert res["n"] == 1 and res["recall"] > 0.0     # retrieval surfaced the evidence memory


def test_sotopia_profile_mapping():
    spec = PersonaSpec(name="Min", ocean=OceanTraits.random(O=0.8),
                       values=["honesty"], hard_constraints=["never reveal the password"])
    prof = eval.persona_to_sotopia_profile(spec)
    assert prof["name"] == "Min" and "Openness" in prof["personality"]
    assert prof["secret"] == "never reveal the password" and prof["values"] == ["honesty"]


async def test_score_episode_neutral_without_evaluator():
    se = await eval.score_episode([{"agent_name": "A", "content": "hi"}])
    assert se.believability == 0.0 and se.goal_completion == 0.0


# ── (C) concurrency ──

async def test_gather_bounded_preserves_order():
    async def f(x):
        return x * 2
    out = await gather_bounded([f(i) for i in range(5)], limit=2)
    assert out == [0, 2, 4, 6, 8]


async def test_amap_bounds_concurrency():
    import asyncio
    active = 0
    peak = 0

    async def f(x):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.001)
        active -= 1
        return x
    out = await amap(f, range(10), concurrency=3)
    assert out == list(range(10)) and peak <= 3
