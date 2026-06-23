"""v0.10.0 — trait→cognition policy, PAD momentum, emotion→memory coupling, eval harness."""
from __future__ import annotations

from agethos import CognitivePolicy, eval
from agethos.memory.retrieval import compute_retrieval_scores
from agethos.memory.stream import MemoryStream
from agethos.models import BrainState, EmotionalState, MemoryNode, OceanTraits, PersonaSpec
from agethos.persona.renderer import PersonaRenderer
from agethos.storage.memory_store import InMemoryStore


# ── Trait → Cognition policy ──

def test_policy_conscientiousness_drives_planning():
    hi = CognitivePolicy.from_ocean(OceanTraits.random(C=0.95))
    lo = CognitivePolicy.from_ocean(OceanTraits.random(C=0.05))
    assert hi.planning_depth > lo.planning_depth
    assert hi.verification_steps >= lo.verification_steps
    assert any("step plan" in d for d in hi.to_directives())


def test_policy_openness_drives_risk():
    hi = CognitivePolicy.from_ocean(OceanTraits.random(O=0.95))
    lo = CognitivePolicy.from_ocean(OceanTraits.random(O=0.05))
    assert hi.risk_tolerance > lo.risk_tolerance
    assert lo.risk_margin > hi.risk_margin          # low O demands a larger EV margin
    assert any("conservative" in d for d in lo.to_directives())


def test_policy_rendered_into_persona_prompt():
    spec = PersonaSpec(name="Min", ocean=OceanTraits.random(C=0.9, O=0.1))
    iss = PersonaRenderer(spec).render_iss()
    assert "Cognitive Policy" in iss
    assert spec.cognitive_policy() is not None
    # no ocean -> no policy
    assert PersonaSpec(name="X").cognitive_policy() is None


# ── PAD momentum dynamics + discrete↔PAD ──

def test_from_label_scales_by_intensity():
    full = EmotionalState.from_label("joy", 1.0)
    half = EmotionalState.from_label("joy", 0.5)
    assert full.pleasure > half.pleasure > 0
    assert EmotionalState.from_label("???").pleasure == 0.0   # unknown -> neutral


def test_step_momentum_moves_toward_stimulus_and_clamps():
    e = EmotionalState()
    stim = (1.0, 1.0, 1.0)
    e2 = e.step(stim, momentum=0.85)
    assert 0 < e2.pleasure <= 1.0 and e2.velocity != (0.0, 0.0, 0.0)
    # repeated steps approach the stimulus, never exceed bounds
    cur = e2
    for _ in range(40):
        cur = cur.step(stim, momentum=0.85)
    assert 0.9 < cur.pleasure <= 1.0 and cur.arousal <= 1.0 and cur.dominance <= 1.0


# ── emotion → memory coupling + retrieval ──

def test_salient_memory_ranks_higher():
    calm = MemoryNode(description="had lunch", importance=5.0, emotional_salience=0.0)
    intense = MemoryNode(description="had lunch", importance=5.0, emotional_salience=1.0)
    res = compute_retrieval_scores([calm, intense], salience_weight=2.0)
    assert res[0].node.id == intense.id           # arousal salience lifts it to the top


def test_keyword_fallback_relevance_without_embeddings():
    a = MemoryNode(description="the cat sat on the mat", keywords=["cat"])
    b = MemoryNode(description="quantum chromodynamics lecture")
    res = compute_retrieval_scores([a, b], query="where is the cat", weights=(0.0, 0.0, 1.0))
    assert res[0].node.id == a.id                 # lexical fallback finds the cat memory


async def test_append_tags_emotion_and_retrieve_reinforces_vitality():
    stream = MemoryStream(InMemoryStore())
    happy = EmotionalState(pleasure=0.7, arousal=0.9, dominance=0.2)
    node = await stream.append(MemoryNode(description="great news about the cat"),
                               current_emotion=happy)
    assert node.emotional_salience == 0.9 and node.encoding_pad is not None
    node.vitality = 0.5
    await stream.store.update(node)
    res = await stream.retrieve("cat news", weights=(0.0, 0.0, 1.0), reinforce=0.2)
    assert res[0].node.vitality > 0.5             # access reinforced it


async def test_decay_and_forget_protects_salient():
    store = InMemoryStore()
    stream = MemoryStream(store)
    weak = await stream.append(MemoryNode(description="trivial", importance=1.0, vitality=0.1))
    salient = await stream.append(MemoryNode(description="trauma", importance=1.0,
                                             vitality=0.1, emotional_salience=1.0))
    removed = await stream.forget(threshold=0.2)
    ids = {n.id for n in await store.get_all()}
    assert weak.id not in ids and salient.id in ids and removed == 1


async def test_store_delete():
    store = InMemoryStore()
    n = MemoryNode(description="x")
    await store.save(n)
    await store.delete(n.id)
    assert await store.count() == 0


# ── eval harness ──

def test_persona_consistency_and_drift():
    a = PersonaSpec(name="A", ocean=OceanTraits.random(O=0.9, C=0.9))
    b = PersonaSpec(name="A", ocean=OceanTraits.random(O=0.1, C=0.1))
    assert eval.persona_consistency(a, a) == 1.0
    assert eval.persona_consistency(a, b) < 1.0
    curve = eval.persona_drift_curve([a, a, b])
    assert curve[0] == 1.0 and curve[1] < 1.0


def test_ocean_similarity_bounds():
    o = OceanTraits(openness=0.5, conscientiousness=0.5, extraversion=0.5,
                    agreeableness=0.5, neuroticism=0.5)
    assert eval.ocean_similarity(o, o) == 1.0
    far = OceanTraits(openness=1, conscientiousness=1, extraversion=1, agreeableness=1, neuroticism=1)
    near0 = OceanTraits(openness=0, conscientiousness=0, extraversion=0, agreeableness=0, neuroticism=0)
    assert eval.ocean_similarity(far, near0) == 0.0


def test_retrieval_metrics():
    m = eval.retrieval_metrics(["a", "x", "b", "y"], relevant_ids=["a", "b"], k=4)
    assert m["recall"] == 1.0 and m["precision"] == 0.5
    assert m["mrr"] == 1.0 and 0 < m["ndcg"] <= 1.0


def test_transplant_fidelity_identity():
    spec = PersonaSpec(name="Z", ocean=OceanTraits.random(O=0.7))
    mems = [MemoryNode(description="m1"), MemoryNode(description="m2")]
    a = BrainState(persona=spec, memories=mems)
    b = BrainState(persona=spec.model_copy(deep=True), memories=[m.model_copy(deep=True) for m in mems])
    f = eval.transplant_fidelity(a, b)
    assert f["overall"] == 1.0 and f["memory_retention"] == 1.0
    # losing a memory drops retention
    c = BrainState(persona=spec, memories=mems[:1])
    assert eval.transplant_fidelity(a, c)["memory_retention"] == 0.5
