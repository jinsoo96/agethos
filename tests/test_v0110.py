"""v0.11.0 — Mem0 arbitrated write, A-MEM evolution, SimToM, relationship dynamics, ACE playbook."""
from __future__ import annotations

from agethos.cognition.relationship import RelationshipBook
from agethos.cognition.tom import TheoryOfMind
from agethos.learning.playbook import Playbook
from agethos.memory.arbiter import MemoryArbiter, remember
from agethos.memory.evolve import link_and_evolve
from agethos.memory.stream import MemoryStream
from agethos.models import Lesson, MemoryNode, Relationship, RelationshipType
from agethos.storage.memory_store import InMemoryStore


class _StubLLM:
    """Minimal async LLM adapter for offline tests."""

    def __init__(self, text: str = "", js: dict | None = None):
        self.text, self.js = text, js or {}
        self.calls = 0

    async def generate(self, system_prompt: str = "", user_prompt: str = "", temperature: float = 0.7) -> str:
        self.calls += 1
        return self.text

    async def generate_json(self, system_prompt: str = "", user_prompt: str = "", temperature: float = 0.2) -> dict:
        self.calls += 1
        return self.js


# ── (1) Mem0-style arbitrated write ──

async def test_arbiter_fallback_noop_update_add():
    arb = MemoryArbiter()
    new = MemoryNode(description="alpha beta gamma delta epsilon zeta")
    dup = MemoryNode(description="alpha beta gamma delta epsilon zeta")           # identical
    near = MemoryNode(description="alpha beta gamma delta epsilon zeta eta")      # 6/7 ≈ 0.857
    far = MemoryNode(description="completely unrelated lecture on plasma physics")
    assert (await arb.decide(new, [dup]))[0] == "NOOP"
    assert (await arb.decide(new, [near]))[0] == "UPDATE"
    assert (await arb.decide(new, [far]))[0] == "ADD"
    assert (await arb.decide(new, []))[0] == "ADD"


async def test_remember_dedups_and_adds():
    stream = MemoryStream(InMemoryStore())
    await stream.append(MemoryNode(description="the cat is black"))
    op, _ = await remember(stream, MemoryNode(description="the cat is black"))
    assert op == "NOOP" and await stream.count() == 1            # duplicate not stored
    op, _ = await remember(stream, MemoryNode(description="the dog barks loudly"))
    assert op == "ADD" and await stream.count() == 2


# ── (2) A-MEM memory evolution ──

async def test_link_and_evolve_links_and_merges_keywords():
    store = InMemoryStore()
    stream = MemoryStream(store)
    a = await stream.append(MemoryNode(description="the cat sat on the mat", keywords=["cat"]))
    b = await stream.append(MemoryNode(description="the cat chased a mouse", keywords=["mouse"]))
    neighbors = await link_and_evolve(stream, b, k=3)
    assert a.id in [n.id for n in neighbors]
    assert b.id in a.links and a.id in b.links                   # bidirectional link
    a2 = next(n for n in await store.get_all() if n.id == a.id)
    assert "mouse" in a2.keywords                                # neighbor keywords evolved


# ── (3) SimToM perspective-taking ──

async def test_simtom_two_stage_routes_through_filter():
    llm = _StubLLM(text="Sam knows the ball is in the box.")
    tom = TheoryOfMind(llm)
    filtered = await tom.perspective_filter("Sam saw the ball put in the box. Sam left. It moved to the basket.", "Sam")
    assert "box" in filtered
    ans = await tom.answer_as("Sam", "Where will Sam look for the ball?", "context...")
    assert ans and llm.calls >= 2                                # filter + answer = 2 calls


# ── (4) Relationship dynamics ──

def test_relationship_strength_evolves_and_clamps():
    book = RelationshipBook()
    r = book.record("alice", valence=4.0, gain=5.0)              # +20 -> 70
    assert r.strength == 70.0 and r.interactions == 1
    book.record("alice", valence=-3.0)                          # -15 -> 55
    assert book.get("alice").strength == 55.0
    for _ in range(20):
        book.record("alice", valence=5.0)
    assert book.get("alice").strength == 100.0                   # clamped
    assert book.tier("alice") == "close"


def test_relationship_decay_toward_neutral():
    book = RelationshipBook([Relationship(target="bob", strength=90.0, last_updated=0.0)])
    book.decay(rate=5.0, now=10**12, idle_hours=24.0)            # very idle
    assert 50.0 <= book.get("bob").strength < 90.0


# ── (5) ACE playbook ──

def test_playbook_dedup_and_refine():
    pb = Playbook()
    pb.add("ask a clarifying question when the request is ambiguous")
    again = pb.add("Ask a clarifying question when the request is ambiguous.")  # dup (norm-equal)
    assert again.helpful == 2 and len(pb.all()) == 1            # merged, counter incremented

    bad = pb.add("interrupt the user often")
    pb.record_outcome(bad.id, helpful=False)
    pb.record_outcome(bad.id, helpful=False)
    assert bad.net < 0
    dropped = pb.refine(min_net=0)
    assert dropped == 1 and all(le.net >= 0 for le in pb.all())
    assert "clarifying question" in pb.render()


def test_lesson_net():
    le = Lesson(text="x", helpful=3, harmful=1)
    assert le.net == 2
