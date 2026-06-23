"""Mem0-style arbitrated memory write — ADD / UPDATE / DELETE / NOOP.

Instead of blindly appending every observation (uncontrolled growth, duplicates,
contradictions), each new memory is checked against the top-K most similar existing
memories and an arbiter decides what to do. With an LLM the decision is made by the
model (dedup/merge/contradiction handling, Mem0); without one it falls back to a
deterministic similarity rule, so it works zero-infra.
"""
from __future__ import annotations

from agethos.memory.retrieval import _jaccard, _tokens, compute_retrieval_scores
from agethos.models import EmotionalState, MemoryNode

_ARBITER_PROMPT = """\
A new memory is being written. Decide how it relates to existing similar memories.

New memory: {new}

Existing similar memories:
{candidates}

Choose ONE operation:
- ADD: it is genuinely new information.
- UPDATE: it refines/extends an existing memory (give its id).
- DELETE: it contradicts/obsoletes an existing memory that should be removed (give its id), then the new one is added.
- NOOP: it is already captured by an existing memory (give its id); do not store it.

Respond in JSON: {{"op": "ADD|UPDATE|DELETE|NOOP", "target_id": "<id or empty>"}}"""


class MemoryArbiter:
    """Decide ADD/UPDATE/DELETE/NOOP for a new memory vs similar ones."""

    def __init__(self, llm=None, sim_threshold: float = 0.85, dup_threshold: float = 0.95):
        self.llm = llm
        self.sim_threshold = sim_threshold
        self.dup_threshold = dup_threshold

    async def decide(self, new_node: MemoryNode, candidates: list[MemoryNode]) -> tuple[str, str | None]:
        if not candidates:
            return ("ADD", None)
        if self.llm is not None:
            try:
                listing = "\n".join(f"- [{c.id}] {c.description}" for c in candidates)
                data = await self.llm.generate_json(
                    system_prompt="You curate an agent's memory store, deduping and merging.",
                    user_prompt=_ARBITER_PROMPT.format(new=new_node.description, candidates=listing),
                )
                op = str(data.get("op", "ADD")).upper()
                tid = data.get("target_id") or None
                if op in ("ADD", "UPDATE", "DELETE", "NOOP"):
                    if op == "ADD":
                        return ("ADD", None)
                    if tid and any(c.id == tid for c in candidates):
                        return (op, tid)
            except Exception:
                pass
        # deterministic fallback: similarity rule
        nt = _tokens(new_node.description)
        best = max(candidates, key=lambda c: _jaccard(nt, _tokens(c.description)))
        sim = _jaccard(nt, _tokens(best.description))
        if sim >= self.dup_threshold:
            return ("NOOP", best.id)
        if sim >= self.sim_threshold:
            return ("UPDATE", best.id)
        return ("ADD", None)


async def remember(
    stream,
    node: MemoryNode,
    arbiter: MemoryArbiter | None = None,
    *,
    top_k: int = 8,
    current_emotion: EmotionalState | None = None,
) -> tuple[str, MemoryNode]:
    """Write a memory through the arbiter. Returns (operation, resulting_node).

    ADD → appended; UPDATE → merged into the target; DELETE → target removed then added;
    NOOP → nothing stored (target returned)."""
    arbiter = arbiter or MemoryArbiter()
    all_nodes = await stream.store.get_all()
    candidates: list[MemoryNode] = []
    if all_nodes:
        scored = compute_retrieval_scores(all_nodes, query=node.description, weights=(0.0, 0.0, 1.0))
        candidates = [r.node for r in scored[:top_k] if r.relevance_score > 0][:top_k]

    op, target_id = await arbiter.decide(node, candidates)
    target = next((c for c in candidates if c.id == target_id), None)

    if op == "NOOP" and target is not None:
        return ("NOOP", target)
    if op == "UPDATE" and target is not None:
        target.description = node.description
        target.keywords = list(dict.fromkeys([*target.keywords, *node.keywords]))
        target.importance = max(target.importance, node.importance)
        if node.id not in target.links:
            target.links.append(node.id)
        await stream.store.update(target)
        return ("UPDATE", target)
    if op == "DELETE" and target is not None:
        try:
            await stream.store.delete(target.id)
        except NotImplementedError:
            pass
    await stream.append(node, current_emotion=current_emotion)
    return ("ADD" if op != "DELETE" else "DELETE", node)
