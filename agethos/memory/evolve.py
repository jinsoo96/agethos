"""A-MEM style memory evolution — link a new memory to its neighbors and let them
evolve.

When a memory is written, find its nearest neighbors, form bidirectional associative
links (a Zettelkasten graph over the memory stream), and update the neighbors' keyword
sets so the graph is *generative*, not just associative. With an LLM the neighbor
context is rewritten; without one, keywords are merged (deterministic).
"""
from __future__ import annotations

from agethos.concurrency import amap
from agethos.memory.retrieval import compute_retrieval_scores
from agethos.models import MemoryNode

_EVOLVE_PROMPT = """\
A new memory was linked to an existing one. Refine the existing memory's keywords so it
reflects the connection (keep it concise, <= 8 keywords).

New memory: {new}
Existing memory: {old}
Existing keywords: {keywords}

Respond in JSON: {{"keywords": ["...", "..."]}}"""


async def link_and_evolve(
    stream,
    node: MemoryNode,
    llm=None,
    *,
    k: int = 5,
    min_relevance: float = 0.0,
    max_keywords: int = 8,
) -> list[MemoryNode]:
    """Link ``node`` to its k nearest neighbors and evolve their keywords. Returns neighbors.

    ``node`` should already be in the store. Links are bidirectional; neighbor keywords
    gain the new memory's keywords (LLM-rewritten if ``llm`` given, else merged)."""
    all_nodes = [n for n in await stream.store.get_all() if n.id != node.id]
    if not all_nodes:
        return []
    scored = compute_retrieval_scores(all_nodes, query=node.description, weights=(0.0, 0.0, 1.0))
    neighbors = [r.node for r in scored[:k] if r.relevance_score > min_relevance]
    if not neighbors:
        return []

    node.links = list(dict.fromkeys([*node.links, *(n.id for n in neighbors)]))
    await stream.store.update(node)

    async def _evolve_neighbor(nb: MemoryNode) -> None:
        if node.id not in nb.links:
            nb.links.append(node.id)
        new_keywords = None
        if llm is not None:
            try:
                data = await llm.generate_json(
                    system_prompt="You refine an agent's memory keywords as its memory graph grows.",
                    user_prompt=_EVOLVE_PROMPT.format(
                        new=node.description, old=nb.description,
                        keywords=", ".join(nb.keywords) or "none"),
                )
                kws = data.get("keywords")
                if isinstance(kws, list) and kws:
                    new_keywords = [str(x) for x in kws][:max_keywords]
            except Exception:
                new_keywords = None
        if new_keywords is None:
            new_keywords = list(dict.fromkeys([*nb.keywords, *node.keywords]))[:max_keywords]
        nb.keywords = new_keywords
        await stream.store.update(nb)

    # neighbors are independent → evolve concurrently (bounded)
    await amap(_evolve_neighbor, neighbors)
    return neighbors
