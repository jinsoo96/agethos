"""LoCoMo-style long-term memory evaluation harness.

LoCoMo (Maharana et al. 2024) is the standard long conversational-memory benchmark. The
dataset is external; this ships the *harness*: ingest a conversation into a MemoryStream,
then measure whether retrieval surfaces the evidence each question needs (recall / NDCG /
MRR). Plug an LLM answerer/judge for end-to-end accuracy; the retrieval scoring is
deterministic and offline-testable.
"""
from __future__ import annotations

import json

from agethos.eval.metrics import retrieval_metrics
from agethos.models import MemoryNode


async def ingest_conversation(stream, turns: list, current_emotion=None) -> list[MemoryNode]:
    """Append conversation turns to the memory stream.

    ``turns`` items may be strings or dicts ({"text"/"content"/"speaker"})."""
    nodes: list[MemoryNode] = []
    for t in turns:
        if isinstance(t, dict):
            speaker = t.get("speaker") or t.get("sender") or ""
            text = t.get("text") or t.get("content") or t.get("utterance") or ""
            desc = f"{speaker}: {text}" if speaker else text
        else:
            desc = str(t)
        if not desc.strip():
            continue
        nodes.append(await stream.append(MemoryNode(description=desc), current_emotion=current_emotion))
    return nodes


async def evaluate_recall(stream, qa_items: list[dict], top_k: int = 10) -> dict:
    """Mean retrieval metrics over QA items.

    Each item: ``{"question": str, "evidence": [substrings that mark a relevant memory]}``.
    A retrieved memory is relevant if it contains any evidence substring."""
    agg = {"precision": 0.0, "recall": 0.0, "mrr": 0.0, "ndcg": 0.0}
    if not qa_items:
        return {**agg, "n": 0}
    for qa in qa_items:
        results = await stream.retrieve(qa["question"], top_k=top_k, weights=(1.0, 0.5, 2.0))
        ranked_ids = [r.node.id for r in results]
        evidence = [e.lower() for e in qa.get("evidence", [])]
        relevant = [r.node.id for r in results
                    if any(ev in r.node.description.lower() for ev in evidence)]
        m = retrieval_metrics(ranked_ids, relevant, k=top_k)
        for key in agg:
            agg[key] += m[key]
    n = len(qa_items)
    return {**{k: round(v / n, 4) for k, v in agg.items()}, "n": n}


def load_locomo(path: str) -> list[dict]:
    """Load a LoCoMo JSON file into a loose list of samples (format-tolerant)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("samples", [data])
