"""기억 검색 점수 계산 — 5축 복합 스코어링.

score = w_r * recency + w_i * importance + w_v * relevance + w_vit * vitality + w_ctx * context
각 축은 후보군 내에서 min-max 정규화 후 가중합산.
synaptic-memory의 5-axis resonance scoring 참고.
"""

from __future__ import annotations

import math
import re
import time

from agethos.models import MemoryNode, RetrievalResult

_WORD = re.compile(r"[\w']+", re.UNICODE)


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _WORD.findall(text or "") if len(t) > 1}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """코사인 유사도 계산."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _min_max_normalize(values: list[float]) -> list[float]:
    """Min-max 정규화 → [0, 1]."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def compute_retrieval_scores(
    nodes: list[MemoryNode],
    query_embedding: list[float] | None = None,
    now: float | None = None,
    decay_factor: float = 0.995,
    weights: tuple[float, ...] = (1.0, 1.0, 1.0),
    context_tags: list[str] | None = None,
    query: str | None = None,
    salience_weight: float = 0.0,
) -> list[RetrievalResult]:
    """5축 복합 점수 기반 기억 검색.

    5-axis scoring (synaptic-memory 참고):
    - Recency: 시간 감쇠 (0.995^hours)
    - Importance: 중요도 (1-10)
    - Relevance: 의미적 유사도 (코사인)
    - Vitality: 활력도 (시간에 따라 감쇠)
    - Context: 맥락 일치도 (키워드 겹침)

    Args:
        nodes: 후보 기억 노드 목록.
        query_embedding: 쿼리 임베딩 벡터 (없으면 relevance=0).
        now: 현재 시간 (default: time.time()).
        decay_factor: recency 지수 감쇠 계수 (default: 0.995).
        weights: (recency, importance, relevance[, vitality, context]) 가중치.
            3-tuple이면 vitality=0, context=0으로 처리 (하위 호환).
        context_tags: 현재 세션의 맥락 키워드 (context 점수 계산용).

    Returns:
        점수 내림차순 RetrievalResult 목록.
    """
    if not nodes:
        return []

    if now is None:
        now = time.time()

    # Backward compatible: 3-tuple → pad to 5
    w = tuple(weights) + (0.0,) * (5 - len(weights)) if len(weights) < 5 else tuple(weights)
    w_r, w_i, w_v, w_vit, w_ctx = w[0], w[1], w[2], w[3], w[4]

    query_tokens = _tokens(query) if query else set()

    # Raw scores
    raw_recency = []
    raw_importance = []
    raw_relevance = []
    raw_vitality = []
    raw_context = []
    raw_salience = []

    for node in nodes:
        salience = max(0.0, min(1.0, getattr(node, "emotional_salience", 0.0)))
        raw_salience.append(salience)

        # Arousal-modulated decay: emotionally salient memories fade more slowly
        # (Emotional RAG / generative-agents gap). decay -> 1.0 as salience -> 1.0.
        eff_decay = decay_factor + (1.0 - decay_factor) * salience
        hours_since = max(0, (now - node.last_accessed) / 3600)
        raw_recency.append(eff_decay ** hours_since)
        raw_importance.append(node.importance / 10.0)

        if query_embedding and node.embedding:
            raw_relevance.append(cosine_similarity(query_embedding, node.embedding))
        elif query_tokens:
            # No embedding → keyword/lexical fallback (so relevance isn't always 0)
            node_tokens = _tokens(node.description) | {k.lower() for k in node.keywords}
            raw_relevance.append(_jaccard(query_tokens, node_tokens))
        else:
            raw_relevance.append(0.0)

        # Vitality (new axis)
        raw_vitality.append(getattr(node, 'vitality', 1.0))

        # Context match (Jaccard-like keyword overlap)
        if context_tags and node.keywords:
            node_kw = set(k.lower() for k in node.keywords)
            ctx_kw = set(k.lower() for k in context_tags)
            intersection = len(node_kw & ctx_kw)
            union = len(node_kw | ctx_kw)
            raw_context.append(intersection / union if union > 0 else 0.0)
        else:
            raw_context.append(0.0)

    # Normalize
    norm_recency = _min_max_normalize(raw_recency)
    norm_importance = _min_max_normalize(raw_importance)
    norm_relevance = _min_max_normalize(raw_relevance)
    norm_vitality = _min_max_normalize(raw_vitality)
    norm_context = _min_max_normalize(raw_context)
    norm_salience = _min_max_normalize(raw_salience)

    # Weighted sum
    results = []
    for i, node in enumerate(nodes):
        score = (
            w_r * norm_recency[i]
            + w_i * norm_importance[i]
            + w_v * norm_relevance[i]
            + w_vit * norm_vitality[i]
            + w_ctx * norm_context[i]
            + salience_weight * norm_salience[i]
        )
        results.append(
            RetrievalResult(
                node=node,
                score=score,
                recency_score=norm_recency[i],
                importance_score=norm_importance[i],
                relevance_score=norm_relevance[i],
                vitality_score=norm_vitality[i],
                context_score=norm_context[i],
                salience_score=norm_salience[i],
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)
    return results
