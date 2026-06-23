"""기억 스트림 — 모든 경험의 시간순 저장소."""

from __future__ import annotations

import time

from agethos.embedding.base import EmbeddingAdapter
from agethos.memory.retrieval import compute_retrieval_scores
from agethos.memory.store import StorageBackend
from agethos.models import EmotionalState, MemoryNode, RetrievalResult


class MemoryStream:
    """기억 스트림.

    append() → 기록
    retrieve() → recency + importance + relevance 복합 점수로 검색
    get_recent() → 최근 N개
    importance_since() → 특정 시점 이후 importance 합산
    """

    def __init__(
        self,
        store: StorageBackend,
        embedder: EmbeddingAdapter | None = None,
        decay_factor: float = 0.995,
    ):
        self._store = store
        self._embedder = embedder
        self._decay_factor = decay_factor

    @property
    def store(self) -> StorageBackend:
        return self._store

    async def append(
        self,
        node: MemoryNode,
        current_emotion: EmotionalState | None = None,
    ) -> MemoryNode:
        """기억 추가. 임베딩 자동 생성 + (선택) 인코딩 시점 감정 태깅.

        current_emotion 을 주면 encoding_pad 와 emotional_salience(=|arousal|)를 기록 →
        각성도 높은 사건이 검색에서 더 잘 살아남고 천천히 잊힌다(감정→기억 결합)."""
        if self._embedder and node.embedding is None:
            embeddings = await self._embedder.embed([node.description])
            node.embedding = embeddings[0]
        if current_emotion is not None and node.encoding_pad is None:
            node.encoding_pad = (current_emotion.pleasure, current_emotion.arousal,
                                 current_emotion.dominance)
            node.emotional_salience = abs(current_emotion.arousal)
        await self._store.save(node)
        return node

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
        salience_weight: float = 0.0,
        reinforce: float = 0.1,
    ) -> list[RetrievalResult]:
        """복합 점수 검색. 임베딩이 없으면 키워드(lexical) 폴백으로 relevance 계산.

        salience_weight>0 이면 감정 각성도(salience) 축을 가산. 검색된 노드는 vitality 가
        reinforce 만큼 강화된다(접근 강화 — Ebbinghaus, 자주 쓰는 기억은 오래 남는다)."""
        all_nodes = await self._store.get_all()
        if not all_nodes:
            return []

        query_embedding: list[float] | None = None
        if self._embedder:
            embeddings = await self._embedder.embed([query])
            query_embedding = embeddings[0]

        results = compute_retrieval_scores(
            nodes=all_nodes,
            query_embedding=query_embedding,
            decay_factor=self._decay_factor,
            weights=weights,
            query=query,
            salience_weight=salience_weight,
        )

        # Update last_accessed + reinforce vitality for retrieved nodes
        now = time.time()
        for r in results[:top_k]:
            r.node.last_accessed = now
            r.node.access_count += 1
            if reinforce:
                r.node.vitality = min(1.0, r.node.vitality + reinforce)
            await self._store.update(r.node)

        return results[:top_k]

    async def decay_vitality(self, rate: float = 0.02) -> None:
        """전체 기억의 vitality 를 rate 만큼 감쇠 (시간/주기 호출). 망각의 연료."""
        for node in await self._store.get_all():
            node.vitality = max(0.0, node.vitality - rate)
            await self._store.update(node)

    async def forget(self, threshold: float = 0.15) -> int:
        """약하고 감정가 낮은 기억을 가지치기. strength = vitality + 0.5·salience + importance/20.

        감정 각성도가 높거나 중요/최근 접근된 기억은 보호된다(LUFY식 각성 게이팅). 삭제 수 반환."""
        removed = 0
        for node in await self._store.get_all():
            strength = node.vitality + 0.5 * node.emotional_salience + node.importance / 20.0
            if strength < threshold:
                try:
                    await self._store.delete(node.id)
                    removed += 1
                except NotImplementedError:
                    break
        return removed

    async def get_recent(self, n: int = 100) -> list[MemoryNode]:
        """최근 N개 기억 반환."""
        return await self._store.get_recent(n)

    async def importance_since(self, timestamp: float) -> float:
        """시점 이후 추가된 기억의 importance 합산."""
        nodes = await self._store.get_since(timestamp)
        return sum(n.importance for n in nodes)

    async def get_by_ids(self, ids: list[str]) -> list[MemoryNode]:
        """ID 목록으로 기억 조회."""
        return await self._store.get_by_ids(ids)

    async def count(self) -> int:
        """총 기억 수."""
        return await self._store.count()
