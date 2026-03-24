"""기억 스트림 — 모든 경험의 시간순 저장소."""

from __future__ import annotations

import time

from agethos.embedding.base import EmbeddingAdapter
from agethos.memory.retrieval import compute_retrieval_scores
from agethos.memory.store import StorageBackend
from agethos.models import MemoryNode, RetrievalResult


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

    async def append(self, node: MemoryNode) -> MemoryNode:
        """기억 추가. 임베딩 자동 생성."""
        if self._embedder and node.embedding is None:
            embeddings = await self._embedder.embed([node.description])
            node.embedding = embeddings[0]
        await self._store.save(node)
        return node

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> list[RetrievalResult]:
        """복합 점수 검색."""
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
        )

        # Update last_accessed for retrieved nodes
        now = time.time()
        for r in results[:top_k]:
            r.node.last_accessed = now
            r.node.access_count += 1
            await self._store.update(r.node)

        return results[:top_k]

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
