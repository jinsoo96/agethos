"""인메모리 저장소 구현."""

from __future__ import annotations

from agethos.memory.store import StorageBackend
from agethos.models import MemoryNode


class InMemoryStore(StorageBackend):
    """리스트 기반 인메모리 저장소. 개발/테스트용."""

    def __init__(self) -> None:
        self._nodes: list[MemoryNode] = []
        self._index: dict[str, int] = {}

    async def save(self, node: MemoryNode) -> None:
        self._index[node.id] = len(self._nodes)
        self._nodes.append(node)

    async def get_all(self) -> list[MemoryNode]:
        return list(self._nodes)

    async def get_recent(self, n: int) -> list[MemoryNode]:
        sorted_nodes = sorted(self._nodes, key=lambda x: x.created_at, reverse=True)
        return sorted_nodes[:n]

    async def get_by_ids(self, ids: list[str]) -> list[MemoryNode]:
        id_set = set(ids)
        return [n for n in self._nodes if n.id in id_set]

    async def get_since(self, timestamp: float) -> list[MemoryNode]:
        return [n for n in self._nodes if n.created_at >= timestamp]

    async def update(self, node: MemoryNode) -> None:
        if node.id in self._index:
            idx = self._index[node.id]
            self._nodes[idx] = node

    async def count(self) -> int:
        return len(self._nodes)
