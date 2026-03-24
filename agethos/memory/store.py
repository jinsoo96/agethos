"""기억 저장소 추상 인터페이스."""

from __future__ import annotations

from abc import ABC, abstractmethod

from agethos.models import MemoryNode


class StorageBackend(ABC):
    """기억 저장소 프로토콜."""

    @abstractmethod
    async def save(self, node: MemoryNode) -> None:
        """기억 노드 저장."""
        ...

    @abstractmethod
    async def get_all(self) -> list[MemoryNode]:
        """전체 기억 반환."""
        ...

    @abstractmethod
    async def get_recent(self, n: int) -> list[MemoryNode]:
        """최근 N개 기억 반환 (created_at 내림차순)."""
        ...

    @abstractmethod
    async def get_by_ids(self, ids: list[str]) -> list[MemoryNode]:
        """ID 목록으로 기억 조회."""
        ...

    @abstractmethod
    async def get_since(self, timestamp: float) -> list[MemoryNode]:
        """특정 시점 이후 기억 반환."""
        ...

    @abstractmethod
    async def update(self, node: MemoryNode) -> None:
        """기억 노드 업데이트."""
        ...

    @abstractmethod
    async def count(self) -> int:
        """총 기억 수."""
        ...
