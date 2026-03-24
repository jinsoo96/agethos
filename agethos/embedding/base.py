"""임베딩 추상 인터페이스."""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingAdapter(ABC):
    """텍스트 → 벡터 임베딩 추상화."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """텍스트 목록 → 임베딩 벡터 목록."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """임베딩 벡터 차원."""
        ...
