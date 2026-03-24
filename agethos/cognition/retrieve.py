"""기억 검색 모듈."""

from __future__ import annotations

from agethos.memory.stream import MemoryStream
from agethos.models import RetrievalResult


class Retriever:
    """기억 검색. MemoryStream 래퍼."""

    def __init__(
        self,
        memory: MemoryStream,
        default_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):
        self._memory = memory
        self._default_weights = default_weights

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        weights: tuple[float, float, float] | None = None,
    ) -> list[RetrievalResult]:
        """쿼리로 관련 기억 검색."""
        return await self._memory.retrieve(
            query=query,
            top_k=top_k,
            weights=weights or self._default_weights,
        )

    async def retrieve_for_reflection(
        self,
        focal_points: list[str],
        per_focal_k: int = 5,
    ) -> dict[str, list[RetrievalResult]]:
        """각 focal point에 대해 관련 기억 검색."""
        results: dict[str, list[RetrievalResult]] = {}
        for fp in focal_points:
            results[fp] = await self.retrieve(query=fp, top_k=per_focal_k)
        return results
