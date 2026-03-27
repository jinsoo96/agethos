"""기억 검색 모듈 — 상황별 가중치 프리셋 포함.

Generative Agents 코드의 실제 가중치 [0.5, 3, 2] 참고.
synaptic-memory의 intent-aware 검색 개념 적용.
"""

from __future__ import annotations

from agethos.memory.stream import MemoryStream
from agethos.models import RetrievalResult


# Intent-aware retrieval weight presets
# (recency, importance, relevance[, vitality, context])
# 3-tuple presets: backward compatible, vitality/context default to 0
# 5-tuple presets: full 5-axis scoring
RETRIEVAL_PRESETS: dict[str, tuple[float, ...]] = {
    "default": (1.0, 1.0, 1.0),                  # 균등 (3-axis)
    "recall": (0.5, 2.0, 3.0),                   # 회상: relevance 최우선
    "planning": (2.0, 1.5, 0.5),                  # 계획: 최근 기억 우선
    "reflection": (0.5, 3.0, 2.0),                # 반성: importance 최우선
    "observation": (1.0, 0.5, 2.0),               # 관찰: relevance + 최근
    "conversation": (1.5, 1.0, 2.0),              # 대화: relevance + recency
    "failure_analysis": (0.5, 2.0, 3.0),          # 실패 분석: relevance 집중
    "exploration": (1.0, 0.5, 1.0),               # 탐색: 넓은 범위
    # 5-axis presets (v0.7.0)
    "deep_recall": (0.5, 2.0, 3.0, 1.0, 0.5),   # 심층 회상: vitality 포함
    "contextual": (1.0, 1.0, 2.0, 0.5, 2.0),     # 맥락 기반: context 축 활용
    "social": (1.5, 1.5, 2.0, 1.0, 1.5),         # 사회적: 5축 균형
    "past_failures": (0.5, 2.5, 3.0, 0.5, 0.0),  # 과거 실패: importance 강화
}


class Retriever:
    """기억 검색. MemoryStream 래퍼 + 상황별 가중치 프리셋.

    Usage::

        retriever = Retriever(memory)

        # 기본 검색
        results = await retriever.retrieve("query")

        # 프리셋 사용
        results = await retriever.retrieve("query", preset="recall")
        results = await retriever.retrieve("query", preset="planning")

        # 커스텀 가중치
        results = await retriever.retrieve("query", weights=(0.5, 3.0, 2.0))
    """

    def __init__(
        self,
        memory: MemoryStream,
        default_weights: tuple[float, ...] = (1.0, 1.0, 1.0),
    ):
        self._memory = memory
        self._default_weights = default_weights

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        weights: tuple[float, ...] | None = None,
        preset: str | None = None,
    ) -> list[RetrievalResult]:
        """쿼리로 관련 기억 검색.

        Args:
            query: 검색 쿼리.
            top_k: 반환할 최대 결과 수.
            weights: (recency, importance, relevance) 직접 지정.
            preset: 프리셋 이름 (weights보다 우선).
        """
        if preset and preset in RETRIEVAL_PRESETS:
            w = RETRIEVAL_PRESETS[preset]
        elif weights:
            w = weights
        else:
            w = self._default_weights

        return await self._memory.retrieve(
            query=query,
            top_k=top_k,
            weights=w,
        )

    async def retrieve_for_reflection(
        self,
        focal_points: list[str],
        per_focal_k: int = 5,
    ) -> dict[str, list[RetrievalResult]]:
        """각 focal point에 대해 관련 기억 검색 (reflection 프리셋 사용)."""
        results: dict[str, list[RetrievalResult]] = {}
        for fp in focal_points:
            results[fp] = await self.retrieve(query=fp, top_k=per_focal_k, preset="reflection")
        return results
