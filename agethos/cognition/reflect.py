"""반성 모듈 — 축적된 경험에서 고수준 통찰 추출.

Generative Agents 논문의 Reflection 메커니즘 구현:
1. importance 합산이 threshold 초과 시 트리거
2. 최근 기억에서 focal points (핵심 질문) 생성
3. 각 focal point로 관련 기억 검색
4. 관련 기억에서 insight 생성 → depth=2 MemoryNode로 저장
"""

from __future__ import annotations

import time

from agethos.cognition.retrieve import Retriever
from agethos.llm.base import LLMAdapter
from agethos.memory.stream import MemoryStream
from agethos.models import MemoryNode, NodeType

_FOCAL_PROMPT = """\
Here are recent experiences:

{memories}

Based on these experiences, generate {n} questions that are most important and worth deep reflection.

Respond in JSON: {{"questions": ["question1", "question2", "question3"]}}"""

_INSIGHT_PROMPT = """\
Given the following question, derive an insight based on the related memories.

Question: {question}

Related memories:
{memories}

Respond in JSON:
{{"insight": "<insight content>", "evidence_indices": [<evidence memory indices>]}}"""


class Reflector:
    """기억 반성 엔진."""

    def __init__(
        self,
        llm: LLMAdapter,
        memory: MemoryStream,
        retriever: Retriever,
        threshold: float = 150.0,
    ):
        self._llm = llm
        self._memory = memory
        self._retriever = retriever
        self._threshold = threshold
        self._last_reflection_at: float = time.time()

    @property
    def last_reflection_at(self) -> float:
        return self._last_reflection_at

    async def should_reflect(self) -> bool:
        """마지막 reflection 이후 importance 합이 threshold 초과인지 확인."""
        total = await self._memory.importance_since(self._last_reflection_at)
        return total >= self._threshold

    async def reflect(self) -> list[MemoryNode]:
        """반성 실행 → insight MemoryNode 목록 반환 및 저장."""
        # 1. Generate focal points
        recent = await self._memory.get_recent(100)
        if not recent:
            return []

        focal_points = await self._generate_focal_points(recent)
        if not focal_points:
            return []

        # 2. Retrieve related memories per focal point
        focal_memories = await self._retriever.retrieve_for_reflection(focal_points)

        # 3. Generate insights
        insights: list[MemoryNode] = []
        for question, results in focal_memories.items():
            if not results:
                continue

            memory_text = "\n".join(
                f"{i}. [{r.node.node_type.value}] {r.node.description}"
                for i, r in enumerate(results)
            )

            try:
                data = await self._llm.generate_json(
                    system_prompt="You are a helper that analyzes experiences and derives insights.",
                    user_prompt=_INSIGHT_PROMPT.format(
                        question=question,
                        memories=memory_text,
                    ),
                )

                evidence_indices = data.get("evidence_indices", [])
                evidence_ids = [
                    results[i].node.id
                    for i in evidence_indices
                    if 0 <= i < len(results)
                ]

                # Max depth of evidence + 1
                max_depth = max((results[i].node.depth for i in evidence_indices if 0 <= i < len(results)), default=1)

                insight_node = MemoryNode(
                    node_type=NodeType.THOUGHT,
                    description=data.get("insight", ""),
                    importance=8.0,  # Reflections are high importance
                    depth=max_depth + 1,
                    evidence_ids=evidence_ids,
                )

                await self._memory.append(insight_node)
                insights.append(insight_node)
            except Exception:
                continue

        self._last_reflection_at = time.time()
        return insights

    async def _generate_focal_points(
        self,
        recent_memories: list[MemoryNode],
        n: int = 3,
    ) -> list[str]:
        """최근 기억에서 핵심 질문 N개 생성."""
        memory_text = "\n".join(
            f"- [{m.node_type.value}] {m.description}" for m in recent_memories[:50]
        )

        try:
            data = await self._llm.generate_json(
                system_prompt="You are a helper that analyzes experiences.",
                user_prompt=_FOCAL_PROMPT.format(memories=memory_text, n=n),
            )
            return data.get("questions", [])[:n]
        except Exception:
            return []
