"""인지 모듈 — 외부 입력을 MemoryNode로 변환."""

from __future__ import annotations

from agethos.llm.base import LLMAdapter
from agethos.models import MemoryNode, NodeType

_IMPORTANCE_PROMPT = """\
Rate the importance of the following observation on a scale of 1 to 10.
1 = mundane and trivial (e.g. brushing teeth)
10 = dramatic and life-changing (e.g. breakup, promotion)

Observation: {observation}

Respond in JSON: {{"importance": <integer>, "subject": "<subject>", "predicate": "<verb>", "object": "<object>", "keywords": ["<keyword1>", "<keyword2>"]}}"""


class Perceiver:
    """외부 입력을 MemoryNode로 변환.

    LLM을 사용해 importance(1-10), SPO triple, keywords를 추출한다.
    """

    def __init__(self, llm: LLMAdapter):
        self._llm = llm

    async def perceive(
        self,
        observation: str,
        node_type: NodeType = NodeType.EVENT,
    ) -> MemoryNode:
        """관찰 → MemoryNode."""
        try:
            data = await self._llm.generate_json(
                system_prompt="You are a helper that analyzes observations.",
                user_prompt=_IMPORTANCE_PROMPT.format(observation=observation),
            )
            return MemoryNode(
                node_type=node_type,
                description=observation,
                subject=data.get("subject", ""),
                predicate=data.get("predicate", ""),
                obj=data.get("object", ""),
                importance=float(data.get("importance", 5)),
                keywords=data.get("keywords", []),
            )
        except Exception:
            # LLM 실패 시 기본값으로 생성
            return MemoryNode(
                node_type=node_type,
                description=observation,
                importance=5.0,
            )
