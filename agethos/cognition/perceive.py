"""인지 모듈 — 외부 입력을 MemoryNode로 변환."""

from __future__ import annotations

from agethos.llm.base import LLMAdapter
from agethos.models import MemoryNode, NodeType

_IMPORTANCE_PROMPT = """\
다음 관찰/사건의 중요도를 1~10 정수로 평가하세요.
1 = 일상적이고 사소한 (예: 이를 닦음)
10 = 극적이고 인생을 바꾸는 (예: 이별, 승진)

관찰: {observation}

JSON 형식으로 응답: {{"importance": <정수>, "subject": "<주어>", "predicate": "<동사>", "object": "<목적어>", "keywords": ["<키워드1>", "<키워드2>"]}}"""


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
                system_prompt="당신은 관찰을 분석하는 도우미입니다.",
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
