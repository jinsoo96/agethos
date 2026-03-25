"""관찰 학습 모듈 — 외부 채팅을 관찰하여 사회적 패턴 추출.

Autopilot이 "참여형 자율주행"이라면, Observer는 "관찰형 자율주행".
에이전트가 발언하지 않고 대화를 관찰하며 사회적 규범/전략 패턴을 추출.

3단계 파이프라인:
1. Observe: 채팅 기록을 수집하고 각 발언의 맥락 분석
2. Extract: 반복되는 사회적 패턴 추출 (LLM reflection)
3. Internalize: 추출된 패턴을 SocialPattern으로 구조화
"""

from __future__ import annotations

from agethos.environment import Environment
from agethos.llm.base import LLMAdapter
from agethos.models import EnvironmentEvent, MemoryNode, NodeType, SocialPattern

_OBSERVE_PROMPT = """\
You are analyzing a conversation from the community "{community}".
Your goal is to understand what social behaviors work well and what doesn't.

Conversation chunk:
{conversation}

Analyze the social dynamics and identify patterns. For each pattern, describe:
1. The context/situation
2. What strategy was effective
3. What strategy was ineffective (if observed)

Respond in JSON:
{{
  "patterns": [
    {{
      "context": "<situation description>",
      "effective_strategy": "<what worked>",
      "counterexample": "<what didn't work, or null>",
      "confidence": <0.0 to 1.0>
    }}
  ]
}}

Focus on generalizable social patterns, not specific content. Return 1-5 patterns."""


class Observer:
    """관찰 학습 엔진 — 외부 대화에서 사회적 패턴 추출.

    Usage::

        observer = Observer(brain=brain, llm=llm, community_name="Python Discord")
        patterns = await observer.observe(env, max_messages=500)
    """

    def __init__(
        self,
        brain,
        llm: LLMAdapter,
        community_name: str = "",
        chunk_size: int = 20,
    ):
        self._brain = brain
        self._llm = llm
        self._community_name = community_name
        self._chunk_size = chunk_size

    async def observe(
        self,
        env: Environment,
        max_messages: int = 500,
    ) -> list[SocialPattern]:
        """환경에서 대화를 관찰하고 사회적 패턴 추출.

        Args:
            env: 채팅 소스 Environment.
            max_messages: 최대 관찰 메시지 수.

        Returns:
            추출된 SocialPattern 목록.
        """
        # Stage 1: Collect messages
        messages: list[EnvironmentEvent] = []
        collected = 0
        while collected < max_messages:
            events = await env.poll()
            if not events:
                break
            messages.extend(events)
            collected += len(events)

        if not messages:
            return []

        # Store observations in memory (silent — no response)
        for msg in messages:
            node = MemoryNode(
                node_type=NodeType.EVENT,
                description=f"[Observed in {self._community_name}] {msg.sender}: {msg.content}",
                importance=2.0,  # Observations are low importance individually
                subject=msg.sender,
                predicate="said",
                obj=msg.content[:100],
            )
            await self._brain.memory.append(node)

        # Stage 2: Extract patterns from chunks
        all_patterns: list[SocialPattern] = []
        chunks = self._chunk_messages(messages)

        for chunk in chunks:
            patterns = await self._extract_patterns(chunk)
            all_patterns.extend(patterns)

        # Stage 3: Merge duplicate patterns
        merged = self._merge_patterns(all_patterns)

        return merged

    def _chunk_messages(self, messages: list[EnvironmentEvent]) -> list[list[EnvironmentEvent]]:
        """메시지를 분석 가능한 청크로 분할."""
        chunks = []
        for i in range(0, len(messages), self._chunk_size):
            chunk = messages[i:i + self._chunk_size]
            if chunk:
                chunks.append(chunk)
        return chunks

    async def _extract_patterns(self, messages: list[EnvironmentEvent]) -> list[SocialPattern]:
        """메시지 청크에서 사회적 패턴 추출."""
        conversation = "\n".join(
            f"{msg.sender}: {msg.content}" for msg in messages
        )

        try:
            data = await self._llm.generate_json(
                system_prompt="You analyze social dynamics in conversations to extract behavioral patterns.",
                user_prompt=_OBSERVE_PROMPT.format(
                    community=self._community_name or "unknown",
                    conversation=conversation,
                ),
            )

            patterns = []
            for p in data.get("patterns", []):
                pattern = SocialPattern(
                    context=p.get("context", ""),
                    effective_strategy=p.get("effective_strategy", ""),
                    counterexample=p.get("counterexample"),
                    confidence=max(0.0, min(1.0, float(p.get("confidence", 0.5)))),
                    community=self._community_name,
                )
                if pattern.context and pattern.effective_strategy:
                    patterns.append(pattern)
            return patterns
        except Exception:
            return []

    def _merge_patterns(self, patterns: list[SocialPattern]) -> list[SocialPattern]:
        """유사 패턴 병합 — context가 겹치면 evidence_count 증가."""
        if not patterns:
            return []

        merged: dict[str, SocialPattern] = {}
        for p in patterns:
            # Simple key: lowercase context prefix
            key = p.context.lower().strip()[:50]
            if key in merged:
                existing = merged[key]
                existing.evidence_count += 1
                existing.confidence = min(1.0, existing.confidence + 0.1)
                if p.counterexample and not existing.counterexample:
                    existing.counterexample = p.counterexample
            else:
                merged[key] = p

        return list(merged.values())
