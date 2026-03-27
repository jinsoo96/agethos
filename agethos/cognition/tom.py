"""Theory of Mind 모듈 — 상대방의 내면 모델 구축.

대화 중 상대의 믿음, 의도, 감정 상태를 추론하고 MentalModel로 구조화.
눈치(SocialCognition)가 "분위기"를 읽는다면, ToM은 "개인"을 읽는다.
"""

from __future__ import annotations

import time

from agethos.llm.base import LLMAdapter
from agethos.models import MentalModel, RelationshipType

_TOM_PROMPT = """\
You are analyzing a conversation to build a mental model of "{target}".

Conversation:
{conversation}

Based on what {target} said and how others responded, infer:
1. What are {target}'s likely goals or intentions?
2. What does {target} seem to know or not know?
3. What emotion is {target} likely feeling?
4. How would you summarize the relationship dynamics?

Respond in JSON:
{{
  "believed_goals": ["goal1", "goal2"],
  "believed_knowledge": ["knows X", "doesn't seem to know Y"],
  "believed_emotion": "<emotion label>",
  "relationship_summary": "<brief summary>",
  "confidence": <0.0 to 1.0>
}}"""

_TOM_UPDATE_PROMPT = """\
You are updating your mental model of "{target}".

Previous model:
- Goals: {prev_goals}
- Knowledge: {prev_knowledge}
- Emotion: {prev_emotion}
- Relationship: {prev_relationship}

New conversation:
{conversation}

Update the mental model based on new information. Keep previous inferences if still valid.

Respond in JSON:
{{
  "believed_goals": ["goal1", "goal2"],
  "believed_knowledge": ["knows X", "doesn't seem to know Y"],
  "believed_emotion": "<emotion label>",
  "relationship_summary": "<brief summary>",
  "confidence": <0.0 to 1.0>
}}"""

# Relationship → ToM inference depth (SOTOPIA)
_RELATIONSHIP_DEPTH = {
    RelationshipType.STRANGER: 1,      # 기본 추론만
    RelationshipType.ACQUAINTANCE: 2,  # 목표 + 감정
    RelationshipType.FRIEND: 3,        # 목표 + 감정 + 지식
    RelationshipType.FAMILY: 4,        # 전체 + recursive
    RelationshipType.ROMANTIC: 4,      # 전체 + recursive
}

_RECURSIVE_TOM_PROMPT = """\
You are performing recursive Theory of Mind analysis.

{name} is interacting with {target}.

Current mental model of {target}:
- Goals: {goals}
- Knowledge: {knowledge}
- Emotion: {emotion}

Now infer: What does {target} think {name} is thinking/feeling/wanting?
This is a second-order belief — {target}'s model of {name}.

Respond in JSON:
{{
  "recursive_belief": "<what {target} likely thinks about {name}'s state>",
  "confidence": <0.0 to 1.0>
}}"""


class TheoryOfMind:
    """Theory of Mind 엔진 — 상대방의 내면 모델 구축/갱신.

    Usage::

        tom = TheoryOfMind(llm=llm_adapter)
        model = await tom.infer("alice", conversation_text)
        # → MentalModel(target="alice", believed_goals=[...], ...)

        # Update with new conversation
        updated = await tom.update(model, new_conversation)
    """

    def __init__(self, llm: LLMAdapter):
        self._llm = llm

    async def infer(self, target: str, conversation: str) -> MentalModel:
        """대화에서 상대의 멘탈 모델 추론."""
        try:
            data = await self._llm.generate_json(
                system_prompt="You analyze conversations to infer people's mental states, goals, and knowledge.",
                user_prompt=_TOM_PROMPT.format(target=target, conversation=conversation),
            )
            return MentalModel(
                target=target,
                believed_goals=data.get("believed_goals", []),
                believed_knowledge=data.get("believed_knowledge", []),
                believed_emotion=data.get("believed_emotion", "neutral"),
                relationship_summary=data.get("relationship_summary", ""),
                confidence=max(0.0, min(1.0, float(data.get("confidence", 0.5)))),
            )
        except Exception:
            return MentalModel(target=target)

    async def update(self, model: MentalModel, conversation: str) -> MentalModel:
        """기존 멘탈 모델을 새 대화 정보로 갱신."""
        try:
            data = await self._llm.generate_json(
                system_prompt="You update mental models of people based on new conversation data.",
                user_prompt=_TOM_UPDATE_PROMPT.format(
                    target=model.target,
                    prev_goals=", ".join(model.believed_goals) or "unknown",
                    prev_knowledge=", ".join(model.believed_knowledge) or "unknown",
                    prev_emotion=model.believed_emotion,
                    prev_relationship=model.relationship_summary or "unknown",
                    conversation=conversation,
                ),
            )
            return MentalModel(
                target=model.target,
                believed_goals=data.get("believed_goals", model.believed_goals),
                believed_knowledge=data.get("believed_knowledge", model.believed_knowledge),
                believed_emotion=data.get("believed_emotion", model.believed_emotion),
                relationship_summary=data.get("relationship_summary", model.relationship_summary),
                confidence=max(0.0, min(1.0, float(data.get("confidence", model.confidence)))),
                last_updated=time.time(),
            )
        except Exception:
            return model

    async def infer_recursive(
        self,
        name: str,
        model: MentalModel,
    ) -> str:
        """2차 ToM — 상대가 나를 어떻게 보는지 추론 (Recursive ToM)."""
        try:
            data = await self._llm.generate_json(
                system_prompt="You perform recursive Theory of Mind — modeling what others think about you.",
                user_prompt=_RECURSIVE_TOM_PROMPT.format(
                    name=name,
                    target=model.target,
                    goals=", ".join(model.believed_goals) or "unknown",
                    knowledge=", ".join(model.believed_knowledge) or "unknown",
                    emotion=model.believed_emotion,
                ),
            )
            return data.get("recursive_belief", "")
        except Exception:
            return ""

    def get_inference_depth(self, relationship: RelationshipType) -> int:
        """관계 유형별 추론 깊이 반환."""
        return _RELATIONSHIP_DEPTH.get(relationship, 1)

    def to_prompt(self, model: MentalModel) -> str:
        """멘탈 모델을 시스템 프롬프트에 주입 가능한 텍스트로 변환."""
        parts = [f"Your understanding of {model.target} (relationship: {model.relationship_type.value}):"]
        if model.believed_goals:
            parts.append(f"- Goals: {', '.join(model.believed_goals)}")
        if model.believed_knowledge:
            parts.append(f"- Knowledge: {', '.join(model.believed_knowledge)}")
        parts.append(f"- Emotional state: {model.believed_emotion}")
        if model.relationship_summary:
            parts.append(f"- Relationship: {model.relationship_summary}")
        if model.recursive_belief:
            parts.append(f"- They likely think about you: {model.recursive_belief}")
        parts.append(f"- Confidence: {model.confidence:.0%}")
        return "\n".join(parts)
