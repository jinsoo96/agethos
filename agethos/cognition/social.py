"""사회적 인지 모듈 — 대화 맥락에서 '눈치'를 읽고 성격에 맞는 사회적 전략을 결정."""

from __future__ import annotations

from pydantic import BaseModel

from agethos.llm.base import LLMAdapter
from agethos.models import OceanTraits


_SOCIAL_READ_PROMPT = """\
You are analyzing a workplace conversation to determine the social dynamics.

Conversation:
{conversation}

Analyze and respond in JSON:
{{
  "atmosphere": "<tense|neutral|friendly|urgent|formal>",
  "tension_level": <0.0 to 1.0>,
  "key_dynamics": "<brief description of what's happening socially>",
  "unresolved": "<any unresolved issues or implicit expectations>",
  "emotional_undercurrent": "<what people might be feeling but not saying>"
}}"""

_STRATEGY_PROMPT = """\
You are {name}, with this personality:
- Extraversion: {E:.2f} ({E_desc})
- Agreeableness: {A:.2f} ({A_desc})
- Conscientiousness: {C:.2f} ({C_desc})
- Neuroticism: {N:.2f} ({N_desc})
- Openness: {O:.2f} ({O_desc})

Role/Position: {role}

Social context of the conversation:
- Atmosphere: {atmosphere}
- Tension: {tension:.0%}
- Dynamics: {dynamics}
- Unresolved: {unresolved}
- Undercurrent: {undercurrent}

Recent conversation:
{conversation}

Based on your personality and the social context, decide your response strategy.

Respond in JSON:
{{
  "strategy": "<agree|empathize|challenge|inform|deflect|support|take_charge|stay_silent>",
  "tone": "<description of how you'd speak>",
  "mirror_style": <true if you should match the other person's communication style>,
  "initiative_level": <0.0 to 1.0, how proactively you'd jump in>,
  "response": "<your actual response in Korean, natural workplace language>",
  "reasoning": "<why this strategy fits your personality and the situation>"
}}

Strategy guide:
- agree: 동조, 맞장구
- empathize: 공감, 위로
- challenge: 반론, 질문 제기
- inform: 정보 전달, 보고
- deflect: 화제 전환, 넘어가기
- support: 지원 제안, 도움
- take_charge: 주도적으로 정리, 지시
- stay_silent: 침묵 (할 말 없음)"""


class SocialContext(BaseModel):
    """대화의 사회적 맥락."""

    atmosphere: str = "neutral"
    tension_level: float = 0.0
    key_dynamics: str = ""
    unresolved: str = ""
    emotional_undercurrent: str = ""


class SocialStrategy(BaseModel):
    """성격 기반 사회적 전략."""

    strategy: str = "stay_silent"
    tone: str = ""
    mirror_style: bool = False
    initiative_level: float = 0.5
    response: str = ""
    reasoning: str = ""


class SocialCognition:
    """눈치 — 대화 맥락을 읽고 성격에 맞는 사회적 전략 결정.

    동작 원리:
    - 대화 전체 분위기/긴장도/암묵적 기대를 파악 (read_context)
    - 성격(OCEAN)에 따라 동조/반박/공감/주도 등 전략 결정 (decide_strategy)

    성격별 경향:
    - A 높음 → 동조, 공감, 갈등 회피
    - A 낮음 → 직설, 반론, 문제 지적
    - E 높음 → 적극 개입, 주도
    - E 낮음 → 침묵, 필요할 때만 발언
    - C 높음 → 정리, 체계적 보고
    - N 높음 → 긴장에 민감, 방어적
    - O 높음 → 유연한 전략 전환
    """

    def __init__(self, llm: LLMAdapter, name: str, ocean: OceanTraits | None = None, role: str = ""):
        self._llm = llm
        self._name = name
        self._ocean = ocean or OceanTraits()
        self._role = role
        self._context_history: list[SocialContext] = []

    async def read_context(self, conversation: str) -> SocialContext:
        """대화에서 사회적 맥락을 읽는다."""
        try:
            data = await self._llm.generate_json(
                system_prompt="You analyze social dynamics in workplace conversations.",
                user_prompt=_SOCIAL_READ_PROMPT.format(conversation=conversation),
            )
            ctx = SocialContext(
                atmosphere=data.get("atmosphere", "neutral"),
                tension_level=max(0, min(1, float(data.get("tension_level", 0)))),
                key_dynamics=data.get("key_dynamics", ""),
                unresolved=data.get("unresolved", ""),
                emotional_undercurrent=data.get("emotional_undercurrent", ""),
            )
            self._context_history.append(ctx)
            return ctx
        except Exception:
            return SocialContext()

    async def decide_strategy(self, conversation: str, context: SocialContext | None = None) -> SocialStrategy:
        """성격과 맥락에 기반한 사회적 전략 결정."""
        if context is None:
            context = await self.read_context(conversation)

        o = self._ocean

        def _desc(val: float, high: str, low: str) -> str:
            if val > 0.66:
                return high
            if val < 0.33:
                return low
            return "balanced"

        try:
            data = await self._llm.generate_json(
                system_prompt="You decide social strategies based on personality.",
                user_prompt=_STRATEGY_PROMPT.format(
                    name=self._name,
                    E=o.extraversion,
                    A=o.agreeableness,
                    C=o.conscientiousness,
                    N=o.neuroticism,
                    O=o.openness,
                    E_desc=_desc(o.extraversion, "outgoing, takes initiative", "reserved, waits"),
                    A_desc=_desc(o.agreeableness, "cooperative, agreeable", "direct, challenges"),
                    C_desc=_desc(o.conscientiousness, "organized, thorough", "spontaneous, flexible"),
                    N_desc=_desc(o.neuroticism, "sensitive to tension", "calm under pressure"),
                    O_desc=_desc(o.openness, "flexible, open to change", "prefers routine"),
                    role=self._role or "team member",
                    atmosphere=context.atmosphere,
                    tension=context.tension_level,
                    dynamics=context.key_dynamics,
                    unresolved=context.unresolved,
                    undercurrent=context.emotional_undercurrent,
                    conversation=conversation,
                ),
            )
            return SocialStrategy(
                strategy=data.get("strategy", "stay_silent"),
                tone=data.get("tone", ""),
                mirror_style=bool(data.get("mirror_style", False)),
                initiative_level=max(0, min(1, float(data.get("initiative_level", 0.5)))),
                response=data.get("response", ""),
                reasoning=data.get("reasoning", ""),
            )
        except Exception:
            return SocialStrategy()

    async def universalize_check(self, action: str, context: str = "") -> dict:
        """칸트 보편화 원리 — "모두가 이걸 하면 어떻게 되나?"

        이기적 전략을 억제하고 협력적 행동을 유도.
        GovSim 연구에서 이 프롬프트만으로 지속 가능한 협력 달성 확인.

        Args:
            action: 평가할 행동.
            context: 상황 설명 (선택).

        Returns:
            {"should_proceed": bool, "reasoning": str, "impact": str}
        """
        prompt = (
            f"Consider this action: {action}\n"
            + (f"Context: {context}\n" if context else "")
            + "\nApply the universalization test: What if EVERYONE in this community did this?\n"
            "Would it lead to a sustainable, positive outcome for the group?\n\n"
            "Respond in JSON:\n"
            '{"should_proceed": <true/false>, "reasoning": "<why>", "impact": "<predicted group impact>"}'
        )
        try:
            return await self._llm.generate_json(
                system_prompt="You evaluate actions using Kant's universalization principle for group cooperation.",
                user_prompt=prompt,
            )
        except Exception:
            return {"should_proceed": True, "reasoning": "evaluation failed", "impact": "unknown"}

    @property
    def context_history(self) -> list[SocialContext]:
        return list(self._context_history)
