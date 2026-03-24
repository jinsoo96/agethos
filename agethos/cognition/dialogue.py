"""대화 연속성 판단 모듈 — 대화 흐름을 추적하고 이어갈지 판단."""

from __future__ import annotations

from pydantic import BaseModel, Field

from agethos.llm.base import LLMAdapter
from agethos.models import OceanTraits

_DIALOGUE_JUDGE_PROMPT = """\
You are judging the flow of a conversation from {name}'s perspective.

Personality:
- Extraversion: {E:.2f} ({"talkative, seeks interaction" if E > 0.6 else "reserved, prefers listening" if E < 0.4 else "balanced"})
- Agreeableness: {A:.2f} ({"cooperative, avoids conflict" if A > 0.6 else "direct, comfortable with disagreement" if A < 0.4 else "balanced"})
- Openness: {O:.2f} ({"curious, explores new topics" if O > 0.6 else "prefers familiar topics" if O < 0.4 else "balanced"})

Current emotion: {emotion}
Topic so far: {topic}
Turn count: {turns}
Last {history_count} messages:
{recent_history}

Based on this character's personality, decide what they would naturally do next.

Respond in JSON:
{{
  "action": "<continue|redirect|disengage|initiate>",
  "topic": "<current or new topic summary>",
  "energy": <0.0 to 1.0, how engaged the character feels>,
  "reason": "<brief reasoning>"
}}

Actions:
- continue: keep talking about the current topic
- redirect: steer toward a new topic (character is bored or curious about something else)
- disengage: stop talking (conversation is done, or character has nothing to add)
- initiate: proactively start a new conversation (only when no active dialogue)"""


class DialogueState(BaseModel):
    """대화 흐름 상태 추적."""

    topic: str = ""
    turn_count: int = 0
    energy: float = 1.0
    last_action: str = "initiate"
    idle_turns: int = 0

    def record_turn(self) -> None:
        self.turn_count += 1
        self.idle_turns = 0

    def record_idle(self) -> None:
        self.idle_turns += 1


class DialogueManager:
    """대화 연속성 판단 — 성격 기반으로 대화 흐름 컨트롤.

    - E 높음: 대화 오래 유지, 먼저 말 걸기
    - E 낮음: 빨리 disengage, 침묵 선호
    - O 높음: 화제 전환 잘 함
    - O 낮음: 한 주제에 집중
    - A 높음: 상대에 맞춰줌
    - A 낮음: 본인 할 말 없으면 끊음
    """

    def __init__(self, llm: LLMAdapter, name: str, ocean: OceanTraits | None = None):
        self._llm = llm
        self._name = name
        self._ocean = ocean or OceanTraits()
        self.state = DialogueState()

    async def judge(
        self,
        recent_history: list[dict[str, str]],
        emotion_label: str = "neutral",
    ) -> dict:
        """대화 흐름 판단 — continue / redirect / disengage / initiate.

        Returns:
            {"action": str, "topic": str, "energy": float, "reason": str}
        """
        # 대화 없으면 성격 기반 빠른 판단
        if not recent_history and self.state.idle_turns > 0:
            return self._quick_initiate_check()

        history_text = "\n".join(
            f"  [{m['role']}]: {m['content'][:150]}"
            for m in recent_history[-6:]
        )

        try:
            data = await self._llm.generate_json(
                system_prompt="You judge conversation flow based on personality.",
                user_prompt=_DIALOGUE_JUDGE_PROMPT.format(
                    name=self._name,
                    E=self._ocean.extraversion,
                    A=self._ocean.agreeableness,
                    O=self._ocean.openness,
                    E_desc="talkative, seeks interaction" if self._ocean.extraversion > 0.6
                        else "reserved, prefers listening" if self._ocean.extraversion < 0.4
                        else "balanced",
                    emotion=emotion_label,
                    topic=self.state.topic or "(no topic yet)",
                    turns=self.state.turn_count,
                    history_count=min(len(recent_history), 6),
                    recent_history=history_text or "(no messages yet)",
                ),
            )

            action = data.get("action", "continue")
            self.state.topic = data.get("topic", self.state.topic)
            self.state.energy = max(0, min(1, float(data.get("energy", 0.5))))
            self.state.last_action = action

            return {
                "action": action,
                "topic": self.state.topic,
                "energy": self.state.energy,
                "reason": data.get("reason", ""),
            }
        except Exception:
            return {"action": "continue", "topic": self.state.topic, "energy": 0.5, "reason": "fallback"}

    def _quick_initiate_check(self) -> dict:
        """LLM 없이 성격 기반 빠른 판단 — 먼저 말 걸지 여부."""
        e = self._ocean.extraversion

        # E 높을수록 빨리 먼저 말 걸음
        threshold = max(1, int(5 - e * 4))  # E=1.0 → 1턴, E=0.0 → 5턴

        if self.state.idle_turns >= threshold:
            return {
                "action": "initiate",
                "topic": "",
                "energy": 0.3 + e * 0.5,
                "reason": f"idle for {self.state.idle_turns} turns, E={e:.2f}",
            }
        return {
            "action": "disengage",
            "topic": "",
            "energy": 0.2,
            "reason": f"waiting (idle {self.state.idle_turns}/{threshold})",
        }

    def reset(self) -> None:
        """대화 상태 초기화."""
        self.state = DialogueState()
