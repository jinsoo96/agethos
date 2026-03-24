"""Autopilot — 자율주행 모드.

Brain을 환경에 연결하고, 성격 기반 트리거 + 대화 연속성 판단으로
에이전트가 알아서 판단하고 행동하게 한다.

Usage::

    brain = Brain.build(persona={...}, llm="openai")
    env = QueueEnvironment()
    pilot = Autopilot(brain=brain, env=env)

    # 수동 1틱
    actions = await pilot.step()

    # 자율 루프
    await pilot.run()
"""

from __future__ import annotations

import asyncio

from agethos.cognition.dialogue import DialogueManager
from agethos.cognition.emotion import EmotionDetector
from agethos.environment import Environment
from agethos.models import Action, EnvironmentEvent

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agethos.brain import Brain


class Autopilot:
    """자율주행 — 환경 이벤트에 성격 기반으로 반응.

    트리거 규칙 (OCEAN 기반):
    - E(외향성) 높음 → 적극 개입, 먼저 말 걸기, 대화 오래 유지
    - E 낮음 → 필요할 때만 응답, 침묵 선호
    - N(신경성) 높음 → 감정 반응 크게, 부정 이벤트에 민감
    - N 낮음 → 차분, 감정 변화 작음
    - O(개방성) 높음 → 화제 전환 자유, 새로운 주제 탐구
    - A(친화성) 높음 → 상대에 맞춰줌, 갈등 회피
    """

    def __init__(
        self,
        brain: Brain,
        env: Environment,
        tick_interval: float = 1.0,
        auto_emotion: bool = True,
        emotion_decay_rate: float = 0.05,
    ):
        self._brain = brain
        self._env = env
        self._tick_interval = tick_interval
        self._auto_emotion = auto_emotion
        self._emotion_decay_rate = emotion_decay_rate
        self._running = False
        self._tick_count = 0

        # 인지 모듈
        self._emotion_detector = EmotionDetector(brain._llm)
        self._dialogue = DialogueManager(
            llm=brain._llm,
            name=brain.persona.name,
            ocean=brain.persona.ocean,
        )

    async def step(self) -> list[Action]:
        """1틱 실행.

        1. 환경에서 이벤트 수집
        2. 이벤트별: 관찰 → 감정 감지 → 대화 판단 → 행동
        3. 이벤트 없으면: idle 판단 (먼저 말 걸지?)
        4. 감정 감쇠
        """
        events = await self._env.poll()
        actions: list[Action] = []

        if events:
            for event in events:
                action = await self._handle_event(event)
                if action and action.type != "silent":
                    await self._env.execute(action)
                    actions.append(action)
        else:
            # 이벤트 없음 → idle 처리
            action = await self._handle_idle()
            if action and action.type != "silent":
                await self._env.execute(action)
                actions.append(action)

        # 감정 감쇠
        if self._emotion_decay_rate > 0:
            self._brain.decay_emotion(self._emotion_decay_rate)

        self._tick_count += 1
        return actions

    async def _handle_event(self, event: EnvironmentEvent) -> Action | None:
        """이벤트 처리 — 관찰, 감정, 대화 판단, 응답."""

        # 1. 관찰 기록
        await self._brain.observe(event.content)

        # 2. 자동 감정 감지
        if self._auto_emotion:
            pad = await self._emotion_detector.detect_pad(event.content)
            self._brain.apply_event_emotion(pad)

        # 3. 대화 연속성 판단
        emotion_label = (
            self._brain.emotion.closest_emotion()
            if self._brain.emotion else "neutral"
        )
        self._dialogue.state.record_turn()

        judgement = await self._dialogue.judge(
            recent_history=self._brain.history,
            emotion_label=emotion_label,
        )

        dialogue_action = judgement["action"]

        # 4. 행동 결정
        if event.type == "message":
            # 메시지면 대화 판단에 따라 처리
            if dialogue_action == "disengage":
                return Action(type="silent")

            response = await self._brain.chat(event.content)
            return Action(type="speak", content=response, target=event.sender)

        else:
            # 관찰/환경 이벤트 → 성격 기반 개입 판단
            if dialogue_action in ("continue", "initiate"):
                response = await self._brain.chat(event.content)
                return Action(type="speak", content=response, target=event.sender)
            elif dialogue_action == "redirect":
                topic = judgement.get("topic", "")
                response = await self._brain.chat(
                    f"(Regarding: {topic}) {event.content}" if topic else event.content
                )
                return Action(type="speak", content=response, target=event.sender)
            else:
                return Action(type="silent")

    async def _handle_idle(self) -> Action | None:
        """이벤트 없을 때 — 성격 기반으로 먼저 말 걸지 판단."""
        self._dialogue.state.record_idle()

        judgement = self._dialogue._quick_initiate_check()

        if judgement["action"] == "initiate":
            # 먼저 말 걸기 — 대화 생성
            response = await self._brain.chat(
                "(You have a moment of free time. Say something or start a conversation.)"
            )
            self._dialogue.state.record_turn()
            return Action(type="speak", content=response)

        return Action(type="silent")

    async def run(self) -> None:
        """자율 루프 — stop() 호출까지 계속."""
        self._running = True
        while self._running:
            await self.step()
            await asyncio.sleep(self._tick_interval)

    def stop(self) -> None:
        """루프 중지."""
        self._running = False

    def reset_dialogue(self) -> None:
        """대화 상태 초기화."""
        self._dialogue.reset()

    @property
    def tick_count(self) -> int:
        return self._tick_count

    @property
    def dialogue_state(self) -> dict:
        """현재 대화 상태."""
        s = self._dialogue.state
        return {
            "topic": s.topic,
            "turn_count": s.turn_count,
            "energy": s.energy,
            "last_action": s.last_action,
            "idle_turns": s.idle_turns,
        }
