"""환경 추상화 — 에이전트에게 이벤트를 공급하고 행동을 수신."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

from agethos.models import Action, EnvironmentEvent


class Environment(ABC):
    """환경 인터페이스 — poll()로 이벤트 공급, execute()로 행동 수신."""

    @abstractmethod
    async def poll(self) -> list[EnvironmentEvent]:
        """대기 중인 이벤트 반환. 비차단."""
        ...

    async def execute(self, action: Action) -> None:
        """에이전트의 행동을 환경에서 실행. 필요시 오버라이드."""
        pass


class QueueEnvironment(Environment):
    """인메모리 큐 기반 환경. 테스트 및 간단한 연동에 적합.

    Usage::

        env = QueueEnvironment()
        await env.push(EnvironmentEvent(type="message", content="안녕!", sender="user"))
        events = await env.poll()  # [EnvironmentEvent(...)]
    """

    def __init__(self) -> None:
        self._event_queue: asyncio.Queue[EnvironmentEvent] = asyncio.Queue()
        self._action_log: list[Action] = []

    async def push(self, event: EnvironmentEvent) -> None:
        """이벤트를 큐에 추가."""
        await self._event_queue.put(event)

    async def poll(self) -> list[EnvironmentEvent]:
        events: list[EnvironmentEvent] = []
        while not self._event_queue.empty():
            events.append(self._event_queue.get_nowait())
        return events

    async def execute(self, action: Action) -> None:
        self._action_log.append(action)

    @property
    def actions(self) -> list[Action]:
        """실행된 행동 로그."""
        return list(self._action_log)

    def clear_actions(self) -> None:
        self._action_log.clear()
