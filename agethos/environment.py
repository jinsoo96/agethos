"""환경 추상화 — 에이전트에게 이벤트를 공급하고 행동을 수신."""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from pathlib import Path

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


class ChatLogEnvironment(Environment):
    """정적 채팅 기록 기반 환경. 관찰 학습용.

    JSON/JSONL 파일에서 채팅 기록을 읽어 이벤트로 공급.
    한 번 poll()하면 모든 메시지를 반환하고 소진됨.

    지원 포맷::

        # JSON array
        [
            {"sender": "alice", "content": "안녕", "type": "message"},
            {"sender": "bob", "content": "반가워", "type": "message"}
        ]

        # JSONL (한 줄에 하나)
        {"sender": "alice", "content": "안녕"}
        {"sender": "bob", "content": "반가워"}

    Usage::

        env = ChatLogEnvironment.from_file("chat_log.json")
        events = await env.poll()  # 전체 반환
        events = await env.poll()  # [] (소진)
    """

    def __init__(self, events: list[EnvironmentEvent]) -> None:
        self._events = list(events)
        self._consumed = False

    @classmethod
    def from_file(cls, path: str) -> ChatLogEnvironment:
        """파일에서 채팅 로그 읽기. JSON 또는 JSONL 지원."""
        p = Path(path)
        text = p.read_text(encoding="utf-8")

        records: list[dict] = []
        # Try JSON array first
        try:
            data = json.loads(text)
            if isinstance(data, list):
                records = data
            else:
                records = [data]
        except json.JSONDecodeError:
            # Try JSONL
            for line in text.strip().splitlines():
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        events = []
        for r in records:
            events.append(EnvironmentEvent(
                type=r.get("type", "message"),
                content=r.get("content", r.get("text", r.get("message", ""))),
                sender=r.get("sender", r.get("author", r.get("user", "unknown"))),
                metadata={k: v for k, v in r.items() if k not in ("type", "content", "text", "message", "sender", "author", "user")},
            ))

        return cls(events)

    @classmethod
    def from_list(cls, messages: list[dict[str, str]]) -> ChatLogEnvironment:
        """딕셔너리 리스트에서 직접 생성.

        Args:
            messages: [{"sender": "alice", "content": "hello"}, ...]
        """
        events = [
            EnvironmentEvent(
                type=m.get("type", "message"),
                content=m.get("content", ""),
                sender=m.get("sender", "unknown"),
            )
            for m in messages
        ]
        return cls(events)

    async def poll(self) -> list[EnvironmentEvent]:
        if self._consumed:
            return []
        self._consumed = True
        return list(self._events)

    @property
    def total_messages(self) -> int:
        return len(self._events)
