"""Observer 모듈 + ChatLogEnvironment 테스트."""

import json
import os
import tempfile

import pytest

from agethos.environment import ChatLogEnvironment
from agethos.models import EnvironmentEvent


# ── ChatLogEnvironment ──


class TestChatLogEnvironment:
    def test_from_list(self):
        env = ChatLogEnvironment.from_list([
            {"sender": "alice", "content": "안녕하세요"},
            {"sender": "bob", "content": "반갑습니다"},
            {"sender": "alice", "content": "오늘 날씨 좋네요"},
        ])
        assert env.total_messages == 3

    @pytest.mark.asyncio
    async def test_poll_returns_all_then_empty(self):
        env = ChatLogEnvironment.from_list([
            {"sender": "alice", "content": "hello"},
            {"sender": "bob", "content": "hi"},
        ])
        events = await env.poll()
        assert len(events) == 2
        assert events[0].sender == "alice"
        assert events[0].content == "hello"
        assert events[1].sender == "bob"

        # Second poll returns empty (consumed)
        events2 = await env.poll()
        assert events2 == []

    def test_from_json_file(self):
        data = [
            {"sender": "alice", "content": "test 1"},
            {"sender": "bob", "content": "test 2"},
        ]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(data, f)
            path = f.name

        try:
            env = ChatLogEnvironment.from_file(path)
            assert env.total_messages == 2
        finally:
            os.unlink(path)

    def test_from_jsonl_file(self):
        lines = [
            '{"sender": "alice", "content": "line 1"}',
            '{"sender": "bob", "content": "line 2"}',
            '{"sender": "charlie", "content": "line 3"}',
        ]
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            f.write("\n".join(lines))
            path = f.name

        try:
            env = ChatLogEnvironment.from_file(path)
            assert env.total_messages == 3
        finally:
            os.unlink(path)

    def test_from_file_flexible_keys(self):
        """다양한 키 이름 지원 (author, text, user, message)."""
        data = [
            {"author": "alice", "text": "hello"},
            {"user": "bob", "message": "world"},
        ]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(data, f)
            path = f.name

        try:
            env = ChatLogEnvironment.from_file(path)
            assert env.total_messages == 2
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_poll_content_mapping(self):
        """text/message 키도 content로 매핑."""
        env = ChatLogEnvironment.from_list([
            {"sender": "a", "content": "direct"},
        ])
        events = await env.poll()
        assert events[0].content == "direct"


# ── Observer (unit tests without LLM) ──


class TestObserverMerge:
    """Observer._merge_patterns 단위 테스트."""

    def test_merge_duplicate_patterns(self):
        from agethos.models import SocialPattern
        from agethos.cognition.observer import Observer

        # Create observer with minimal deps
        observer = Observer.__new__(Observer)
        observer._community_name = "test"

        patterns = [
            SocialPattern(context="Code review feedback", effective_strategy="Ask questions", confidence=0.5),
            SocialPattern(context="code review feedback", effective_strategy="Be constructive", confidence=0.6),
            SocialPattern(context="Onboarding new members", effective_strategy="Be welcoming", confidence=0.7),
        ]

        merged = observer._merge_patterns(patterns)
        # "code review feedback" should merge (case-insensitive prefix match)
        assert len(merged) == 2

        # Find the merged code review pattern
        code_review = next(p for p in merged if "code review" in p.context.lower())
        assert code_review.evidence_count == 2
        assert code_review.confidence > 0.5  # Increased

    def test_merge_empty(self):
        from agethos.cognition.observer import Observer
        observer = Observer.__new__(Observer)
        observer._community_name = "test"
        assert observer._merge_patterns([]) == []

    def test_chunk_messages(self):
        from agethos.cognition.observer import Observer
        observer = Observer.__new__(Observer)
        observer._chunk_size = 3

        messages = [EnvironmentEvent(content=f"msg {i}", sender="user") for i in range(8)]
        chunks = observer._chunk_messages(messages)
        assert len(chunks) == 3  # 3 + 3 + 2
        assert len(chunks[0]) == 3
        assert len(chunks[2]) == 2
