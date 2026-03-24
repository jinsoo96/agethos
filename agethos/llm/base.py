"""LLM 호출 추상 인터페이스."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod


class LLMAdapter(ABC):
    """텍스트 생성용 LLM 추상화."""

    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
    ) -> str:
        """텍스트 생성 → 문자열 반환."""
        ...

    async def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
    ) -> dict:
        """JSON 생성 → dict 반환."""
        raw = await self.generate(
            system_prompt=system_prompt + "\n\nYou MUST respond with valid JSON only. No markdown, no explanation.",
            user_prompt=user_prompt,
            temperature=temperature,
        )
        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]  # remove opening fence
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        return json.loads(text)
