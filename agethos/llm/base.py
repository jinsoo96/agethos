"""LLM call abstract interface."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod


class LLMAdapter(ABC):
    """Text generation LLM abstraction."""

    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
    ) -> str:
        """Generate text → return string."""
        ...

    async def generate_with_history(
        self,
        system_prompt: str,
        history: list[dict[str, str]],
        user_prompt: str,
        temperature: float = 0.7,
    ) -> str:
        """Generate with multi-turn conversation history.

        Default implementation ignores history and falls back to single-turn.
        Subclasses should override for proper multi-turn support.

        Args:
            system_prompt: System prompt.
            history: List of {"role": "user"|"assistant", "content": "..."} dicts.
            user_prompt: Current user message.
            temperature: Sampling temperature.
        """
        return await self.generate(system_prompt, user_prompt, temperature)

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
