"""Anthropic LLM adapter."""

from __future__ import annotations

from agethos.llm.base import LLMAdapter


class AnthropicAdapter(LLMAdapter):
    """Anthropic Claude API LLM adapter."""

    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: str | None = None):
        try:
            from anthropic import AsyncAnthropic
        except ImportError as e:
            raise ImportError("pip install agethos[anthropic]") from e
        self._client = AsyncAnthropic(api_key=api_key)
        self._model = model

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
    ) -> str:
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=temperature,
        )
        return response.content[0].text

    async def generate_with_history(
        self,
        system_prompt: str,
        history: list[dict[str, str]],
        user_prompt: str,
        temperature: float = 0.7,
    ) -> str:
        messages = list(history)
        messages.append({"role": "user", "content": user_prompt})
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=system_prompt,
            messages=messages,
            temperature=temperature,
        )
        return response.content[0].text
