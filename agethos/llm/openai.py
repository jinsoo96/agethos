"""OpenAI LLM 어댑터."""

from __future__ import annotations

from agethos.llm.base import LLMAdapter


class OpenAIAdapter(LLMAdapter):
    """OpenAI API를 사용하는 LLM 어댑터."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError("pip install agethos[openai]") from e
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
    ) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content or ""
