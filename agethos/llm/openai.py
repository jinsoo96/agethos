"""OpenAI LLM adapter."""

from __future__ import annotations

from agethos.llm.base import LLMAdapter


class OpenAIAdapter(LLMAdapter):
    """OpenAI-compatible API adapter.

    Works with OpenAI, Qwen, Ollama, vLLM, Together AI, LM Studio, and
    any provider that exposes an OpenAI-compatible chat completions endpoint.

    Examples::

        # OpenAI (default)
        adapter = OpenAIAdapter()

        # Ollama (local)
        adapter = OpenAIAdapter(model="qwen2.5:7b", base_url="http://localhost:11434/v1")

        # Qwen via DashScope
        adapter = OpenAIAdapter(
            model="qwen-plus",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key="sk-your-dashscope-key",
        )

        # vLLM / LM Studio
        adapter = OpenAIAdapter(model="meta-llama/...", base_url="http://localhost:8000/v1")
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError("pip install agethos[openai]") from e
        kwargs: dict = {}
        if api_key is not None:
            kwargs["api_key"] = api_key
        if base_url is not None:
            kwargs["base_url"] = base_url
        self._client = AsyncOpenAI(**kwargs)
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

    async def generate_with_history(
        self,
        system_prompt: str,
        history: list[dict[str, str]],
        user_prompt: str,
        temperature: float = 0.7,
    ) -> str:
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_prompt})
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""
