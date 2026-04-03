"""LiteLLM adapter — 100+ LLM providers via a single interface.

Supports OpenAI, Anthropic, Google Gemini, Groq, Mistral, Cohere,
Azure, AWS Bedrock, Together AI, Fireworks, and many more.

See https://docs.litellm.ai/docs/providers for full list.

Examples::

    # Google Gemini
    adapter = LiteLLMAdapter(model="gemini/gemini-2.0-flash")

    # Groq (ultra-fast inference)
    adapter = LiteLLMAdapter(model="groq/llama3-70b-8192")

    # Mistral
    adapter = LiteLLMAdapter(model="mistral/mistral-large-latest")

    # AWS Bedrock
    adapter = LiteLLMAdapter(model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0")

    # Azure OpenAI
    adapter = LiteLLMAdapter(model="azure/my-deployment", api_base="https://xxx.openai.azure.com")
"""

from __future__ import annotations

from agethos.llm.base import LLMAdapter


class LiteLLMAdapter(LLMAdapter):
    """LiteLLM universal LLM adapter.

    Uses litellm.acompletion which routes to the correct provider
    based on the model string prefix (e.g. ``gemini/``, ``groq/``).
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs,
    ):
        try:
            import litellm  # noqa: F401
        except ImportError as e:
            raise ImportError("pip install agethos[litellm]") from e
        self._model = model
        self._api_key = api_key
        self._api_base = api_base
        self._extra = kwargs

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
    ) -> str:
        from litellm import acompletion

        response = await acompletion(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            api_key=self._api_key,
            api_base=self._api_base,
            **self._extra,
        )
        return response.choices[0].message.content or ""

    async def generate_with_history(
        self,
        system_prompt: str,
        history: list[dict[str, str]],
        user_prompt: str,
        temperature: float = 0.7,
    ) -> str:
        from litellm import acompletion

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_prompt})
        response = await acompletion(
            model=self._model,
            messages=messages,
            temperature=temperature,
            api_key=self._api_key,
            api_base=self._api_base,
            **self._extra,
        )
        return response.choices[0].message.content or ""
