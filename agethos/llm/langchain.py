"""LangChain LLM adapter — use any LangChain chat model as agethos LLM.

Wraps a LangChain ``BaseChatModel`` so it can be used anywhere agethos
expects an ``LLMAdapter``.

Examples::

    from langchain_openai import ChatOpenAI
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_anthropic import ChatAnthropic

    # LangChain ChatOpenAI → agethos
    lc_llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    adapter = LangChainAdapter(lc_llm)

    # LangChain Gemini → agethos
    lc_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    adapter = LangChainAdapter(lc_llm)

    # Use with Brain
    brain = Brain(persona=spec, llm=LangChainAdapter(lc_llm))
"""

from __future__ import annotations

from agethos.llm.base import LLMAdapter


class LangChainAdapter(LLMAdapter):
    """Adapter that wraps a LangChain BaseChatModel."""

    def __init__(self, chat_model):
        """Wrap a LangChain chat model.

        Args:
            chat_model: Any LangChain ``BaseChatModel`` instance
                (e.g. ``ChatOpenAI``, ``ChatAnthropic``, ``ChatGoogleGenerativeAI``).
        """
        try:
            from langchain_core.language_models.chat_models import BaseChatModel
        except ImportError as e:
            raise ImportError("pip install agethos[langchain]") from e
        if not isinstance(chat_model, BaseChatModel):
            raise TypeError(
                f"Expected a LangChain BaseChatModel, got {type(chat_model).__name__}"
            )
        self._chat_model = chat_model

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
    ) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        response = await self._chat_model.ainvoke(
            messages, temperature=temperature
        )
        return response.content or ""

    async def generate_with_history(
        self,
        system_prompt: str,
        history: list[dict[str, str]],
        user_prompt: str,
        temperature: float = 0.7,
    ) -> str:
        from langchain_core.messages import (
            AIMessage,
            HumanMessage,
            SystemMessage,
        )

        _role_map = {"user": HumanMessage, "assistant": AIMessage}
        messages = [SystemMessage(content=system_prompt)]
        for msg in history:
            cls = _role_map.get(msg["role"], HumanMessage)
            messages.append(cls(content=msg["content"]))
        messages.append(HumanMessage(content=user_prompt))
        response = await self._chat_model.ainvoke(
            messages, temperature=temperature
        )
        return response.content or ""
