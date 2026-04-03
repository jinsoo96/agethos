from agethos.llm.base import LLMAdapter

__all__ = ["LLMAdapter"]


def _lazy_import(name: str):
    """Lazy import to avoid pulling optional deps at module load."""
    if name == "OpenAIAdapter":
        from agethos.llm.openai import OpenAIAdapter
        return OpenAIAdapter
    if name == "AnthropicAdapter":
        from agethos.llm.anthropic import AnthropicAdapter
        return AnthropicAdapter
    if name == "LiteLLMAdapter":
        from agethos.llm.litellm import LiteLLMAdapter
        return LiteLLMAdapter
    if name == "LangChainAdapter":
        from agethos.llm.langchain import LangChainAdapter
        return LangChainAdapter
    raise AttributeError(f"module 'agethos.llm' has no attribute {name!r}")


def __getattr__(name: str):
    return _lazy_import(name)
