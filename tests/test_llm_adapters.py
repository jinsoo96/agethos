"""LLM adapter tests — LiteLLM & LangChain adapters + resolve logic."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agethos.llm.base import LLMAdapter


# ── LiteLLM Adapter ──


class TestLiteLLMAdapter:
    def test_import_error_without_litellm(self):
        with patch.dict("sys.modules", {"litellm": None}):
            with pytest.raises(ImportError, match="agethos\\[litellm\\]"):
                # Force re-import
                import importlib
                import agethos.llm.litellm as mod
                importlib.reload(mod)
                mod.LiteLLMAdapter()

    def test_instantiation(self):
        with patch.dict("sys.modules", {"litellm": MagicMock()}):
            from agethos.llm.litellm import LiteLLMAdapter
            adapter = LiteLLMAdapter(model="gemini/gemini-2.0-flash", api_key="test-key")
            assert isinstance(adapter, LLMAdapter)
            assert adapter._model == "gemini/gemini-2.0-flash"
            assert adapter._api_key == "test-key"

    def test_config_stored(self):
        with patch.dict("sys.modules", {"litellm": MagicMock()}):
            from agethos.llm.litellm import LiteLLMAdapter
            adapter = LiteLLMAdapter(
                model="mistral/mistral-large-latest",
                api_key="sk-test",
                api_base="https://custom.endpoint.com",
            )
            assert adapter._model == "mistral/mistral-large-latest"
            assert adapter._api_key == "sk-test"
            assert adapter._api_base == "https://custom.endpoint.com"

    def test_extra_kwargs(self):
        with patch.dict("sys.modules", {"litellm": MagicMock()}):
            from agethos.llm.litellm import LiteLLMAdapter
            adapter = LiteLLMAdapter(
                model="azure/gpt-4",
                max_tokens=1024,
                top_p=0.9,
            )
            assert adapter._extra == {"max_tokens": 1024, "top_p": 0.9}


# ── LangChain Adapter ──


class TestLangChainAdapter:
    def _make_mock_chat_model(self):
        """Create a mock that passes isinstance check."""
        from unittest.mock import MagicMock
        mock = MagicMock()
        return mock

    def test_import_error_without_langchain(self):
        with patch.dict("sys.modules", {"langchain_core": None, "langchain_core.language_models": None, "langchain_core.language_models.chat_models": None}):
            with pytest.raises(ImportError, match="agethos\\[langchain\\]"):
                import importlib
                import agethos.llm.langchain as mod
                importlib.reload(mod)
                mod.LangChainAdapter(MagicMock())

    def test_type_check(self):
        """LangChainAdapter should reject non-BaseChatModel objects."""
        try:
            from langchain_core.language_models.chat_models import BaseChatModel
        except ImportError:
            pytest.skip("langchain-core not installed")

        from agethos.llm.langchain import LangChainAdapter
        with pytest.raises(TypeError, match="BaseChatModel"):
            LangChainAdapter("not a chat model")

    def test_instantiation_with_mock(self):
        """Test instantiation by mocking BaseChatModel isinstance check."""
        try:
            from langchain_core.language_models.chat_models import BaseChatModel
        except ImportError:
            pytest.skip("langchain-core not installed")

        from agethos.llm.langchain import LangChainAdapter

        mock_model = MagicMock(spec=BaseChatModel)
        adapter = LangChainAdapter(mock_model)
        assert isinstance(adapter, LLMAdapter)
        assert adapter._chat_model is mock_model


# ── _resolve_llm ──


class TestResolveLLM:
    def test_resolve_litellm(self):
        with patch.dict("sys.modules", {"litellm": MagicMock()}):
            from agethos.brain import _resolve_llm
            adapter = _resolve_llm("litellm", model="gemini/gemini-2.0-flash")
            assert adapter._model == "gemini/gemini-2.0-flash"

    def test_resolve_litellm_with_base_url(self):
        with patch.dict("sys.modules", {"litellm": MagicMock()}):
            from agethos.brain import _resolve_llm
            adapter = _resolve_llm("litellm", model="custom/model", base_url="http://localhost:8000")
            assert adapter._api_base == "http://localhost:8000"

    def test_resolve_unknown_raises(self):
        from agethos.brain import _resolve_llm
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            _resolve_llm("nonexistent_provider")

    def test_resolve_error_message_mentions_litellm(self):
        from agethos.brain import _resolve_llm
        with pytest.raises(ValueError, match="litellm"):
            _resolve_llm("nonexistent_provider")


# ── Lazy import via __init__ ──


class TestLazyImport:
    def test_litellm_lazy_import(self):
        with patch.dict("sys.modules", {"litellm": MagicMock()}):
            from agethos.llm import LiteLLMAdapter
            assert LiteLLMAdapter is not None

    def test_langchain_lazy_import(self):
        try:
            from langchain_core.language_models.chat_models import BaseChatModel  # noqa: F401
        except ImportError:
            pytest.skip("langchain-core not installed")
        from agethos.llm import LangChainAdapter
        assert LangChainAdapter is not None

    def test_unknown_attr_raises(self):
        with pytest.raises(AttributeError):
            from agethos import llm
            llm.NonExistentAdapter
