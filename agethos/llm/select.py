"""LLM selection layer — choose mode (api / subscription / auto), provider, and model.

One config decides how an agethos brain talks to its model:

    Brain.build(persona, llm={"mode": "subscription", "provider": "claude", "model": "sonnet"})
    Brain.build(persona, llm={"mode": "api", "provider": "openai", "model": "gpt-4o"})
    Brain.build(persona, llm="auto")     # env/CLI discovery picks the best available

``mode="auto"`` prefers an API key when one is available (lowest latency) and falls
back to an installed subscription CLI; ``available_backends()`` reports what this
machine can use. Env defaults: ``AGETHOS_LLM_MODE`` / ``AGETHOS_LLM_PROVIDER`` /
``AGETHOS_LLM_MODEL`` / ``AGETHOS_LLM_API_KEY`` / ``AGETHOS_LLM_BASE_URL``.
"""
from __future__ import annotations

import os
import shutil
from typing import Literal

from pydantic import BaseModel, Field

from agethos.llm.base import LLMAdapter

# canonical provider ← aliases
_PROVIDER_ALIASES = {
    "claude": "claude", "anthropic": "claude",
    "openai": "openai", "chatgpt": "openai", "codex": "openai",
    "gemini": "gemini", "google": "gemini",
    "litellm": "litellm",
    "vllm": "vllm", "openai-compatible": "vllm", "ollama": "vllm",
}

_API_KEY_ENV = {
    "claude": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
}

_CLI_BINARY = {"claude": "claude", "gemini": "gemini", "openai": "codex"}


class LLMConfig(BaseModel):
    """How to reach a model: billing mode + provider + model (+ credentials/endpoint)."""

    mode: Literal["auto", "api", "subscription"] = "auto"
    provider: str = "claude"
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    timeout: float = 300.0
    command: list[str] | None = Field(
        None, description="custom CLI argv for subscription mode ({system_file}/{prompt} placeholders)")

    @classmethod
    def from_env(cls, **overrides) -> "LLMConfig":
        """Build a config from AGETHOS_LLM_* env vars; kwargs override env."""
        env = {
            "mode": os.getenv("AGETHOS_LLM_MODE"),
            "provider": os.getenv("AGETHOS_LLM_PROVIDER"),
            "model": os.getenv("AGETHOS_LLM_MODEL"),
            "api_key": os.getenv("AGETHOS_LLM_API_KEY"),
            "base_url": os.getenv("AGETHOS_LLM_BASE_URL"),
        }
        merged = {k: v for k, v in env.items() if v} | {k: v for k, v in overrides.items() if v is not None}
        return cls(**merged)


def _canonical(provider: str) -> str:
    p = _PROVIDER_ALIASES.get(provider.lower())
    if p is None:
        return "litellm"  # unknown providers route through LiteLLM's 100+ catalog
    return p


def _has_api_key(provider: str, cfg: LLMConfig) -> bool:
    if cfg.api_key:
        return True
    if provider == "vllm":
        return cfg.base_url is not None  # self-hosted endpoints often need no key
    return any(os.getenv(k) for k in _API_KEY_ENV.get(provider, ()))


def _has_cli(provider: str, cfg: LLMConfig) -> bool:
    if cfg.command:
        return shutil.which(cfg.command[0]) is not None
    binary = _CLI_BINARY.get(provider)
    return binary is not None and shutil.which(binary) is not None


def available_backends() -> dict:
    """Probe this machine: which API keys are set, which subscription CLIs are installed."""
    return {
        "api": {p: any(os.getenv(k) for k in keys) for p, keys in _API_KEY_ENV.items()},
        "subscription": {p: shutil.which(b) is not None for p, b in _CLI_BINARY.items()},
    }


def _api_adapter(provider: str, cfg: LLMConfig) -> LLMAdapter:
    if provider == "claude":
        from agethos.llm.anthropic import AnthropicAdapter
        return AnthropicAdapter(model=cfg.model or "claude-sonnet-4-20250514", api_key=cfg.api_key)
    if provider == "openai":
        from agethos.llm.openai import OpenAIAdapter
        return OpenAIAdapter(model=cfg.model or "gpt-4o-mini", api_key=cfg.api_key, base_url=cfg.base_url)
    if provider == "vllm":
        if not cfg.base_url:
            raise ValueError("provider 'vllm'/'openai-compatible' needs base_url")
        from agethos.llm.openai import OpenAIAdapter
        return OpenAIAdapter(model=cfg.model or "default", api_key=cfg.api_key or "EMPTY",
                             base_url=cfg.base_url)
    if provider == "gemini":
        from agethos.llm.litellm import LiteLLMAdapter
        model = cfg.model or "gemini-2.0-flash"
        if "/" not in model:
            model = f"gemini/{model}"
        kwargs = {"api_key": cfg.api_key} if cfg.api_key else {}
        return LiteLLMAdapter(model=model, **kwargs)
    # litellm catch-all: model string carries the provider prefix ("groq/llama-3.3-70b", ...)
    from agethos.llm.litellm import LiteLLMAdapter
    kwargs = {"api_key": cfg.api_key} if cfg.api_key else {}
    if cfg.base_url:
        kwargs["api_base"] = cfg.base_url
    return LiteLLMAdapter(model=cfg.model or "gpt-4o-mini", **kwargs)


def _subscription_adapter(provider: str, cfg: LLMConfig) -> LLMAdapter:
    from agethos.llm.cli import ClaudeCodeAdapter, CLIAdapter, CodexCLIAdapter, GeminiCLIAdapter
    if cfg.command:
        return CLIAdapter(cfg.command, timeout=cfg.timeout)
    if provider == "claude":
        return ClaudeCodeAdapter(model=cfg.model, timeout=cfg.timeout)
    if provider == "gemini":
        return GeminiCLIAdapter(model=cfg.model, timeout=cfg.timeout)
    if provider == "openai":
        return CodexCLIAdapter(model=cfg.model, timeout=cfg.timeout)
    raise ValueError(
        f"No subscription CLI mapping for provider {provider!r} — "
        "use provider 'claude'/'gemini'/'openai' or pass a custom `command`."
    )


def resolve_llm(config: LLMConfig | dict | str | None = None, **overrides) -> LLMAdapter:
    """Config → adapter. Accepts an LLMConfig, a dict, or None/'auto' (env + discovery).

    ``mode='auto'``: use the API when a key is available (faster), otherwise fall back
    to an installed subscription CLI; raise with the discovery report when neither works."""
    if isinstance(config, str):
        if config != "auto":
            raise ValueError(f"resolve_llm only accepts 'auto' as a string, got {config!r}")
        config = None
    if config is None:
        cfg = LLMConfig.from_env(**overrides)
    elif isinstance(config, dict):
        cfg = LLMConfig(**{**config, **{k: v for k, v in overrides.items() if v is not None}})
    else:
        cfg = config.model_copy(update={k: v for k, v in overrides.items() if v is not None})

    provider = _canonical(cfg.provider)

    if cfg.mode == "api":
        return _api_adapter(provider, cfg)
    if cfg.mode == "subscription":
        return _subscription_adapter(provider, cfg)

    # auto
    if _has_api_key(provider, cfg):
        return _api_adapter(provider, cfg)
    if _has_cli(provider, cfg):
        return _subscription_adapter(provider, cfg)
    raise RuntimeError(
        f"No usable backend for provider {cfg.provider!r}: no API key "
        f"({' / '.join(_API_KEY_ENV.get(provider, ('api_key',)))}) and no CLI on PATH. "
        f"Available on this machine: {available_backends()}"
    )
