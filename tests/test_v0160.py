"""v0.16.0 — LLM selection layer: mode (api/subscription/auto) × provider × model."""
from __future__ import annotations

import pytest

from agethos.brain import Brain, _coerce_llm
from agethos.llm.cli import ClaudeCodeAdapter, CodexCLIAdapter, GeminiCLIAdapter
from agethos.llm.select import LLMConfig, available_backends, resolve_llm
from agethos.models import PersonaSpec

ENV_KEYS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
            "AGETHOS_LLM_MODE", "AGETHOS_LLM_PROVIDER", "AGETHOS_LLM_MODEL",
            "AGETHOS_LLM_API_KEY", "AGETHOS_LLM_BASE_URL")


@pytest.fixture()
def clean_env(monkeypatch):
    for k in ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


# ── config ──

def test_config_from_env_and_overrides(clean_env):
    clean_env.setenv("AGETHOS_LLM_MODE", "subscription")
    clean_env.setenv("AGETHOS_LLM_PROVIDER", "gemini")
    clean_env.setenv("AGETHOS_LLM_MODEL", "gemini-2.0-flash")
    cfg = LLMConfig.from_env()
    assert (cfg.mode, cfg.provider, cfg.model) == ("subscription", "gemini", "gemini-2.0-flash")
    cfg2 = LLMConfig.from_env(provider="claude", model=None)
    assert cfg2.provider == "claude" and cfg2.model == "gemini-2.0-flash"  # None doesn't override


# ── explicit modes ──

def test_subscription_mode_provider_matrix():
    assert isinstance(resolve_llm({"mode": "subscription", "provider": "claude"}), ClaudeCodeAdapter)
    assert isinstance(resolve_llm({"mode": "subscription", "provider": "gemini"}), GeminiCLIAdapter)
    assert isinstance(resolve_llm({"mode": "subscription", "provider": "chatgpt"}), CodexCLIAdapter)
    with pytest.raises(ValueError, match="subscription CLI"):
        resolve_llm({"mode": "subscription", "provider": "groq"})


def test_subscription_custom_command():
    from agethos.llm.cli import CLIAdapter
    llm = resolve_llm({"mode": "subscription", "provider": "claude",
                       "command": ["mycli", "--sys", "{system_file}", "{prompt}"]})
    assert type(llm) is CLIAdapter and llm._command[0] == "mycli"


def test_api_mode_openai_and_vllm():
    from agethos.llm.openai import OpenAIAdapter
    llm = resolve_llm({"mode": "api", "provider": "openai", "model": "gpt-4o", "api_key": "k"})
    assert isinstance(llm, OpenAIAdapter)
    llm2 = resolve_llm({"mode": "api", "provider": "vllm",
                        "base_url": "http://localhost:8000/v1", "model": "qwen3"})
    assert isinstance(llm2, OpenAIAdapter)
    with pytest.raises(ValueError, match="base_url"):
        resolve_llm({"mode": "api", "provider": "vllm"})


def test_api_mode_anthropic():
    pytest.importorskip("anthropic")
    from agethos.llm.anthropic import AnthropicAdapter
    llm = resolve_llm({"mode": "api", "provider": "anthropic", "api_key": "k"})
    assert isinstance(llm, AnthropicAdapter)


# ── auto mode ──

def test_auto_prefers_api_key(clean_env):
    from agethos.llm.openai import OpenAIAdapter
    clean_env.setenv("OPENAI_API_KEY", "k")
    llm = resolve_llm({"mode": "auto", "provider": "openai"})
    assert isinstance(llm, OpenAIAdapter)


def test_auto_falls_back_to_cli(clean_env, monkeypatch):
    import agethos.llm.select as select
    monkeypatch.setattr(select.shutil, "which", lambda name: "C:/fake/" + name)
    llm = resolve_llm({"mode": "auto", "provider": "claude"})
    assert isinstance(llm, ClaudeCodeAdapter)


def test_auto_raises_with_discovery_report(clean_env, monkeypatch):
    import agethos.llm.select as select
    monkeypatch.setattr(select.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="No usable backend"):
        resolve_llm({"mode": "auto", "provider": "claude"})


def test_available_backends_shape():
    report = available_backends()
    assert set(report) == {"api", "subscription"}
    assert set(report["subscription"]) == {"claude", "gemini", "openai"}
    assert all(isinstance(v, bool) for v in report["api"].values())


# ── Brain wiring ──

def test_brain_build_accepts_config_dict():
    brain = Brain.build(
        persona={"name": "S"},
        llm={"mode": "subscription", "provider": "claude", "model": "sonnet"},
    )
    assert isinstance(brain._llm, ClaudeCodeAdapter)


def test_brain_build_accepts_llmconfig_and_auto(clean_env):
    from agethos.llm.openai import OpenAIAdapter
    brain = Brain.build(persona={"name": "S"},
                        llm=LLMConfig(mode="api", provider="openai", api_key="k"))
    assert isinstance(brain._llm, OpenAIAdapter)
    clean_env.setenv("OPENAI_API_KEY", "k")
    clean_env.setenv("AGETHOS_LLM_PROVIDER", "openai")
    brain2 = Brain.build(persona={"name": "S"}, llm="auto")
    assert isinstance(brain2._llm, OpenAIAdapter)


def test_coerce_llm_passes_adapter_through():
    llm = ClaudeCodeAdapter()
    assert _coerce_llm(llm) is llm
