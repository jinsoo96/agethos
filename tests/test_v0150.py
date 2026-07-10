"""v0.15.0 — subscription CLI adapters (Claude Code / Gemini / Codex CLIs)."""
from __future__ import annotations

import sys

import pytest

from agethos.brain import _resolve_llm
from agethos.llm.cli import ClaudeCodeAdapter, CLIAdapter, CodexCLIAdapter, GeminiCLIAdapter

PY = sys.executable


async def test_system_placeholder_and_stdin_prompt():
    llm = CLIAdapter([PY, "-c",
                      "import sys; print('S=' + sys.argv[1] + '|P=' + sys.stdin.read().strip())",
                      "{system}"])
    out = await llm.generate("SYS", "hello")
    assert out == "S=SYS|P=hello"


async def test_system_folded_when_no_placeholder():
    llm = CLIAdapter([PY, "-c", "import sys; print(sys.stdin.read().strip())"])
    out = await llm.generate("BE TERSE", "hello")
    assert "[System instructions]" in out and "BE TERSE" in out and "hello" in out


async def test_prompt_arg_mode_skips_stdin():
    llm = CLIAdapter([PY, "-c", "import sys; print('ARG=' + sys.argv[1])", "{prompt}"])
    out = await llm.generate("", "hi there")
    assert out == "ARG=hi there"


async def test_history_folded_into_prompt():
    llm = CLIAdapter([PY, "-c", "import sys; print(sys.stdin.read().strip())"])
    out = await llm.generate_with_history(
        "SYS",
        [{"role": "user", "content": "first"}, {"role": "assistant", "content": "reply"}],
        "second",
    )
    assert "user: first" in out and "assistant: reply" in out and "user: second" in out


async def test_nonzero_exit_raises():
    llm = CLIAdapter([PY, "-c", "import sys; sys.exit(3)"])
    with pytest.raises(RuntimeError, match="exited 3"):
        await llm.generate("", "x")


async def test_generate_json_strips_fences():
    llm = CLIAdapter([PY, "-c", r"print('```json\n{\"x\": 1}\n```')"])
    data = await llm.generate_json("", "x")
    assert data == {"x": 1}


async def test_system_file_mode_writes_and_cleans_temp():
    llm = CLIAdapter([PY, "-c",
                      "import sys; print('S=' + open(sys.argv[1], encoding='utf-8').read()"
                      " + '|P=' + sys.stdin.read().strip())",
                      "{system_file}"])
    out = await llm.generate('multi\nline "quoted" system', "hello")
    assert out == 'S=multi\nline "quoted" system|P=hello'


async def test_generate_json_extracts_object_from_prose():
    llm = CLIAdapter([PY, "-c", "print('Here is the JSON: {\"x\": 2} hope it helps')"])
    assert await llm.generate_json("", "x") == {"x": 2}


def test_presets_build_expected_argv():
    c = ClaudeCodeAdapter(model="sonnet")
    assert "--system-prompt-file" in c._command and "{system_file}" in c._command
    assert "--model" in c._command and "sonnet" in c._command
    g = GeminiCLIAdapter()
    assert "{prompt}" in g._command and not g._uses_system_arg
    x = CodexCLIAdapter()
    assert x._command[1] == "exec" and "{prompt}" in x._command


def test_resolve_llm_subscription_providers():
    assert isinstance(_resolve_llm("claude-code"), ClaudeCodeAdapter)
    assert isinstance(_resolve_llm("subscription", model="sonnet"), ClaudeCodeAdapter)
    assert isinstance(_resolve_llm("gemini-cli"), GeminiCLIAdapter)
    assert isinstance(_resolve_llm("codex-cli"), CodexCLIAdapter)
