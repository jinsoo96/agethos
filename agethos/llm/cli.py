"""Subscription CLI adapters — inference through a locally-authenticated AI CLI.

A flat-rate subscription (e.g. Claude Pro/Max) authenticates the ``claude`` binary via
OAuth; ``claude -p`` then answers headlessly with no per-token API billing. These
adapters shell out to such a CLI per call, so every agethos feature (forge, verify,
steered chat) runs on a subscription instead of a metered key. Latency is higher than a
raw API (process startup per call); quality is the same model.

``CLIAdapter`` is generic: give it an argv where ``{system}`` / ``{prompt}`` are
placeholder tokens. If ``{prompt}`` is absent the user prompt is piped to stdin; if
``{system}`` is absent the system prompt is folded into the prompt text (for CLIs
without a system-prompt flag). ``temperature`` is accepted but ignored — headless CLIs
don't expose sampling controls.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile

from agethos.llm.base import LLMAdapter


class CLIAdapter(LLMAdapter):
    """Run an AI CLI per call and return its stdout.

    Examples::

        # Any CLI, explicitly
        llm = CLIAdapter(["claude", "-p", "--output-format", "text",
                          "--system-prompt", "{system}"])

        # Presets
        llm = ClaudeCodeAdapter(model="sonnet")   # Claude Pro/Max subscription
        llm = GeminiCLIAdapter()                  # gemini CLI
        llm = CodexCLIAdapter()                   # codex CLI
    """

    def __init__(self, command: list[str], timeout: float = 300.0):
        if not command:
            raise ValueError("command must be a non-empty argv list")
        self._command = list(command)
        self._timeout = timeout
        self._uses_prompt_arg = any("{prompt}" in tok for tok in self._command)
        self._uses_system_arg = any("{system}" in tok for tok in self._command)
        self._uses_system_file = any("{system_file}" in tok for tok in self._command)

    def _build(self, system_prompt: str, prompt: str) -> tuple[list[str], str | None, str | None]:
        """Substitute placeholders → (argv, stdin_text or None, temp system file or None).

        ``{system_file}`` writes the system prompt to a temp file and substitutes its
        path — multiline prompts survive Windows .cmd shims that mangle quoted argv."""
        if not self._uses_system_arg and not self._uses_system_file and system_prompt:
            prompt = f"[System instructions]\n{system_prompt}\n\n{prompt}"
        system_file: str | None = None
        if self._uses_system_file:
            fd, system_file = tempfile.mkstemp(suffix=".txt", prefix="agethos_sys_")
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
                f.write(system_prompt)
        argv = [
            tok.replace("{system_file}", system_file or "")
               .replace("{system}", system_prompt)
               .replace("{prompt}", prompt)
            for tok in self._command
        ]
        return argv, (None if self._uses_prompt_arg else prompt), system_file

    async def _run(self, argv: list[str], stdin_text: str | None) -> str:
        exe = shutil.which(argv[0]) or argv[0]
        # Windows npm shims are .cmd batch files — CreateProcess needs cmd.exe for those
        if sys.platform == "win32" and exe.lower().endswith((".cmd", ".bat")):
            argv = ["cmd", "/c", exe, *argv[1:]]
        else:
            argv = [exe, *argv[1:]]
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE if stdin_text is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(
                proc.communicate(stdin_text.encode("utf-8") if stdin_text is not None else None),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            raise TimeoutError(f"CLI call exceeded {self._timeout}s: {argv[0]}")
        if proc.returncode != 0:
            raise RuntimeError(
                f"CLI exited {proc.returncode}: {err.decode('utf-8', 'replace')[:500]}"
            )
        return out.decode("utf-8", "replace").replace("\r\n", "\n").strip()

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
    ) -> str:
        argv, stdin_text, system_file = self._build(system_prompt, user_prompt)
        try:
            return await self._run(argv, stdin_text)
        finally:
            if system_file:
                try:
                    os.unlink(system_file)
                except OSError:
                    pass

    async def generate_with_history(
        self,
        system_prompt: str,
        history: list[dict[str, str]],
        user_prompt: str,
        temperature: float = 0.7,
    ) -> str:
        """Fold the multi-turn history into the prompt (headless CLIs are single-shot)."""
        if not history:
            return await self.generate(system_prompt, user_prompt, temperature)
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in history)
        folded = (
            f"Conversation so far:\n{transcript}\n\n"
            f"user: {user_prompt}\n\nReply as the assistant (reply text only)."
        )
        return await self.generate(system_prompt, folded, temperature)


class ClaudeCodeAdapter(CLIAdapter):
    """Claude Code CLI — runs on the user's Claude Pro/Max subscription (OAuth), no API key.

    ``--strict-mcp-config`` skips configured MCP servers for a fast cold start."""

    def __init__(
        self,
        model: str | None = None,
        executable: str = "claude",
        timeout: float = 300.0,
        extra_args: list[str] | None = None,
    ):
        cmd = [executable, "-p", "--output-format", "text", "--strict-mcp-config",
               "--system-prompt-file", "{system_file}"]
        if model:
            cmd += ["--model", model]
        cmd += extra_args or []
        super().__init__(cmd, timeout)


class GeminiCLIAdapter(CLIAdapter):
    """Gemini CLI (no system-prompt flag — the system prompt is folded into the prompt)."""

    def __init__(self, model: str | None = None, executable: str = "gemini",
                 timeout: float = 300.0, extra_args: list[str] | None = None):
        cmd = [executable, "-p", "{prompt}"]
        if model:
            cmd += ["--model", model]
        cmd += extra_args or []
        super().__init__(cmd, timeout)


class CodexCLIAdapter(CLIAdapter):
    """OpenAI Codex CLI (ChatGPT subscription) — system prompt folded into the prompt."""

    def __init__(self, model: str | None = None, executable: str = "codex",
                 timeout: float = 300.0, extra_args: list[str] | None = None):
        cmd = [executable, "exec", "{prompt}"]
        if model:
            cmd += ["--model", model]
        cmd += extra_args or []
        super().__init__(cmd, timeout)
