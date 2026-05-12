"""Tests for the LLM client layer."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import pytest

from tailor_core.llm.client import (
    DEFAULT_MODEL,
    ClaudeCliClient,
    FakeLLMClient,
    LLMError,
)

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


# -- FakeLLMClient -----------------------------------------------------------


def test_fake_returns_scripted_response_for_known_prompt() -> None:
    fake = FakeLLMClient(responses={"hello": "world"})
    assert fake.complete(system="s", user="hello") == "world"
    assert fake.calls == [("s", "hello", None)]


def test_fake_returns_default_for_unknown_prompt() -> None:
    fake = FakeLLMClient(default_response="{}")
    assert fake.complete(system="s", user="anything") == "{}"


def test_fake_raises_for_unknown_prompt_when_strict() -> None:
    fake = FakeLLMClient(raise_for_unknown=True)
    with pytest.raises(LLMError, match="no scripted response"):
        fake.complete(system="s", user="surprise")


def test_fake_records_model() -> None:
    fake = FakeLLMClient()
    fake.complete(system="s", user="u", model="claude-sonnet-4-6")
    assert fake.calls == [("s", "u", "claude-sonnet-4-6")]


# -- ClaudeCliClient ---------------------------------------------------------


def _completed(stdout: str = "", stderr: str = "", returncode: int = 0):  # type: ignore[no-untyped-def]
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def test_claude_cli_returns_stdout_on_success(mocker: MockerFixture) -> None:
    mocker.patch("shutil.which", return_value="/usr/local/bin/claude")
    run = mocker.patch("subprocess.run", return_value=_completed(stdout="response"))

    client = ClaudeCliClient()
    result = client.complete(system="sys", user="user prompt body")

    assert result == "response"
    cmd = run.call_args.args[0]
    assert cmd[0] == "/usr/local/bin/claude"
    assert "--print" in cmd
    # System prompt is written to a temp file (avoids ARG_MAX / E2BIG when
    # the prompt is large) and the path is passed via --system-prompt-file.
    assert "--system-prompt-file" in cmd
    file_arg = cmd[cmd.index("--system-prompt-file") + 1]
    assert file_arg.endswith(".txt")
    assert "--model" in cmd
    assert DEFAULT_MODEL in cmd
    # The user prompt is piped via stdin, not appended to argv.
    assert "user prompt body" not in cmd
    assert run.call_args.kwargs.get("input") == "user prompt body"


def test_claude_cli_writes_system_prompt_to_temp_file_then_cleans_up(
    mocker: MockerFixture,
) -> None:
    """The temp file holding the system prompt is unlinked after the call."""
    from pathlib import Path  # noqa: PLC0415

    captured: dict[str, str] = {}

    def fake_run(
        cmd: list[str],
        **_: object,
    ) -> subprocess.CompletedProcess[str]:
        path_str = cmd[cmd.index("--system-prompt-file") + 1]
        captured["path"] = path_str
        captured["content"] = Path(path_str).read_text(encoding="utf-8")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    mocker.patch("shutil.which", return_value="/usr/local/bin/claude")
    mocker.patch("subprocess.run", side_effect=fake_run)

    ClaudeCliClient().complete(system="my system prompt", user="u")

    assert captured["content"] == "my system prompt"
    # File is removed after complete() returns -- prompt content doesn't
    # linger in /tmp once the call is done.
    assert not Path(captured["path"]).exists()


def test_claude_cli_uses_supplied_model(mocker: MockerFixture) -> None:
    mocker.patch("shutil.which", return_value="/usr/local/bin/claude")
    run = mocker.patch("subprocess.run", return_value=_completed(stdout="ok"))

    ClaudeCliClient().complete(system="s", user="u", model="claude-sonnet-4-6")

    assert "claude-sonnet-4-6" in run.call_args.args[0]


def test_claude_cli_falls_back_to_bin_name_when_which_returns_none(
    mocker: MockerFixture,
) -> None:
    mocker.patch("shutil.which", return_value=None)
    run = mocker.patch("subprocess.run", return_value=_completed(stdout="ok"))

    ClaudeCliClient(claude_bin="my-claude").complete(system="s", user="u")

    assert run.call_args.args[0][0] == "my-claude"


def test_claude_cli_raises_on_missing_binary(mocker: MockerFixture) -> None:
    mocker.patch("shutil.which", return_value=None)
    mocker.patch("subprocess.run", side_effect=FileNotFoundError())

    with pytest.raises(LLMError, match="not found"):
        ClaudeCliClient().complete(system="s", user="u")


def test_claude_cli_raises_on_timeout(mocker: MockerFixture) -> None:
    mocker.patch("shutil.which", return_value="/usr/local/bin/claude")
    mocker.patch(
        "subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=120),
    )

    with pytest.raises(LLMError, match="timed out"):
        ClaudeCliClient().complete(system="s", user="u")


def test_claude_cli_raises_on_non_zero_exit(mocker: MockerFixture) -> None:
    mocker.patch("shutil.which", return_value="/usr/local/bin/claude")
    mocker.patch(
        "subprocess.run",
        return_value=_completed(stdout="", stderr="auth missing", returncode=1),
    )

    with pytest.raises(LLMError, match=r"exited with 1.*auth missing"):
        ClaudeCliClient().complete(system="s", user="u")


def test_claude_cli_handles_empty_stderr_on_failure(mocker: MockerFixture) -> None:
    mocker.patch("shutil.which", return_value="/usr/local/bin/claude")
    mocker.patch(
        "subprocess.run",
        return_value=_completed(stdout="", stderr="", returncode=2),
    )

    with pytest.raises(LLMError, match=r"\(no stderr\)"):
        ClaudeCliClient().complete(system="s", user="u")
