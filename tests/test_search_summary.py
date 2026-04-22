"""Tests for `documind search`'s natural-language summary mode.

Covered cases:

* `--no-summary` never touches the LLM (even if ``ping()`` would explode).
* Auto path with the LLM unreachable falls back to snippets cleanly.
* Auto path with a fake LLM that yields tokens prints the answer and a
  ``Sources`` header.
"""

from __future__ import annotations

import textwrap
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from documind import cli as cli_mod
from documind.cli import app

runner = CliRunner()


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    """Create a tiny project with a distinctive symbol to search for."""
    (tmp_path / "app.py").write_text(
        textwrap.dedent(
            """
            class RateLimiter:
                '''Simple token bucket.'''

                def is_allowed(self, key: str) -> bool:
                    return True
            """
        ).strip(),
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text(
        "# Demo\n\nA tiny RateLimiter for testing.\n", encoding="utf-8"
    )
    return tmp_path


class _FakeLLM:
    """Minimal stand-in for ``OllamaClient``."""

    def __init__(
        self,
        *,
        up: bool = True,
        model_pulled: bool = True,
        answer: str = "RateLimiter is a simple token bucket.",
    ) -> None:
        self._up = up
        self._pulled = model_pulled
        self._answer = answer

    def ping(self) -> bool:
        return self._up

    def model_available(self, _model: str | None = None) -> bool:
        return self._pulled

    def chat_stream(
        self, _messages, *, model: str | None = None
    ) -> Iterator[str]:
        # Stream a few tokens so the Live panel exercises multiple updates.
        for chunk in self._answer.split(" "):
            yield chunk + " "


def _install_fake_llm(monkeypatch: pytest.MonkeyPatch, fake: _FakeLLM) -> None:
    """Replace the ``OllamaClient`` symbol used by ``documind.cli`` with ``fake``."""
    monkeypatch.setattr(cli_mod, "OllamaClient", lambda _cfg: fake)


def test_no_summary_never_touches_llm(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--no-summary must not instantiate or call the LLM at all."""
    def explode(_cfg):
        raise RuntimeError("LLM should not have been constructed")

    monkeypatch.setattr(cli_mod, "OllamaClient", explode)

    result = runner.invoke(
        app,
        [
            "search", "RateLimiter",
            "--path", str(project),
            "--auto-index",
            "--no-summary",
            "--k", "3",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Snippets from" in result.output
    assert "Answer" not in result.output


def test_auto_with_llm_down_falls_back_to_snippets(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Auto mode + LLM unreachable -> snippet view, no crash, exit 0."""
    _install_fake_llm(monkeypatch, _FakeLLM(up=False))

    result = runner.invoke(
        app,
        [
            "search", "RateLimiter",
            "--path", str(project),
            "--auto-index",
            "--k", "3",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Snippets from" in result.output
    assert "documind setup" in result.output  # the tip line
    assert "Answer" not in result.output


def test_auto_with_ready_llm_prints_answer_and_sources(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Auto mode + ready LLM -> streamed answer + Sources header."""
    fake = _FakeLLM(answer="RateLimiter implements a token bucket.")
    _install_fake_llm(monkeypatch, fake)

    result = runner.invoke(
        app,
        [
            "search", "what does RateLimiter do?",
            "--path", str(project),
            "--auto-index",
            "--k", "3",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Answer" in result.output
    assert "Sources" in result.output
    assert "RateLimiter implements a token bucket." in result.output
