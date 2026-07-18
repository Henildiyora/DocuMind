"""Phase 1: post-index setup offer and friendly typed errors."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from documind import cli as cli_mod
from documind.cli import app
from documind.config import load_config, update_user_config
from documind.errors import ModelNotPulled, OllamaNotInstalled, format_user_error

runner = CliRunner()


@pytest.fixture()
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    return home


@pytest.fixture()
def tiny_project(tmp_path: Path) -> Path:
    (tmp_path / "app.py").write_text("def hello():\n    return 1\n", encoding="utf-8")
    return tmp_path


def test_index_skips_setup_offer_when_not_tty(
    tiny_project: Path,
    fake_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli_mod, "_stdin_is_tty", lambda: False)
    offered = {"called": False}

    def boom(_root):
        offered["called"] = True
        return 0

    monkeypatch.setattr("documind.setup.run_setup", boom)
    r = runner.invoke(app, ["index", str(tiny_project)])
    assert r.exit_code == 0, r.output
    assert offered["called"] is False
    assert "Want AI-powered" not in r.output


def test_index_decline_writes_flag_and_does_not_reprompt(
    tiny_project: Path,
    fake_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli_mod, "_stdin_is_tty", lambda: True)
    monkeypatch.setattr(cli_mod.Confirm, "ask", lambda *a, **k: False)

    r1 = runner.invoke(app, ["index", str(tiny_project)])
    assert r1.exit_code == 0, r1.output
    assert "documind setup" in r1.output

    cfg = load_config()
    assert cfg.offer_setup_after_index is False

    asked = {"n": 0}

    def track(*_a, **_k):
        asked["n"] += 1
        return False

    monkeypatch.setattr(cli_mod.Confirm, "ask", track)
    r2 = runner.invoke(app, ["index", str(tiny_project)])
    assert r2.exit_code == 0, r2.output
    assert asked["n"] == 0


def test_index_accept_runs_setup(
    tiny_project: Path,
    fake_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli_mod, "_stdin_is_tty", lambda: True)
    monkeypatch.setattr(cli_mod.Confirm, "ask", lambda *a, **k: True)
    called = {"n": 0}

    def fake_setup(root, **_kwargs):
        called["n"] += 1
        update_user_config({"model": "qwen2.5-coder:1.5b", "setup_done": True})
        return 0

    monkeypatch.setattr("documind.setup.run_setup", fake_setup)
    r = runner.invoke(app, ["index", str(tiny_project)])
    assert r.exit_code == 0, r.output
    assert called["n"] == 1
    assert load_config().setup_done is True


def test_ask_ollama_missing_friendly_no_traceback(
    tiny_project: Path,
    fake_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner.invoke(app, ["index", str(tiny_project)])
    monkeypatch.setattr(cli_mod, "ollama_installed", lambda: False)
    # Ensure setup offer does not interfere
    update_user_config({"offer_setup_after_index": False})

    r = runner.invoke(
        app,
        ["ask", "what", "is", "this", "--path", str(tiny_project), "--no-stale-warn"],
    )
    assert r.exit_code == 2
    assert "Ollama isn't installed" in r.output
    assert "Traceback" not in r.output
    assert "brew install ollama" in r.output or "install.sh" in r.output or "ollama.com" in r.output


def test_ask_debug_flag_in_help() -> None:
    r = runner.invoke(app, ["--help"])
    assert r.exit_code == 0
    assert "--debug" in r.output or "-v" in r.output


def test_format_user_error_messages() -> None:
    msg = format_user_error(OllamaNotInstalled())
    assert "Ollama isn't installed" in msg
    assert "documind search" in msg

    msg2 = format_user_error(ModelNotPulled("gemma3:4b"))
    assert "gemma3:4b" in msg2
    assert "documind setup --pull" in msg2
