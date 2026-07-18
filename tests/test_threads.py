"""Persistent chat threads and variadic ask."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from documind.cli import app
from documind.config import load_config
from documind.threads import (
    append_turn,
    delete_thread,
    list_threads,
    load_thread,
    rename_thread,
    save_thread,
)

runner = CliRunner()


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    (tmp_path / "app.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    return tmp_path


def test_thread_roundtrip(project: Path) -> None:
    cfg = load_config()
    t = append_turn(project, "default", "hello?", "world!", cfg=cfg)
    assert t.name == "default"
    assert len(t.messages) == 2

    loaded = load_thread(project, "default", cfg)
    assert loaded.messages[0]["content"] == "hello?"
    assert loaded.preview().startswith("world")

    rename_thread(project, "default", "auth", cfg)
    names = [x.name for x in list_threads(project, cfg)]
    assert "auth" in names
    assert "default" not in names

    delete_thread(project, "auth", cfg)
    assert list_threads(project, cfg) == []


def test_invalid_thread_name(project: Path) -> None:
    with pytest.raises(ValueError):
        save_thread(project, load_thread(project, "bad name!"))


def test_ask_joins_trailing_args(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Index first (non-interactive)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    r_idx = runner.invoke(app, ["index", str(project)])
    assert r_idx.exit_code == 0, r_idx.output

    r = runner.invoke(
        app,
        [
            "ask",
            "why",
            "does",
            "add",
            "exist",
            "--path",
            str(project),
            "--no-llm",
            "--no-stale-warn",
        ],
    )
    assert r.exit_code == 0, r.output
    assert "app.py" in r.output


def test_ask_without_question_opens_chat(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = {"n": 0}

    def fake_chat(root, cfg, thread_name="default"):
        called["n"] += 1
        assert root == project.resolve()

    monkeypatch.setattr("documind.chat.run_chat", fake_chat)
    r = runner.invoke(app, ["ask", "--path", str(project)])
    assert r.exit_code == 0, r.output
    assert called["n"] == 1
