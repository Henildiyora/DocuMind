"""CLI behavior: stale index vs disk, and re-index picks up new tokens."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from typer.testing import CliRunner

from documind.cli import app

runner = CliRunner()


@pytest.fixture()
def tiny_project(tmp_path: Path) -> Path:
    (tmp_path / "app.py").write_text(
        textwrap.dedent(
            '''
            # TOKEN_ALPHA_UNIQUE
            def hello():
                return 1
            '''
        ).strip(),
        encoding="utf-8",
    )
    return tmp_path


def test_search_without_reindex_misses_new_token(tiny_project: Path) -> None:
    r1 = runner.invoke(
        app,
        ["index", str(tiny_project)],
    )
    assert r1.exit_code == 0, r1.output

    (tiny_project / "app.py").write_text(
        textwrap.dedent(
            '''
            # TOKEN_BETA_UNIQUE
            def hello():
                return 2
            '''
        ).strip(),
        encoding="utf-8",
    )

    r2 = runner.invoke(
        app,
        [
            "search",
            "TOKEN_BETA_UNIQUE",
            "--path", str(tiny_project),
            "--no-summary",
            "--no-stale-warn",
            "--k", "3",
        ],
    )
    assert r2.exit_code == 0, r2.output
    assert "TOKEN_BETA_UNIQUE" not in r2.output


def test_search_after_index_finds_new_token(tiny_project: Path) -> None:
    runner.invoke(app, ["index", str(tiny_project)])
    (tiny_project / "app.py").write_text(
        textwrap.dedent(
            '''
            # TOKEN_BETA_UNIQUE
            def hello():
                return 2
            '''
        ).strip(),
        encoding="utf-8",
    )
    r_index = runner.invoke(app, ["index", str(tiny_project)])
    assert r_index.exit_code == 0, r_index.output

    r2 = runner.invoke(
        app,
        [
            "search",
            "TOKEN_BETA_UNIQUE",
            "--path", str(tiny_project),
            "--no-summary",
            "--no-stale-warn",
            "--k", "3",
        ],
    )
    assert r2.exit_code == 0, r2.output
    assert "TOKEN_BETA_UNIQUE" in r2.output


def test_stale_warn_prints_when_disk_newer(tiny_project: Path) -> None:
    runner.invoke(app, ["index", str(tiny_project)])
    (tiny_project / "app.py").write_text("# edited\nx = 1\n", encoding="utf-8")

    r = runner.invoke(
        app,
        [
            "search",
            "hello",
            "--path", str(tiny_project),
            "--no-summary",
            "--k", "2",
        ],
    )
    assert r.exit_code == 0, r.output
    assert "Your working tree looks newer than the last index" in r.output


def test_watch_help_runs() -> None:
    r = runner.invoke(app, ["watch", "--help"])
    assert r.exit_code == 0
    assert "debounce" in r.output.lower()
