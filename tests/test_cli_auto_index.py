"""Smoke tests for the CLI auto-index flow on `documind search`."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from typer.testing import CliRunner

from documind.cli import app

runner = CliRunner()


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    """Create a trivial project with one well-known identifier to find."""
    (tmp_path / "app.py").write_text(
        textwrap.dedent(
            """
            class IngestionPipeline:
                '''Pipeline that does the thing.'''

                def run(self, path: str) -> None:
                    print("ingesting", path)
            """
        ).strip(),
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("# Demo project", encoding="utf-8")
    return tmp_path


def test_search_auto_indexes_on_missing_index(project: Path) -> None:
    assert not (project / ".documind").exists()

    result = runner.invoke(
        app,
        [
            "search",
            "IngestionPipeline",
            "--path", str(project),
            "--auto-index",
            "--no-code",
            "--k", "3",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (project / ".documind").exists(), "index should have been built"
    assert "app.py" in result.output


def test_search_respects_no_auto_index(project: Path) -> None:
    assert not (project / ".documind").exists()

    result = runner.invoke(
        app,
        [
            "search",
            "IngestionPipeline",
            "--path", str(project),
            "--no-auto-index",
        ],
    )
    assert result.exit_code != 0
    assert "No index found" in result.output
    assert not (project / ".documind").exists()
