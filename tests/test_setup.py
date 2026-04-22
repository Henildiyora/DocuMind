"""`documind setup` must succeed even when Ollama is unreachable, as long as
--no-pull is requested (or the user opts out of pulling)."""

from __future__ import annotations

from pathlib import Path

import pytest

from documind import setup as setup_mod


@pytest.fixture()
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect both XDG_CONFIG_HOME and HOME so we never touch the real config."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    return home


def test_setup_succeeds_with_ollama_unreachable_and_no_pull(
    tmp_path: Path,
    fake_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "main.py").write_text("print('hi')\n", encoding="utf-8")

    # Pretend Ollama isn't installed — setup must still return 0 with --no-pull.
    monkeypatch.setattr("shutil.which", lambda _name: None)

    code = setup_mod.run_setup(
        project,
        tier="tiny",
        yes=True,
        pull=False,
    )
    assert code == 0

    cfg_path = fake_home / ".config" / "documind" / "config.toml"
    assert cfg_path.exists(), "setup should always persist the model preference"
    body = cfg_path.read_text(encoding="utf-8")
    assert "qwen2.5-coder:1.5b" in body


def test_setup_with_pull_but_ollama_missing_still_exits_zero_when_pull_defaulted(
    tmp_path: Path,
    fake_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # pull=None is the "default, non-interactive" path which must be forgiving.
    project = tmp_path / "proj"
    project.mkdir()
    (project / "main.py").write_text("print('hi')\n", encoding="utf-8")

    monkeypatch.setattr("shutil.which", lambda _name: None)

    # Force non-interactive branch so _should_pull resolves to False.
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    code = setup_mod.run_setup(project, tier="tiny", yes=False, pull=None)
    assert code == 0
