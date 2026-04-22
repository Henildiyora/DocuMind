"""Generated artifacts (htmlcov, coverage.xml, lockfiles, *.min.js) must not
make it into the index."""

from __future__ import annotations

from pathlib import Path

from documind.chunker import iter_source_files
from documind.config import load_config


def _write(path: Path, body: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def test_htmlcov_and_coverage_are_excluded(tmp_path: Path) -> None:
    cfg = load_config()

    _write(tmp_path / "app.py", "def hello():\n    return 'hi'\n")
    _write(tmp_path / "README.md", "# Project")

    # Things we expect to be filtered out:
    _write(tmp_path / "htmlcov" / "index.html", "<html>gen</html>")
    _write(tmp_path / "htmlcov" / "function_index.html", "<html>gen</html>")
    _write(tmp_path / "coverage.xml", "<coverage/>")
    _write(tmp_path / ".coverage", "binary")
    _write(tmp_path / "package-lock.json", "{}")
    _write(tmp_path / "bundle.min.js", "var x=1;")
    _write(tmp_path / "site" / "index.html", "<html>generated docs</html>")
    _write(tmp_path / "node_modules" / "leftpad" / "index.js", "module.exports=1;")

    picked = sorted(
        p.relative_to(tmp_path).as_posix()
        for p in iter_source_files(tmp_path, cfg.max_file_bytes)
    )

    assert "app.py" in picked
    assert "README.md" in picked

    forbidden = {
        "htmlcov/index.html",
        "htmlcov/function_index.html",
        "coverage.xml",
        ".coverage",
        "package-lock.json",
        "bundle.min.js",
        "site/index.html",
        "node_modules/leftpad/index.js",
    }
    assert forbidden.isdisjoint(picked), f"unexpected files got through: {forbidden & set(picked)}"
