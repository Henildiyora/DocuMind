"""Retrieval quality: Dockerfile filename boost + overview entrypoint ranking."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from documind.config import Config
from documind.index import DocuMindIndex
from documind.query_understand import (
    detect_ambiguity,
    is_overview_intent,
    refine_hits,
    rewrite_query,
)
from documind.search import SearchHit, search

DOCKERFILE_MARKER = "FROM python:3.12-slim AS documind_unique_stage"
README_MARKER = "AI_GitHub_Profile_Summarizer unique overview marker"


@pytest.fixture()
def docker_project(tmp_path: Path) -> Path:
    (tmp_path / "Dockerfile").write_text(
        textwrap.dedent(
            f"""
            # Production image
            {DOCKERFILE_MARKER}
            WORKDIR /app
            COPY . .
            CMD ["python", "main.py"]
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    # Noise that is still indexed (not under generated_reports / report.json).
    noise = {
        "questions": [
            "Explain your experience with Docker and Kubernetes.",
            "How would you containerize a microservice?",
            "Describe CI/CD with docker compose and helm charts.",
        ]
        * 20,
        "notes": "interview prep about containers, not this project's Dockerfile",
    }
    (tmp_path / "misc").mkdir()
    (tmp_path / "misc" / "container_interview.json").write_text(
        json.dumps(noise, indent=2), encoding="utf-8"
    )
    (tmp_path / "main.py").write_text("print('hi')\n", encoding="utf-8")
    return tmp_path


@pytest.fixture()
def overview_project(tmp_path: Path) -> Path:
    (tmp_path / "README.md").write_text(
        f"# Profile Summarizer\n\n{README_MARKER}\n\n"
        "This app analyzes GitHub profiles and writes candidate reports.\n",
        encoding="utf-8",
    )
    (tmp_path / "main.py").write_text(
        "def main():\n"
        "    '''CLI entrypoint for the profile summarizer.'''\n"
        "    print('summarize')\n",
        encoding="utf-8",
    )
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "models.py").write_text(
        "class Candidate:\n    pass\n", encoding="utf-8"
    )
    # Indexed noise that would otherwise dominate vague "project" queries.
    blob = {
        "questions": [
            "Explain AWS Elastic Beanstalk and SageMaker experience.",
            "Describe C++ Java Python tradeoffs on your resume projects.",
        ]
        * 30
    }
    (tmp_path / "misc").mkdir()
    (tmp_path / "misc" / "resume_interview.json").write_text(
        json.dumps(blob, indent=2), encoding="utf-8"
    )
    # These must be ignored by the walker (regression for the real bug).
    (tmp_path / "generated_reports" / "x").mkdir(parents=True)
    (tmp_path / "generated_reports" / "x" / "report.json").write_text(
        json.dumps(blob), encoding="utf-8"
    )
    return tmp_path


@pytest.fixture()
def built_docker_index(docker_project: Path):
    cfg = Config()
    idx = DocuMindIndex(docker_project, cfg)
    idx.build_or_update()
    yield idx, cfg
    idx.destroy()
    idx.close()


@pytest.fixture()
def built_overview_index(overview_project: Path):
    cfg = Config()
    idx = DocuMindIndex(overview_project, cfg)
    idx.build_or_update()
    yield idx, cfg, overview_project
    idx.destroy()
    idx.close()


def test_docker_file_query_ranks_dockerfile_first(built_docker_index) -> None:
    idx, cfg = built_docker_index
    query = "can you explain me the docker file"
    hits = refine_hits(search(idx, query, cfg, top_k=8), query, cfg)

    assert hits, "expected at least one hit"
    top = hits[0]
    assert Path(top.rel_path).name.lower() == "dockerfile"
    assert DOCKERFILE_MARKER in top.text


def test_overview_query_prefers_readme_or_main(built_overview_index) -> None:
    idx, cfg, root = built_overview_index
    # generated_reports must not be indexed at all
    all_paths = {c["rel_path"] for c in idx.all_chunks()}
    assert not any("generated_reports" in p for p in all_paths)
    assert not any(Path(p).name == "report.json" for p in all_paths)

    query = "can you explain me this project"
    hits = refine_hits(search(idx, query, cfg, top_k=8), query, cfg)
    assert hits
    top_name = Path(hits[0].rel_path).name.lower()
    assert top_name in {"readme.md", "main.py"}, (
        f"expected README or main.py first, got {hits[0].rel_path}"
    )
    # Artifact demotion: interview dump should not be #1
    assert "resume_interview" not in hits[0].rel_path


def test_overview_intent_skips_ambiguity_clarify() -> None:
    assert is_overview_intent("can you explain me this project")
    assert is_overview_intent("how many files are there in this project")
    hits = [
        SearchHit("1", "a.py", "python", 1, 2, "x", 0.1),
        SearchHit("2", "b.py", "python", 1, 2, "y", 0.09),
    ]
    assert detect_ambiguity(hits, "explain this project") is False


def test_rewrite_query_fallback_without_valid_json() -> None:
    class FakeLLM:
        cfg = Config()

        def chat(self, messages, model=None, keep_alive=None):
            return "not json at all"

    normalized, alts = rewrite_query(FakeLLM(), "explian docker file", Config())  # type: ignore[arg-type]
    assert normalized == "explian docker file"
    assert alts == []


def test_rewrite_adds_overview_alternates() -> None:
    class FakeLLM:
        cfg = Config()

        def chat(self, messages, model=None, keep_alive=None):
            return json.dumps({"normalized": "explain this project", "alternates": []})

    normalized, alts = rewrite_query(
        FakeLLM(), "can you explain me this project", Config()  # type: ignore[arg-type]
    )
    assert normalized == "explain this project"
    assert any("README" in a for a in alts)
