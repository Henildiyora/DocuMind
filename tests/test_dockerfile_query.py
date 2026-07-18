"""Phase 4 regression: 'docker file' query should rank Dockerfile first."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from documind.config import Config
from documind.index import DocuMindIndex
from documind.query_understand import boost_filename_matches, rewrite_query
from documind.search import filter_relevant, search

DOCKERFILE_MARKER = "FROM python:3.12-slim AS documind_unique_stage"


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
    # Noisy interview / report content that previously polluted RRF top-K.
    report = {
        "questions": [
            "Explain your experience with Docker and Kubernetes.",
            "How would you containerize a microservice?",
            "Describe CI/CD with docker compose and helm charts.",
        ]
        * 20,
        "notes": "interview prep about containers, not this project's Dockerfile",
    }
    (tmp_path / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (tmp_path / "main.py").write_text("print('hi')\n", encoding="utf-8")
    return tmp_path


@pytest.fixture()
def built_index(docker_project: Path):
    cfg = Config()
    idx = DocuMindIndex(docker_project, cfg)
    idx.build_or_update()
    yield idx, cfg
    idx.destroy()
    idx.close()


def test_docker_file_query_ranks_dockerfile_first(built_index) -> None:
    idx, cfg = built_index
    query = "can you explain me the docker file"
    hits = search(idx, query, cfg, top_k=8)
    hits = boost_filename_matches(hits, query)
    hits = filter_relevant(hits, cfg) or hits

    assert hits, "expected at least one hit"
    top = hits[0]
    assert Path(top.rel_path).name.lower() == "dockerfile"
    assert DOCKERFILE_MARKER in top.text


def test_rewrite_query_fallback_without_valid_json() -> None:
    class FakeLLM:
        cfg = Config()

        def chat(self, messages, model=None, keep_alive=None):
            return "not json at all"

    normalized, alts = rewrite_query(FakeLLM(), "explian docker file", Config())  # type: ignore[arg-type]
    assert normalized == "explian docker file"
    assert alts == []


def test_rewrite_query_parses_json() -> None:
    class FakeLLM:
        cfg = Config()

        def chat(self, messages, model=None, keep_alive=None):
            return json.dumps(
                {
                    "normalized": "explain the Dockerfile",
                    "alternates": ["Dockerfile contents", "docker image build file"],
                }
            )

    normalized, alts = rewrite_query(FakeLLM(), "explian docker file", Config())  # type: ignore[arg-type]
    assert normalized == "explain the Dockerfile"
    assert "Dockerfile contents" in alts
