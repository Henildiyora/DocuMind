"""Smoke tests for the DocuMind hybrid search.

These tests build a tiny on-disk index in a pytest tmpdir and verify:
    - exact-keyword queries return the expected chunk
    - misspelled queries still retrieve the right file via fuzzy expansion
    - unrelated queries don't accidentally rank the fixture first
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from documind.config import load_config
from documind.index import DocuMindIndex
from documind.search import search

FIXTURE_PY = textwrap.dedent(
    '''
    """Ingestion pipeline.

    This module loads documents, splits them into chunks, and stores them
    into a hybrid vector + keyword index for later retrieval.
    """


    class IngestionPipeline:
        """Pipeline that ingests and embeds project documents."""

        def run(self, directory_path: str) -> None:
            """Walk the directory and embed every supported file."""
            for file in self.scan(directory_path):
                chunks = self.split(file)
                self.embed(chunks)


    class Retriever:
        """BM25 plus dense-vector retrieval, merged by reciprocal rank fusion."""

        def query(self, text: str) -> list[str]:
            return []
    '''
).strip()


FIXTURE_MD = textwrap.dedent(
    """
    # Authentication

    Session expiration is handled by the `SessionStore` class which evicts
    tokens older than one hour. This is unrelated to ingestion or retrieval.
    """
).strip()


@pytest.fixture(scope="module")
def built_index(tmp_path_factory) -> DocuMindIndex:
    root: Path = tmp_path_factory.mktemp("project")
    (root / "pipeline.py").write_text(FIXTURE_PY, encoding="utf-8")
    (root / "auth.md").write_text(FIXTURE_MD, encoding="utf-8")

    cfg = load_config()
    idx = DocuMindIndex(root, cfg)
    stats = idx.build_or_update()
    assert stats.total_chunks >= 1
    assert stats.new_files == 2
    yield idx
    idx.destroy()


def test_exact_keyword_hits_pipeline(built_index: DocuMindIndex) -> None:
    cfg = load_config()
    hits = search(built_index, "IngestionPipeline", cfg, top_k=3)
    assert hits, "Expected at least one hit for an exact class name"
    assert hits[0].rel_path == "pipeline.py"


def test_typo_query_still_finds_ingestion(built_index: DocuMindIndex) -> None:
    cfg = load_config()
    # Misspelled "ingestion"
    hits = search(built_index, "ingesion pipelin", cfg, top_k=5)
    assert hits, "Expected fuzzy expansion to rescue a misspelled query"
    paths = {h.rel_path for h in hits}
    assert "pipeline.py" in paths


def test_semantic_query_finds_retrieval_code(built_index: DocuMindIndex) -> None:
    cfg = load_config()
    hits = search(built_index, "how do we combine bm25 with embeddings", cfg, top_k=5)
    assert hits
    assert any(h.rel_path == "pipeline.py" for h in hits)


def test_unrelated_file_not_top_ranked(built_index: DocuMindIndex) -> None:
    cfg = load_config()
    hits = search(built_index, "IngestionPipeline class", cfg, top_k=3)
    assert hits[0].rel_path == "pipeline.py"


def test_incremental_reindex_is_noop(built_index: DocuMindIndex) -> None:
    stats = built_index.build_or_update()
    assert stats.new_files == 0
    assert stats.changed_files == 0
    assert stats.unchanged_files >= 2
