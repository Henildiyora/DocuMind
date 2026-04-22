"""Hybrid retrieval: typo-tolerant BM25 + dense vectors, merged by RRF."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from .config import Config
from .embeddings import embed_query
from .index import DocuMindIndex

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}")


@dataclass
class SearchHit:
    """A single retrieval result."""

    chunk_id: str
    rel_path: str
    language: str
    start_line: int
    end_line: int
    text: str
    score: float
    # per-source ranks (None if the hit didn't appear there)
    bm25_rank: int | None = None
    vector_rank: int | None = None


# --------------------------------------------------------------------- helpers


def _tokenize(query: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(query)]


def _expand_query_fuzzy(
    query: str,
    vocab: list[str],
    cfg: Config,
) -> str:
    """Return an expanded query string with near-misspelling matches appended.

    Handles typos like "ingesion" -> "ingestion" by pulling the closest
    vocabulary terms via rapidfuzz and OR'ing them into the BM25 query.
    """
    try:
        from rapidfuzz import fuzz, process
    except ImportError:
        return query

    terms = _tokenize(query)
    if not terms or not vocab:
        return query

    vocab_set = set(vocab)
    additions: list[str] = []
    for term in terms:
        if term in vocab_set:
            continue
        matches = process.extract(
            term,
            vocab,
            scorer=fuzz.WRatio,
            limit=cfg.fuzzy_expand_per_term,
            score_cutoff=cfg.fuzzy_threshold,
        )
        for cand, _score, _idx in matches:
            if cand != term:
                additions.append(cand)

    if not additions:
        return query
    return query + " " + " ".join(additions)


def _rrf_merge(
    rankings: list[list[str]],
    k: int,
    top_n: int,
) -> list[tuple[str, float, list[int | None]]]:
    """Reciprocal Rank Fusion across N ranked id lists.

    Returns a list of (id, score, [rank_in_list_i ...]) sorted by score desc.
    """
    scores: dict[str, float] = {}
    per_source_ranks: dict[str, list[int | None]] = {}

    for src_idx, ranking in enumerate(rankings):
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in per_source_ranks:
                per_source_ranks[cid] = [None] * len(rankings)
            per_source_ranks[cid][src_idx] = rank

    sorted_ids = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return [(cid, score, per_source_ranks[cid]) for cid, score in sorted_ids]


def _vocab_from_bm25(retriever) -> list[str]:
    """Extract the bm25s vocabulary (list of tokens) for fuzzy expansion."""
    vocab_attr = getattr(retriever, "vocab_dict", None)
    if isinstance(vocab_attr, dict):
        return list(vocab_attr.keys())
    vocab_attr = getattr(retriever, "vocabulary", None)
    if isinstance(vocab_attr, dict):
        return list(vocab_attr.keys())
    if isinstance(vocab_attr, list):
        return list(vocab_attr)
    return []


# ------------------------------------------------------------------ searches


def _bm25_search(
    idx: DocuMindIndex,
    query: str,
    cfg: Config,
    limit: int,
) -> list[str]:
    import bm25s

    retriever, chunk_ids = idx.load_bm25()
    if retriever is None or not chunk_ids:
        return []

    vocab = _vocab_from_bm25(retriever)
    expanded = _expand_query_fuzzy(query, vocab, cfg) if vocab else query

    try:
        query_tokens = bm25s.tokenize(
            [expanded], stopwords="en", stemmer=None, show_progress=False
        )
        results, _scores = retriever.retrieve(
            query_tokens,
            k=min(limit, len(chunk_ids)),
            show_progress=False,
        )
    except Exception:
        return []

    row = results[0] if len(results) else []
    ordered_ids: list[str] = []
    for item in row:
        doc_idx = int(item) if not isinstance(item, (int, np.integer)) else int(item)
        if 0 <= doc_idx < len(chunk_ids):
            ordered_ids.append(chunk_ids[doc_idx])
    return ordered_ids


def _vector_search(
    idx: DocuMindIndex,
    query: str,
    cfg: Config,
    limit: int,
) -> list[str]:
    table = idx._open_lance_table(create=False)
    if table is None:
        return []
    try:
        qvec = embed_query(query, cfg)
    except Exception:
        return []

    try:
        df = (
            table.search(qvec.tolist())
            .limit(limit)
            .select(["chunk_id", "_distance"])
            .to_list()
        )
    except Exception:
        return []
    return [row["chunk_id"] for row in df if row.get("chunk_id")]


# ------------------------------------------------------------------- public


def search(
    idx: DocuMindIndex,
    query: str,
    cfg: Config,
    top_k: int | None = None,
) -> list[SearchHit]:
    """Run hybrid BM25 + vector search and return RRF-merged SearchHits."""
    k = top_k or cfg.top_k
    pool = max(k * 4, 20)

    bm25_ids = _bm25_search(idx, query, cfg, pool)
    vec_ids = _vector_search(idx, query, cfg, pool)

    if not bm25_ids and not vec_ids:
        return []

    merged = _rrf_merge([bm25_ids, vec_ids], cfg.rrf_k, k)
    lookup = idx.chunks_by_ids([cid for cid, _, _ in merged])

    hits: list[SearchHit] = []
    for cid, score, per_ranks in merged:
        row = lookup.get(cid)
        if not row:
            continue
        hits.append(
            SearchHit(
                chunk_id=cid,
                rel_path=row["rel_path"],
                language=row["language"],
                start_line=row["start_line"],
                end_line=row["end_line"],
                text=row["text"],
                score=score,
                bm25_rank=per_ranks[0],
                vector_rank=per_ranks[1],
            )
        )
    return hits


def format_snippet(hit: SearchHit, cfg: Config) -> str:
    """Trim a hit's text for display while preserving leading code context."""
    text = hit.text.rstrip()
    if len(text) <= cfg.snippet_chars:
        return text
    return text[: cfg.snippet_chars].rstrip() + "\n..."


def hits_to_context(hits: Iterable[SearchHit], max_chars: int = 8000) -> str:
    """Concatenate hits into a bounded context string for the LLM."""
    parts: list[str] = []
    used = 0
    for h in hits:
        header = f"### {h.rel_path}:{h.start_line}-{h.end_line} ({h.language})"
        block = f"{header}\n```{h.language if h.language != 'text' else ''}\n{h.text}\n```"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block) + 1
    return "\n\n".join(parts)
