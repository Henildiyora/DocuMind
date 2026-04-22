"""Text embeddings via fastembed (ONNX, no torch required).

Uses the BGE small English model by default (384d, tiny, very fast on CPU).
The underlying model is downloaded and cached by fastembed itself on first use.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from .config import Config


@dataclass
class _EmbedderState:
    model_name: str | None = None
    model: object | None = None  # fastembed.TextEmbedding instance


_state = _EmbedderState()


def _get_model(model_name: str):
    """Lazy-load the fastembed TextEmbedding model (cached per process)."""
    if _state.model is not None and _state.model_name == model_name:
        return _state.model

    try:
        from fastembed import TextEmbedding
    except ImportError as exc:  # pragma: no cover - import error path
        raise RuntimeError(
            "fastembed is required. Install with: pip install fastembed"
        ) from exc

    _state.model = TextEmbedding(model_name=model_name)
    _state.model_name = model_name
    return _state.model


def embed_texts(
    texts: Iterable[str],
    cfg: Config,
    batch_size: int = 64,
) -> np.ndarray:
    """Embed an iterable of texts and return an (N, dim) float32 ndarray.

    Vectors are L2-normalized so cosine similarity == dot product.
    """
    texts = [t if t else " " for t in texts]
    if not texts:
        return np.zeros((0, cfg.embedding_dim), dtype=np.float32)

    model = _get_model(cfg.embedding_model)
    vectors = list(model.embed(texts, batch_size=batch_size))
    arr = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return arr / norms


def embed_query(query: str, cfg: Config) -> np.ndarray:
    """Embed a single query string and return a 1-D float32 ndarray."""
    model = _get_model(cfg.embedding_model)
    vec = next(iter(model.query_embed([query])))
    arr = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(arr))
    return arr / n if n else arr
