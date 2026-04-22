# Contributing to DocuMind

Thanks for your interest. DocuMind is a small, self-contained project that
welcomes bug fixes, docs improvements, and well-scoped features.

## Dev setup

```bash
git clone https://github.com/Henildiyora/DocuMind.git
cd DocuMind
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running tests and linters

```bash
pytest
ruff check src/ tests/
```

The test suite bootstraps a tiny on-disk index in `tmp_path` (see
[tests/test_search.py](tests/test_search.py)); no network or Ollama is needed.

## Manual end-to-end smoke check

```bash
documind index .
documind search "reciprocal rank fusion"
documind doctor
```

If you have Ollama running locally, also try:

```bash
documind ask "how is the index built?"
```

## Project layout

- [src/documind/cli.py](src/documind/cli.py)        Typer CLI entrypoints.
- [src/documind/index.py](src/documind/index.py)      Hybrid index (SQLite + LanceDB + BM25).
- [src/documind/search.py](src/documind/search.py)     Hybrid retrieval + fuzzy expansion + RRF.
- [src/documind/chunker.py](src/documind/chunker.py)    File walking and line-aware chunking.
- [src/documind/embeddings.py](src/documind/embeddings.py) Fastembed wrapper (BGE small).
- [src/documind/llm.py](src/documind/llm.py)        Ollama client with streaming.
- [src/documind/chat.py](src/documind/chat.py)       Rich-based REPL.
- [src/documind/setup.py](src/documind/setup.py)      `documind setup` flow (model recommender).
- [src/documind/models.py](src/documind/models.py)     Curated Ollama model tiers.
- [src/documind/config.py](src/documind/config.py)     TOML config loader.
- [src/documind/prompts.py](src/documind/prompts.py)    RAG synthesis prompt.

## Pull request checklist

- [ ] `pytest` passes.
- [ ] `ruff check src/ tests/` is clean.
- [ ] New behavior has a test, or a clear explanation of why it can't.
- [ ] README or CONTRIBUTING updated if the UX or install path changed.

## Filing issues

Please include:
- DocuMind version (`documind --version`)
- OS and Python version
- Minimal reproduction (a sample directory structure and the query)
- Output of `documind doctor`
