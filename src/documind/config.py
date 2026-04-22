"""Configuration for DocuMind.

Loads defaults, merges with `~/.config/documind/config.toml` when present,
and exposes a small `Config` dataclass used throughout the package.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]


SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({
    ".py", ".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs",
    ".java", ".kt", ".scala",
    ".cpp", ".cc", ".c", ".h", ".hpp", ".cs",
    ".go", ".rs", ".php", ".rb", ".swift", ".m",
    ".json", ".yaml", ".yml", ".toml", ".xml", ".ini",
    ".html", ".css", ".scss", ".sass",
    ".sql", ".md", ".mdx", ".rst", ".txt",
    ".sh", ".bash", ".zsh", ".fish",
    ".dockerfile", ".makefile",
    ".pdf",
})

IGNORE_DIRS: frozenset[str] = frozenset({
    # VCS
    ".git", ".hg", ".svn",
    # Package / dependency dirs
    "node_modules", "bower_components",
    "venv", ".venv", "env", "DocuMind_venv",
    "site-packages", "dist-packages",
    "vendor",  # Go / PHP
    "Pods",    # iOS / CocoaPods
    # Caches
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".cache", ".tox", ".eggs",
    "hf_cache", ".parcel-cache", ".turbo", ".svelte-kit",
    # Editors / IDEs
    ".idea", ".vscode",
    # Build / generated output
    "build", "dist", "target", "out", "out-tsc",
    ".next", ".nuxt", ".vercel", ".gradle",
    "DerivedData", ".terraform",
    # Docs / coverage artifacts (the stuff that polluted search before)
    "htmlcov", "coverage", ".nyc_output",
    "site", "_site", "_build", "public",
    # DocuMind itself
    ".documind",
})

IGNORE_FILES: frozenset[str] = frozenset({
    # Coverage / test artifacts
    "coverage.xml", "coverage.json", ".coverage", "lcov.info",
    # Lockfiles (huge, not useful for semantic search)
    "package-lock.json", "pnpm-lock.yaml", "yarn.lock",
    "poetry.lock", "Pipfile.lock",
    "Cargo.lock", "go.sum",
    # OS / editor scratch
    ".DS_Store", "Thumbs.db",
})

IGNORE_FILE_GLOBS: tuple[str, ...] = (
    "*.min.js",
    "*.min.css",
    "*.map",
    "*.bundle.js",
    "*.bundle.css",
    "*.lock",
)


def _config_path() -> Path:
    """Return the path to the user's DocuMind config file."""
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "documind" / "config.toml"


@dataclass(frozen=True)
class Config:
    """Runtime configuration for DocuMind.

    All fields can be overridden by values in `~/.config/documind/config.toml`
    or by passing an explicit `Config` instance in code.
    """

    # Local LLM (Ollama) settings
    model: str = "gemma3:4b"
    ollama_base_url: str = "http://localhost:11434"
    llm_temperature: float = 0.1
    llm_num_ctx: int = 8192

    # Embedding model (fastembed ONNX, no torch required)
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # Retrieval / fusion
    top_k: int = 8
    rrf_k: int = 60  # constant used in Reciprocal Rank Fusion
    fuzzy_threshold: int = 82  # rapidfuzz score_cutoff (0-100)
    fuzzy_expand_per_term: int = 3

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 120
    max_file_bytes: int = 2_000_000  # skip files bigger than this

    # Index layout (project-local)
    index_dir_name: str = ".documind"

    # CLI / output
    show_line_ranges: bool = True
    snippet_chars: int = 320

    def to_dict(self) -> dict:
        return asdict(self)


def _load_toml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except Exception:
        return {}


def load_config(overrides: dict | None = None) -> Config:
    """Load config with precedence: defaults < file < explicit overrides."""
    data = _load_toml(_config_path())
    if overrides:
        data.update({k: v for k, v in overrides.items() if v is not None})

    valid = {f.name for f in fields(Config)}
    clean = {k: v for k, v in data.items() if k in valid}
    return Config(**clean)


def write_default_config() -> Path:
    """Write a commented default config file and return its path."""
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path

    cfg = Config()
    text = (
        "# DocuMind configuration\n"
        f'model = "{cfg.model}"                # Ollama model (run: ollama pull gemma3:4b)\n'
        f'ollama_base_url = "{cfg.ollama_base_url}"\n'
        f"llm_temperature = {cfg.llm_temperature}\n"
        f"llm_num_ctx = {cfg.llm_num_ctx}\n"
        "\n"
        f'embedding_model = "{cfg.embedding_model}"\n'
        f"embedding_dim = {cfg.embedding_dim}\n"
        "\n"
        f"top_k = {cfg.top_k}\n"
        f"fuzzy_threshold = {cfg.fuzzy_threshold}\n"
        f"fuzzy_expand_per_term = {cfg.fuzzy_expand_per_term}\n"
        "\n"
        f"chunk_size = {cfg.chunk_size}\n"
        f"chunk_overlap = {cfg.chunk_overlap}\n"
    )
    path.write_text(text, encoding="utf-8")
    return path


def index_dir_for(project_root: Path, cfg: Config | None = None) -> Path:
    """Return the per-project index directory (e.g. `./.documind`)."""
    cfg = cfg or load_config()
    return project_root / cfg.index_dir_name


def user_config_path() -> Path:
    """Expose the user-level config path for CLI/debug use."""
    return _config_path()


def update_user_config(updates: dict) -> Path:
    """Merge `updates` into the user-level TOML and write it.

    Preserves existing keys, writes a simple key-per-line TOML file. Only
    whitelisted fields (those present on `Config`) are persisted.
    """
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    valid = {f.name for f in fields(Config)}
    existing = _load_toml(path)
    merged = {k: v for k, v in existing.items() if k in valid}
    for k, v in updates.items():
        if k in valid and v is not None:
            merged[k] = v

    lines = ["# DocuMind configuration (managed by `documind setup`)"]
    for key in sorted(merged.keys()):
        val = merged[key]
        if isinstance(val, bool):
            lines.append(f"{key} = {'true' if val else 'false'}")
        elif isinstance(val, (int, float)):
            lines.append(f"{key} = {val}")
        else:
            escaped = str(val).replace('"', '\\"')
            lines.append(f'{key} = "{escaped}"')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
