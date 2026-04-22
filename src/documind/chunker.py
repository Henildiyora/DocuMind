"""File walking and chunking.

Walks a directory, filters by extension/ignore rules, extracts text (including
PDFs), and splits into overlapping, line-aware chunks with metadata. Each chunk
carries its source path, line range, language, and file hash for incremental
indexing.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from .config import IGNORE_DIRS, SUPPORTED_EXTENSIONS, Config

LANG_BY_EXT: dict[str, str] = {
    ".py": "python", ".pyi": "python",
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".java": "java", ".kt": "kotlin", ".scala": "scala",
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".hpp": "cpp",
    ".cs": "csharp",
    ".go": "go", ".rs": "rust", ".rb": "ruby", ".php": "php",
    ".swift": "swift", ".m": "objc",
    ".json": "json", ".yaml": "yaml", ".yml": "yaml",
    ".toml": "toml", ".xml": "xml", ".ini": "ini",
    ".html": "html", ".css": "css", ".scss": "scss", ".sass": "sass",
    ".sql": "sql",
    ".md": "markdown", ".mdx": "markdown", ".rst": "rst", ".txt": "text",
    ".sh": "shell", ".bash": "shell", ".zsh": "shell", ".fish": "shell",
    ".pdf": "pdf",
}


@dataclass
class FileRecord:
    """Metadata for a single source file picked up by the walker."""

    path: str          # absolute POSIX path
    rel_path: str      # path relative to project root (POSIX)
    size: int
    mtime: float
    file_hash: str     # sha1 of content
    language: str


@dataclass
class Chunk:
    """A chunk produced from a file, ready for embedding + indexing."""

    chunk_id: str       # stable id: f"{file_hash}:{index}"
    rel_path: str
    language: str
    start_line: int     # 1-based, inclusive
    end_line: int       # 1-based, inclusive
    text: str
    file_hash: str


def _sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(data, usedforsecurity=False).hexdigest()


def _detect_language(path: Path) -> str:
    name = path.name.lower()
    if name == "dockerfile":
        return "dockerfile"
    if name in {"makefile", "gnumakefile"}:
        return "makefile"
    return LANG_BY_EXT.get(path.suffix.lower(), "text")


def _should_include(path: Path) -> bool:
    if path.suffix.lower() in SUPPORTED_EXTENSIONS:
        return True
    lname = path.name.lower()
    return lname in {"dockerfile", "makefile", "gnumakefile"}


def iter_source_files(root: Path, max_bytes: int) -> Iterator[Path]:
    """Yield candidate source files under `root`, honoring ignore rules."""
    root = root.resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS and not d.startswith(".")]
        for fname in filenames:
            if fname.startswith("."):
                continue
            p = Path(dirpath) / fname
            if not _should_include(p):
                continue
            try:
                if p.stat().st_size > max_bytes:
                    continue
            except OSError:
                continue
            yield p


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""
    try:
        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                continue
        return "\n\n".join(pages)
    except Exception:
        return ""


def _read_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return _read_pdf(path)
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    if b"\x00" in data[:4096]:
        return ""
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return ""


def scan_files(root: Path, cfg: Config) -> list[FileRecord]:
    """Walk the project and return FileRecord metadata (no text loaded yet)."""
    records: list[FileRecord] = []
    root = root.resolve()
    for p in iter_source_files(root, cfg.max_file_bytes):
        try:
            stat = p.stat()
            data = p.read_bytes()
        except OSError:
            continue
        records.append(
            FileRecord(
                path=p.as_posix(),
                rel_path=p.relative_to(root).as_posix(),
                size=stat.st_size,
                mtime=stat.st_mtime,
                file_hash=_sha1_bytes(data),
                language=_detect_language(p),
            )
        )
    return records


def _split_lines_into_chunks(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[int, int, str]]:
    """Line-aware chunking targeting `chunk_size` characters per chunk.

    Returns (start_line, end_line, text) tuples with 1-based inclusive lines.
    """
    if not text:
        return []
    lines = text.splitlines(keepends=True)
    chunks: list[tuple[int, int, str]] = []

    i = 0
    n = len(lines)
    while i < n:
        cur_len = 0
        j = i
        while j < n and cur_len + len(lines[j]) <= chunk_size:
            cur_len += len(lines[j])
            j += 1
        if j == i:
            j = i + 1
        start_line = i + 1
        end_line = j
        chunk_text = "".join(lines[i:j]).strip("\n")
        if chunk_text.strip():
            chunks.append((start_line, end_line, chunk_text))

        if j >= n:
            break

        # Step back by `chunk_overlap` worth of characters (line-aligned)
        back = 0
        k = j
        while k > i and back < chunk_overlap:
            k -= 1
            back += len(lines[k])
        i = max(k, i + 1)

    return chunks


def chunks_for_file(record: FileRecord, cfg: Config) -> list[Chunk]:
    """Read a file and split it into chunks carrying metadata."""
    text = _read_text(Path(record.path))
    if not text:
        return []

    raw = _split_lines_into_chunks(text, cfg.chunk_size, cfg.chunk_overlap)
    out: list[Chunk] = []
    for idx, (s, e, body) in enumerate(raw):
        out.append(
            Chunk(
                chunk_id=f"{record.file_hash}:{idx}",
                rel_path=record.rel_path,
                language=record.language,
                start_line=s,
                end_line=e,
                text=body,
                file_hash=record.file_hash,
            )
        )
    return out


def chunks_for_files(records: Iterable[FileRecord], cfg: Config) -> Iterator[Chunk]:
    """Yield chunks for a sequence of file records."""
    for rec in records:
        yield from chunks_for_file(rec, cfg)
