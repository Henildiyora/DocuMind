"""Per-project hybrid index.

Layout inside `<project>/.documind/`:
    meta.sqlite    - files + chunks tables (ground truth for metadata/text)
    lance/         - LanceDB directory for dense vectors
    bm25/          - saved bm25s retriever (keyword index)
    state.json     - versioning / model info

Supports incremental reindex: only files whose content hash changed are
re-embedded; deleted files have their chunks purged; BM25 is rebuilt from the
chunks table (cheap and keeps IDF correct).
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa

from .chunker import Chunk, FileRecord, chunks_for_file, scan_files
from .config import Config, index_dir_for
from .embeddings import embed_texts

SCHEMA_VERSION = 1


def _extract_table_names(db) -> set[str]:
    """Normalize the varied return types of `lancedb.Connection.list_tables`.

    Newer lancedb versions return a `ListTablesResponse` pydantic object whose
    iterator yields `(field_name, value)` tuples instead of raw names.
    """
    out: set[str] = set()
    # Pydantic-style response
    tables_attr = getattr(db, "list_tables", None)
    if callable(tables_attr):
        try:
            resp = db.list_tables()
        except Exception:
            resp = None
        if resp is not None:
            maybe_tables = getattr(resp, "tables", None)
            if isinstance(maybe_tables, (list, tuple)):
                out.update(str(x) for x in maybe_tables)
                return out
            if isinstance(resp, (list, tuple, set)):
                for x in resp:
                    if isinstance(x, str):
                        out.add(x)
                return out
    # Older API
    table_names_attr = getattr(db, "table_names", None)
    if callable(table_names_attr):
        try:
            for x in db.table_names():
                if isinstance(x, str):
                    out.add(x)
        except Exception:
            pass
    return out

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS files (
    rel_path   TEXT PRIMARY KEY,
    file_hash  TEXT NOT NULL,
    mtime      REAL NOT NULL,
    size       INTEGER NOT NULL,
    language   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id   TEXT PRIMARY KEY,
    rel_path   TEXT NOT NULL,
    file_hash  TEXT NOT NULL,
    language   TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line   INTEGER NOT NULL,
    text       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_rel_path ON chunks(rel_path);
CREATE INDEX IF NOT EXISTS idx_chunks_file_hash ON chunks(file_hash);
"""


@dataclass
class IndexStats:
    scanned_files: int = 0
    new_files: int = 0
    changed_files: int = 0
    unchanged_files: int = 0
    removed_files: int = 0
    total_chunks: int = 0
    embedded_chunks: int = 0


class DocuMindIndex:
    """Manages the per-project hybrid index (vectors + BM25 + metadata)."""

    def __init__(self, project_root: Path, cfg: Config):
        self.project_root = project_root.resolve()
        self.cfg = cfg
        self.index_dir = index_dir_for(self.project_root, cfg)
        self.meta_path = self.index_dir / "meta.sqlite"
        self.lance_path = self.index_dir / "lance"
        self.bm25_path = self.index_dir / "bm25"
        self.state_path = self.index_dir / "state.json"
        self._conn: sqlite3.Connection | None = None

    # --------------------------------------------------------------- plumbing

    def _ensure_dirs(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._ensure_dirs()
            self._conn = sqlite3.connect(self.meta_path)
            self._conn.executescript(_SCHEMA_SQL)
            self._conn.commit()
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def exists(self) -> bool:
        return self.meta_path.exists() and self.lance_path.exists()

    def _write_state(self) -> None:
        self.state_path.write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "embedding_model": self.cfg.embedding_model,
                    "embedding_dim": self.cfg.embedding_dim,
                    "project_root": self.project_root.as_posix(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def read_state(self) -> dict:
        if not self.state_path.exists():
            return {}
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    # ------------------------------------------------------------ vector DB

    def _open_lance_table(self, create: bool = False):
        import lancedb

        db = lancedb.connect(str(self.lance_path))
        names = _extract_table_names(db)
        if "chunks" in names:
            return db.open_table("chunks")
        if not create:
            return None

        schema = pa.schema([
            pa.field("chunk_id", pa.string()),
            pa.field("file_hash", pa.string()),
            pa.field("rel_path", pa.string()),
            pa.field(
                "vector",
                pa.list_(pa.float32(), list_size=self.cfg.embedding_dim),
            ),
        ])
        return db.create_table("chunks", schema=schema)

    def _lance_delete_by_file_hash(self, file_hashes: Iterable[str]) -> None:
        hashes = [h for h in file_hashes if h]
        if not hashes:
            return
        table = self._open_lance_table(create=False)
        if table is None:
            return
        quoted = ", ".join(f"'{h}'" for h in hashes)
        table.delete(f"file_hash IN ({quoted})")

    def _lance_add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if not chunks:
            return
        table = self._open_lance_table(create=True)
        rows = pa.table({
            "chunk_id": [c.chunk_id for c in chunks],
            "file_hash": [c.file_hash for c in chunks],
            "rel_path": [c.rel_path for c in chunks],
            "vector": pa.array(
                [v.tolist() for v in vectors],
                type=pa.list_(pa.float32(), list_size=self.cfg.embedding_dim),
            ),
        })
        table.add(rows)

    # ---------------------------------------------------------------- SQLite

    def _get_known_files(self) -> dict[str, tuple[str, float]]:
        cur = self.conn.execute("SELECT rel_path, file_hash, mtime FROM files")
        return {row[0]: (row[1], row[2]) for row in cur.fetchall()}

    def _upsert_file_record(self, rec: FileRecord) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO files (rel_path, file_hash, mtime, size, language) "
            "VALUES (?, ?, ?, ?, ?)",
            (rec.rel_path, rec.file_hash, rec.mtime, rec.size, rec.language),
        )

    def _delete_file_rows(self, rel_paths: Iterable[str]) -> list[str]:
        """Delete rows for given rel_paths. Returns affected file_hashes."""
        hashes: list[str] = []
        for rel in rel_paths:
            cur = self.conn.execute("SELECT file_hash FROM files WHERE rel_path = ?", (rel,))
            row = cur.fetchone()
            if row:
                hashes.append(row[0])
            self.conn.execute("DELETE FROM chunks WHERE rel_path = ?", (rel,))
            self.conn.execute("DELETE FROM files  WHERE rel_path = ?", (rel,))
        return hashes

    def _insert_chunks(self, chunks: list[Chunk]) -> None:
        self.conn.executemany(
            "INSERT OR REPLACE INTO chunks "
            "(chunk_id, rel_path, file_hash, language, start_line, end_line, text) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    c.chunk_id, c.rel_path, c.file_hash, c.language,
                    c.start_line, c.end_line, c.text,
                )
                for c in chunks
            ],
        )

    def all_chunks(self) -> list[dict]:
        """Return all chunks as dicts; used by search + BM25 rebuild."""
        cur = self.conn.execute(
            "SELECT chunk_id, rel_path, file_hash, language, start_line, end_line, text "
            "FROM chunks ORDER BY rel_path, start_line"
        )
        cols = ["chunk_id", "rel_path", "file_hash", "language", "start_line", "end_line", "text"]
        return [dict(zip(cols, row, strict=False)) for row in cur.fetchall()]

    def chunks_by_ids(self, chunk_ids: Iterable[str]) -> dict[str, dict]:
        ids = list(chunk_ids)
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        cur = self.conn.execute(
            f"SELECT chunk_id, rel_path, file_hash, language, start_line, end_line, text "
            f"FROM chunks WHERE chunk_id IN ({placeholders})",
            ids,
        )
        cols = ["chunk_id", "rel_path", "file_hash", "language", "start_line", "end_line", "text"]
        return {row[0]: dict(zip(cols, row, strict=False)) for row in cur.fetchall()}

    # -------------------------------------------------------------- BM25

    def rebuild_bm25(self) -> int:
        """Rebuild BM25 from the current chunks table. Returns doc count."""
        import bm25s

        rows = self.conn.execute(
            "SELECT chunk_id, text FROM chunks ORDER BY chunk_id"
        ).fetchall()
        if not rows:
            if self.bm25_path.exists():
                shutil.rmtree(self.bm25_path, ignore_errors=True)
            return 0

        ids = [r[0] for r in rows]
        texts = [r[1] for r in rows]
        tokens = bm25s.tokenize(
            texts, stopwords="en", stemmer=None, show_progress=False
        )

        retriever = bm25s.BM25()
        try:
            retriever.index(tokens, show_progress=False)
        except TypeError:
            retriever.index(tokens)

        if self.bm25_path.exists():
            shutil.rmtree(self.bm25_path, ignore_errors=True)
        self.bm25_path.mkdir(parents=True, exist_ok=True)
        retriever.save(str(self.bm25_path))
        (self.bm25_path / "chunk_ids.json").write_text(json.dumps(ids), encoding="utf-8")
        return len(ids)

    def load_bm25(self):
        """Load saved BM25 retriever + chunk id mapping. Returns (retriever, ids)."""
        import bm25s

        if not (self.bm25_path / "chunk_ids.json").exists():
            return None, []
        ids = json.loads((self.bm25_path / "chunk_ids.json").read_text(encoding="utf-8"))
        retriever = bm25s.BM25.load(str(self.bm25_path))
        return retriever, ids

    # -------------------------------------------------------------- build

    def build_or_update(
        self,
        progress: Callable[[str, int, int], None] | None = None,
    ) -> IndexStats:
        """Full incremental indexing pass.

        Args:
            progress: optional callback(phase, done, total).

        Returns:
            IndexStats summary.
        """
        self._ensure_dirs()
        stats = IndexStats()

        # 1) Scan
        records = scan_files(self.project_root, self.cfg)
        stats.scanned_files = len(records)
        known = self._get_known_files()

        new_or_changed: list[FileRecord] = []
        removed: list[str] = []

        current_paths = {r.rel_path for r in records}
        for rel in known.keys() - current_paths:
            removed.append(rel)

        for rec in records:
            prev = known.get(rec.rel_path)
            if prev is None:
                stats.new_files += 1
                new_or_changed.append(rec)
            elif prev[0] != rec.file_hash:
                stats.changed_files += 1
                new_or_changed.append(rec)
            else:
                stats.unchanged_files += 1

        stats.removed_files = len(removed)

        # 2) Remove stale rows/vectors
        to_purge_paths = removed + [r.rel_path for r in new_or_changed]
        stale_hashes = self._delete_file_rows(to_purge_paths)
        self._lance_delete_by_file_hash(stale_hashes)

        # 3) Chunk + embed new/changed files
        all_new_chunks: list[Chunk] = []
        for idx, rec in enumerate(new_or_changed):
            if progress:
                progress("chunking", idx, len(new_or_changed))
            chunks = chunks_for_file(rec, self.cfg)
            all_new_chunks.extend(chunks)
            self._upsert_file_record(rec)

        stats.embedded_chunks = len(all_new_chunks)

        if all_new_chunks:
            if progress:
                progress("embedding", 0, len(all_new_chunks))
            vectors = embed_texts([c.text for c in all_new_chunks], self.cfg)
            if progress:
                progress("embedding", len(all_new_chunks), len(all_new_chunks))

            self._insert_chunks(all_new_chunks)
            self._lance_add(all_new_chunks, vectors)

        self.conn.commit()

        # 4) Rebuild BM25 from final state
        if progress:
            progress("bm25", 0, 1)
        total = self.rebuild_bm25()
        stats.total_chunks = total
        if progress:
            progress("bm25", 1, 1)

        self._write_state()
        return stats

    # -------------------------------------------------------------- delete

    def destroy(self) -> None:
        """Remove the entire `.documind/` directory for this project."""
        self.close()
        if self.index_dir.exists():
            shutil.rmtree(self.index_dir, ignore_errors=True)
