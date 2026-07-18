// Package store owns the per-project SQLite metadata database. It is the ground
// truth for file records and chunk text/metadata; the vector and keyword indexes
// are derived from it. The schema mirrors meta.sqlite from the Python build
// (files, chunks) with added symbol/symbol_kind/has_vector columns for the Go
// code-navigation features.
package store

import (
	"database/sql"
	"strings"

	"github.com/Henildiyora/DocuMind/internal/chunker"
	_ "modernc.org/sqlite" // pure-Go SQLite driver (database/sql)
)

const schemaSQL = `
CREATE TABLE IF NOT EXISTS files (
    rel_path   TEXT PRIMARY KEY,
    file_hash  TEXT NOT NULL,
    mtime      REAL NOT NULL,
    size       INTEGER NOT NULL,
    language   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT PRIMARY KEY,
    rel_path    TEXT NOT NULL,
    file_hash   TEXT NOT NULL,
    language    TEXT NOT NULL,
    start_line  INTEGER NOT NULL,
    end_line    INTEGER NOT NULL,
    symbol      TEXT NOT NULL DEFAULT '',
    symbol_kind TEXT NOT NULL DEFAULT '',
    text        TEXT NOT NULL,
    has_vector  INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_chunks_rel_path ON chunks(rel_path);
CREATE INDEX IF NOT EXISTS idx_chunks_file_hash ON chunks(file_hash);
CREATE INDEX IF NOT EXISTS idx_chunks_symbol ON chunks(symbol);
`

// DB wraps the SQLite connection for a project.
type DB struct {
	sql *sql.DB
}

// ChunkRow is a chunk as stored, returned by lookups and search hydration.
type ChunkRow struct {
	ChunkID    string
	RelPath    string
	FileHash   string
	Language   string
	StartLine  int
	EndLine    int
	Symbol     string
	SymbolKind string
	Text       string
	HasVector  bool
}

// Open opens (creating if needed) the SQLite database at path and ensures the
// schema exists.
func Open(path string) (*DB, error) {
	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, err
	}
	// Serialize writes; the indexer is single-writer but WAL keeps reads fast.
	if _, err := db.Exec("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;"); err != nil {
		db.Close()
		return nil, err
	}
	if _, err := db.Exec(schemaSQL); err != nil {
		db.Close()
		return nil, err
	}
	return &DB{sql: db}, nil
}

// Close closes the underlying database.
func (d *DB) Close() error {
	if d == nil || d.sql == nil {
		return nil
	}
	return d.sql.Close()
}

// KnownFiles returns rel_path -> FileRecord for every indexed file, used for the
// incremental mtime/size fast path.
func (d *DB) KnownFiles() (map[string]chunker.FileRecord, error) {
	rows, err := d.sql.Query("SELECT rel_path, file_hash, mtime, size, language FROM files")
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := map[string]chunker.FileRecord{}
	for rows.Next() {
		var r chunker.FileRecord
		if err := rows.Scan(&r.RelPath, &r.FileHash, &r.Mtime, &r.Size, &r.Language); err != nil {
			return nil, err
		}
		out[r.RelPath] = r
	}
	return out, rows.Err()
}

// UpsertFile inserts or replaces a file record.
func (d *DB) UpsertFile(rec chunker.FileRecord) error {
	_, err := d.sql.Exec(
		"INSERT OR REPLACE INTO files (rel_path, file_hash, mtime, size, language) VALUES (?, ?, ?, ?, ?)",
		rec.RelPath, rec.FileHash, rec.Mtime, rec.Size, rec.Language,
	)
	return err
}

// DeleteFiles removes file + chunk rows for the given rel_paths and returns the
// affected file hashes (so callers can purge derived indexes by hash).
func (d *DB) DeleteFiles(relPaths []string) ([]string, error) {
	var hashes []string
	for _, rel := range relPaths {
		var hash string
		err := d.sql.QueryRow("SELECT file_hash FROM files WHERE rel_path = ?", rel).Scan(&hash)
		if err == nil && hash != "" {
			hashes = append(hashes, hash)
		} else if err != nil && err != sql.ErrNoRows {
			return hashes, err
		}
		if _, err := d.sql.Exec("DELETE FROM chunks WHERE rel_path = ?", rel); err != nil {
			return hashes, err
		}
		if _, err := d.sql.Exec("DELETE FROM files WHERE rel_path = ?", rel); err != nil {
			return hashes, err
		}
	}
	return hashes, nil
}

// InsertChunks inserts or replaces chunk rows in a single transaction. New
// chunks start with has_vector = 0.
func (d *DB) InsertChunks(chunks []chunker.Chunk) error {
	if len(chunks) == 0 {
		return nil
	}
	tx, err := d.sql.Begin()
	if err != nil {
		return err
	}
	stmt, err := tx.Prepare(
		"INSERT OR REPLACE INTO chunks " +
			"(chunk_id, rel_path, file_hash, language, start_line, end_line, symbol, symbol_kind, text, has_vector) " +
			"VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)",
	)
	if err != nil {
		tx.Rollback()
		return err
	}
	defer stmt.Close()
	for _, c := range chunks {
		if _, err := stmt.Exec(
			c.ID, c.RelPath, c.FileHash, c.Language,
			c.StartLine, c.EndLine, c.Symbol, c.SymbolKind, c.Text,
		); err != nil {
			tx.Rollback()
			return err
		}
	}
	return tx.Commit()
}

// MarkVectored flags the given chunk ids as having a dense vector.
func (d *DB) MarkVectored(chunkIDs []string) error {
	if len(chunkIDs) == 0 {
		return nil
	}
	tx, err := d.sql.Begin()
	if err != nil {
		return err
	}
	stmt, err := tx.Prepare("UPDATE chunks SET has_vector = 1 WHERE chunk_id = ?")
	if err != nil {
		tx.Rollback()
		return err
	}
	defer stmt.Close()
	for _, id := range chunkIDs {
		if _, err := stmt.Exec(id); err != nil {
			tx.Rollback()
			return err
		}
	}
	return tx.Commit()
}

// ChunksWithoutVectors returns chunks that still need embedding (has_vector = 0).
// Used to backfill vectors once Ollama becomes available.
func (d *DB) ChunksWithoutVectors() ([]ChunkRow, error) {
	return d.queryChunks("WHERE has_vector = 0")
}

// AllChunks returns every chunk ordered by path then start line.
func (d *DB) AllChunks() ([]ChunkRow, error) {
	return d.queryChunks("ORDER BY rel_path, start_line")
}

// ChunksByIDs hydrates chunk rows for the given ids.
func (d *DB) ChunksByIDs(ids []string) (map[string]ChunkRow, error) {
	out := map[string]ChunkRow{}
	if len(ids) == 0 {
		return out, nil
	}
	placeholders := strings.TrimSuffix(strings.Repeat("?,", len(ids)), ",")
	args := make([]any, len(ids))
	for i, id := range ids {
		args[i] = id
	}
	rows, err := d.sql.Query(
		"SELECT chunk_id, rel_path, file_hash, language, start_line, end_line, symbol, symbol_kind, text, has_vector "+
			"FROM chunks WHERE chunk_id IN ("+placeholders+")",
		args...,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	for rows.Next() {
		r, err := scanChunk(rows)
		if err != nil {
			return nil, err
		}
		out[r.ChunkID] = r
	}
	return out, rows.Err()
}

// CountChunks returns the total number of chunks.
func (d *DB) CountChunks() (int, error) {
	var n int
	err := d.sql.QueryRow("SELECT COUNT(*) FROM chunks").Scan(&n)
	return n, err
}

// CountVectored returns how many chunks have a dense vector.
func (d *DB) CountVectored() (int, error) {
	var n int
	err := d.sql.QueryRow("SELECT COUNT(*) FROM chunks WHERE has_vector = 1").Scan(&n)
	return n, err
}

func (d *DB) queryChunks(clause string) ([]ChunkRow, error) {
	q := "SELECT chunk_id, rel_path, file_hash, language, start_line, end_line, symbol, symbol_kind, text, has_vector FROM chunks " + clause
	rows, err := d.sql.Query(q)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []ChunkRow
	for rows.Next() {
		r, err := scanChunk(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, r)
	}
	return out, rows.Err()
}

func scanChunk(rows *sql.Rows) (ChunkRow, error) {
	var r ChunkRow
	var hasVec int
	err := rows.Scan(
		&r.ChunkID, &r.RelPath, &r.FileHash, &r.Language,
		&r.StartLine, &r.EndLine, &r.Symbol, &r.SymbolKind, &r.Text, &hasVec,
	)
	r.HasVector = hasVec != 0
	return r, err
}

// HasSymbolColumn reports whether the chunks table has the Go-only symbol
// column. Used by doctor to detect a Python-era database. Always true for
// databases created by this build; the check exists for migration messaging.
func (d *DB) HasSymbolColumn() bool {
	rows, err := d.sql.Query("PRAGMA table_info(chunks)")
	if err != nil {
		return false
	}
	defer rows.Close()
	for rows.Next() {
		var cid int
		var name, ctype string
		var notnull, pk int
		var dflt sql.NullString
		if err := rows.Scan(&cid, &name, &ctype, &notnull, &dflt, &pk); err != nil {
			return false
		}
		if name == "symbol" {
			return true
		}
	}
	return false
}
