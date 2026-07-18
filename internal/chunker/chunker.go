// Package chunker walks a project, filters by extension/ignore rules, and
// splits files into overlapping, metadata-rich chunks. Source files in a
// supported language are chunked at AST boundaries (functions, classes, etc.)
// via tree-sitter; everything else falls back to a line-window splitter ported
// from the Python implementation.
package chunker

import (
	"crypto/sha1"
	"encoding/hex"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	"github.com/Henildiyora/DocuMind/internal/config"
	"github.com/Henildiyora/DocuMind/internal/ignorerules"
)

// langByExt maps file extensions to a language label. Ported from LANG_BY_EXT
// in chunker.py. The label is stored on every chunk and drives syntax
// highlighting and tree-sitter grammar selection.
var langByExt = map[string]string{
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
	".md":  "markdown", ".mdx": "markdown", ".rst": "rst", ".txt": "text",
	".sh": "shell", ".bash": "shell", ".zsh": "shell", ".fish": "shell",
}

// FileRecord holds metadata for a source file discovered by the walker. No file
// content is loaded at scan time.
type FileRecord struct {
	Path     string  // absolute path
	RelPath  string  // path relative to project root (slash-separated)
	Size     int64   // bytes
	Mtime    float64 // unix mtime seconds (float to match Python's stat mtime)
	FileHash string  // sha1 of content (hex)
	Language string
}

// Chunk is a slice of a file ready for embedding + indexing. Symbol and
// SymbolKind are populated for AST-derived chunks (empty for fallback chunks).
type Chunk struct {
	ID         string // stable id: "<file_hash>:<index>"
	RelPath    string
	Language   string
	StartLine  int // 1-based, inclusive
	EndLine    int // 1-based, inclusive
	Symbol     string
	SymbolKind string
	Text       string
	FileHash   string
}

// DetectLanguage returns the language label for a path, with the same
// Dockerfile/Makefile special-casing as chunker.py.
func DetectLanguage(path string) string {
	name := strings.ToLower(filepath.Base(path))
	switch name {
	case "dockerfile":
		return "dockerfile"
	case "makefile", "gnumakefile":
		return "makefile"
	}
	if lang, ok := langByExt[strings.ToLower(filepath.Ext(path))]; ok {
		return lang
	}
	return "text"
}

func sha1Hex(data []byte) string {
	sum := sha1.Sum(data)
	return hex.EncodeToString(sum[:])
}

// ScanFiles walks the project and returns FileRecord metadata. When known maps
// rel_path -> prior record and forceRehash is false, files whose mtime and size
// are unchanged reuse the stored hash without re-reading bytes (the SHA1 fast
// path from scan_files in index.py).
func ScanFiles(root string, cfg config.Config, rules *ignorerules.Rules, known map[string]FileRecord, forceRehash bool) ([]FileRecord, error) {
	absRoot, err := filepath.Abs(root)
	if err != nil {
		return nil, err
	}

	var records []FileRecord
	walkErr := filepath.WalkDir(absRoot, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			// Skip unreadable entries rather than aborting the whole walk.
			if d != nil && d.IsDir() {
				return fs.SkipDir
			}
			return nil
		}
		if d.IsDir() {
			// Never descend into ignored directories. The root itself is kept
			// even if its basename would otherwise be ignored.
			if path != absRoot && rules.IsIgnoredDir(d.Name()) {
				return fs.SkipDir
			}
			return nil
		}

		name := d.Name()
		if rules.IsIgnoredFile(name) || !ignorerules.ShouldInclude(name) {
			return nil
		}

		info, statErr := d.Info()
		if statErr != nil {
			return nil
		}
		if info.Size() > int64(cfg.MaxFileBytes) {
			return nil
		}

		rel, relErr := filepath.Rel(absRoot, path)
		if relErr != nil {
			return nil
		}
		rel = filepath.ToSlash(rel)
		mtime := float64(info.ModTime().UnixNano()) / 1e9

		var hash string
		if prev, ok := known[rel]; !forceRehash && ok && prev.Mtime == mtime && prev.Size == info.Size() {
			hash = prev.FileHash
		} else {
			data, readErr := os.ReadFile(path)
			if readErr != nil {
				return nil
			}
			hash = sha1Hex(data)
		}

		records = append(records, FileRecord{
			Path:     path,
			RelPath:  rel,
			Size:     info.Size(),
			Mtime:    mtime,
			FileHash: hash,
			Language: DetectLanguage(path),
		})
		return nil
	})
	if walkErr != nil {
		return records, walkErr
	}
	return records, nil
}

// readText reads a file as UTF-8 text, skipping binary content (a NUL byte in
// the first 4KB). Ported from _read_text in chunker.py (PDF support dropped).
func readText(path string) (string, bool) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", false
	}
	limit := 4096
	if len(data) < limit {
		limit = len(data)
	}
	for _, b := range data[:limit] {
		if b == 0 {
			return "", false
		}
	}
	return string(data), true
}

// ChunksForFile reads a file and splits it into chunks. AST-aware chunking is
// used for supported languages; other files use the line-window fallback.
func ChunksForFile(rec FileRecord, cfg config.Config) []Chunk {
	text, ok := readText(rec.Path)
	if !ok || strings.TrimSpace(text) == "" {
		return nil
	}

	var pieces []chunkPiece
	if lang, ok := treeSitterLang(rec.Language); ok {
		pieces = astChunks([]byte(text), lang, rec.Language, cfg)
	}
	if len(pieces) == 0 {
		// Fallback for unsupported languages, parse failures, or empty ASTs.
		pieces = fallbackChunks(text, cfg.ChunkSize, cfg.ChunkOverlap)
	}

	out := make([]Chunk, 0, len(pieces))
	for i, p := range pieces {
		body := strings.Trim(p.text, "\n")
		if strings.TrimSpace(body) == "" {
			continue
		}
		out = append(out, Chunk{
			ID:         fmt.Sprintf("%s:%d", rec.FileHash, i),
			RelPath:    rec.RelPath,
			Language:   rec.Language,
			StartLine:  p.startLine,
			EndLine:    p.endLine,
			Symbol:     p.symbol,
			SymbolKind: p.symbolKind,
			Text:       body,
			FileHash:   rec.FileHash,
		})
	}
	return out
}

// chunkPiece is an intermediate chunk before it gets a stable id.
type chunkPiece struct {
	startLine  int
	endLine    int
	symbol     string
	symbolKind string
	text       string
}

// fallbackChunks is a line-aware splitter targeting chunkSize characters per
// chunk with chunkOverlap character overlap. Ported from
// _split_lines_into_chunks in chunker.py, preserving its stepping behavior.
func fallbackChunks(text string, chunkSize, chunkOverlap int) []chunkPiece {
	if text == "" {
		return nil
	}
	lines := splitKeepEnds(text)
	var chunks []chunkPiece

	i := 0
	n := len(lines)
	for i < n {
		curLen := 0
		j := i
		for j < n && curLen+len(lines[j]) <= chunkSize {
			curLen += len(lines[j])
			j++
		}
		if j == i {
			j = i + 1
		}
		startLine := i + 1
		endLine := j
		body := strings.Join(lines[i:j], "")
		if strings.TrimSpace(body) != "" {
			chunks = append(chunks, chunkPiece{startLine: startLine, endLine: endLine, text: body})
		}

		if j >= n {
			break
		}
		// Step back by chunkOverlap worth of characters (line-aligned).
		back := 0
		k := j
		for k > i && back < chunkOverlap {
			k--
			back += len(lines[k])
		}
		if k > i {
			i = k
		} else {
			i = i + 1
		}
	}
	return chunks
}

// splitKeepEnds splits text into lines while keeping the trailing newline on
// each line, matching Python's str.splitlines(keepends=True) closely enough for
// character-budget accounting.
func splitKeepEnds(text string) []string {
	var lines []string
	start := 0
	for i := 0; i < len(text); i++ {
		if text[i] == '\n' {
			lines = append(lines, text[start:i+1])
			start = i + 1
		}
	}
	if start < len(text) {
		lines = append(lines, text[start:])
	}
	return lines
}

// ChunksForFiles yields chunks for a sequence of file records.
func ChunksForFiles(records []FileRecord, cfg config.Config) []Chunk {
	var all []Chunk
	for _, rec := range records {
		all = append(all, ChunksForFile(rec, cfg)...)
	}
	return all
}
