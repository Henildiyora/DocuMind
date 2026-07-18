// Package indexer orchestrates the indexing pipeline: walk -> chunk -> (embed)
// -> store. It writes three derived stores under <project>/.documind/: the
// SQLite metadata db (ground truth), the chromem-go vector store (dense), and
// the bleve keyword index (sparse). Dense vectors are optional: if Ollama is not
// reachable, indexing completes keyword-only and backfills vectors on a later
// run once Ollama is available.
package indexer

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/Henildiyora/DocuMind/internal/chunker"
	"github.com/Henildiyora/DocuMind/internal/config"
	"github.com/Henildiyora/DocuMind/internal/ignorerules"
	"github.com/Henildiyora/DocuMind/internal/kwindex"
	"github.com/Henildiyora/DocuMind/internal/ollama"
	"github.com/Henildiyora/DocuMind/internal/store"
	"github.com/Henildiyora/DocuMind/internal/vectorstore"
	"golang.org/x/sync/errgroup"
)

// ErrLocked is returned when another indexer holds the project lock.
var ErrLocked = errors.New("another documind index is running (project locked)")

const (
	schemaVersion   = 1
	lockTimeout     = 10 * time.Minute
	embedBatchSize  = 32 // texts per /api/embed request
	embedConcurrent = 4  // parallel embed requests
)

// Stats summarizes an indexing pass.
type Stats struct {
	ScannedFiles   int
	NewFiles       int
	ChangedFiles   int
	UnchangedFiles int
	RemovedFiles   int
	TotalChunks    int
	EmbeddedChunks int    // chunks that received a vector this run
	VectorsSkipped bool   // true when dense embedding was not performed
	SkipReason     string // human-readable reason vectors were skipped
}

// Options tunes a build.
type Options struct {
	ForceRehash bool                                // re-hash every file, ignoring mtime/size
	Rebuild     bool                                // wipe .documind/ before building
	Progress    func(phase string, done, total int) // optional progress callback
}

type state struct {
	SchemaVersion  int     `json:"schema_version"`
	EmbeddingModel string  `json:"embedding_model"`
	EmbeddingDim   int     `json:"embedding_dim"`
	ProjectRoot    string  `json:"project_root"`
	IndexedAt      string  `json:"indexed_at"`
	MaxMtime       float64 `json:"max_mtime_at_index"`
	ChunkCount     int     `json:"chunk_count"`
}

func (o Options) progress(phase string, done, total int) {
	if o.Progress != nil {
		o.Progress(phase, done, total)
	}
}

// Paths bundles the on-disk locations for a project's index.
type Paths struct {
	IndexDir string
	Meta     string
	Chroma   string
	Bleve    string
	State    string
}

// PathsFor computes the index paths for a project.
func PathsFor(root string, cfg config.Config) Paths {
	dir := cfg.IndexDirFor(root)
	return Paths{
		IndexDir: dir,
		Meta:     filepath.Join(dir, "meta.sqlite"),
		Chroma:   filepath.Join(dir, "chroma"),
		Bleve:    filepath.Join(dir, "bleve"),
		State:    filepath.Join(dir, "state.json"),
	}
}

// Exists reports whether a usable index is present (metadata db + keyword index).
func Exists(root string, cfg config.Config) bool {
	p := PathsFor(root, cfg)
	if _, err := os.Stat(p.Meta); err != nil {
		return false
	}
	if _, err := os.Stat(p.Bleve); err != nil {
		return false
	}
	return true
}

// Reset deletes the entire .documind/ directory for a project.
func Reset(root string, cfg config.Config) error {
	return os.RemoveAll(cfg.IndexDirFor(root))
}

// BuildOrUpdate runs a full incremental indexing pass under the project lock.
func BuildOrUpdate(ctx context.Context, root string, cfg config.Config, opts Options) (Stats, error) {
	absRoot, err := filepath.Abs(root)
	if err != nil {
		return Stats{}, err
	}
	var stats Stats
	runErr := withIndexLock(absRoot, lockTimeout, func() error {
		var innerErr error
		stats, innerErr = buildLocked(ctx, absRoot, cfg, opts)
		return innerErr
	})
	return stats, runErr
}

func buildLocked(ctx context.Context, root string, cfg config.Config, opts Options) (Stats, error) {
	paths := PathsFor(root, cfg)
	var stats Stats

	if opts.Rebuild {
		if err := os.RemoveAll(paths.IndexDir); err != nil {
			return stats, err
		}
	}
	if err := os.MkdirAll(paths.IndexDir, 0o755); err != nil {
		return stats, err
	}

	// A change in embedding model/dim invalidates existing vectors: wipe and
	// rebuild so dimensions stay consistent.
	if prev, ok := readState(paths.State); ok {
		if prev.EmbeddingModel != cfg.EmbeddingModel || prev.EmbeddingDim != cfg.EmbeddingDim {
			if err := os.RemoveAll(paths.IndexDir); err != nil {
				return stats, err
			}
			if err := os.MkdirAll(paths.IndexDir, 0o755); err != nil {
				return stats, err
			}
		}
	}

	db, err := store.Open(paths.Meta)
	if err != nil {
		return stats, err
	}
	defer db.Close()

	known, err := db.KnownFiles()
	if err != nil {
		return stats, err
	}

	rules := ignorerules.New(cfg.ExtraIgnoreDirs, cfg.ExtraIgnoreFiles, cfg.ExtraIgnoreGlobs)
	records, err := chunker.ScanFiles(root, cfg, rules, known, opts.ForceRehash)
	if err != nil {
		return stats, err
	}
	stats.ScannedFiles = len(records)

	// Diff against the previous index.
	current := make(map[string]struct{}, len(records))
	var newOrChanged []chunker.FileRecord
	for _, rec := range records {
		current[rec.RelPath] = struct{}{}
		prev, ok := known[rec.RelPath]
		switch {
		case !ok:
			stats.NewFiles++
			newOrChanged = append(newOrChanged, rec)
		case prev.FileHash != rec.FileHash:
			stats.ChangedFiles++
			newOrChanged = append(newOrChanged, rec)
		default:
			stats.UnchangedFiles++
		}
	}
	var removed []string
	for rel := range known {
		if _, ok := current[rel]; !ok {
			removed = append(removed, rel)
		}
	}
	stats.RemovedFiles = len(removed)

	// Purge chunks for removed + changed files from metadata and the vector
	// store (keyed by the stale file hashes).
	toPurge := append(append([]string{}, removed...), relPaths(newOrChanged)...)
	staleHashes, err := db.DeleteFiles(toPurge)
	if err != nil {
		return stats, err
	}

	vec, err := vectorstore.Open(paths.Chroma)
	if err != nil {
		return stats, err
	}
	if err := vec.DeleteByFileHash(ctx, staleHashes); err != nil {
		return stats, err
	}

	// Chunk changed files and persist to metadata.
	var newChunks []chunker.Chunk
	for i, rec := range newOrChanged {
		opts.progress("chunking", i, max(len(newOrChanged), 1))
		chunks := chunker.ChunksForFile(rec, cfg)
		newChunks = append(newChunks, chunks...)
		if err := db.UpsertFile(rec); err != nil {
			return stats, err
		}
	}
	if err := db.InsertChunks(newChunks); err != nil {
		return stats, err
	}

	// Embed: only if Ollama is reachable. Otherwise complete keyword-only and
	// leave has_vector=0 so a later run backfills.
	pending, err := db.ChunksWithoutVectors()
	if err != nil {
		return stats, err
	}
	client := ollama.New(cfg)
	if len(pending) > 0 {
		switch {
		case !ollama.Installed() || !client.Ping():
			stats.VectorsSkipped = true
			stats.SkipReason = "Ollama not available"
		case !client.ModelAvailable(cfg.EmbeddingModel):
			stats.VectorsSkipped = true
			stats.SkipReason = "embedding model '" + cfg.EmbeddingModel + "' not pulled (run: ollama pull " + cfg.EmbeddingModel + ")"
		default:
			embedded, err := embedChunks(ctx, client, vec, db, pending, opts)
			if err != nil {
				return stats, err
			}
			stats.EmbeddedChunks = embedded
		}
	}

	// Rebuild the keyword index from the full chunk set (cheap, keeps IDF
	// correct and avoids stale-id bookkeeping).
	opts.progress("keyword", 0, 1)
	allChunks, err := db.AllChunks()
	if err != nil {
		return stats, err
	}
	kw, err := kwindex.Open(paths.Bleve)
	if err != nil {
		return stats, err
	}
	if err := kw.Rebuild(toKwDocs(allChunks)); err != nil {
		kw.Close()
		return stats, err
	}
	if err := kw.Close(); err != nil {
		return stats, err
	}
	opts.progress("keyword", 1, 1)

	stats.TotalChunks = len(allChunks)

	// Persist state for freshness + migration checks.
	writeState(paths.State, state{
		SchemaVersion:  schemaVersion,
		EmbeddingModel: cfg.EmbeddingModel,
		EmbeddingDim:   cfg.EmbeddingDim,
		ProjectRoot:    root,
		IndexedAt:      time.Now().UTC().Format(time.RFC3339),
		MaxMtime:       maxMtime(records),
		ChunkCount:     stats.TotalChunks,
	})

	return stats, nil
}

// embedChunks embeds pending chunks in parallel batches through Ollama, adds the
// vectors to the store, and marks them vectored. Returns the count embedded.
func embedChunks(ctx context.Context, client *ollama.Client, vec *vectorstore.Store, db *store.DB, pending []store.ChunkRow, opts Options) (int, error) {
	total := len(pending)
	opts.progress("embedding", 0, total)

	var (
		mu       sync.Mutex
		done     int
		vectored []string
	)

	g, gctx := errgroup.WithContext(ctx)
	g.SetLimit(embedConcurrent)

	for start := 0; start < total; start += embedBatchSize {
		end := min(start+embedBatchSize, total)
		batch := pending[start:end]
		g.Go(func() error {
			texts := make([]string, len(batch))
			for i, c := range batch {
				texts[i] = c.Text
			}
			vectors, err := client.EmbedBatch(gctx, texts)
			if err != nil {
				return err
			}
			if len(vectors) != len(batch) {
				return errors.New("indexer: embedding count mismatch")
			}
			items := make([]vectorstore.Item, len(batch))
			ids := make([]string, len(batch))
			for i, c := range batch {
				items[i] = vectorstore.Item{
					ChunkID:   c.ChunkID,
					FileHash:  c.FileHash,
					RelPath:   c.RelPath,
					Embedding: vectors[i],
					Content:   c.Text,
				}
				ids[i] = c.ChunkID
			}
			mu.Lock()
			defer mu.Unlock()
			if err := vec.Add(gctx, items, 1); err != nil {
				return err
			}
			vectored = append(vectored, ids...)
			done += len(batch)
			opts.progress("embedding", done, total)
			return nil
		})
	}
	if err := g.Wait(); err != nil {
		// Persist whatever succeeded so a re-run backfills only the rest.
		_ = db.MarkVectored(vectored)
		return len(vectored), err
	}
	if err := db.MarkVectored(vectored); err != nil {
		return len(vectored), err
	}
	return len(vectored), nil
}

func toKwDocs(rows []store.ChunkRow) []kwindex.Doc {
	docs := make([]kwindex.Doc, len(rows))
	for i, r := range rows {
		docs[i] = kwindex.Doc{
			ID:       r.ChunkID,
			Text:     r.Text,
			RelPath:  r.RelPath,
			Symbol:   r.Symbol,
			Language: r.Language,
		}
	}
	return docs
}

func relPaths(records []chunker.FileRecord) []string {
	out := make([]string, len(records))
	for i, r := range records {
		out[i] = r.RelPath
	}
	return out
}

func maxMtime(records []chunker.FileRecord) float64 {
	var m float64
	for _, r := range records {
		if r.Mtime > m {
			m = r.Mtime
		}
	}
	return m
}

func readState(path string) (state, bool) {
	data, err := os.ReadFile(path)
	if err != nil {
		return state{}, false
	}
	var s state
	if err := json.Unmarshal(data, &s); err != nil {
		return state{}, false
	}
	return s, true
}

func writeState(path string, s state) {
	data, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return
	}
	_ = os.WriteFile(path, data, 0o644)
}
