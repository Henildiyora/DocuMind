// Package kwindex wraps bleve for DocuMind's keyword (BM25-style) retrieval
// half. This index has zero external dependencies and is always built during
// `documind index`, so search works even when Ollama is unavailable. Data
// persists under <project>/.documind/bleve/.
package kwindex

import (
	"os"

	"github.com/blevesearch/bleve/v2"
	"github.com/blevesearch/bleve/v2/mapping"
)

// Doc is a chunk to be keyword-indexed. The ID is the chunk id; the other fields
// feed the analyzer so filename and symbol tokens are searchable alongside body
// text.
type Doc struct {
	ID       string
	Text     string
	RelPath  string
	Symbol   string
	Language string
}

// Index wraps a bleve index at a fixed path.
type Index struct {
	path  string
	bleve bleve.Index
}

func buildMapping() mapping.IndexMapping {
	m := bleve.NewIndexMapping()
	return m
}

// Open opens an existing bleve index, or creates an empty one if none exists.
func Open(path string) (*Index, error) {
	idx, err := bleve.Open(path)
	if err == bleve.ErrorIndexPathDoesNotExist {
		idx, err = bleve.New(path, buildMapping())
	}
	if err != nil {
		return nil, err
	}
	return &Index{path: path, bleve: idx}, nil
}

// Close closes the underlying index.
func (i *Index) Close() error {
	if i == nil || i.bleve == nil {
		return nil
	}
	return i.bleve.Close()
}

// Rebuild recreates the index from scratch with the given documents. Bleve
// document ids are chunk ids that embed the file hash, so a full rebuild is the
// simplest way to keep the keyword index consistent with the metadata store
// after incremental file changes (mirrors the Python BM25 rebuild).
func (i *Index) Rebuild(docs []Doc) error {
	if err := i.bleve.Close(); err != nil {
		return err
	}
	if err := os.RemoveAll(i.path); err != nil {
		return err
	}
	idx, err := bleve.New(i.path, buildMapping())
	if err != nil {
		return err
	}
	i.bleve = idx

	batch := idx.NewBatch()
	const batchSize = 1000
	n := 0
	for _, d := range docs {
		if err := batch.Index(d.ID, map[string]any{
			"text":     d.Text,
			"rel_path": d.RelPath,
			"symbol":   d.Symbol,
			"language": d.Language,
		}); err != nil {
			return err
		}
		n++
		if n%batchSize == 0 {
			if err := idx.Batch(batch); err != nil {
				return err
			}
			batch = idx.NewBatch()
		}
	}
	if batch.Size() > 0 {
		if err := idx.Batch(batch); err != nil {
			return err
		}
	}
	return nil
}

// Count returns the number of indexed documents.
func (i *Index) Count() (uint64, error) {
	return i.bleve.DocCount()
}

// Search returns up to limit chunk ids ranked by keyword relevance. An empty
// query or empty index returns no results (never an error).
func (i *Index) Search(query string, limit int) ([]string, error) {
	if query == "" || limit <= 0 {
		return nil, nil
	}
	q := bleve.NewMatchQuery(query)
	req := bleve.NewSearchRequestOptions(q, limit, 0, false)
	req.Fields = nil
	res, err := i.bleve.Search(req)
	if err != nil {
		return nil, err
	}
	ids := make([]string, 0, len(res.Hits))
	for _, hit := range res.Hits {
		ids = append(ids, hit.ID)
	}
	return ids, nil
}
