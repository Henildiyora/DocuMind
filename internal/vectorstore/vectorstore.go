// Package vectorstore wraps chromem-go, a pure-Go embedded vector database, for
// DocuMind's dense-retrieval half. Vectors are computed externally (via Ollama)
// and stored precomputed, so chromem never needs its own embedding backend.
// Data persists under <project>/.documind/chroma/.
package vectorstore

import (
	"context"
	"errors"

	chromem "github.com/philippgille/chromem-go"
)

const collectionName = "chunks"

// Store is a persistent chromem-go collection of chunk vectors.
type Store struct {
	db         *chromem.DB
	collection *chromem.Collection
}

// stubEmbed is installed as the collection's embedding function. We always pass
// precomputed embeddings, so this must never actually be called; if it is, that
// is a programming error we surface loudly.
func stubEmbed(_ context.Context, _ string) ([]float32, error) {
	return nil, errors.New("vectorstore: embedding function should not be called (precomputed vectors expected)")
}

// Open opens (creating if needed) the persistent vector store at path.
func Open(path string) (*Store, error) {
	db, err := chromem.NewPersistentDB(path, false)
	if err != nil {
		return nil, err
	}
	coll, err := db.GetOrCreateCollection(collectionName, nil, stubEmbed)
	if err != nil {
		return nil, err
	}
	return &Store{db: db, collection: coll}, nil
}

// Item is a chunk vector to be stored. Metadata carries rel_path and file_hash
// so we can filter/delete by file.
type Item struct {
	ChunkID   string
	FileHash  string
	RelPath   string
	Embedding []float32
	Content   string
}

// Add inserts or updates the given vectors. Concurrency controls parallel writes
// inside chromem.
func (s *Store) Add(ctx context.Context, items []Item, concurrency int) error {
	if len(items) == 0 {
		return nil
	}
	docs := make([]chromem.Document, 0, len(items))
	for _, it := range items {
		docs = append(docs, chromem.Document{
			ID:        it.ChunkID,
			Metadata:  map[string]string{"file_hash": it.FileHash, "rel_path": it.RelPath},
			Embedding: it.Embedding,
			Content:   it.Content,
		})
	}
	if concurrency < 1 {
		concurrency = 1
	}
	return s.collection.AddDocuments(ctx, docs, concurrency)
}

// DeleteByFileHash removes all vectors whose file_hash metadata matches any of
// the given hashes.
func (s *Store) DeleteByFileHash(ctx context.Context, hashes []string) error {
	for _, h := range hashes {
		if h == "" {
			continue
		}
		if err := s.collection.Delete(ctx, map[string]string{"file_hash": h}, nil); err != nil {
			return err
		}
	}
	return nil
}

// Count returns the number of stored vectors.
func (s *Store) Count() int {
	if s.collection == nil {
		return 0
	}
	return s.collection.Count()
}

// Query returns up to nResults chunk ids ranked by similarity to the given query
// embedding, most similar first. An empty collection returns no results (never
// an error) so BM25-only retrieval can proceed.
func (s *Store) Query(ctx context.Context, embedding []float32, nResults int) ([]string, error) {
	count := s.collection.Count()
	if count == 0 || len(embedding) == 0 {
		return nil, nil
	}
	if nResults > count {
		nResults = count
	}
	results, err := s.collection.QueryEmbedding(ctx, embedding, nResults, nil, nil)
	if err != nil {
		return nil, err
	}
	ids := make([]string, 0, len(results))
	for _, r := range results {
		ids = append(ids, r.ID)
	}
	return ids, nil
}
