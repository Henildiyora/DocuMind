package cli

import (
	"github.com/Henildiyora/DocuMind/internal/config"
	"github.com/Henildiyora/DocuMind/internal/indexer"
	"github.com/Henildiyora/DocuMind/internal/kwindex"
	"github.com/Henildiyora/DocuMind/internal/ollama"
	"github.com/Henildiyora/DocuMind/internal/search"
	"github.com/Henildiyora/DocuMind/internal/store"
	"github.com/Henildiyora/DocuMind/internal/vectorstore"
)

// retrieverBundle owns the open handles behind a Retriever so callers can close
// them together.
type retrieverBundle struct {
	Retriever *search.Retriever
	db        *store.DB
	kw        *kwindex.Index
}

func (b *retrieverBundle) Close() {
	if b.kw != nil {
		b.kw.Close()
	}
	if b.db != nil {
		b.db.Close()
	}
}

// openRetriever opens the metadata, keyword, and vector stores for a project and
// wires a Retriever. The Ollama client is attached only when the daemon is
// reachable, so vector retrieval is best-effort and keyword search always works.
func openRetriever(root string, cfg config.Config) (*retrieverBundle, error) {
	paths := indexer.PathsFor(root, cfg)

	db, err := store.Open(paths.Meta)
	if err != nil {
		return nil, err
	}
	kw, err := kwindex.Open(paths.Bleve)
	if err != nil {
		db.Close()
		return nil, err
	}
	vec, err := vectorstore.Open(paths.Chroma)
	if err != nil {
		kw.Close()
		db.Close()
		return nil, err
	}

	var client *ollama.Client
	if ollama.Installed() {
		c := ollama.New(cfg)
		if c.Ping() {
			client = c
		}
	}

	return &retrieverBundle{
		Retriever: &search.Retriever{DB: db, KW: kw, Vec: vec, Client: client, Cfg: cfg},
		db:        db,
		kw:        kw,
	}, nil
}

// llmReady reports whether the daemon is reachable and the chat model is pulled,
// so answer synthesis can proceed. Returns the client when ready.
func llmReady(cfg config.Config) (*ollama.Client, bool) {
	c, ready, _ := llmStatus(cfg)
	return c, ready
}

// llmStatus reports LLM readiness along with a short, actionable hint typed by
// the failure mode (not installed / not running / model not pulled). The hint is
// empty when ready. Kept to 1-2 plain lines so `ask`/`chat` can degrade cleanly.
func llmStatus(cfg config.Config) (*ollama.Client, bool, string) {
	if !ollama.Installed() {
		return nil, false, "Ollama isn't installed. Install it from https://ollama.com/download, then run `documind setup`."
	}
	c := ollama.New(cfg)
	if !c.Ping() {
		return nil, false, "Ollama isn't running. Start it with `ollama serve`, then retry."
	}
	if !c.ModelAvailable(cfg.Model) {
		return c, false, "Model \"" + cfg.Model + "\" isn't pulled. Run `documind setup` or `ollama pull " + cfg.Model + "`."
	}
	return c, true, ""
}

// llmHint returns just the actionable hint (empty when ready).
func llmHint(cfg config.Config) string {
	_, _, hint := llmStatus(cfg)
	return hint
}

// indexReady reports whether a usable index exists for the project.
func indexReady(root string, cfg config.Config) bool {
	return indexer.Exists(root, cfg)
}
