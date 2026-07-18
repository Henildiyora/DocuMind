package cli

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/Henildiyora/DocuMind/internal/cliui"
	"github.com/Henildiyora/DocuMind/internal/config"
	"github.com/Henildiyora/DocuMind/internal/indexer"
	"github.com/Henildiyora/DocuMind/internal/ollama"
	"github.com/Henildiyora/DocuMind/internal/store"
	"github.com/spf13/cobra"
)

func newDoctorCmd() *cobra.Command {
	var path string
	cmd := &cobra.Command{
		Use:   "doctor",
		Short: "Check environment, index health, and Python-era migration state.",
		RunE: func(cmd *cobra.Command, args []string) error {
			root, err := resolveRoot(path)
			if err != nil {
				return err
			}
			cfg := config.Load()

			cliui.Info("%s", cliui.Bold("DocuMind doctor"))
			checkEnvironment(cfg)
			checkConfig(cfg)
			checkIndex(root, cfg)
			checkMigration(root, cfg)
			return nil
		},
	}
	cmd.Flags().StringVarP(&path, "path", "p", "", "Project root.")
	return cmd
}

func ok(msg string, args ...any)   { cliui.Success("  "+msg, args...) }
func warn(msg string, args ...any) { cliui.Warn("  "+msg, args...) }
func note(msg string, args ...any) { cliui.Info("  %s", fmt.Sprintf(cliui.Dim(msg), args...)) }

func checkEnvironment(cfg config.Config) {
	cliui.Info("%s", cliui.Bold("Environment"))
	if !ollama.Installed() {
		warn("Ollama not installed — search/index still work; `ask`/`chat` need it.")
		note("Install: https://ollama.com/download")
		return
	}
	ok("Ollama installed")
	c := ollama.New(cfg)
	if !c.Ping() {
		warn("Ollama daemon not running. Start with: ollama serve")
		return
	}
	ok("Ollama daemon reachable at %s", cfg.OllamaBaseURL)
	if c.ModelAvailable(cfg.Model) {
		ok("Chat model pulled: %s", cfg.Model)
	} else {
		warn("Chat model not pulled: %s (run `documind setup` or `ollama pull %s`)", cfg.Model, cfg.Model)
	}
	if c.ModelAvailable(cfg.EmbeddingModel) {
		ok("Embedding model pulled: %s", cfg.EmbeddingModel)
	} else {
		warn("Embedding model not pulled: %s (vectors skipped until `ollama pull %s`)", cfg.EmbeddingModel, cfg.EmbeddingModel)
	}
}

func checkConfig(cfg config.Config) {
	cliui.Info("%s", cliui.Bold("Config"))
	cfgPath := config.UserConfigPath()
	if _, err := os.Stat(cfgPath); err != nil {
		note("No user config yet (%s). Using defaults.", cfgPath)
	} else {
		ok("Config: %s", cfgPath)
	}
	if cfg.SetupDone {
		ok("Setup complete (model: %s, keep_alive: %s)", cfg.Model, cfg.KeepAlive)
	} else {
		note("Setup not run yet. Run `documind setup` to pick a model.")
	}
	// Python-era configs stored embedding_model as a HuggingFace repo id
	// (e.g. "BAAI/bge-small-en-v1.5"); the Go build needs an Ollama tag.
	if strings.Contains(cfg.EmbeddingModel, "/") {
		warn("embedding_model %q looks like a Python/HuggingFace id.", cfg.EmbeddingModel)
		note("Set it to an Ollama tag, e.g.: embedding_model = \"nomic-embed-text\"")
	}
}

func checkIndex(root string, cfg config.Config) {
	cliui.Info("%s", cliui.Bold("Index"))
	if !indexer.Exists(root, cfg) {
		note("No index for this project. Run: documind index %s", root)
		return
	}
	paths := indexer.PathsFor(root, cfg)
	db, err := store.Open(paths.Meta)
	if err != nil {
		warn("Cannot open metadata db: %v", err)
		return
	}
	defer db.Close()

	chunks, _ := db.CountChunks()
	vectored, _ := db.CountVectored()
	ok("Chunks: %d (with vectors: %d)", chunks, vectored)
	if vectored == 0 && chunks > 0 {
		note("Keyword-only index. Pull the embedding model and re-run `documind index` to add vectors.")
	}
	if !db.HasSymbolColumn() {
		warn("Index schema predates code-navigation. Rebuild with: documind index --rebuild")
	}
}

func checkMigration(root string, cfg config.Config) {
	cliui.Info("%s", cliui.Bold("Migration (Python -> Go)"))
	dir := cfg.IndexDirFor(root)
	if _, err := os.Stat(dir); err != nil {
		note("No .documind/ directory; nothing to migrate.")
		return
	}
	// A Python-era index used LanceDB (*.lance dirs / lance files) and had no
	// meta.sqlite/bleve. Detect leftovers and advise a clean rebuild.
	legacy := detectLegacyIndex(dir)
	if len(legacy) == 0 {
		ok("No Python-era index artifacts detected.")
		return
	}
	warn("Detected Python-era index artifacts:")
	for _, l := range legacy {
		note("- %s", l)
	}
	note("Migrate with: documind index --rebuild (rebuilds cleanly in the Go format)")
}

// detectLegacyIndex returns human-readable names of Python/LanceDB leftovers
// found inside a .documind directory.
func detectLegacyIndex(dir string) []string {
	var found []string
	entries, err := os.ReadDir(dir)
	if err != nil {
		return found
	}
	for _, e := range entries {
		name := e.Name()
		lower := strings.ToLower(name)
		switch {
		case strings.HasSuffix(lower, ".lance"):
			found = append(found, name)
		case lower == "lance" || strings.Contains(lower, "lancedb"):
			found = append(found, name)
		case lower == "bm25" || strings.HasSuffix(lower, ".npz"): // bm25s artifacts
			found = append(found, name)
		}
	}
	// A .documind that has content but no Go metadata db is also suspect.
	if len(found) == 0 {
		if _, err := os.Stat(filepath.Join(dir, "meta.sqlite")); err != nil {
			if len(entries) > 0 {
				found = append(found, "non-empty .documind/ without meta.sqlite (unknown/old format)")
			}
		}
	}
	return found
}
