package cli

import (
	"context"
	"errors"
	"fmt"
	"os"

	"github.com/Henildiyora/DocuMind/internal/cliui"
	"github.com/Henildiyora/DocuMind/internal/config"
	"github.com/Henildiyora/DocuMind/internal/indexer"
	"github.com/spf13/cobra"
)

func newIndexCmd() *cobra.Command {
	var rebuild bool
	var forceRehash bool

	cmd := &cobra.Command{
		Use:   "index [path]",
		Short: "Index a project (incremental by default).",
		Long: "Walk a project, chunk supported files, and build the keyword index. " +
			"Dense vectors are added when Ollama is reachable; otherwise indexing " +
			"completes keyword-only and backfills vectors on a later run.",
		Args: cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			path := ""
			if len(args) == 1 {
				path = args[0]
			}
			root, err := resolveRoot(path)
			if err != nil {
				return err
			}
			cfg := config.Load()

			if rebuild {
				cliui.Warn("Rebuilding index at %s", cfg.IndexDirFor(root))
			}
			cliui.Info("%s %s", cliui.Bold("Indexing"), root)

			progress := newProgressPrinter()
			stats, err := indexer.BuildOrUpdate(context.Background(), root, cfg, indexer.Options{
				ForceRehash: forceRehash,
				Rebuild:     rebuild,
				Progress:    progress.update,
			})
			progress.finish()
			if err != nil {
				if errors.Is(err, indexer.ErrLocked) {
					return fmt.Errorf("another indexing process holds the project lock at %s/.documind-index.lock", root)
				}
				return err
			}

			printStats(stats)
			idle := stats.NewFiles == 0 && stats.ChangedFiles == 0 && stats.RemovedFiles == 0 && stats.EmbeddedChunks == 0
			if idle && stats.TotalChunks > 0 {
				cliui.Success("Index already up to date (%d chunks) at %s", stats.TotalChunks, cfg.IndexDirFor(root))
			} else {
				cliui.Success("Index ready at %s", cfg.IndexDirFor(root))
			}
			if stats.VectorsSkipped {
				reason := stats.SkipReason
				if reason == "" {
					reason = "Ollama not available"
				}
				cliui.Warn("vector embeddings skipped: %s (keyword search still works; re-run `documind index` to backfill vectors)", reason)
			}

			// One-time inline offer: if the user has never run setup, invite them
			// to pick a model so `ask`/`chat` work. Only on an interactive TTY.
			maybeOfferSetup(root, cfg)
			return nil
		},
	}
	cmd.Flags().BoolVar(&rebuild, "rebuild", false, "Delete and rebuild the index from scratch.")
	cmd.Flags().BoolVar(&forceRehash, "force-rehash", false, "Re-read and re-hash every file (ignore mtime/size fast path).")
	return cmd
}

// maybeOfferSetup shows a one-time prompt after indexing to help the user pick a
// local model. It is skipped when setup already ran, when the offer was already
// shown, or on a non-interactive terminal. The offer flag is cleared afterward
// so it never nags twice.
func maybeOfferSetup(root string, cfg config.Config) {
	if cfg.SetupDone || !cfg.OfferSetupAfterIndex || !cliui.IsTTY() {
		return
	}
	// Clear the flag first so we never prompt again regardless of the answer.
	_, _ = config.UpdateUser(map[string]any{"offer_setup_after_index": false})

	cliui.Info("%s", cliui.Dim("Tip: `documind search` and `documind ask` work now. `ask`/`chat` synthesis needs a local model."))
	if !cliui.Confirm("Run setup to pick a local model now?", false) {
		cliui.Info("%s", cliui.Dim("You can run `documind setup` anytime."))
		return
	}
	if err := runSetup(root, setupOptions{}); err != nil {
		cliui.Errorln("setup failed: %v", err)
	}
}

func printStats(s indexer.Stats) {
	rows := [][2]string{
		{"Scanned files", fmt.Sprintf("%d", s.ScannedFiles)},
		{"New", cliui.Green(fmt.Sprintf("%d", s.NewFiles))},
		{"Changed", cliui.Yellow(fmt.Sprintf("%d", s.ChangedFiles))},
		{"Unchanged", fmt.Sprintf("%d", s.UnchangedFiles)},
		{"Removed", cliui.Red(fmt.Sprintf("%d", s.RemovedFiles))},
		{"Embedded chunks", fmt.Sprintf("%d", s.EmbeddedChunks)},
		{"Total chunks", fmt.Sprintf("%d", s.TotalChunks)},
	}
	for _, r := range rows {
		fmt.Fprintf(os.Stdout, "  %-18s %s\n", r[0], r[1])
	}
}

// progressPrinter renders single-line, in-place progress for each phase.
type progressPrinter struct {
	lastPhase string
	active    bool
}

func newProgressPrinter() *progressPrinter { return &progressPrinter{} }

func (p *progressPrinter) update(phase string, done, total int) {
	label := map[string]string{
		"chunking":  "Chunking",
		"embedding": "Embedding",
		"keyword":   "Building keyword index",
	}[phase]
	if label == "" {
		label = phase
	}
	p.active = true
	if phase != p.lastPhase && p.lastPhase != "" {
		fmt.Fprint(os.Stdout, "\n")
	}
	p.lastPhase = phase
	fmt.Fprintf(os.Stdout, "\r  %s %d/%d ", label, done, total)
}

func (p *progressPrinter) finish() {
	if p.active {
		fmt.Fprint(os.Stdout, "\n")
	}
}
