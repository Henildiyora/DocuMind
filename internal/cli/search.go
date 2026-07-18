package cli

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/Henildiyora/DocuMind/internal/cliui"
	"github.com/Henildiyora/DocuMind/internal/config"
	"github.com/Henildiyora/DocuMind/internal/search"
	"github.com/spf13/cobra"
)

func newSearchCmd() *cobra.Command {
	var path string
	var k int
	var showCode bool

	cmd := &cobra.Command{
		Use:   "search <query>",
		Short: "Hybrid keyword + vector search over the project index.",
		Long:  "Fast, zero-config hybrid search (keyword + optional dense vectors). No LLM required.",
		Args:  cobra.ArbitraryArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			query := strings.TrimSpace(strings.Join(args, " "))
			if query == "" {
				return fmt.Errorf("provide a search query, e.g. documind search rate limiter")
			}
			root, err := resolveRoot(path)
			if err != nil {
				return err
			}
			cfg := config.Load()
			if k > 0 {
				cfg.TopK = k
			}
			if !indexReady(root, cfg) {
				return fmt.Errorf("no index found for this project. Run: documind index %s", root)
			}

			bundle, err := openRetriever(root, cfg)
			if err != nil {
				return err
			}
			defer bundle.Close()

			hits, err := bundle.Retriever.Search(context.Background(), query, cfg.TopK)
			if err != nil {
				return err
			}
			hits = search.Refine(hits, query, cfg)
			if len(hits) == 0 {
				cliui.Warn("No matches.")
				return nil
			}

			cliui.Info("%s %s", cliui.Bold("Snippets from"), cliui.Cyan(root))
			printHits(hits, cfg, showCode)
			return nil
		},
	}
	cmd.Flags().StringVarP(&path, "path", "p", "", "Project root.")
	cmd.Flags().IntVarP(&k, "k", "k", 0, "Number of results.")
	cmd.Flags().BoolVar(&showCode, "code", true, "Print code snippets (default: on).")
	return cmd
}

// printHits renders ranked search hits with per-source ranks and optional code.
func printHits(hits []search.SearchHit, cfg config.Config, showCode bool) {
	for i, h := range hits {
		bm25 := "--"
		if h.BM25Rank > 0 {
			bm25 = fmt.Sprintf("bm25#%d", h.BM25Rank)
		}
		vec := "--"
		if h.VectorRank > 0 {
			vec = fmt.Sprintf("vec#%d", h.VectorRank)
		}
		sym := ""
		if h.Symbol != "" {
			sym = " " + cliui.Dim(h.SymbolKind+" "+h.Symbol)
		}
		fmt.Fprintf(os.Stdout, "%s %s:%s%s %s\n",
			cliui.Bold(fmt.Sprintf("%2d", i+1)),
			cliui.Green(h.RelPath),
			cliui.Cyan(fmt.Sprintf("%d-%d", h.StartLine, h.EndLine)),
			sym,
			cliui.Dim(fmt.Sprintf("(%s, %s, rrf=%.4f)", bm25, vec, h.Score)),
		)
		if showCode {
			fmt.Fprintln(os.Stdout, search.FormatSnippet(h, cfg))
			fmt.Fprintln(os.Stdout)
		}
	}
}
