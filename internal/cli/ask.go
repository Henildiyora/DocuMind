package cli

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/Henildiyora/DocuMind/internal/cliui"
	"github.com/Henildiyora/DocuMind/internal/config"
	"github.com/Henildiyora/DocuMind/internal/ollama"
	"github.com/Henildiyora/DocuMind/internal/prompts"
	"github.com/Henildiyora/DocuMind/internal/router"
	"github.com/Henildiyora/DocuMind/internal/search"
	"github.com/Henildiyora/DocuMind/internal/structural"
	"github.com/Henildiyora/DocuMind/internal/threads"
	"github.com/spf13/cobra"
)

func newAskCmd() *cobra.Command {
	var path, model, keepAlive, file string
	var k int
	var noLLM, showCode, noClarify bool

	cmd := &cobra.Command{
		Use:   "ask [question...]",
		Short: "Ask a grounded question about the project.",
		Long: "Ask a grounded question. Trailing words are joined so quotes are optional:\n\n" +
			"    documind ask why does the rate limiter reset early\n\n" +
			"The query is routed to structural facts, code navigation (--file), or\n" +
			"content retrieval, and answered with a local model when available.",
		Args: cobra.ArbitraryArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			query := strings.TrimSpace(strings.Join(args, " "))
			if query == "" {
				return fmt.Errorf("provide a question, e.g. documind ask how does indexing work")
			}
			root, err := resolveRoot(path)
			if err != nil {
				return err
			}
			cfg := config.Load()
			if model != "" {
				cfg.Model = model
			}
			if k > 0 {
				cfg.TopK = k
			}
			if cmd.Flags().Changed("keep-alive") {
				cfg.KeepAlive = keepAlive
			}
			if !indexReady(root, cfg) {
				return fmt.Errorf("no index found for this project. Run: documind index %s", root)
			}

			bundle, err := openRetriever(root, cfg)
			if err != nil {
				return err
			}
			defer bundle.Close()

			// --no-llm: pure retrieval, print ranked snippets.
			if noLLM {
				hits, err := bundle.Retriever.Search(context.Background(), query, cfg.TopK)
				if err != nil {
					return err
				}
				hits = search.Refine(hits, query, cfg)
				if len(hits) == 0 {
					cliui.Warn("No matches.")
					return nil
				}
				printHits(hits, cfg, true)
				return nil
			}

			client, ready := llmReady(cfg)

			opts := router.Options{HasFile: file != ""}
			var classifyClient *ollama.Client
			if ready {
				classifyClient = client
			}
			decision := router.Classify(context.Background(), classifyClient, query, opts)

			switch decision.Category {
			case router.Structural:
				return runStructural(root, cfg, query, client, ready)
			case router.CodeNavigation:
				return runCodeNavigation(root, cfg, query, file, showCode, bundle, client, ready)
			default:
				return runContentQA(root, cfg, query, bundle, client, ready, !noClarify)
			}
		},
	}
	cmd.Flags().StringVarP(&path, "path", "p", "", "Project root.")
	cmd.Flags().StringVarP(&model, "model", "m", "", "Ollama model override.")
	cmd.Flags().StringVar(&keepAlive, "keep-alive", "", "How long Ollama keeps the model in RAM (e.g. 5m, 0).")
	cmd.Flags().StringVar(&file, "file", "", "Focus a specific file for code navigation.")
	cmd.Flags().IntVarP(&k, "k", "k", 0, "Number of snippets.")
	cmd.Flags().BoolVar(&noLLM, "no-llm", false, "Print ranked hits only (skip LLM).")
	cmd.Flags().BoolVar(&showCode, "code", false, "Include scoped code snippets in the answer.")
	cmd.Flags().BoolVar(&noClarify, "no-clarify", false, "Skip interactive clarification when ambiguous.")
	return cmd
}

// runStructural answers an overview/structural question from exact facts.
func runStructural(root string, cfg config.Config, query string, client *ollama.Client, ready bool) error {
	facts, err := structural.GatherFacts(root, cfg)
	if err != nil {
		return err
	}
	if !ready {
		// No model: print the facts directly (still useful, still grounded).
		cliui.Info("%s", cliui.Bold("Project facts"))
		fmt.Fprintln(os.Stdout, facts.Summary())
		if hint := llmHint(cfg); hint != "" {
			cliui.Info("%s", cliui.Dim(hint))
		}
		return nil
	}
	cliui.Info("%s %s", cliui.Bold("Answer"), cliui.Dim("(local model: "+cfg.Model+")"))
	messages := structuralMessages(query, facts.Summary())
	_, err = streamAnswer(client, messages, cfg)
	return err
}

// structuralMessages picks the right prompt for a structural question: a friendly
// overview for "explain the project"-style intent, or a terse factual answer for
// pointed asks like "how many folders".
func structuralMessages(query, facts string) []ollama.Message {
	if router.IsExplainIntent(query) {
		return prompts.BuildOverviewMessages(query, facts)
	}
	return prompts.BuildStructuralMessages(query, facts)
}

// runContentQA handles the default retrieval path with an optional clarify loop.
func runContentQA(root string, cfg config.Config, query string, bundle *retrieverBundle, client *ollama.Client, ready, allowClarify bool) error {
	ctx := context.Background()
	hits, err := bundle.Retriever.Search(ctx, query, cfg.TopK)
	if err != nil {
		return err
	}
	hits = search.Refine(hits, query, cfg)
	if len(hits) == 0 {
		cliui.Warn("No matches.")
		return nil
	}

	effective := query

	// Ambiguity: only when a model is available, clarification is allowed, and
	// the results are a genuine basename collision / named-file miss.
	if ready && allowClarify && router.DetectAmbiguity(hits, query) {
		clar := router.RequestClarification(ctx, client, query, topPaths(hits, 6), cfg)
		if clar != nil {
			choice, ok := cliui.Pick(clar.Question, clar.Options)
			if ok && choice != "" {
				effective = effective + " — focusing on: " + choice
				hits, err = bundle.Retriever.SearchMulti(ctx, []string{effective, choice}, cfg.TopK)
				if err != nil {
					return err
				}
				hits = search.Refine(hits, effective, cfg)
			}
		}
	}

	if !ready {
		cliui.Info("%s %s", cliui.Bold("Snippets from"), cliui.Cyan(root))
		printHits(hits, cfg, true)
		if hint := llmHint(cfg); hint != "" {
			cliui.Info("%s", cliui.Dim(hint))
		}
		return nil
	}

	cliui.Info("%s %s", cliui.Bold("Answer"), cliui.Dim("(local model: "+cfg.Model+")"))
	messages := prompts.BuildAnswerMessages(effective, search.HitsToContext(hits, 8000))
	answer, err := streamAnswer(client, messages, cfg)
	if err != nil {
		return err
	}
	printSources(hits)
	// One-shot ask persists to the default thread so `chat`/`ask` can continue it.
	_ = threads.AppendTurn(root, threads.DefaultThread, query, answer, cfg)
	return nil
}

// streamAnswer streams an answer to stdout token-by-token and returns the full
// accumulated text.
func streamAnswer(client *ollama.Client, messages []ollama.Message, cfg config.Config) (string, error) {
	var b strings.Builder
	err := client.ChatStream(context.Background(), messages, cfg.Model, "", func(tok string) {
		b.WriteString(tok)
		fmt.Fprint(os.Stdout, tok)
	})
	fmt.Fprintln(os.Stdout)
	return b.String(), err
}

func printSources(hits []search.SearchHit) {
	fmt.Fprintln(os.Stdout)
	cliui.Info("%s", cliui.Bold("Sources"))
	for _, h := range hits {
		fmt.Fprintf(os.Stdout, "  %s:%s\n", cliui.Green(h.RelPath), cliui.Cyan(fmt.Sprintf("%d-%d", h.StartLine, h.EndLine)))
	}
}

func topPaths(hits []search.SearchHit, n int) []string {
	var out []string
	for i, h := range hits {
		if i >= n {
			break
		}
		out = append(out, fmt.Sprintf("%s:%d-%d", h.RelPath, h.StartLine, h.EndLine))
	}
	return out
}
