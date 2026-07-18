package cli

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/Henildiyora/DocuMind/internal/cliui"
	"github.com/Henildiyora/DocuMind/internal/config"
	"github.com/Henildiyora/DocuMind/internal/navigation"
	"github.com/Henildiyora/DocuMind/internal/ollama"
	"github.com/Henildiyora/DocuMind/internal/prompts"
)

// runCodeNavigation resolves the symbol a question is about within --file, traces
// its definition and usages across the project, and answers prose-first with
// line-scoped citations. When no file is given (LLM misclassification) or no
// symbol resolves, it falls back to content retrieval.
func runCodeNavigation(root string, cfg config.Config, query, file string, showCode bool, bundle *retrieverBundle, client *ollama.Client, ready bool) error {
	if file == "" {
		// CodeNavigation without a target file: fall back to content Q&A.
		return runContentQA(root, cfg, query, bundle, client, ready, false)
	}
	absFile, err := resolveFile(root, file)
	if err != nil {
		return err
	}
	if _, err := os.Stat(absFile); err != nil {
		return fmt.Errorf("file not found: %s", file)
	}

	best, candidates, err := navigation.ResolveSymbol(absFile, query)
	if err != nil {
		return err
	}

	// Ambiguous symbol match: ask the user to pick (interactive only).
	if best == nil && len(candidates) > 1 {
		labels := make([]string, len(candidates))
		for i, c := range candidates {
			labels[i] = fmt.Sprintf("%s (%s, line %d)", c.Name, c.Kind, c.StartLine)
		}
		choice, ok := cliui.Pick("Which symbol did you mean?", labels)
		if ok {
			for i, l := range labels {
				if l == choice {
					best = &candidates[i]
					break
				}
			}
		}
	}

	if best == nil {
		cliui.Warn("Could not resolve a specific symbol in %s from your question; falling back to content search.", file)
		return runContentQA(root, cfg, query, bundle, client, ready, false)
	}

	usages, err := navigation.TraceUsages(bundle.db, bundle.kw, *best, cfg.TopK+4)
	if err != nil {
		return err
	}
	if len(usages) == 0 {
		cliui.Warn("No definition or usages found for %s.", best.Name)
		return nil
	}

	relFile, _ := filepath.Rel(root, absFile)
	cliui.Info("%s %s %s", cliui.Bold("Symbol"), cliui.Cyan(best.Name), cliui.Dim("("+best.Kind+" in "+relFile+")"))

	if !ready {
		// No model: print a scoped, grounded trace directly.
		printUsages(usages, cfg, root, true)
		cliui.Info("%s", cliui.Dim("Tip: run `documind setup` for a natural-language trace."))
		return nil
	}

	cliui.Info("%s %s", cliui.Bold("Trace"), cliui.Dim("(local model: "+cfg.Model+")"))
	context := buildTraceContext(usages, root, 8000)
	messages := prompts.BuildTraceMessages(query+" (symbol: "+best.Name+")", context)
	if err := streamAnswerNav(client, messages, cfg); err != nil {
		return err
	}
	if showCode {
		fmt.Fprintln(os.Stdout)
		cliui.Info("%s", cliui.Bold("Scoped snippets"))
		printUsages(usages, cfg, root, true)
	} else {
		printUsages(usages, cfg, root, false)
	}
	return nil
}

func streamAnswerNav(client *ollama.Client, messages []ollama.Message, cfg config.Config) error {
	err := client.ChatStream(context.Background(), messages, cfg.Model, "", func(tok string) {
		fmt.Fprint(os.Stdout, tok)
	})
	fmt.Fprintln(os.Stdout)
	return err
}

// resolveFile resolves a --file argument relative to cwd first, then the project
// root, returning an absolute path.
func resolveFile(root, file string) (string, error) {
	if filepath.IsAbs(file) {
		return file, nil
	}
	if abs, err := filepath.Abs(file); err == nil {
		if _, statErr := os.Stat(abs); statErr == nil {
			return abs, nil
		}
	}
	return filepath.Join(root, file), nil
}

// buildTraceContext renders definition + usage snippets as a bounded, labeled
// context for the trace prompt (scoped to the stored line ranges, never full
// files).
func buildTraceContext(usages []navigation.Usage, root string, maxChars int) string {
	var b strings.Builder
	used := 0
	for _, u := range usages {
		label := "USAGE"
		if u.IsDef {
			label = "DEFINITION"
		}
		header := fmt.Sprintf("### %s %s:%d-%d\n", label, u.RelPath, u.StartLine, u.EndLine)
		block := header + "```\n" + u.Text + "\n```\n\n"
		if used+len(block) > maxChars {
			break
		}
		b.WriteString(block)
		used += len(block)
	}
	return b.String()
}

func printUsages(usages []navigation.Usage, cfg config.Config, root string, showCode bool) {
	for _, u := range usages {
		tag := cliui.Dim("usage")
		if u.IsDef {
			tag = cliui.Green("def")
		}
		fmt.Fprintf(os.Stdout, "  %s %s:%s\n", tag, cliui.Green(u.RelPath), cliui.Cyan(fmt.Sprintf("%d-%d", u.StartLine, u.EndLine)))
		if showCode {
			snippet := u.Text
			if len(snippet) > cfg.SnippetChars {
				snippet = strings.TrimRight(snippet[:cfg.SnippetChars], " \n\t") + "\n..."
			}
			fmt.Fprintln(os.Stdout, snippet)
			fmt.Fprintln(os.Stdout)
		}
	}
}
