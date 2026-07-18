// Package router classifies an incoming query into one of four handling
// categories and detects when a query is too ambiguous to answer. Classification
// is heuristic-first (fast, deterministic, zero-cost); only when the heuristic is
// unsure does it make exactly one single-purpose LLM call that returns a label
// and nothing else. This package never synthesizes an answer.
package router

import (
	"context"
	"encoding/json"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/Henildiyora/DocuMind/internal/config"
	"github.com/Henildiyora/DocuMind/internal/ollama"
	"github.com/Henildiyora/DocuMind/internal/prompts"
	"github.com/Henildiyora/DocuMind/internal/search"
)

// Category is the handling bucket for a query.
type Category int

const (
	// ContentQA answers from file content via hybrid retrieval (the default).
	ContentQA Category = iota
	// Structural answers from LLM-free directory facts.
	Structural
	// CodeNavigation resolves and traces a symbol (only when --file is set).
	CodeNavigation
	// Ambiguous requires a clarifying question before answering.
	Ambiguous
)

func (c Category) String() string {
	switch c {
	case Structural:
		return "STRUCTURAL"
	case CodeNavigation:
		return "CODE_NAVIGATION"
	case Ambiguous:
		return "AMBIGUOUS"
	default:
		return "CONTENT_QA"
	}
}

// Options controls classification.
type Options struct {
	HasFile bool // true when --file is set (forces CodeNavigation)
}

// Decision is the result of Classify.
type Decision struct {
	Category  Category
	Heuristic bool // true when decided without an LLM call
	LLMLabel  string
}

// overviewRe matches broad project-overview / structural questions. Ported and
// expanded from is_overview_intent in query_understand.py to also catch folder
// and directory counting.
var overviewRe = regexp.MustCompile(`(?i)\b(` +
	`how\s+many\s+(files|folders|directories|dirs|packages|modules|lines)|` +
	`(project|repo|repository|codebase)\s+(structure|overview|layout)|` +
	`directory\s+structure|folder\s+structure|` +
	`explain\s+(me\s+)?(this\s+)?(project|repo|repository|codebase)|` +
	`what\s+is\s+this(\s+(project|repo|repository|codebase))?|` +
	`how\s+does\s+this\s+(project|repo|repository|codebase)\s+work|` +
	`overview|summarize\s+(this\s+)?(project|repo|repository|codebase)|` +
	`tell\s+me\s+about\s+(this\s+)?(project|repo|repository|codebase)|` +
	`what\s+(languages|does\s+this\s+(app|repo|codebase|project)\s+do)` +
	`)\b`)

// IsOverviewIntent reports whether a query is a broad structural/overview ask.
func IsOverviewIntent(query string) bool {
	return overviewRe.MatchString(query)
}

// Classify determines the category for a query. Heuristics decide the common
// cases with no LLM call; otherwise a single classification call is made when a
// client is available.
func Classify(ctx context.Context, client *ollama.Client, query string, opts Options) Decision {
	if opts.HasFile {
		return Decision{Category: CodeNavigation, Heuristic: true}
	}
	if IsOverviewIntent(query) {
		return Decision{Category: Structural, Heuristic: true}
	}
	// Heuristic not confident. Fall back to a single-purpose LLM classify call
	// when possible; otherwise default to content Q&A.
	if client == nil {
		return Decision{Category: ContentQA, Heuristic: true}
	}
	label := llmClassify(ctx, client, query)
	switch label {
	case "STRUCTURAL":
		return Decision{Category: Structural, LLMLabel: label}
	case "CODE_NAVIGATION":
		return Decision{Category: CodeNavigation, LLMLabel: label}
	case "CONTENT_QA":
		return Decision{Category: ContentQA, LLMLabel: label}
	default:
		return Decision{Category: ContentQA, Heuristic: true}
	}
}

func llmClassify(ctx context.Context, client *ollama.Client, query string) string {
	raw, err := client.Chat(ctx, prompts.BuildClassifyMessages(query), "", "json")
	if err != nil {
		return ""
	}
	obj := extractJSONObject(raw)
	if obj == nil {
		return ""
	}
	if cat, ok := obj["category"].(string); ok {
		return strings.ToUpper(strings.TrimSpace(cat))
	}
	return ""
}

// DetectAmbiguity reports whether ContentQA results are genuinely ambiguous:
// real basename collisions in the top hits, or a named file in the query that
// does not match the top hit. Overview queries are never ambiguous. Ported from
// detect_ambiguity in query_understand.py.
func DetectAmbiguity(hits []search.SearchHit, query string) bool {
	if len(hits) == 0 || IsOverviewIntent(query) {
		return false
	}
	limit := 8
	if len(hits) < limit {
		limit = len(hits)
	}
	counts := map[string]int{}
	var basenames []string
	for _, h := range hits[:limit] {
		name := strings.ToLower(filepath.Base(h.RelPath))
		basenames = append(basenames, name)
		counts[name]++
	}
	for _, c := range counts {
		if c > 1 {
			return true
		}
	}
	hints := search.FilenameHints(query)
	if len(hints) == 0 {
		return false
	}
	top := ""
	if len(basenames) > 0 {
		top = basenames[0]
	}
	for _, h := range hints {
		if strings.Contains(top, h) || strings.HasPrefix(top, h) {
			return false
		}
	}
	return true
}

// Clarification is a short multiple-choice question produced for ambiguous
// queries.
type Clarification struct {
	Question string
	Options  []string
}

// RequestClarification makes a single-purpose clarify call. Returns nil when the
// model judges the question clear or the reply is unusable.
func RequestClarification(ctx context.Context, client *ollama.Client, query string, topPaths []string, cfg config.Config) *Clarification {
	if client == nil {
		return nil
	}
	raw, err := client.Chat(ctx, prompts.BuildClarifyMessages(query, topPaths), "", "json")
	if err != nil {
		return nil
	}
	obj := extractJSONObject(raw)
	if obj == nil {
		return nil
	}
	if amb, ok := obj["ambiguous"].(bool); ok && !amb {
		return nil
	}
	rawOpts, ok := obj["options"].([]any)
	if !ok || len(rawOpts) < 2 {
		return nil
	}
	var options []string
	for _, o := range rawOpts {
		if s, ok := o.(string); ok {
			if s = strings.TrimSpace(s); s != "" {
				options = append(options, s)
			}
		}
	}
	if len(options) < 2 {
		return nil
	}
	if len(options) > 4 {
		options = options[:4]
	}
	question := "Which of these did you mean?"
	if q, ok := obj["question"].(string); ok && strings.TrimSpace(q) != "" {
		question = strings.TrimSpace(q)
	}
	return &Clarification{Question: question, Options: options}
}

// extractJSONObject best-effort extracts a JSON object from an LLM reply,
// tolerating code fences and surrounding prose.
func extractJSONObject(text string) map[string]any {
	text = strings.TrimSpace(text)
	if text == "" {
		return nil
	}
	if fence := regexp.MustCompile("(?s)```(?:json)?\\s*(\\{.*?\\})\\s*```").FindStringSubmatch(text); len(fence) == 2 {
		text = fence[1]
	} else {
		start := strings.Index(text, "{")
		end := strings.LastIndex(text, "}")
		if start >= 0 && end > start {
			text = text[start : end+1]
		}
	}
	var obj map[string]any
	if err := json.Unmarshal([]byte(text), &obj); err != nil {
		return nil
	}
	return obj
}
