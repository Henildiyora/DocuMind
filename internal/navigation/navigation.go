// Package navigation implements --file code navigation: it resolves which symbol
// a question is about within a given file (via tree-sitter), then traces that
// symbol's definition and usages across the whole project (via the metadata
// store and keyword index). The answer is prose-first with line-scoped
// citations; full files are never dumped.
package navigation

import (
	"context"
	"os"
	"regexp"
	"sort"
	"strings"

	"github.com/Henildiyora/DocuMind/internal/chunker"
	"github.com/Henildiyora/DocuMind/internal/kwindex"
	"github.com/Henildiyora/DocuMind/internal/store"
	sitter "github.com/smacker/go-tree-sitter"
)

// Symbol is a resolved code symbol within a file.
type Symbol struct {
	Name      string
	Kind      string // function, method, class, type, parameter, ...
	File      string // absolute path of the file it was found in
	StartLine int
	EndLine   int
}

// definitionTypes maps AST node types to a symbol kind for definitions we can
// navigate to. A superset of the chunker's significant types, walked recursively
// so methods inside classes are captured too.
var definitionTypes = map[string]string{
	"function_declaration":           "function",
	"generator_function_declaration": "function",
	"function_definition":            "function",
	"function_item":                  "function",
	"method_declaration":             "method",
	"method_definition":              "method",
	"constructor_declaration":        "constructor",
	"class_declaration":              "class",
	"class_definition":               "class",
	"class_specifier":                "class",
	"struct_specifier":               "struct",
	"struct_item":                    "struct",
	"interface_declaration":          "interface",
	"trait_item":                     "trait",
	"enum_declaration":               "enum",
	"enum_item":                      "enum",
	"type_declaration":               "type",
	"type_alias_declaration":         "type",
	"type_item":                      "type",
}

// parameterListTypes are the node types that hold a callable's parameters.
var parameterListTypes = map[string]bool{
	"parameter_list":    true,
	"formal_parameters": true,
	"parameters":        true,
}

// ExtractSymbols parses a file and returns its navigable symbols (definitions and
// their parameters). Returns nil for unsupported languages.
func ExtractSymbols(absPath string) ([]Symbol, error) {
	source, err := os.ReadFile(absPath)
	if err != nil {
		return nil, err
	}
	lang, ok := chunker.GrammarFor(chunker.DetectLanguage(absPath))
	if !ok {
		return nil, nil
	}
	parser := sitter.NewParser()
	parser.SetLanguage(lang)
	tree, err := parser.ParseCtx(context.Background(), nil, source)
	if err != nil || tree == nil {
		return nil, err
	}
	defer tree.Close()

	var symbols []Symbol
	seen := map[string]bool{}
	var walk func(n *sitter.Node)
	walk = func(n *sitter.Node) {
		if n == nil {
			return
		}
		if kind, ok := definitionTypes[n.Type()]; ok {
			name := nameOf(n, source)
			if name != "" {
				key := name + "|" + kind
				if !seen[key] {
					seen[key] = true
					symbols = append(symbols, Symbol{
						Name:      name,
						Kind:      kind,
						File:      absPath,
						StartLine: int(n.StartPoint().Row) + 1,
						EndLine:   int(n.EndPoint().Row) + 1,
					})
				}
			}
			// Capture parameters of callables.
			for _, p := range parametersOf(n, source) {
				key := p + "|parameter"
				if !seen[key] {
					seen[key] = true
					symbols = append(symbols, Symbol{
						Name:      p,
						Kind:      "parameter",
						File:      absPath,
						StartLine: int(n.StartPoint().Row) + 1,
						EndLine:   int(n.EndPoint().Row) + 1,
					})
				}
			}
		}
		count := int(n.NamedChildCount())
		for i := 0; i < count; i++ {
			walk(n.NamedChild(i))
		}
	}
	walk(tree.RootNode())
	return symbols, nil
}

func nameOf(n *sitter.Node, source []byte) string {
	if named := n.ChildByFieldName("name"); named != nil {
		return strings.TrimSpace(named.Content(source))
	}
	count := int(n.NamedChildCount())
	for i := 0; i < count; i++ {
		child := n.NamedChild(i)
		if child == nil {
			continue
		}
		switch child.Type() {
		case "identifier", "type_identifier", "field_identifier", "constant":
			return strings.TrimSpace(child.Content(source))
		}
	}
	return ""
}

// parametersOf collects parameter identifier names for a callable node.
func parametersOf(n *sitter.Node, source []byte) []string {
	var params []string
	count := int(n.NamedChildCount())
	for i := 0; i < count; i++ {
		child := n.NamedChild(i)
		if child == nil {
			continue
		}
		if parameterListTypes[child.Type()] {
			collectIdentifiers(child, source, &params)
		}
	}
	return params
}

// collectIdentifiers gathers identifier tokens directly relevant to parameters,
// avoiding type identifiers where possible by taking the first identifier in each
// parameter subtree.
func collectIdentifiers(n *sitter.Node, source []byte, out *[]string) {
	count := int(n.NamedChildCount())
	for i := 0; i < count; i++ {
		child := n.NamedChild(i)
		if child == nil {
			continue
		}
		if id := firstIdentifier(child, source); id != "" {
			*out = append(*out, id)
		}
	}
}

func firstIdentifier(n *sitter.Node, source []byte) string {
	if n.Type() == "identifier" {
		return strings.TrimSpace(n.Content(source))
	}
	// Prefer a "name"/"pattern" field when present.
	for _, field := range []string{"name", "pattern"} {
		if f := n.ChildByFieldName(field); f != nil {
			if f.Type() == "identifier" {
				return strings.TrimSpace(f.Content(source))
			}
		}
	}
	count := int(n.NamedChildCount())
	for i := 0; i < count; i++ {
		if id := firstIdentifier(n.NamedChild(i), source); id != "" {
			return id
		}
	}
	return ""
}

// ResolveSymbol picks the symbol in a file that best matches the query. It
// returns the single best match when confident, or a list of candidates (2+)
// when the match is ambiguous, so the caller can ask a clarifying question.
func ResolveSymbol(absPath, query string) (best *Symbol, candidates []Symbol, err error) {
	symbols, err := ExtractSymbols(absPath)
	if err != nil {
		return nil, nil, err
	}
	if len(symbols) == 0 {
		return nil, nil, nil
	}
	tokens := tokenize(query)

	type scoredSym struct {
		sym   Symbol
		score int
	}
	var scored []scoredSym
	for _, s := range symbols {
		score := scoreSymbol(s.Name, tokens)
		if score > 0 {
			scored = append(scored, scoredSym{sym: s, score: score})
		}
	}
	if len(scored) == 0 {
		return nil, nil, nil
	}
	sort.SliceStable(scored, func(i, j int) bool { return scored[i].score > scored[j].score })

	top := scored[0].score
	var topSyms []Symbol
	for _, s := range scored {
		if s.score == top {
			topSyms = append(topSyms, s.sym)
		}
	}
	if len(topSyms) == 1 {
		return &topSyms[0], nil, nil
	}
	return nil, topSyms, nil
}

// scoreSymbol scores a symbol name against query tokens using exact-token,
// substring, and split-token overlap. No external fuzzy-match dependency.
func scoreSymbol(name string, queryTokens []string) int {
	lname := strings.ToLower(name)
	best := 0
	for _, t := range queryTokens {
		switch {
		case t == lname:
			if best < 5 {
				best = 5
			}
		case strings.Contains(lname, t) || strings.Contains(t, lname):
			if best < 3 {
				best = 3
			}
		}
	}
	// Split-identifier overlap (camelCase / snake_case) contributes a low score.
	nameParts := splitIdentifier(name)
	for _, np := range nameParts {
		for _, t := range queryTokens {
			if np == t && best < 2 {
				best = 2
			}
		}
	}
	return best
}

// Usage is a place a symbol appears in the project.
type Usage struct {
	RelPath    string
	StartLine  int
	EndLine    int
	Symbol     string
	SymbolKind string
	Text       string
	IsDef      bool
}

// TraceUsages returns the symbol's definition chunks (matched by the stored
// symbol column) plus other chunks that reference the identifier project-wide
// (via a keyword term query).
func TraceUsages(db *store.DB, kw *kwindex.Index, symbol Symbol, limit int) ([]Usage, error) {
	if limit <= 0 {
		limit = 12
	}
	// Definitions: chunks whose stored symbol matches (exact) across files.
	all, err := db.AllChunks()
	if err != nil {
		return nil, err
	}
	var usages []Usage
	defSeen := map[string]bool{}
	for _, c := range all {
		if strings.EqualFold(c.Symbol, symbol.Name) {
			key := c.ChunkID
			if !defSeen[key] {
				defSeen[key] = true
				usages = append(usages, Usage{
					RelPath: c.RelPath, StartLine: c.StartLine, EndLine: c.EndLine,
					Symbol: c.Symbol, SymbolKind: c.SymbolKind, Text: c.Text, IsDef: true,
				})
			}
		}
	}

	// Usages: keyword search for the identifier, then keep chunks that actually
	// mention it as a whole word.
	ids, err := kw.Search(symbol.Name, limit*3)
	if err != nil {
		return usages, err
	}
	rows, err := db.ChunksByIDs(ids)
	if err != nil {
		return usages, err
	}
	wordRe := regexp.MustCompile(`\b` + regexp.QuoteMeta(symbol.Name) + `\b`)
	for _, id := range ids {
		c, ok := rows[id]
		if !ok || defSeen[c.ChunkID] {
			continue
		}
		if !wordRe.MatchString(c.Text) {
			continue
		}
		usages = append(usages, Usage{
			RelPath: c.RelPath, StartLine: c.StartLine, EndLine: c.EndLine,
			Symbol: c.Symbol, SymbolKind: c.SymbolKind, Text: c.Text, IsDef: false,
		})
		if len(usages) >= limit {
			break
		}
	}
	return usages, nil
}

var tokenRe = regexp.MustCompile(`[A-Za-z_][A-Za-z0-9_]+`)

func tokenize(s string) []string {
	var out []string
	for _, m := range tokenRe.FindAllString(strings.ToLower(s), -1) {
		out = append(out, m)
	}
	return out
}

// splitIdentifier breaks camelCase / snake_case / kebab-case into lowercase
// parts for overlap scoring.
func splitIdentifier(name string) []string {
	var withUnderscores strings.Builder
	for i, r := range name {
		if i > 0 && r >= 'A' && r <= 'Z' {
			withUnderscores.WriteByte('_')
		}
		withUnderscores.WriteRune(r)
	}
	fields := strings.FieldsFunc(withUnderscores.String(), func(r rune) bool {
		return r == '_' || r == '-' || r == ' '
	})
	var out []string
	for _, f := range fields {
		if f != "" {
			out = append(out, strings.ToLower(f))
		}
	}
	return out
}
