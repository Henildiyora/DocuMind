package chunker

import (
	"context"
	"strings"

	"github.com/Henildiyora/DocuMind/internal/config"
	sitter "github.com/smacker/go-tree-sitter"
	"github.com/smacker/go-tree-sitter/c"
	"github.com/smacker/go-tree-sitter/cpp"
	"github.com/smacker/go-tree-sitter/golang"
	"github.com/smacker/go-tree-sitter/java"
	"github.com/smacker/go-tree-sitter/javascript"
	"github.com/smacker/go-tree-sitter/python"
	"github.com/smacker/go-tree-sitter/rust"
	typescript "github.com/smacker/go-tree-sitter/typescript/typescript"
)

// GrammarFor returns the tree-sitter grammar for a language label, if one is
// compiled in. Exposed for the navigation package's symbol resolution.
func GrammarFor(language string) (*sitter.Language, bool) {
	return treeSitterLang(language)
}

// treeSitterLang returns the tree-sitter grammar for a language label, if we
// have one compiled in. Languages without a grammar use the line-window
// fallback chunker.
func treeSitterLang(language string) (*sitter.Language, bool) {
	switch language {
	case "go":
		return golang.GetLanguage(), true
	case "python":
		return python.GetLanguage(), true
	case "javascript":
		return javascript.GetLanguage(), true
	case "typescript":
		return typescript.GetLanguage(), true
	case "java":
		return java.GetLanguage(), true
	case "c":
		return c.GetLanguage(), true
	case "cpp":
		return cpp.GetLanguage(), true
	case "rust":
		return rust.GetLanguage(), true
	default:
		return nil, false
	}
}

// significantTypes lists, per language, the top-level AST node types that become
// their own chunk (functions, classes, structs, etc.). Everything else at the
// top level (imports, stray statements) is grouped into fallback-style chunks so
// no source is lost.
var significantTypes = map[string]map[string]bool{
	"go": {
		"function_declaration": true,
		"method_declaration":   true,
		"type_declaration":     true,
	},
	"python": {
		"function_definition":  true,
		"class_definition":     true,
		"decorated_definition": true,
	},
	"javascript": {
		"function_declaration":           true,
		"generator_function_declaration": true,
		"class_declaration":              true,
		"method_definition":              true,
		"lexical_declaration":            true,
		"variable_declaration":           true,
		"export_statement":               true,
	},
	"typescript": {
		"function_declaration":           true,
		"generator_function_declaration": true,
		"class_declaration":              true,
		"abstract_class_declaration":     true,
		"method_definition":              true,
		"interface_declaration":          true,
		"type_alias_declaration":         true,
		"enum_declaration":               true,
		"lexical_declaration":            true,
		"variable_declaration":           true,
		"export_statement":               true,
	},
	"java": {
		"class_declaration":       true,
		"interface_declaration":   true,
		"enum_declaration":        true,
		"record_declaration":      true,
		"method_declaration":      true,
		"constructor_declaration": true,
	},
	"c": {
		"function_definition": true,
		"struct_specifier":    true,
		"enum_specifier":      true,
		"union_specifier":     true,
		"type_definition":     true,
	},
	"cpp": {
		"function_definition":  true,
		"class_specifier":      true,
		"struct_specifier":     true,
		"enum_specifier":       true,
		"union_specifier":      true,
		"namespace_definition": true,
		"template_declaration": true,
	},
	"rust": {
		"function_item":    true,
		"struct_item":      true,
		"impl_item":        true,
		"trait_item":       true,
		"enum_item":        true,
		"mod_item":         true,
		"type_item":        true,
		"macro_definition": true,
	},
}

// kindByType maps an AST node type to a short human-readable symbol kind.
var kindByType = map[string]string{
	"function_declaration":           "function",
	"generator_function_declaration": "function",
	"function_definition":            "function",
	"function_item":                  "function",
	"method_declaration":             "method",
	"method_definition":              "method",
	"constructor_declaration":        "constructor",
	"class_declaration":              "class",
	"abstract_class_declaration":     "class",
	"class_definition":               "class",
	"class_specifier":                "class",
	"struct_specifier":               "struct",
	"struct_item":                    "struct",
	"interface_declaration":          "interface",
	"trait_item":                     "trait",
	"impl_item":                      "impl",
	"enum_declaration":               "enum",
	"enum_item":                      "enum",
	"enum_specifier":                 "enum",
	"union_specifier":                "union",
	"record_declaration":             "record",
	"type_declaration":               "type",
	"type_alias_declaration":         "type",
	"type_item":                      "type",
	"type_definition":                "type",
	"namespace_definition":           "namespace",
	"mod_item":                       "module",
	"macro_definition":               "macro",
	"lexical_declaration":            "binding",
	"variable_declaration":           "binding",
}

// astChunks parses source with the given grammar and returns chunks aligned to
// top-level declaration boundaries. langLabel selects the significant node-type
// set. It returns nil on parse failure or an empty tree so the caller can fall
// back to line-window chunking.
func astChunks(source []byte, lang *sitter.Language, langLabel string, cfg config.Config) []chunkPiece {
	parser := sitter.NewParser()
	parser.SetLanguage(lang)
	tree, err := parser.ParseCtx(context.Background(), nil, source)
	if err != nil || tree == nil {
		return nil
	}
	defer tree.Close()

	root := tree.RootNode()
	if root == nil || root.NamedChildCount() == 0 {
		return nil
	}

	sig := significantTypes[langLabel]

	var pieces []chunkPiece
	// buffer accumulates consecutive non-significant top-level nodes so leading
	// imports and stray statements still get indexed.
	var buf []string
	bufStart, bufEnd := 0, 0
	flush := func() {
		if len(buf) == 0 {
			return
		}
		text := strings.Join(buf, "\n")
		if strings.TrimSpace(text) != "" {
			pieces = append(pieces, chunkPiece{startLine: bufStart, endLine: bufEnd, text: text})
		}
		buf = nil
	}

	count := int(root.NamedChildCount())
	for i := 0; i < count; i++ {
		node := root.NamedChild(i)
		if node == nil {
			continue
		}
		nodeType := node.Type()
		startLine := int(node.StartPoint().Row) + 1
		endLine := endLineOf(node)

		if sig[nodeType] {
			flush()
			target := unwrapDeclaration(node)
			symbol := symbolName(target, source)
			kind := kindByType[target.Type()]
			if kind == "" {
				kind = kindByType[nodeType]
			}
			text := node.Content(source)
			maxChars := cfg.ChunkSize * 6
			if len(text) > maxChars && maxChars > 0 {
				// Split a very large declaration but keep its symbol on each
				// part, offsetting line numbers into the original file.
				for _, part := range fallbackChunks(text, cfg.ChunkSize, cfg.ChunkOverlap) {
					pieces = append(pieces, chunkPiece{
						startLine:  startLine + part.startLine - 1,
						endLine:    startLine + part.endLine - 1,
						symbol:     symbol,
						symbolKind: kind,
						text:       part.text,
					})
				}
			} else {
				pieces = append(pieces, chunkPiece{
					startLine:  startLine,
					endLine:    endLine,
					symbol:     symbol,
					symbolKind: kind,
					text:       text,
				})
			}
			continue
		}

		// Non-significant: accumulate into the buffer.
		if len(buf) == 0 {
			bufStart = startLine
		}
		bufEnd = endLine
		buf = append(buf, node.Content(source))
	}
	flush()
	return pieces
}

// unwrapDeclaration digs through wrapper nodes (Python decorators, JS/TS export
// statements) to the underlying declaration so we tag the real symbol.
func unwrapDeclaration(node *sitter.Node) *sitter.Node {
	switch node.Type() {
	case "decorated_definition", "export_statement":
		count := int(node.NamedChildCount())
		for i := 0; i < count; i++ {
			child := node.NamedChild(i)
			if child == nil {
				continue
			}
			if kindByType[child.Type()] != "" {
				return child
			}
		}
	}
	return node
}

// symbolName extracts the declared identifier from a declaration node. It first
// tries the grammar's "name" field, then falls back to the first identifier-like
// named child.
func symbolName(node *sitter.Node, source []byte) string {
	if node == nil {
		return ""
	}
	if named := node.ChildByFieldName("name"); named != nil {
		return strings.TrimSpace(named.Content(source))
	}
	count := int(node.NamedChildCount())
	for i := 0; i < count; i++ {
		child := node.NamedChild(i)
		if child == nil {
			continue
		}
		switch child.Type() {
		case "identifier", "type_identifier", "field_identifier", "constant", "name":
			return strings.TrimSpace(child.Content(source))
		}
	}
	return ""
}

// endLineOf returns the 1-based inclusive last line covered by a node. When a
// node ends exactly at a line boundary (column 0) its content stops on the
// previous line, so we do not count the boundary line.
func endLineOf(node *sitter.Node) int {
	end := node.EndPoint()
	if end.Column == 0 && end.Row > node.StartPoint().Row {
		return int(end.Row)
	}
	return int(end.Row) + 1
}
