package ollama

import "testing"

// TestModelMatches guards the tag-matching fix: a requested tag must be present
// exactly, while a bare (untagged) request stays lenient so "nomic-embed-text"
// still matches the "nomic-embed-text:latest" that Ollama actually stores.
func TestModelMatches(t *testing.T) {
	cases := []struct {
		name   string // as returned by /api/tags (fully-qualified)
		target string // as requested by DocuMind config
		want   bool
	}{
		{"qwen2.5-coder:3b", "qwen2.5-coder:3b", true}, // exact tag
		{"qwen2.5-coder:1.5b", "qwen2.5-coder:3b", false}, // the bug: different tag
		{"gemma3:4b", "gemma3:12b", false},
		{"nomic-embed-text:latest", "nomic-embed-text", true}, // bare target -> latest
		{"nomic-embed-text:latest", "nomic-embed-text:latest", true},
		{"qwen2.5-coder:3b", "qwen2.5-coder", true}, // bare target -> any tag of base
		{"qwen2.5-coder:3b", "", false},
		{"", "qwen2.5-coder:3b", false},
		{"llama3.1:8b", "llama3.2:3b", false},
	}
	for _, c := range cases {
		if got := modelMatches(c.name, c.target); got != c.want {
			t.Errorf("modelMatches(%q, %q) = %v, want %v", c.name, c.target, got, c.want)
		}
	}
}
