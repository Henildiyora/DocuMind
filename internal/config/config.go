// Package config loads and persists DocuMind's runtime configuration.
//
// Configuration precedence is: built-in defaults < user TOML file < explicit
// overrides passed on the command line. The file lives at the same location the
// Python version used ($XDG_CONFIG_HOME/documind/config.toml, falling back to
// ~/.config/documind/config.toml) so a shared config keeps working across the
// Python and Go builds.
package config

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	toml "github.com/pelletier/go-toml/v2"
)

// Config is the full runtime configuration. Field names use snake_case TOML
// keys so the file format matches the Python implementation. Where semantics
// diverge it is called out inline (notably embedding_model, see below).
type Config struct {
	// Local LLM (Ollama) settings.
	Model         string  `toml:"model"`
	OllamaBaseURL string  `toml:"ollama_base_url"`
	LLMTemp       float64 `toml:"llm_temperature"`
	LLMNumCtx     int     `toml:"llm_num_ctx"`
	// KeepAlive controls how long Ollama keeps a model resident after a call
	// (e.g. "5m", "0" to unload immediately). Lower idle RAM/energy use.
	KeepAlive string `toml:"keep_alive"`
	// SetupDone is true only after a successful `documind setup`.
	SetupDone bool `toml:"setup_done"`
	// OfferSetupAfterIndex gates the one-time post-index setup prompt.
	OfferSetupAfterIndex bool `toml:"offer_setup_after_index"`

	// Embedding model. NOTE: unlike the Python build (which used a fastembed
	// HuggingFace repo id like "BAAI/bge-small-en-v1.5"), this is an Ollama
	// model tag pulled locally. Default is nomic-embed-text (768-dim).
	EmbeddingModel string `toml:"embedding_model"`
	EmbeddingDim   int    `toml:"embedding_dim"`

	// Retrieval / fusion.
	TopK          int     `toml:"top_k"`
	RRFK          int     `toml:"rrf_k"` // constant used in Reciprocal Rank Fusion
	MinScoreRatio float64 `toml:"min_score_ratio"`

	// Chat / ask memory: last N user+assistant pairs kept in context.
	ChatHistoryTurns int `toml:"chat_history_turns"`

	// Chunking.
	ChunkSize    int `toml:"chunk_size"`
	ChunkOverlap int `toml:"chunk_overlap"`
	MaxFileBytes int `toml:"max_file_bytes"`

	// Index layout (project-local).
	IndexDirName string `toml:"index_dir_name"`

	// CLI / output.
	ShowLineRanges bool `toml:"show_line_ranges"`
	SnippetChars   int  `toml:"snippet_chars"`

	// Ignore-rule extensions. These EXTEND (never replace) the built-in
	// defaults in the ignorerules package.
	ExtraIgnoreDirs  []string `toml:"extra_ignore_dirs"`
	ExtraIgnoreFiles []string `toml:"extra_ignore_files"`
	ExtraIgnoreGlobs []string `toml:"extra_ignore_globs"`
}

// Default returns a Config populated with built-in defaults.
func Default() Config {
	return Config{
		Model:                "gemma3:4b",
		OllamaBaseURL:        "http://localhost:11434",
		LLMTemp:              0.1,
		LLMNumCtx:            8192,
		KeepAlive:            "5m",
		SetupDone:            false,
		OfferSetupAfterIndex: true,

		EmbeddingModel: "nomic-embed-text",
		EmbeddingDim:   768,

		TopK:          8,
		RRFK:          60,
		MinScoreRatio: 0.35,

		ChatHistoryTurns: 4,

		ChunkSize:    800,
		ChunkOverlap: 120,
		MaxFileBytes: 2_000_000,

		IndexDirName: ".documind",

		ShowLineRanges: true,
		SnippetChars:   320,
	}
}

// UserConfigPath returns the path to the user-level config file, honoring
// $XDG_CONFIG_HOME just like the Python build.
func UserConfigPath() string {
	base := os.Getenv("XDG_CONFIG_HOME")
	if base == "" {
		home, err := os.UserHomeDir()
		if err == nil {
			base = filepath.Join(home, ".config")
		} else {
			base = ".config"
		}
	}
	return filepath.Join(base, "documind", "config.toml")
}

// Load reads the user config file (if present) over the built-in defaults.
// A missing or malformed file is not an error: defaults are returned instead so
// zero-config search/index keeps working.
func Load() Config {
	cfg := Default()
	data, err := os.ReadFile(UserConfigPath())
	if err != nil {
		return cfg
	}
	// go-toml only overwrites keys present in the file, so pre-seeding cfg with
	// defaults yields the desired "defaults < file" precedence automatically.
	_ = toml.Unmarshal(data, &cfg)
	return cfg
}

// Overrides carries optional CLI flag values that take precedence over the
// file. Nil pointers mean "not set" so we can distinguish an explicit 0/"" from
// an unspecified flag.
type Overrides struct {
	Model     *string
	TopK      *int
	KeepAlive *string
}

// LoadWith loads config and applies non-nil overrides on top.
func LoadWith(ov Overrides) Config {
	cfg := Load()
	if ov.Model != nil && *ov.Model != "" {
		cfg.Model = *ov.Model
	}
	if ov.TopK != nil && *ov.TopK > 0 {
		cfg.TopK = *ov.TopK
	}
	if ov.KeepAlive != nil {
		cfg.KeepAlive = *ov.KeepAlive
	}
	return cfg
}

// IndexDirFor returns the per-project index directory (e.g. <root>/.documind).
func (c Config) IndexDirFor(projectRoot string) string {
	name := c.IndexDirName
	if name == "" {
		name = ".documind"
	}
	return filepath.Join(projectRoot, name)
}

// UpdateUser merges the given key/value updates into the user TOML file and
// writes it back. Only known keys are persisted. This mirrors the Python
// update_user_config: a simple, stable, one-key-per-line TOML file sorted by
// key so diffs stay readable.
func UpdateUser(updates map[string]any) (string, error) {
	path := UserConfigPath()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return path, err
	}

	// Start from whatever is already on disk (as a generic map) so we preserve
	// keys we do not model explicitly, then apply updates.
	merged := map[string]any{}
	if data, err := os.ReadFile(path); err == nil {
		_ = toml.Unmarshal(data, &merged)
	}
	for k, v := range updates {
		if v != nil {
			merged[k] = v
		}
	}

	keys := make([]string, 0, len(merged))
	for k := range merged {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	var b strings.Builder
	b.WriteString("# DocuMind configuration (managed by `documind setup`)\n")
	for _, k := range keys {
		b.WriteString(k)
		b.WriteString(" = ")
		b.WriteString(formatTOMLValue(merged[k]))
		b.WriteByte('\n')
	}
	if err := os.WriteFile(path, []byte(b.String()), 0o644); err != nil {
		return path, err
	}
	return path, nil
}

// WriteDefault writes a commented default config file if none exists, returning
// its path. An existing file is left untouched.
func WriteDefault() (string, error) {
	path := UserConfigPath()
	if _, err := os.Stat(path); err == nil {
		return path, nil
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return path, err
	}
	cfg := Default()
	text := "# DocuMind configuration\n" +
		"model = \"" + cfg.Model + "\"                # Ollama chat model (run: ollama pull " + cfg.Model + ")\n" +
		"ollama_base_url = \"" + cfg.OllamaBaseURL + "\"\n" +
		"llm_temperature = " + strconv.FormatFloat(cfg.LLMTemp, 'g', -1, 64) + "\n" +
		"llm_num_ctx = " + strconv.Itoa(cfg.LLMNumCtx) + "\n\n" +
		"embedding_model = \"" + cfg.EmbeddingModel + "\"   # Ollama embedding model (ollama pull " + cfg.EmbeddingModel + ")\n" +
		"embedding_dim = " + strconv.Itoa(cfg.EmbeddingDim) + "\n\n" +
		"top_k = " + strconv.Itoa(cfg.TopK) + "\n" +
		"chunk_size = " + strconv.Itoa(cfg.ChunkSize) + "\n" +
		"chunk_overlap = " + strconv.Itoa(cfg.ChunkOverlap) + "\n"
	if err := os.WriteFile(path, []byte(text), 0o644); err != nil {
		return path, err
	}
	return path, nil
}

func formatTOMLValue(v any) string {
	switch val := v.(type) {
	case bool:
		if val {
			return "true"
		}
		return "false"
	case int:
		return strconv.Itoa(val)
	case int64:
		return strconv.FormatInt(val, 10)
	case float64:
		return strconv.FormatFloat(val, 'g', -1, 64)
	case string:
		return "\"" + strings.ReplaceAll(val, "\"", "\\\"") + "\""
	default:
		return "\"" + strings.ReplaceAll(fmt.Sprintf("%v", val), "\"", "\\\"") + "\""
	}
}
