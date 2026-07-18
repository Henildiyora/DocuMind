// Package ignorerules centralizes which directories and files are skipped while
// walking a project. The default sets are ported directly from the Python
// implementation (config.py) including the hard-won fixes for generated report
// dumps and virtualenv directories. Config-supplied extras EXTEND these
// defaults; they never replace them.
package ignorerules

import (
	"path/filepath"
	"strings"
)

// SupportedExtensions is the set of file extensions DocuMind will index. Ported
// from SUPPORTED_EXTENSIONS in config.py.
var SupportedExtensions = toSet([]string{
	".py", ".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs",
	".java", ".kt", ".scala",
	".cpp", ".cc", ".c", ".h", ".hpp", ".cs",
	".go", ".rs", ".php", ".rb", ".swift", ".m",
	".json", ".yaml", ".yml", ".toml", ".xml", ".ini",
	".html", ".css", ".scss", ".sass",
	".sql", ".md", ".mdx", ".rst", ".txt",
	".sh", ".bash", ".zsh", ".fish",
	".dockerfile", ".makefile",
})

// defaultIgnoreDirs mirrors IGNORE_DIRS in config.py.
var defaultIgnoreDirs = toSet([]string{
	// VCS
	".git", ".hg", ".svn",
	// Package / dependency dirs
	"node_modules", "bower_components",
	"venv", ".venv", "env", "DocuMind_venv",
	"site-packages", "dist-packages",
	"vendor", // Go / PHP
	"Pods",   // iOS / CocoaPods
	// Caches
	"__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
	".cache", ".tox", ".eggs",
	"hf_cache", ".parcel-cache", ".turbo", ".svelte-kit",
	// Editors / IDEs
	".idea", ".vscode",
	// Build / generated output
	"build", "dist", "target", "out", "out-tsc",
	".next", ".nuxt", ".vercel", ".gradle",
	"DerivedData", ".terraform",
	// Docs / coverage artifacts (the stuff that polluted search before)
	"htmlcov", "coverage", ".nyc_output",
	"site", "_site", "_build", "public",
	// Generated report dumps (interview/RAG output, not source of truth)
	"generated_reports", "generated",
	// DocuMind itself
	".documind",
})

// defaultIgnoreFiles mirrors IGNORE_FILES in config.py.
var defaultIgnoreFiles = toSet([]string{
	"coverage.xml", "coverage.json", ".coverage", "lcov.info",
	"package-lock.json", "pnpm-lock.yaml", "yarn.lock",
	"poetry.lock", "Pipfile.lock",
	"Cargo.lock", "go.sum",
	"report.json",
	".DS_Store", "Thumbs.db",
})

// defaultIgnoreGlobs mirrors IGNORE_FILE_GLOBS in config.py.
var defaultIgnoreGlobs = []string{
	"*.min.js",
	"*.min.css",
	"*.map",
	"*.bundle.js",
	"*.bundle.css",
	"*.lock",
}

// Rules is an ignore matcher with defaults plus any config-supplied extras.
type Rules struct {
	dirs  map[string]struct{}
	files map[string]struct{}
	globs []string
}

// New builds a Rules value from the defaults extended by the provided extras.
func New(extraDirs, extraFiles, extraGlobs []string) *Rules {
	r := &Rules{
		dirs:  cloneSet(defaultIgnoreDirs),
		files: cloneSet(defaultIgnoreFiles),
		globs: append([]string(nil), defaultIgnoreGlobs...),
	}
	for _, d := range extraDirs {
		if d = strings.TrimSpace(d); d != "" {
			r.dirs[d] = struct{}{}
		}
	}
	for _, f := range extraFiles {
		if f = strings.TrimSpace(f); f != "" {
			r.files[f] = struct{}{}
		}
	}
	for _, g := range extraGlobs {
		if g = strings.TrimSpace(g); g != "" {
			r.globs = append(r.globs, g)
		}
	}
	return r
}

// IsIgnoredDir reports whether a directory name should be skipped. Exact names
// in the ignore set are skipped, plus any dotfile directory and any directory
// ending in _venv/-venv (e.g. "github_summarizer_venv"). Ported from
// is_ignored_dirname in config.py.
func (r *Rules) IsIgnoredDir(name string) bool {
	if _, ok := r.dirs[name]; ok {
		return true
	}
	if strings.HasPrefix(name, ".") {
		return true
	}
	return strings.HasSuffix(name, "_venv") || strings.HasSuffix(name, "-venv")
}

// IsIgnoredFile reports whether a filename should be skipped: dotfiles, exact
// names in the ignore set, or any configured glob match.
func (r *Rules) IsIgnoredFile(name string) bool {
	if strings.HasPrefix(name, ".") {
		return true
	}
	if _, ok := r.files[name]; ok {
		return true
	}
	for _, g := range r.globs {
		if ok, _ := filepath.Match(g, name); ok {
			return true
		}
	}
	return false
}

// ShouldInclude reports whether a file is a candidate for indexing based on its
// extension, with the same Dockerfile/Makefile special-casing as chunker.py.
func ShouldInclude(name string) bool {
	lower := strings.ToLower(name)
	switch lower {
	case "dockerfile", "makefile", "gnumakefile":
		return true
	}
	ext := strings.ToLower(filepath.Ext(name))
	_, ok := SupportedExtensions[ext]
	return ok
}

func toSet(items []string) map[string]struct{} {
	m := make(map[string]struct{}, len(items))
	for _, it := range items {
		m[it] = struct{}{}
	}
	return m
}

func cloneSet(src map[string]struct{}) map[string]struct{} {
	dst := make(map[string]struct{}, len(src))
	for k := range src {
		dst[k] = struct{}{}
	}
	return dst
}
