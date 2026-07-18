// Package structural gathers factual, LLM-free information about a project's
// layout: folder/file counts, per-language breakdown, README contents, and
// detected entry points. STRUCTURAL queries ("how many folders", "what is this
// project") are answered from these exact facts; the LLM only phrases them and
// is explicitly told not to invent numbers.
package structural

import (
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/Henildiyora/DocuMind/internal/chunker"
	"github.com/Henildiyora/DocuMind/internal/config"
	"github.com/Henildiyora/DocuMind/internal/ignorerules"
)

// Facts is the computed structural summary of a project.
type Facts struct {
	Root           string
	FileCount      int            // indexable (supported) files
	FolderCount    int            // non-ignored directories (excluding the root)
	LanguageCounts map[string]int // language label -> file count
	ReadmePath     string
	ReadmeExcerpt  string
	EntryPoints    []string // known entry-point files found at the root
	TopLevelDirs   []string // immediate child directory names (non-ignored)
}

// knownEntryPoints are files that signal a project's type/entry, checked at the
// repository root. Mirrors the entry-point list in query_understand.py.
var knownEntryPoints = []string{
	"main.go", "go.mod",
	"main.py", "app.py", "pyproject.toml", "setup.py", "requirements.txt",
	"index.js", "index.ts", "package.json",
	"Cargo.toml",
	"Dockerfile", "Makefile",
	"pom.xml", "build.gradle",
}

var readmeNames = []string{"README.md", "README.rst", "README.txt", "README", "readme.md"}

// GatherFacts walks the project (honoring ignore rules) and returns its facts.
func GatherFacts(root string, cfg config.Config) (Facts, error) {
	absRoot, err := filepath.Abs(root)
	if err != nil {
		return Facts{}, err
	}
	rules := ignorerules.New(cfg.ExtraIgnoreDirs, cfg.ExtraIgnoreFiles, cfg.ExtraIgnoreGlobs)

	facts := Facts{
		Root:           absRoot,
		LanguageCounts: map[string]int{},
	}
	var topDirs []string

	walkErr := filepath.WalkDir(absRoot, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			if d != nil && d.IsDir() {
				return fs.SkipDir
			}
			return nil
		}
		if d.IsDir() {
			if path == absRoot {
				return nil
			}
			if rules.IsIgnoredDir(d.Name()) {
				return fs.SkipDir
			}
			facts.FolderCount++
			if filepath.Dir(path) == absRoot {
				topDirs = append(topDirs, d.Name())
			}
			return nil
		}
		name := d.Name()
		if rules.IsIgnoredFile(name) || !ignorerules.ShouldInclude(name) {
			return nil
		}
		facts.FileCount++
		facts.LanguageCounts[chunker.DetectLanguage(path)]++
		return nil
	})
	if walkErr != nil {
		return facts, walkErr
	}

	sort.Strings(topDirs)
	facts.TopLevelDirs = topDirs

	// Entry points present at the root.
	for _, ep := range knownEntryPoints {
		if _, err := os.Stat(filepath.Join(absRoot, ep)); err == nil {
			facts.EntryPoints = append(facts.EntryPoints, ep)
		}
	}

	// README excerpt from the first match at the root.
	for _, rn := range readmeNames {
		p := filepath.Join(absRoot, rn)
		if data, err := os.ReadFile(p); err == nil {
			facts.ReadmePath = rn
			facts.ReadmeExcerpt = excerpt(string(data), 2000)
			break
		}
	}

	return facts, nil
}

// Summary renders the facts as a compact, deterministic text block suitable for
// feeding to the answer-synthesis prompt as ground truth.
func (f Facts) Summary() string {
	var b strings.Builder
	b.WriteString("Project root: " + filepath.Base(f.Root) + "\n")
	b.WriteString("Indexable files: " + itoa(f.FileCount) + "\n")
	b.WriteString("Folders (excluding ignored): " + itoa(f.FolderCount) + "\n")

	if len(f.LanguageCounts) > 0 {
		b.WriteString("Language breakdown:\n")
		for _, kv := range sortedCounts(f.LanguageCounts) {
			b.WriteString("  - " + kv.key + ": " + itoa(kv.val) + " files\n")
		}
	}
	if len(f.TopLevelDirs) > 0 {
		b.WriteString("Top-level directories: " + strings.Join(f.TopLevelDirs, ", ") + "\n")
	}
	if len(f.EntryPoints) > 0 {
		b.WriteString("Entry points: " + strings.Join(f.EntryPoints, ", ") + "\n")
	}
	if f.ReadmeExcerpt != "" {
		b.WriteString("\nREADME (" + f.ReadmePath + ") excerpt:\n")
		b.WriteString(f.ReadmeExcerpt + "\n")
	}
	return b.String()
}

type kvPair struct {
	key string
	val int
}

func sortedCounts(m map[string]int) []kvPair {
	pairs := make([]kvPair, 0, len(m))
	for k, v := range m {
		pairs = append(pairs, kvPair{k, v})
	}
	sort.Slice(pairs, func(i, j int) bool {
		if pairs[i].val != pairs[j].val {
			return pairs[i].val > pairs[j].val
		}
		return pairs[i].key < pairs[j].key
	})
	return pairs
}

func excerpt(s string, maxLen int) string {
	s = strings.TrimSpace(s)
	if len(s) <= maxLen {
		return s
	}
	return strings.TrimSpace(s[:maxLen]) + "\n..."
}

func itoa(n int) string {
	return strconv.Itoa(n)
}
