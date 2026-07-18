package router

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/Henildiyora/DocuMind/internal/config"
	"github.com/Henildiyora/DocuMind/internal/structural"
)

// TestStructural_FolderCount verifies that a folder-counting question is
// classified as STRUCTURAL by the heuristic path alone (nil Ollama client, so no
// HTTP call is possible) and that the real folder count is computed.
func TestStructural_FolderCount(t *testing.T) {
	root := t.TempDir()
	// Create a known folder structure: 3 folders (a, a/b, c).
	mkdirs := []string{"a", filepath.Join("a", "b"), "c"}
	for _, d := range mkdirs {
		if err := os.MkdirAll(filepath.Join(root, d), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	// A source file so the walk has something indexable.
	if err := os.WriteFile(filepath.Join(root, "a", "main.go"), []byte("package a\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	query := "how many folders are there in this project"

	// nil client: if the heuristic did NOT decide, Classify would default to
	// ContentQA (it cannot call an LLM), so asserting Structural proves the
	// heuristic path was taken with no LLM involvement.
	decision := Classify(context.Background(), nil, query, Options{})
	if decision.Category != Structural {
		t.Fatalf("expected Structural, got %s", decision.Category)
	}
	if !decision.Heuristic {
		t.Fatalf("expected heuristic classification, got LLM label %q", decision.LLMLabel)
	}

	facts, err := structural.GatherFacts(root, config.Default())
	if err != nil {
		t.Fatal(err)
	}
	if facts.FolderCount != 3 {
		t.Fatalf("expected 3 folders, got %d (dirs: %v)", facts.FolderCount, facts.TopLevelDirs)
	}
}

func TestOverviewIntent(t *testing.T) {
	overview := []string{
		"how many files are in this project",
		"how many folders are there",
		"explain this project",
		"what is this repo",
		"give me an overview",
		"project structure",
		"what languages are used",
	}
	for _, q := range overview {
		if !IsOverviewIntent(q) {
			t.Errorf("expected overview intent for %q", q)
		}
	}
	notOverview := []string{
		"where is the rate limiter defined",
		"explain the docker file",
		"how does the auth middleware validate tokens",
	}
	for _, q := range notOverview {
		if IsOverviewIntent(q) {
			t.Errorf("did not expect overview intent for %q", q)
		}
	}
}

// TestClassify_FileForcesCodeNav confirms --file forces CODE_NAVIGATION via the
// heuristic path.
func TestClassify_FileForcesCodeNav(t *testing.T) {
	d := Classify(context.Background(), nil, "where does this parameter come from", Options{HasFile: true})
	if d.Category != CodeNavigation || !d.Heuristic {
		t.Fatalf("expected heuristic CodeNavigation, got %s (heuristic=%v)", d.Category, d.Heuristic)
	}
}
