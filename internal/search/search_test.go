package search

import (
	"testing"

	"github.com/Henildiyora/DocuMind/internal/config"
)

// TestDockerfileRanksAboveArtifact reproduces the regression where a noisy
// generated report.json outranked a real Dockerfile for "explain the docker
// file". After artifact demotion + filename boosting, the Dockerfile must win.
func TestDockerfileRanksAboveArtifact(t *testing.T) {
	// report.json starts with a HIGHER raw score than the Dockerfile.
	hits := []SearchHit{
		{ChunkID: "1", RelPath: "generated_reports/report.json", Score: 0.9},
		{ChunkID: "2", RelPath: "Dockerfile", Score: 0.5},
		{ChunkID: "3", RelPath: "app/main.go", Score: 0.4},
	}
	cfg := config.Default()
	refined := Refine(hits, "can you explain me the docker file", cfg)
	if len(refined) == 0 {
		t.Fatal("expected hits")
	}
	if refined[0].RelPath != "Dockerfile" {
		t.Fatalf("expected Dockerfile first, got %s", refined[0].RelPath)
	}
}

func TestIsArtifactPath(t *testing.T) {
	cases := map[string]bool{
		"generated_reports/report.json": true,
		"report.json":                   true,
		"htmlcov/index.html":            true,
		"app/main.go":                   false,
		"src/report_builder.go":         false,
	}
	for path, want := range cases {
		if got := IsArtifactPath(path); got != want {
			t.Errorf("IsArtifactPath(%q) = %v, want %v", path, got, want)
		}
	}
}

func TestFilenameHints(t *testing.T) {
	hints := FilenameHints("can you explain me the docker file")
	if !contains(hints, "dockerfile") {
		t.Errorf("expected dockerfile hint, got %v", hints)
	}
	hints = FilenameHints("what does main.go do")
	if !contains(hints, "main.go") {
		t.Errorf("expected main.go hint, got %v", hints)
	}
}

// TestFuseRRF checks that an id appearing high in both lists beats ids appearing
// in only one.
func TestFuseRRF(t *testing.T) {
	bm25 := []string{"a", "b", "c"}
	vec := []string{"b", "d", "a"}
	fused := FuseRRF([][]string{bm25, vec}, 60, 10)
	if len(fused) == 0 {
		t.Fatal("expected fused results")
	}
	// "b" is rank 2 in bm25 and rank 1 in vec; "a" is rank 1 in bm25 and rank 3
	// in vec. Both appear twice; the top result must be one of them.
	top := fused[0].id
	if top != "a" && top != "b" {
		t.Fatalf("expected 'a' or 'b' on top, got %q", top)
	}
	// Single-list ids (c, d) must score below the two-list ids.
	scoreByID := map[string]float64{}
	for _, f := range fused {
		scoreByID[f.id] = f.score
	}
	if scoreByID["a"] <= scoreByID["c"] {
		t.Errorf("expected 'a' to outscore 'c'")
	}
	if scoreByID["b"] <= scoreByID["d"] {
		t.Errorf("expected 'b' to outscore 'd'")
	}
}
