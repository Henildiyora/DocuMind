package models

import "testing"

func TestFilterForRAM(t *testing.T) {
	// Unknown RAM returns the full catalog.
	if got := FilterForRAM(0, 0.6); len(got) != len(Catalog) {
		t.Fatalf("unknown RAM: expected full catalog (%d), got %d", len(Catalog), len(got))
	}

	// A tiny budget still returns at least the three smallest models.
	if got := FilterForRAM(0.5, 0.6); len(got) < 3 {
		t.Fatalf("tiny RAM: expected >=3 fallback models, got %d", len(got))
	}

	// Every returned model must fit the budget when some do.
	budgetRAM := 8.0
	headroom := 0.6
	for _, m := range FilterForRAM(budgetRAM, headroom) {
		if m.RAMGb > budgetRAM*headroom {
			t.Fatalf("model %s (%.0f GB) exceeds budget %.1f", m.Tag, m.RAMGb, budgetRAM*headroom)
		}
	}
}

func TestRecommendForHardware(t *testing.T) {
	// With plenty of RAM the recommendation should be a quality-tier model.
	rec := RecommendForHardware(64, 5000, 2_000_000)
	if rec.Tradeoff != "quality" {
		t.Fatalf("high RAM: expected a quality model, got %s (%s)", rec.Tag, rec.Tradeoff)
	}

	// The recommended model must always fit RAM.
	ram := 8.0
	rec = RecommendForHardware(ram, 300, 100_000)
	if rec.RAMGb > ram*0.6 {
		t.Fatalf("recommended %s needs %.0f GB, over budget for %.0f GB RAM", rec.Tag, rec.RAMGb, ram)
	}

	// A very small project nudges toward a faster model when a lighter option of
	// equal-or-lower tier exists.
	small := RecommendForHardware(8, 10, 500)
	big := RecommendForHardware(8, 5000, 2_000_000)
	if tradeoffRank[small.Tradeoff] > tradeoffRank[big.Tradeoff] {
		t.Fatalf("small project should not pick a heavier tier than a large one: %s vs %s", small.Tag, big.Tag)
	}
}
