// Package models holds the curated Ollama model catalog and hardware-aware
// recommendation logic. All models are free and run locally via Ollama. Ported
// and expanded from the Python models module (models.py).
package models

import (
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
)

// Spec describes a catalog model.
type Spec struct {
	Tier     string  // tiny | small | deep
	Tag      string  // Ollama tag, e.g. "gemma3:4b"
	SizeGB   float64 // approximate on-disk footprint
	RAMGb    float64 // approximate RAM needed while loaded
	Family   string
	BestFor  string
	Tradeoff string // fast | balanced | quality
}

// Catalog is the full set shown by `documind models` and filtered by `setup`.
var Catalog = []Spec{
	{Tier: "tiny", Tag: "qwen2.5-coder:0.5b", SizeGB: 0.4, RAMGb: 1.0, Family: "Qwen2.5 Coder", BestFor: "Smallest, near-instant answers, best on any laptop", Tradeoff: "fast"},
	{Tier: "tiny", Tag: "qwen2.5-coder:1.5b", SizeGB: 1.0, RAMGb: 2.0, Family: "Qwen2.5 Coder", BestFor: "Fast coding answers that still fit tiny machines", Tradeoff: "fast"},
	{Tier: "tiny", Tag: "llama3.2:1b", SizeGB: 1.3, RAMGb: 2.0, Family: "Llama 3.2", BestFor: "Tiny general chat; quick and light", Tradeoff: "fast"},
	{Tier: "tiny", Tag: "phi3.5:3.8b", SizeGB: 2.2, RAMGb: 4.0, Family: "Phi 3.5", BestFor: "Compact reasoning from Microsoft Phi", Tradeoff: "balanced"},
	{Tier: "small", Tag: "llama3.2:3b", SizeGB: 2.0, RAMGb: 4.0, Family: "Llama 3.2", BestFor: "Solid general Q&A without a big footprint", Tradeoff: "balanced"},
	{Tier: "small", Tag: "qwen2.5-coder:3b", SizeGB: 1.9, RAMGb: 4.0, Family: "Qwen2.5 Coder", BestFor: "Better code detail than 1.5b, still laptop-friendly", Tradeoff: "balanced"},
	{Tier: "small", Tag: "gemma3:4b", SizeGB: 3.3, RAMGb: 6.0, Family: "Gemma 3", BestFor: "Richer answers for mid-sized repos", Tradeoff: "balanced"},
	{Tier: "deep", Tag: "qwen2.5-coder:7b", SizeGB: 4.7, RAMGb: 8.0, Family: "Qwen2.5 Coder", BestFor: "Deeper code reasoning on larger repos", Tradeoff: "quality"},
	{Tier: "deep", Tag: "llama3.1:8b", SizeGB: 4.7, RAMGb: 10.0, Family: "Llama 3.1", BestFor: "Strong general answers if your machine has 16GB+", Tradeoff: "quality"},
	{Tier: "deep", Tag: "gemma3:12b", SizeGB: 8.1, RAMGb: 12.0, Family: "Gemma 3", BestFor: "Higher-quality synthesis when you have spare RAM", Tradeoff: "quality"},
}

var tradeoffRank = map[string]int{"fast": 0, "balanced": 1, "quality": 2}

// FindByTag returns the catalog entry for an exact tag, if present.
func FindByTag(tag string) (Spec, bool) {
	tag = strings.TrimSpace(tag)
	for _, m := range Catalog {
		if m.Tag == tag {
			return m, true
		}
	}
	return Spec{}, false
}

// DetectSystemRAMGB returns best-effort total system RAM in GB (stdlib only).
// Returns 0 when it cannot be determined.
func DetectSystemRAMGB() float64 {
	switch runtime.GOOS {
	case "darwin":
		out, err := exec.Command("sysctl", "-n", "hw.memsize").Output()
		if err != nil {
			return 0
		}
		bytes, err := strconv.ParseInt(strings.TrimSpace(string(out)), 10, 64)
		if err != nil {
			return 0
		}
		return float64(bytes) / (1024 * 1024 * 1024)
	case "linux":
		data, err := os.ReadFile("/proc/meminfo")
		if err != nil {
			return 0
		}
		for _, line := range strings.Split(string(data), "\n") {
			if strings.HasPrefix(line, "MemTotal:") {
				fields := strings.Fields(line)
				if len(fields) >= 2 {
					kb, err := strconv.ParseInt(fields[1], 10, 64)
					if err == nil {
						return float64(kb) / (1024 * 1024)
					}
				}
			}
		}
	}
	return 0
}

// DetectGPUHint returns a short GPU hint string if easily detectable, else "".
func DetectGPUHint() string {
	if _, err := exec.LookPath("nvidia-smi"); err == nil {
		out, err := exec.Command("nvidia-smi", "--query-gpu=name", "--format=csv,noheader").Output()
		if err == nil {
			line := strings.TrimSpace(string(out))
			if line != "" {
				return strings.SplitN(line, "\n", 2)[0]
			}
		}
		return "NVIDIA GPU"
	}
	if runtime.GOOS == "darwin" {
		return "Apple Silicon / Metal"
	}
	return ""
}

// FilterForRAM keeps catalog entries that fit in availableRAM*headroom. When RAM
// is unknown (0), the full catalog is returned. Always returns at least the
// three smallest models.
func FilterForRAM(availableRAMGB, headroom float64) []Spec {
	if availableRAMGB <= 0 {
		return append([]Spec(nil), Catalog...)
	}
	budget := availableRAMGB * headroom
	var fitted []Spec
	for _, m := range Catalog {
		if m.RAMGb <= budget {
			fitted = append(fitted, m)
		}
	}
	if len(fitted) == 0 {
		return append([]Spec(nil), Catalog[:3]...)
	}
	return fitted
}

// RecommendForHardware picks the best-quality model that comfortably fits RAM.
// Project size only nudges very small projects one step toward a faster model.
func RecommendForHardware(availableRAMGB float64, fileCount, totalLOC int) Spec {
	candidates := FilterForRAM(availableRAMGB, 0.6)
	ranked := append([]Spec(nil), candidates...)
	sortByQuality(ranked)
	if len(ranked) == 0 {
		return Catalog[0]
	}
	if fileCount <= 200 && totalLOC <= 50_000 && len(ranked) > 1 {
		top, alt := ranked[0], ranked[1]
		if tradeoffRank[top.Tradeoff] > tradeoffRank[alt.Tradeoff] {
			return alt
		}
	}
	return ranked[0]
}

// sortByQuality sorts descending by tradeoff quality, then RAM, then size, so the
// best model within a RAM band comes first.
func sortByQuality(specs []Spec) {
	for i := 1; i < len(specs); i++ {
		for j := i; j > 0 && less(specs[j-1], specs[j]); j-- {
			specs[j-1], specs[j] = specs[j], specs[j-1]
		}
	}
}

// familyPref ranks families by how well they follow DocuMind's "explain, cite,
// do not dump code" instructions at small sizes. Higher is preferred. Used only
// as a tiebreak within the same quality tier and RAM band, so it never picks a
// heavier or lower-quality model - it just favors a better instruction-follower
// (e.g. qwen2.5-coder over phi3.5 at the same ~4 GB tier).
func familyPref(family string) int {
	switch family {
	case "Qwen2.5 Coder":
		return 3
	case "Llama 3.2", "Llama 3.1":
		return 2
	case "Gemma 3":
		return 1
	default: // Phi 3.5, custom, etc.
		return 0
	}
}

// less reports whether a should sort AFTER b (i.e. a is "smaller/worse").
func less(a, b Spec) bool {
	ra, rb := tradeoffRank[a.Tradeoff], tradeoffRank[b.Tradeoff]
	if ra != rb {
		return ra < rb
	}
	if a.RAMGb != b.RAMGb {
		return a.RAMGb < b.RAMGb
	}
	if fa, fb := familyPref(a.Family), familyPref(b.Family); fa != fb {
		return fa < fb
	}
	return a.SizeGB < b.SizeGB
}
