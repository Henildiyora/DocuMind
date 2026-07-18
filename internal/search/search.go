// Package search implements hybrid retrieval: keyword (bleve) and dense vectors
// (chromem-go) fused with Reciprocal Rank Fusion, followed by re-ranking that
// boosts filename matches and demotes generated artifacts. All ranking logic is
// pure and unit-testable; the LLM is never involved here (it only phrases the
// final answer, in a separate call site).
package search

import (
	"context"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/Henildiyora/DocuMind/internal/config"
	"github.com/Henildiyora/DocuMind/internal/kwindex"
	"github.com/Henildiyora/DocuMind/internal/ollama"
	"github.com/Henildiyora/DocuMind/internal/store"
	"github.com/Henildiyora/DocuMind/internal/vectorstore"
)

// SearchHit is a single ranked retrieval result.
type SearchHit struct {
	ChunkID    string
	RelPath    string
	Language   string
	StartLine  int
	EndLine    int
	Symbol     string
	SymbolKind string
	Text       string
	Score      float64
	BM25Rank   int // 1-based rank in keyword results, 0 if absent
	VectorRank int // 1-based rank in vector results, 0 if absent
}

// Retriever bundles the stores needed to answer a query. The Ollama client is
// optional: when nil or unreachable, retrieval degrades to keyword-only.
type Retriever struct {
	DB     *store.DB
	KW     *kwindex.Index
	Vec    *vectorstore.Store
	Client *ollama.Client // may be nil
	Cfg    config.Config
}

// scored is an intermediate fused result.
type scored struct {
	id    string
	score float64
	ranks []int
}

// FuseRRF merges N ranked id lists with Reciprocal Rank Fusion. k is the RRF
// constant (typically 60); topN caps the output. Returns fused entries sorted by
// score descending, each carrying its 1-based rank within every input list (0 =
// absent from that list).
func FuseRRF(rankings [][]string, k, topN int) []scored {
	scores := map[string]float64{}
	perSource := map[string][]int{}
	nLists := len(rankings)

	for srcIdx, ranking := range rankings {
		for rank, id := range ranking {
			scores[id] += 1.0 / float64(k+rank+1)
			if _, ok := perSource[id]; !ok {
				perSource[id] = make([]int, nLists)
			}
			perSource[id][srcIdx] = rank + 1
		}
	}

	fused := make([]scored, 0, len(scores))
	for id, sc := range scores {
		fused = append(fused, scored{id: id, score: sc, ranks: perSource[id]})
	}
	sort.SliceStable(fused, func(i, j int) bool { return fused[i].score > fused[j].score })
	if topN > 0 && len(fused) > topN {
		fused = fused[:topN]
	}
	return fused
}

// Search runs keyword + vector retrieval and returns RRF-merged hits.
func (r *Retriever) Search(ctx context.Context, query string, topK int) ([]SearchHit, error) {
	k := topK
	if k <= 0 {
		k = r.Cfg.TopK
	}
	pool := k * 4
	if pool < 20 {
		pool = 20
	}

	bm25IDs, err := r.KW.Search(query, pool)
	if err != nil {
		return nil, err
	}

	var vecIDs []string
	if r.Client != nil && r.Vec != nil && r.Vec.Count() > 0 {
		if emb, err := r.Client.EmbedQuery(ctx, query); err == nil {
			vecIDs, _ = r.Vec.Query(ctx, emb, pool)
		}
	}

	if len(bm25IDs) == 0 && len(vecIDs) == 0 {
		return nil, nil
	}

	fused := FuseRRF([][]string{bm25IDs, vecIDs}, r.Cfg.RRFK, k)
	return r.hydrate(fused)
}

// SearchMulti runs Search for each query and fuses the ranked id lists. Used for
// query rewriting (a normalized query plus alternates).
func (r *Retriever) SearchMulti(ctx context.Context, queries []string, topK int) ([]SearchHit, error) {
	k := topK
	if k <= 0 {
		k = r.Cfg.TopK
	}
	var cleaned []string
	for _, q := range queries {
		if s := strings.TrimSpace(q); s != "" {
			cleaned = append(cleaned, s)
		}
	}
	if len(cleaned) == 0 {
		return nil, nil
	}
	if len(cleaned) == 1 {
		return r.Search(ctx, cleaned[0], k)
	}

	var rankings [][]string
	for _, q := range cleaned {
		hits, err := r.Search(ctx, q, max(k, 12))
		if err != nil {
			return nil, err
		}
		ids := make([]string, len(hits))
		for i, h := range hits {
			ids[i] = h.ChunkID
		}
		rankings = append(rankings, ids)
	}
	fused := FuseRRF(rankings, r.Cfg.RRFK, k)
	return r.hydrate(fused)
}

func (r *Retriever) hydrate(fused []scored) ([]SearchHit, error) {
	ids := make([]string, len(fused))
	for i, f := range fused {
		ids[i] = f.id
	}
	rows, err := r.DB.ChunksByIDs(ids)
	if err != nil {
		return nil, err
	}
	hits := make([]SearchHit, 0, len(fused))
	for _, f := range fused {
		row, ok := rows[f.id]
		if !ok {
			continue
		}
		hit := SearchHit{
			ChunkID:    row.ChunkID,
			RelPath:    row.RelPath,
			Language:   row.Language,
			StartLine:  row.StartLine,
			EndLine:    row.EndLine,
			Symbol:     row.Symbol,
			SymbolKind: row.SymbolKind,
			Text:       row.Text,
			Score:      f.score,
		}
		if len(f.ranks) > 0 {
			hit.BM25Rank = f.ranks[0]
		}
		if len(f.ranks) > 1 {
			hit.VectorRank = f.ranks[1]
		}
		hits = append(hits, hit)
	}
	return hits, nil
}

// ------------------------------------------------------------ re-ranking

var artifactMarkers = []string{
	"generated_reports/",
	"generated/",
	"htmlcov/",
}

// IsArtifactPath reports whether a path looks like a generated report dump.
// Ported from is_artifact_path in query_understand.py.
func IsArtifactPath(relPath string) bool {
	p := strings.ToLower(strings.ReplaceAll(relPath, "\\", "/"))
	if filepath.Base(p) == "report.json" {
		return true
	}
	for _, m := range artifactMarkers {
		if strings.Contains(p, m) {
			return true
		}
	}
	return false
}

// DemoteArtifactPaths pushes generated_reports / report.json chunks far below
// real source hits by scaling their score.
func DemoteArtifactPaths(hits []SearchHit) []SearchHit {
	return rescore(hits, func(h SearchHit) float64 {
		if IsArtifactPath(h.RelPath) {
			return h.Score * 0.05
		}
		return h.Score
	})
}

var fileHintRe = regexp.MustCompile(`(?i)\b(docker\s*file|dockerfile|make\s*file|makefile|read\s*me|readme|[\w.-]+\.(?:py|js|ts|tsx|jsx|go|rs|java|md|yml|yaml|toml|json|sh))\b`)

// FilenameHints extracts likely filename tokens from a query. Ported from
// _filename_hints in query_understand.py.
func FilenameHints(query string) []string {
	var hints []string
	for _, m := range fileHintRe.FindAllStringSubmatch(query, -1) {
		token := strings.ToLower(regexp.MustCompile(`\s+`).ReplaceAllString(m[1], ""))
		hints = append(hints, token)
	}
	q := strings.ToLower(query)
	if strings.Contains(q, "docker") && strings.Contains(q, "file") && !contains(hints, "dockerfile") {
		hints = append(hints, "dockerfile")
	}
	return hints
}

// BoostFilenameMatches boosts hits whose basename matches a filename mentioned
// in the query. Ported from boost_filename_matches in query_understand.py.
func BoostFilenameMatches(hits []SearchHit, query string) []SearchHit {
	hints := FilenameHints(query)
	if len(hints) == 0 || len(hits) == 0 {
		return hits
	}
	return rescore(hits, func(h SearchHit) float64 {
		if IsArtifactPath(h.RelPath) {
			return h.Score
		}
		name := strings.ToLower(filepath.Base(h.RelPath))
		stem := strings.ToLower(strings.TrimSuffix(name, filepath.Ext(name)))
		for _, hint := range hints {
			if hint == name || hint == stem || strings.HasPrefix(name, hint) || strings.Contains(name, hint) {
				return h.Score + 1.0
			}
			compact := strings.ReplaceAll(strings.ReplaceAll(name, "-", ""), "_", "")
			if strings.Contains(compact, strings.ReplaceAll(hint, "-", "")) {
				return h.Score + 0.5
			}
		}
		return h.Score
	})
}

// FilterByThreshold drops hits whose score is below maxScore*ratio, keeping at
// least the single best hit. Ported from filter_relevant in search.py.
func FilterByThreshold(hits []SearchHit, ratio float64) []SearchHit {
	if len(hits) == 0 {
		return nil
	}
	maxScore := hits[0].Score
	for _, h := range hits {
		if h.Score > maxScore {
			maxScore = h.Score
		}
	}
	if maxScore <= 0 {
		return hits
	}
	threshold := maxScore * ratio
	var kept []SearchHit
	for _, h := range hits {
		if h.Score >= threshold {
			kept = append(kept, h)
		}
	}
	if len(kept) == 0 {
		return hits[:1]
	}
	return kept
}

// Refine applies artifact demotion, filename boosting, then threshold filtering.
func Refine(hits []SearchHit, query string, cfg config.Config) []SearchHit {
	hits = DemoteArtifactPaths(hits)
	hits = BoostFilenameMatches(hits, query)
	filtered := FilterByThreshold(hits, cfg.MinScoreRatio)
	if len(filtered) == 0 {
		return hits
	}
	return filtered
}

func rescore(hits []SearchHit, fn func(SearchHit) float64) []SearchHit {
	out := make([]SearchHit, len(hits))
	copy(out, hits)
	for i := range out {
		out[i].Score = fn(out[i])
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Score > out[j].Score })
	return out
}

// HitsToContext concatenates hits into a bounded context string for the LLM.
func HitsToContext(hits []SearchHit, maxChars int) string {
	if maxChars <= 0 {
		maxChars = 8000
	}
	var parts []string
	used := 0
	for _, h := range hits {
		lang := h.Language
		if lang == "text" {
			lang = ""
		}
		header := "### " + h.RelPath + ":" + strconv.Itoa(h.StartLine) + "-" + strconv.Itoa(h.EndLine) + " (" + h.Language + ")"
		block := header + "\n```" + lang + "\n" + h.Text + "\n```"
		if used+len(block) > maxChars {
			break
		}
		parts = append(parts, block)
		used += len(block) + 1
	}
	return strings.Join(parts, "\n\n")
}

// FormatSnippet trims a hit's text for display.
func FormatSnippet(h SearchHit, cfg config.Config) string {
	text := strings.TrimRight(h.Text, " \n\t")
	if len(text) <= cfg.SnippetChars {
		return text
	}
	return strings.TrimRight(text[:cfg.SnippetChars], " \n\t") + "\n..."
}

func contains(s []string, v string) bool {
	for _, x := range s {
		if x == v {
			return true
		}
	}
	return false
}
