package index

import "sort"

// scored is an (id, score) pair used across the vector index, BM25 index, and
// RRF fusion so all three speak the same shape.
type scored struct {
	id    string
	score float64
}

// vectorIndex is a flat cosine index. Vectors are stored unit-normalized (the
// LLM client and the fake embedder both normalize), so cosine similarity is a
// plain dot product. Flat brute force is fine for the reference store; a
// production backend (pgvector, DuckDB) replaces this without touching callers.
type vectorIndex struct {
	ids  []string
	vecs [][]float64
}

func (vi *vectorIndex) add(id string, v []float64) {
	vi.ids = append(vi.ids, id)
	vi.vecs = append(vi.vecs, v)
}

func (vi *vectorIndex) search(q []float64, n int) []scored {
	out := make([]scored, 0, len(vi.ids))
	for i, id := range vi.ids {
		out = append(out, scored{id: id, score: dot(q, vi.vecs[i])})
	}
	sortScored(out)
	if n > 0 && len(out) > n {
		out = out[:n]
	}
	return out
}

func dot(a, b []float64) float64 {
	// Guard against mismatched widths from a swapped embedding model.
	n := min(len(a), len(b))
	var s float64
	for i := range n {
		s += a[i] * b[i]
	}
	return s
}

// sortScored orders by score descending with a deterministic id tie-break, so
// rankings — and the golden tests over them — are reproducible.
func sortScored(s []scored) {
	sort.Slice(s, func(i, j int) bool {
		if s[i].score != s[j].score {
			return s[i].score > s[j].score
		}
		return s[i].id < s[j].id
	})
}
