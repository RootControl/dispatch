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

// remove drops every entry whose id is in drop, preserving the relative order of
// the rest. Order is load-bearing: Entries and Save both walk this slice, so a
// swap-with-last delete would reshuffle the corpus and make clustering and
// snapshots depend on deletion history. One filtered pass per call keeps a
// whole-document removal linear rather than quadratic.
func (vi *vectorIndex) remove(drop map[string]bool) int {
	if len(drop) == 0 {
		return 0
	}
	ids, vecs, removed := vi.ids[:0], vi.vecs[:0], 0
	for i, id := range vi.ids {
		if drop[id] {
			removed++
			continue
		}
		ids = append(ids, id)
		vecs = append(vecs, vi.vecs[i])
	}
	// Clear the vacated tail so removed vectors are not pinned by the backing
	// array; each is a few KB and a long-lived store would otherwise hold every
	// vector it ever indexed.
	for i := len(vecs); i < len(vi.vecs); i++ {
		vi.vecs[i] = nil
	}
	vi.ids, vi.vecs = ids, vecs
	return removed
}

// search returns the n best-scoring entries that allow accepts. Filtering
// happens during the scan rather than after it, so a filter that excludes most
// of the corpus still returns a full pool of n candidates instead of whatever
// survives from an unfiltered top-n. A nil allow admits everything.
func (vi *vectorIndex) search(q []float64, n int, allow func(string) bool) []scored {
	out := make([]scored, 0, len(vi.ids))
	for i, id := range vi.ids {
		if allow != nil && !allow(id) {
			continue
		}
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
