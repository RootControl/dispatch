package index

// fuseRRF combines several ranked lists by Reciprocal Rank Fusion: each item
// scores sum over lists of 1/(k+rank), rank being 1-based position. RRF needs no
// score calibration between lists — it uses only ranks — which is exactly why it
// fuses cosine similarity and BM25 (incomparable scales) cleanly. k=60 is the
// value from the original RRF paper and the common default.
//
// Each input list must already be sorted best-first. The result is sorted by
// fused score with a deterministic id tie-break, truncated to topN (all if <=0).
// Ranked is one fused result: an id and its RRF score. It is the exported
// shape of the same fusion the in-memory store uses internally.
type Ranked struct {
	ID    string
	Score float64
}

// FuseRankings fuses ranked id lists by RRF and returns the top topN.
//
// It is exported so an alternative backend fuses with this code rather than a
// reimplementation — in SQL, say. Ranking is the one place where two backends
// differing would be invisible in the results, so the reference store and
// index/pgvector run the same function over the same input shape.
func FuseRankings(lists [][]string, topN int) []Ranked {
	converted := make([][]scored, len(lists))
	for i, l := range lists {
		converted[i] = make([]scored, len(l))
		for j, id := range l {
			converted[i][j] = scored{id: id}
		}
	}
	fused := fuseRRF(converted, 60, topN)
	out := make([]Ranked, len(fused))
	for i, f := range fused {
		out[i] = Ranked{ID: f.id, Score: f.score}
	}
	return out
}

func fuseRRF(lists [][]scored, k, topN int) []scored {
	if k <= 0 {
		k = 60
	}
	agg := make(map[string]float64)
	for _, list := range lists {
		for rank, s := range list {
			agg[s.id] += 1.0 / float64(k+rank+1)
		}
	}
	out := make([]scored, 0, len(agg))
	for id, sc := range agg {
		out = append(out, scored{id: id, score: sc})
	}
	sortScored(out)
	if topN > 0 && len(out) > topN {
		out = out[:topN]
	}
	return out
}
