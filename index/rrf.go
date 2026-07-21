package index

// fuseRRF combines several ranked lists by Reciprocal Rank Fusion: each item
// scores sum over lists of 1/(k+rank), rank being 1-based position. RRF needs no
// score calibration between lists — it uses only ranks — which is exactly why it
// fuses cosine similarity and BM25 (incomparable scales) cleanly. k=60 is the
// value from the original RRF paper and the common default.
//
// Each input list must already be sorted best-first. The result is sorted by
// fused score with a deterministic id tie-break, truncated to topN (all if <=0).
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
