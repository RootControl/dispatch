package index

// Ranked is one fused result: an id, its RRF score, and where it came from.
// It is the exported shape of the same fusion the in-memory store uses
// internally.
type Ranked struct {
	ID    string
	Score float64
	// Ranks is the 1-based position this id held in each input list, or 0 where
	// that list did not return it at all.
	Ranks []int
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
	out := make([]Ranked, 0, topN)
	for _, f := range fuseRRF(converted, 60, topN) {
		out = append(out, Ranked{ID: f.id, Score: f.score, Ranks: f.ranks})
	}
	return out
}

// fused is one RRF result together with the rank it held in each input list.
//
// The ranks are carried rather than discarded because the fused score alone
// cannot answer the first question anyone asks of a hybrid index: did this
// chunk come back because the embedding matched, because the terms matched, or
// because both did? Two chunks with near-identical scores can have arrived for
// completely different reasons, and without this there is no way to see it —
// isolating the lexical half in the pgvector integration tests needed elaborate
// scaffolding purely because the store would not say.
type fused struct {
	id    string
	score float64
	ranks []int
}

// fuseRRF combines several ranked lists by Reciprocal Rank Fusion: each item
// scores sum over lists of 1/(k+rank), rank being 1-based position. RRF needs no
// score calibration between lists — it uses only ranks — which is exactly why it
// fuses cosine similarity and BM25 (incomparable scales) cleanly. k=60 is the
// value from the original RRF paper and the common default.
//
// Each input list must already be sorted best-first. The result is sorted by
// fused score with a deterministic id tie-break, truncated to topN (all if <=0).
func fuseRRF(lists [][]scored, k, topN int) []fused {
	if k <= 0 {
		k = 60
	}
	agg := make(map[string]float64)
	ranks := make(map[string][]int)
	for li, list := range lists {
		for rank, s := range list {
			agg[s.id] += 1.0 / float64(k+rank+1)
			if ranks[s.id] == nil {
				ranks[s.id] = make([]int, len(lists))
			}
			ranks[s.id][li] = rank + 1
		}
	}

	ordered := make([]scored, 0, len(agg))
	for id, sc := range agg {
		ordered = append(ordered, scored{id: id, score: sc})
	}
	sortScored(ordered)
	if topN > 0 && len(ordered) > topN {
		ordered = ordered[:topN]
	}

	out := make([]fused, 0, len(ordered))
	for _, s := range ordered {
		out = append(out, fused{id: s.id, score: s.score, ranks: ranks[s.id]})
	}
	return out
}
