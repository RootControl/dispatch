// Package tiers holds the retrievers, one per kind of question. Each satisfies
// core.Retriever, so the agent loop can fan out across them without knowing what
// any of them does internally.
package tiers

import (
	"context"

	"github.com/RootControl/dispatch/core"
	"github.com/RootControl/dispatch/index"
)

// Semantic answers "what does the corpus say about X" by hybrid search over
// contextually-chunked text. It is a thin adapter over a Searcher: the store
// owns retrieval, the tier owns the core.Retriever contract and citation shape.
type Semantic struct {
	store index.Searcher
}

var _ core.Filterable = (*Semantic)(nil)

// NewSemantic wraps a searcher as the semantic tier. It takes the interface
// rather than *index.Store so a production backend — pgvector, DuckDB — can
// serve this tier without any change above it.
func NewSemantic(s index.Searcher) *Semantic { return &Semantic{store: s} }

func (s *Semantic) Tier() core.Tier { return core.TierSemantic }

// HonorsFilter: the store applies Query.Filter during its scan.
func (s *Semantic) HonorsFilter() {}

// Retrieve returns the best-matching chunks. Result.Text carries the situating
// context sentence along with the chunk body — the generator needs that context
// to interpret a chunk that was written to be read in place.
func (s *Semantic) Retrieve(ctx context.Context, q core.Query) ([]core.Result, error) {
	hits, err := s.store.Search(ctx, q)
	if err != nil {
		return nil, err
	}
	out := make([]core.Result, 0, len(hits))
	for _, h := range hits {
		out = append(out, core.Result{
			Tier:     core.TierSemantic,
			SourceID: h.Chunk.ID,
			Text:     h.Chunk.Embedded(),
			Score:    h.Score,
			// "sources" carries which half of the hybrid found this chunk, so a
			// trace can show it. It is metadata rather than a typed field
			// because core.Result is shared by tiers that have no such notion.
			Meta: map[string]string{"doc": h.Chunk.DocID, "sources": h.Sources()},
		})
	}
	return out, nil
}
