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
// contextually-chunked text. It is a thin adapter over index.Store: the store
// owns retrieval, the tier owns the core.Retriever contract and citation shape.
type Semantic struct {
	store *index.Store
}

var _ core.Retriever = (*Semantic)(nil)

// NewSemantic wraps a store as the semantic tier.
func NewSemantic(s *index.Store) *Semantic { return &Semantic{store: s} }

func (s *Semantic) Tier() core.Tier { return core.TierSemantic }

// Retrieve returns the best-matching chunks. Result.Text carries the situating
// context sentence along with the chunk body — the generator needs that context
// to interpret a chunk that was written to be read in place.
func (s *Semantic) Retrieve(ctx context.Context, q core.Query) ([]core.Result, error) {
	hits, err := s.store.Search(ctx, q.Text, q.TopK)
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
			Meta:     map[string]string{"doc": h.Chunk.DocID},
		})
	}
	return out, nil
}
