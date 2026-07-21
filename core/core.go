// Package core holds the domain types and the single Retriever interface that
// every tier satisfies. It depends on nothing else in the module, so both the
// tiers and the agent loop can import it without cycles.
package core

import (
	"context"
	"fmt"
)

// Tier names a kind of retrieval. Each tier answers a different kind of
// question; the router ranks tiers per query and the agent loop fans out over
// the ranking rather than committing to one.
type Tier string

const (
	TierStructured   Tier = "structured"   // text-to-SQL over a relational store
	TierSemantic     Tier = "semantic"     // contextual chunk + hybrid search
	TierRelational   Tier = "relational"   // entity graph, multi-hop
	TierHierarchical Tier = "hierarchical" // RAPTOR summary tree
	TierMemory       Tier = "memory"       // agent write-back archival store
)

// Doc is a source document handed to ingestion. ID must be stable across
// re-ingests so the contextual-chunk cache and citations stay valid.
type Doc struct {
	ID   string
	Text string
	Meta map[string]string
}

// Chunk is a unit of retrievable text produced from a Doc. Context is the
// one-sentence situating sentence written during contextual chunking; it is
// prepended to Text before embedding but tracked separately so callers can show
// the original text in citations.
type Chunk struct {
	ID       string
	DocID    string
	Text     string
	Context  string
	Position int
	Meta     map[string]string
}

// Embedded returns the text that is actually embedded: the situating context
// sentence prepended to the chunk body. When no context was produced this is
// just the body, so the contextual and non-contextual paths share one code path.
func (c Chunk) Embedded() string {
	if c.Context == "" {
		return c.Text
	}
	return c.Context + "\n\n" + c.Text
}

// Query is one retrieval request. TopK is a hint; a tier may return fewer. The
// agent loop rewrites Text as it refines toward an evidence gap, so a single
// user question becomes several Query values over the loop's lifetime.
type Query struct {
	Text string
	TopK int
}

// Result is one piece of retrieved evidence. Score is tier-local and not
// comparable across tiers — the loop treats it as a within-tier ranking only.
type Result struct {
	Tier     Tier
	SourceID string
	Text     string
	Score    float64
	Meta     map[string]string
}

// Cite renders the [tier:source] citation the generator emits and the verifier
// resolves back to a Result.
func (r Result) Cite() string {
	return fmt.Sprintf("[%s:%s]", r.Tier, r.SourceID)
}

// Retriever is the one interface every tier satisfies. Keeping it this narrow is
// what lets the agent loop stay tier-agnostic and lets a production backend
// (pgvector, DuckDB) drop in without changing anything upstream.
type Retriever interface {
	// Tier reports which tier this retriever serves, so results can be cited
	// and the router can address it by name.
	Tier() Tier
	// Retrieve returns evidence for q, best-first. Returning zero results with a
	// nil error is valid and means "nothing relevant here" — the loop reads that
	// as a gap, not a failure.
	Retrieve(ctx context.Context, q Query) ([]Result, error)
}
