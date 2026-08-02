package tiers

import (
	"context"
	"fmt"
	"strconv"
	"strings"

	"github.com/RootControl/dispatch/core"
	"github.com/RootControl/dispatch/index"
	"github.com/RootControl/dispatch/internal/par"
	"github.com/RootControl/dispatch/llm"
)

// Hierarchical answers questions no single chunk can: "recurring themes across
// all docs", "what does the corpus generally say". It is a RAPTOR summary tree —
// leaves are clustered, each cluster is summarized by an LLM, the summaries are
// clustered again, and so on toward a root.
//
// The tier indexes only the SUMMARIES, not the leaves. In a single-index RAPTOR
// the collapsed tree holds both, but here the semantic tier already serves the
// leaves; including them again would return the same chunk under two citations
// and waste evidence budget. The tiers stay complementary: summaries for
// themes, semantic for specifics.
type Hierarchical struct {
	nodes *index.Store
}

var _ core.Retriever = (*Hierarchical)(nil)

func (h *Hierarchical) Tier() core.Tier { return core.TierHierarchical }

// HierarchyOptions tunes tree construction.
type HierarchyOptions struct {
	Branching int // target leaves per cluster; default 5
	MaxLevels int // safety bound on tree height; default 4
	// Cache stores summaries by the content of their members. Without it the
	// tree is the only stage that pays again on every rebuild, while chunking
	// and extraction replay for free — a surprise precisely because the other
	// stages trained you to expect a rebuild to be cheap.
	Cache    *index.Cache
	CacheTag string
	// EmbedTag identifies the embedding model. It caches the summary vectors
	// and is recorded in the saved tree, so loading a hierarchy built under a
	// different embedding model fails loudly instead of searching one embedding
	// space with another's query vector.
	EmbedTag string
	// Parallelism bounds concurrent summary calls within a level; default 4.
	// Levels stay sequential because level N+1 clusters level N's summaries, so
	// there is nothing to overlap between them — the width is inside a level,
	// where the clusters are genuinely independent.
	Parallelism int
}

func (o HierarchyOptions) withDefaults() HierarchyOptions {
	if o.Branching < 2 {
		o.Branching = 5
	}
	if o.MaxLevels <= 0 {
		o.MaxLevels = 4
	}
	return o
}

// BuildStats reports what constructing a tree cost.
type BuildStats struct {
	Levels     int
	Summaries  int
	LLMCalls   int
	CacheHits  int
	EmbedCalls int // summaries sent to the embedder
	EmbedHits  int // summary vectors served from cache
}

// BuildHierarchy constructs the summary tree from a store's existing chunks,
// reusing their embeddings rather than paying to embed the corpus again. Only
// the summaries it creates are embedded — roughly n/4 vectors for n leaves.
func BuildHierarchy(ctx context.Context, l llm.LLM, leaves *index.Store, opts HierarchyOptions) (*Hierarchical, BuildStats, error) {
	opts = opts.withDefaults()
	var stats BuildStats

	chunks, vectors := leaves.Entries()
	if len(chunks) == 0 {
		return nil, stats, fmt.Errorf("tiers: cannot build a hierarchy over an empty store")
	}

	nodes := index.New(index.Config{LLM: l, Cache: opts.Cache, EmbedTag: opts.EmbedTag,
		SourceGen: leaves.Generation()})
	h := &Hierarchical{nodes: nodes}

	// Current level: the text and vectors being clustered. Starts as the leaves.
	texts := make([]string, len(chunks))
	for i, c := range chunks {
		texts[i] = c.Embedded()
	}
	vecs := vectors

	for level := 1; level <= opts.MaxLevels && len(texts) > 1; level++ {
		k := (len(texts) + opts.Branching - 1) / opts.Branching
		// Guarantee progress: a level that does not shrink would loop until
		// MaxLevels, paying for summaries that summarize one item each.
		if k >= len(texts) {
			k = len(texts) / 2
		}
		if k < 1 {
			k = 1
		}

		clusters := kmeans(vecs, k, 10)
		// Summaries are written into their cluster's slot rather than appended,
		// so the tree does not depend on which call returned first. Node IDs are
		// positional (L1-0, L1-1, ...) and citations quote them, so appending in
		// completion order would renumber the tree on every rebuild and make a
		// saved citation point somewhere else.
		summaries := make([]string, len(clusters))
		hits, calls := make([]bool, len(clusters)), make([]bool, len(clusters))
		err := par.ForEach(ctx, len(clusters), parallelism(opts.Parallelism), func(ctx context.Context, ci int) error {
			members := make([]string, 0, len(clusters[ci]))
			for _, i := range clusters[ci] {
				members = append(members, texts[i])
			}
			// Key on the members themselves, so a cluster that survives a
			// rebuild unchanged costs nothing even if its level number moved.
			key := index.Key("summary|"+opts.CacheTag, strconv.Itoa(len(members)), strings.Join(members, "\x00"))
			if cached, ok := opts.Cache.Get(key); ok && strings.TrimSpace(cached) != "" {
				summaries[ci], hits[ci] = cached, true
				return nil
			}
			s, err := summarize(ctx, l, members)
			if err != nil {
				return fmt.Errorf("tiers: summarize level %d: %w", level, err)
			}
			_ = opts.Cache.Put(key, s)
			summaries[ci], calls[ci] = s, true
			return nil
		})
		// Unlike graph extraction, a failed summary is fatal here and was
		// before: a level with a missing node is not a smaller tree, it is a
		// tree with a hole in it, and every level above inherits the hole.
		if err != nil {
			return nil, stats, err
		}
		for i := range clusters {
			if hits[i] {
				stats.CacheHits++
			}
			if calls[i] {
				stats.LLMCalls++
			}
		}

		// One embedding call per level, not per summary, and only for the
		// summaries whose vectors are not already cached. A cached summary that
		// then had to be re-embedded would leave the rebuild paying per level
		// anyway — the two caches have to cover the same work to be worth having.
		embedded, err := embedCached(ctx, l, opts.Cache, opts.EmbedTag, summaries, &stats)
		if err != nil {
			return nil, stats, fmt.Errorf("tiers: embed level %d summaries: %w", level, err)
		}

		for i, s := range summaries {
			nodes.Add(core.Chunk{
				ID:       fmt.Sprintf("L%d-%d", level, i),
				DocID:    fmt.Sprintf("hierarchy-L%d", level),
				Text:     s,
				Position: i,
				Meta:     map[string]string{"level": strconv.Itoa(level), "members": strconv.Itoa(len(clusters[i]))},
			}, embedded[i])
		}

		stats.Levels = level
		stats.Summaries += len(summaries)
		texts, vecs = summaries, embedded
	}

	return h, stats, nil
}

// embedCached returns one vector per text, serving what it can from the cache
// and sending the rest in a single batch.
func embedCached(ctx context.Context, l llm.LLM, cache *index.Cache, embedTag string, texts []string, stats *BuildStats) ([][]float64, error) {
	out := make([][]float64, len(texts))
	var missIdx []int
	var miss []string
	for i, t := range texts {
		if v, ok := cache.GetVector(index.VecKey(embedTag, t)); ok {
			out[i] = v
			stats.EmbedHits++
			continue
		}
		missIdx = append(missIdx, i)
		miss = append(miss, t)
	}
	if len(miss) == 0 {
		return out, nil
	}

	got, err := l.Embed(ctx, miss)
	if err != nil {
		return nil, err
	}
	if len(got) != len(miss) {
		return nil, fmt.Errorf("got %d vectors for %d texts", len(got), len(miss))
	}
	for j, i := range missIdx {
		out[i] = got[j]
		if err := cache.PutVector(index.VecKey(embedTag, miss[j]), got[j]); err != nil {
			return nil, err
		}
	}
	stats.EmbedCalls += len(miss)
	return out, nil
}

const summarySystem = `You write a summary of related excerpts from a document collection.

Capture the themes that run across the excerpts AND the specific facts that support them —
names, figures, dates. A summary that keeps only the abstractions is useless for retrieval,
because the specifics are what a question will match on.

Write prose only. No preamble, no bullet list, no heading.`

func summarize(ctx context.Context, l llm.LLM, members []string) (string, error) {
	var b strings.Builder
	for i, m := range members {
		fmt.Fprintf(&b, "<excerpt %d>\n%s\n</excerpt %d>\n\n", i+1, strings.TrimSpace(m), i+1)
	}
	reply, err := l.Chat(ctx, []llm.Message{
		llm.System(summarySystem),
		llm.User(strings.TrimSpace(b.String())),
	})
	return strings.TrimSpace(reply), err
}

// Retrieve searches every level of the tree at once — the "collapsed tree"
// strategy. A broad question tends to match a high-level summary and a narrow
// one a lower-level summary, so no level needs choosing in advance.
func (h *Hierarchical) Retrieve(ctx context.Context, q core.Query) ([]core.Result, error) {
	if h.nodes.Len() == 0 {
		return nil, nil
	}
	// Filter is deliberately not forwarded: summary nodes are derived from
	// clusters that can span documents, so they carry no document metadata to
	// match on. Hierarchical does not implement core.Filterable, so the loop
	// skips this tier entirely when a filter is set rather than letting it
	// return summaries from outside the filter.
	hits, err := h.nodes.Search(ctx, core.Query{Text: q.Text, TopK: q.TopK})
	if err != nil {
		return nil, err
	}
	out := make([]core.Result, 0, len(hits))
	for _, hit := range hits {
		out = append(out, core.Result{
			Tier:     core.TierHierarchical,
			SourceID: hit.Chunk.ID,
			Text:     hit.Chunk.Text,
			Score:    hit.Score,
			Meta:     hit.Chunk.Meta,
		})
	}
	return out, nil
}

// Len reports the number of summary nodes across all levels.
func (h *Hierarchical) Len() int { return h.nodes.Len() }

// Save writes the tree to path.
func (h *Hierarchical) Save(path string) error { return h.nodes.Save(path) }

// LoadHierarchy reads a tree written by Save. The LLM is needed only to embed
// queries at search time, not to rebuild anything.
func LoadHierarchy(l llm.LLM, path string) (*Hierarchical, error) {
	nodes := index.New(index.Config{LLM: l})
	if err := nodes.Load(path); err != nil {
		return nil, err
	}
	return &Hierarchical{nodes: nodes}, nil
}

// SourceGeneration reports the index generation this tree was built from, or
// "" for a tree written before artifacts recorded it.
func (h *Hierarchical) SourceGeneration() string { return h.nodes.SourceGeneration() }
