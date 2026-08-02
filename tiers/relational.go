package tiers

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/RootControl/dispatch/core"
	"github.com/RootControl/dispatch/index"
	"github.com/RootControl/dispatch/internal/atomicfile"
	"github.com/RootControl/dispatch/llm"
)

// Relational answers questions that a single chunk cannot because the answer
// spans a chain of stated relationships: "who does Bob report to, and what did
// she sign". Entities and relations are extracted per chunk, merged into one
// graph, and traversed outward from the entities named in the question.
//
// Every edge remembers the chunk that asserted it, so a multi-hop answer stays
// grounded: each hop cites a real chunk rather than the graph itself.
type Relational struct {
	graph   *Graph
	MaxHops int // default 2
}

var _ core.Retriever = (*Relational)(nil)

func (r *Relational) Tier() core.Tier { return core.TierRelational }

// GraphOptions tunes extraction.
type GraphOptions struct {
	// Cache stores extractions by content hash. Extraction is one LLM call per
	// chunk — the same cost shape as contextual chunking — so without a cache a
	// rebuild pays for the whole corpus again.
	Cache    *index.Cache
	CacheTag string // model identity folded into cache keys
	MaxHops  int    // traversal depth; default 2
	// NoCoreference disables merging coreferent entity names. Merging costs one
	// LLM call per candidate pair (cached), and a wrong merge invents
	// relationships, so it can be turned off.
	NoCoreference bool
}

// GraphStats reports what building the graph cost and produced.
type GraphStats struct {
	Chunks    int
	Entities  int
	Relations int
	LLMCalls  int
	CacheHits int
	// Skipped counts chunks whose extraction failed. They contribute no
	// entities, which is a smaller loss than discarding the whole build.
	Skipped int
	// Merges records coreference decisions, so an automatic merge can be
	// audited rather than taken on trust.
	Merges []Merge
	// CorefFailures counts adjudications that errored. Zero merges from zero
	// failures means the model rejected the candidates; zero merges from many
	// failures means coreference did not run at all, and the two must not look
	// the same.
	CorefFailures int
	CorefError    error
	// FirstError is the first extraction failure, kept so a build that skipped
	// chunks can explain why rather than only reporting a count.
	FirstError error
}

type extraction struct {
	Entities []struct {
		Name string `json:"name"`
		Type string `json:"type"`
	} `json:"entities"`
	Relations []struct {
		From     string `json:"from"`
		Relation string `json:"relation"`
		To       string `json:"to"`
	} `json:"relations"`
}

const extractSystem = `You extract an entity-relationship graph from a text excerpt.

Rules:
- Extract only entities and relationships STATED in the excerpt. Never infer or add
  outside knowledge.
- Use the exact names as they appear in the text, so the same entity in different
  excerpts produces the same name.
- "relation" is a short verb phrase: "reports to", "depends on", "signed", "owns".
- Types: person, org, system, document, or other.
- If the excerpt states no relationships, return empty lists.

Reply with a JSON object only:
{"entities":[{"name":"...","type":"..."}],"relations":[{"from":"...","relation":"...","to":"..."}]}`

// BuildGraph extracts a graph from every chunk in a store.
func BuildGraph(ctx context.Context, l llm.LLM, store *index.Store, opts GraphOptions) (*Relational, GraphStats, error) {
	var stats GraphStats
	chunks, _ := store.Entries()
	if len(chunks) == 0 {
		return nil, stats, fmt.Errorf("tiers: cannot build a graph over an empty store")
	}
	if opts.CacheTag == "" {
		opts.CacheTag = "default"
	}

	g := newGraph()
	g.SourceGeneration = store.Generation()
	for _, c := range chunks {
		stats.Chunks++
		body := c.Embedded()

		var ex extraction
		key := index.Key("graph|"+opts.CacheTag, c.DocID, body)
		if raw, ok := opts.Cache.Get(key); ok {
			if err := json.Unmarshal([]byte(raw), &ex); err == nil {
				stats.CacheHits++
				apply(g, ex, c.ID, c.Text)
				continue
			}
			// A corrupt cache entry is not worth failing over; re-extract.
		}

		msgs := []llm.Message{
			llm.System(extractSystem),
			llm.User(fmt.Sprintf("<excerpt>\n%s\n</excerpt>", strings.TrimSpace(body))),
		}
		if err := l.ChatJSON(ctx, msgs, &ex); err != nil {
			// One unextractable chunk must not discard the whole build. On a
			// large corpus extraction runs for tens of minutes, and aborting at
			// chunk 60 of 70 throws away every earlier call. A skipped chunk
			// costs its entities; an aborted build costs all of them.
			stats.Skipped++
			if stats.FirstError == nil {
				stats.FirstError = fmt.Errorf("extract from %s: %w", c.ID, err)
			}
			continue
		}
		stats.LLMCalls++
		if encoded, err := json.Marshal(ex); err == nil {
			_ = opts.Cache.Put(key, string(encoded))
		}
		apply(g, ex, c.ID, c.Text)
	}

	// Coreference: lexical candidates, model adjudication. Cheap because the
	// candidate set is tens of pairs, not thousands.
	confirm := func(short, long string) bool { return false }
	var corefFails *int
	var corefErr *error
	if !opts.NoCoreference {
		confirm, corefFails, corefErr = confirmCoreference(ctx, l, opts.Cache, opts.CacheTag)
	}
	stats.Merges = g.canonicalize(confirm)
	if corefFails != nil && *corefFails > 0 {
		stats.CorefFailures = *corefFails
		stats.CorefError = *corefErr
	}
	g.reindex()
	stats.Entities = len(g.Entities)
	stats.Relations = len(g.Edges)
	return &Relational{graph: g, MaxHops: opts.MaxHops}, stats, nil
}

func apply(g *Graph, ex extraction, chunkID, excerpt string) {
	for _, e := range ex.Entities {
		g.addEntity(e.Name, e.Type)
	}
	for _, rel := range ex.Relations {
		g.addEdge(rel.From, rel.To, rel.Relation, chunkID)
	}
	g.Excerpts[chunkID] = excerpt
}

// Retrieve seeds from entities named in the question, traverses outward, and
// returns the supporting chunks with the relationship paths that led to them.
//
// Results are grouped by source chunk rather than emitted per edge: several
// edges often come from one chunk, and one Result per edge would produce
// duplicate citations that the agent loop would dedupe away anyway.
func (r *Relational) Retrieve(ctx context.Context, q core.Query) ([]core.Result, error) {
	if r.graph == nil || len(r.graph.Edges) == 0 {
		return nil, nil
	}
	seeds := r.graph.Seeds(q.Text)
	if len(seeds) == 0 {
		// No known entity in the question. Returning nothing is honest: the
		// loop reads it as a gap and can refine or fall back to another tier.
		return nil, nil
	}

	maxHops := r.MaxHops
	if maxHops <= 0 {
		maxHops = 2
	}
	hits := r.graph.Traverse(seeds, maxHops)
	if len(hits) == 0 {
		return nil, nil
	}

	type group struct {
		paths   []string
		minHop  int
		chunkID string
	}
	groups := map[string]*group{}
	var order []string
	for _, h := range hits {
		gr, ok := groups[h.Edge.ChunkID]
		if !ok {
			gr = &group{minHop: h.Hop, chunkID: h.Edge.ChunkID}
			groups[h.Edge.ChunkID] = gr
			order = append(order, h.Edge.ChunkID)
		}
		gr.paths = append(gr.paths, r.graph.Render(h.Edge))
		gr.minHop = min(gr.minHop, h.Hop)
	}

	// Nearest-hop chunks first; ties broken by ID for reproducibility.
	sort.SliceStable(order, func(i, j int) bool {
		a, b := groups[order[i]], groups[order[j]]
		if a.minHop != b.minHop {
			return a.minHop < b.minHop
		}
		return a.chunkID < b.chunkID
	})

	topK := q.TopK
	if topK <= 0 {
		topK = 5
	}
	out := make([]core.Result, 0, min(len(order), topK))
	for _, id := range order[:min(len(order), topK)] {
		gr := groups[id]
		var b strings.Builder
		b.WriteString("Relationships:\n")
		for _, p := range gr.paths {
			fmt.Fprintf(&b, "  %s\n", p)
		}
		if excerpt := r.graph.Excerpts[id]; excerpt != "" {
			fmt.Fprintf(&b, "\nSource excerpt:\n%s", strings.TrimSpace(excerpt))
		}
		out = append(out, core.Result{
			Tier:     core.TierRelational,
			SourceID: id,
			Text:     b.String(),
			// Closer hops score higher; this is a within-tier ranking only.
			Score: 1.0 / float64(gr.minHop),
			Meta:  map[string]string{"hops": strconv.Itoa(gr.minHop), "edges": strconv.Itoa(len(gr.paths))},
		})
	}
	return out, nil
}

// Stats reports the graph size.
func (r *Relational) Stats() (entities, relations int) {
	if r.graph == nil {
		return 0, 0
	}
	return len(r.graph.Entities), len(r.graph.Edges)
}

// Save writes the graph to path.
func (r *Relational) Save(path string) error {
	if dir := filepath.Dir(path); dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	data, err := json.MarshalIndent(r.graph, "", "  ")
	if err != nil {
		return err
	}
	return atomicfile.Write(path, data, 0o644)
}

// SourceGeneration reports the index generation this graph was extracted from,
// or "" for a graph written before artifacts recorded it.
func (r *Relational) SourceGeneration() string { return r.graph.SourceGeneration }

// LoadGraph reads a graph written by Save.
func LoadGraph(path string, maxHops int) (*Relational, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	g := newGraph()
	if err := json.Unmarshal(data, g); err != nil {
		return nil, fmt.Errorf("tiers: parse graph %s: %w", path, err)
	}
	if g.Entities == nil {
		g.Entities = map[string]Entity{}
	}
	if g.Excerpts == nil {
		g.Excerpts = map[string]string{}
	}
	if g.Aliases == nil {
		g.Aliases = map[string]string{}
	}
	g.reindex()
	return &Relational{graph: g, MaxHops: maxHops}, nil
}
