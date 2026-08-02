package tiers

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strconv"
	"strings"

	"github.com/RootControl/dispatch/core"
	"github.com/RootControl/dispatch/index"
	"github.com/RootControl/dispatch/internal/atomicfile"
	"github.com/RootControl/dispatch/internal/par"
	"github.com/RootControl/dispatch/llm"
)

// parallelism is the shared default for the concurrent ingest passes: enough to
// keep a server with a few slots busy, low enough not to bury one with a single
// slot in queued requests it will serve one at a time anyway.
func parallelism(n int) int {
	if n <= 0 {
		return 4
	}
	return n
}

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
	// Parallelism bounds concurrent extraction calls; default 4.
	//
	// Extraction is the single most expensive pass in this project — measured at
	// ~42s per chunk, p90 65s, roughly 40 minutes over 70 chunks — and it is
	// latency-bound, not CPU-bound. Running it one chunk at a time left the
	// machine idle for almost all of that. Raising it past what the server will
	// serve concurrently buys nothing: Ollama defaults to OLLAMA_NUM_PARALLEL=1.
	Parallelism int
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

	// Extract concurrently, then apply in chunk order.
	//
	// The split is not incidental. addEntity keeps the display name as first
	// seen, so applying extractions in completion order would make Entity.Name —
	// and therefore every rendered relationship an answer cites — depend on
	// which LLM call happened to return first. Two runs over an unchanged corpus
	// would produce different graphs. Holding the results and applying them in
	// order costs one slice and makes the parallel build byte-identical to the
	// sequential one, which is also what lets the existing tests keep meaning
	// something.
	type outcome struct {
		ex     extraction
		hit    bool // served from cache
		called bool // an LLM call was made
		err    error
	}
	results := make([]outcome, len(chunks))
	_ = par.ForEach(ctx, len(chunks), parallelism(opts.Parallelism), func(ctx context.Context, i int) error {
		c := chunks[i]
		body := c.Embedded()
		key := index.Key("graph|"+opts.CacheTag, c.DocID, body)
		if raw, ok := opts.Cache.Get(key); ok {
			var ex extraction
			if err := json.Unmarshal([]byte(raw), &ex); err == nil {
				results[i] = outcome{ex: ex, hit: true}
				return nil
			}
			// A corrupt cache entry is not worth failing over; re-extract.
		}

		msgs := []llm.Message{
			llm.System(extractSystem),
			llm.User(fmt.Sprintf("<excerpt>\n%s\n</excerpt>", strings.TrimSpace(body))),
		}
		var ex extraction
		if err := l.ChatJSON(ctx, msgs, &ex); err != nil {
			// One unextractable chunk must not discard the whole build. On a
			// large corpus extraction runs for tens of minutes, and aborting at
			// chunk 60 of 70 throws away every earlier call. A skipped chunk
			// costs its entities; an aborted build costs all of them. Recording
			// the error here rather than returning it is what keeps that true
			// under par.ForEach, which cancels its siblings on a returned error.
			results[i] = outcome{err: fmt.Errorf("extract from %s: %w", c.ID, err)}
			return nil
		}
		if encoded, err := json.Marshal(ex); err == nil {
			_ = opts.Cache.Put(key, string(encoded))
		}
		results[i] = outcome{ex: ex, called: true}
		return nil
	})

	for i, c := range chunks {
		stats.Chunks++
		r := results[i]
		switch {
		case r.err != nil:
			stats.Skipped++
			if stats.FirstError == nil {
				stats.FirstError = r.err
			}
			continue
		case r.hit:
			stats.CacheHits++
		case r.called:
			stats.LLMCalls++
		}
		apply(g, r.ex, c.ID, c.Text)
	}

	// A cancelled or timed-out build is not a build with some chunks missing.
	// Every remaining chunk fails at once with the same context error, so
	// without this the result is a graph holding whatever finished first,
	// stamped with the current index generation — indistinguishable from a
	// complete one, and the staleness check cannot catch it because the
	// generation is genuinely current. Interrupting an ingest must not silently
	// narrow the graph.
	if err := ctx.Err(); err != nil {
		return nil, stats, fmt.Errorf("tiers: graph build interrupted after %d of %d chunks: %w",
			stats.Chunks-stats.Skipped, len(chunks), err)
	}

	// Nor is a build where EVERY chunk failed a graph. Skipping a chunk is the
	// right call when the rest succeeded — aborting at chunk 60 of 70 throws
	// away an hour of calls — but skipping all of them means the endpoint was
	// down, the model could not produce JSON, or the timeout was too short, and
	// none of those should leave an empty graph on disk stamped with the
	// current index generation. Found by running it: a slow endpoint timed out
	// on both chunks of a two-document corpus, and the result was a graph with
	// zero entities that the staleness check called current, having overwritten
	// whatever was there before.
	if stats.Skipped == stats.Chunks {
		return nil, stats, fmt.Errorf(
			"tiers: every one of %d chunk(s) failed extraction, so this is a failed build rather than an empty graph: %w",
			stats.Chunks, stats.FirstError)
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

// The accessors below exist for inspection — `dispatch graph`. Everything the
// tier decides is derived from this state, and until it could be read back the
// only record of a coreference merge was a line printed at ingest and lost with
// the terminal scrollback. That is the riskiest decision this tier makes, so
// "audit it later" has to be possible.

// EntityList returns every entity with its key, sorted by key.
func (r *Relational) EntityList() []struct {
	Key string
	Entity
} {
	out := make([]struct {
		Key string
		Entity
	}, 0, len(r.graph.Entities))
	for k, e := range r.graph.Entities {
		out = append(out, struct {
			Key string
			Entity
		}{k, e})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Key < out[j].Key })
	return out
}

// Edges returns every stated relationship, in extraction order.
func (r *Relational) Edges() []Edge { return slices.Clone(r.graph.Edges) }

// Aliases returns the coreference merges: variant -> canonical.
func (r *Relational) Aliases() map[string]string { return maps.Clone(r.graph.Aliases) }

// Degree counts the edges touching an entity key.
func (r *Relational) Degree(key string) int {
	var n int
	for _, e := range r.graph.Edges {
		if e.From == key || e.To == key {
			n++
		}
	}
	return n
}

// Seeds reports which entities a question would start a traversal from. This is
// the whole answer to "why did the relational tier return nothing": no seed
// means no traversal, and the tier correctly returns nothing rather than
// guessing.
func (r *Relational) Seeds(question string) []string { return r.graph.Seeds(question) }

// Traverse walks outward from seeds, exposing what the tier itself does.
func (r *Relational) Traverse(seeds []string, maxHops int) []EdgeHit {
	return r.graph.Traverse(seeds, maxHops)
}

// Render describes an edge using display names.
func (r *Relational) Render(e Edge) string { return r.graph.Render(e) }

// Excerpt returns the text of the chunk that asserted an edge.
func (r *Relational) Excerpt(chunkID string) string { return r.graph.Excerpts[chunkID] }

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
