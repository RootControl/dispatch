package tiers

import (
	"slices"
	"sort"
	"strings"
	"unicode"
)

// Entity is a node: a person, org, system, or document named in the corpus.
type Entity struct {
	Name string `json:"name"` // display form, as first seen
	Type string `json:"type"`
}

// Edge is a stated relationship. ChunkID records which chunk asserted it, so
// every hop stays traceable back to a citable source.
type Edge struct {
	From     string `json:"from"` // normalized entity key
	To       string `json:"to"`
	Relation string `json:"relation"`
	ChunkID  string `json:"chunk_id"`
}

// Graph is the entity graph. Keys are normalized names; Entity.Name keeps the
// display form.
type Graph struct {
	Entities map[string]Entity `json:"entities"`
	Edges    []Edge            `json:"edges"`
	Excerpts map[string]string `json:"excerpts"` // chunk ID -> supporting text

	adj map[string][]int // normalized entity -> edge indices, both directions
}

func newGraph() *Graph {
	return &Graph{
		Entities: map[string]Entity{},
		Excerpts: map[string]string{},
		adj:      map[string][]int{},
	}
}

// normalizeEntity collapses surface variations so "The VP of Platform",
// "VP of Platform" and "vp of platform" resolve to one node. This is the whole
// of entity resolution here: no embedding-based coreference, no alias table.
// It handles case and punctuation, and nothing else — "Priya" and "Priya Raman"
// stay distinct, which is the main known limitation of this tier.
//
// Every non-alphanumeric rune is a SEPARATOR, not a deletion. Deleting them
// welds words together: "packages/data-provider" became "packagesdata provider"
// and "@librechat/agents" became "librechatagents", which is unreadable and can
// falsely merge distinct names. Splitting also lets a question written
// "packages data provider" reach an entity written "packages/data-provider".
// This matches the tokenizer used by BM25, so the two agree on word boundaries.
func normalizeEntity(name string) string {
	fields := strings.FieldsFunc(strings.ToLower(strings.TrimSpace(name)), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsNumber(r)
	})
	return strings.TrimPrefix(strings.Join(fields, " "), "the ")
}

// addEntity records an entity, keeping the first display form seen.
func (g *Graph) addEntity(name, typ string) string {
	key := normalizeEntity(name)
	if key == "" {
		return ""
	}
	if _, ok := g.Entities[key]; !ok {
		g.Entities[key] = Entity{Name: strings.TrimSpace(name), Type: typ}
	}
	return key
}

// addEdge records a relationship, ignoring self-loops and exact duplicates.
func (g *Graph) addEdge(from, to, relation, chunkID string) {
	f, t := normalizeEntity(from), normalizeEntity(to)
	if f == "" || t == "" || f == t {
		return
	}
	rel := strings.TrimSpace(relation)
	if rel == "" {
		return
	}
	for _, e := range g.Edges {
		if e.From == f && e.To == t && strings.EqualFold(e.Relation, rel) && e.ChunkID == chunkID {
			return
		}
	}
	// Entities may be referenced by an edge without appearing in the entity
	// list; register them so traversal can reach them.
	if _, ok := g.Entities[f]; !ok {
		g.Entities[f] = Entity{Name: strings.TrimSpace(from)}
	}
	if _, ok := g.Entities[t]; !ok {
		g.Entities[t] = Entity{Name: strings.TrimSpace(to)}
	}
	g.Edges = append(g.Edges, Edge{From: f, To: t, Relation: rel, ChunkID: chunkID})
}

// reindex rebuilds the adjacency map. Traversal is undirected — "Priya reports
// to the VP" should be reachable from either end — while Edge keeps its
// direction for rendering.
func (g *Graph) reindex() {
	g.adj = make(map[string][]int, len(g.Entities))
	for i, e := range g.Edges {
		g.adj[e.From] = append(g.adj[e.From], i)
		g.adj[e.To] = append(g.adj[e.To], i)
	}
}

// Seeds finds entities named in the query. Matching is on normalized text, so
// punctuation and case in the question do not matter. Longer names are
// preferred, so "VP of Platform" wins over a bare "platform" substring.
func (g *Graph) Seeds(query string) []string {
	q := " " + normalizeEntity(query) + " "
	var found []string
	for key := range g.Entities {
		if key == "" {
			continue
		}
		if strings.Contains(q, " "+key+" ") {
			found = append(found, key)
		}
	}
	// Drop a seed fully contained in a longer seed: matching both "platform"
	// and "vp of platform" would traverse from a vaguer node for no gain.
	filtered := make([]string, 0, len(found))
	for _, a := range found {
		contained := slices.ContainsFunc(found, func(b string) bool {
			return a != b && strings.Contains(" "+b+" ", " "+a+" ")
		})
		if !contained {
			filtered = append(filtered, a)
		}
	}
	sort.Strings(filtered) // deterministic traversal order
	return filtered
}

// EdgeHit is an edge reached during traversal, with the hop count at which it
// was first found.
type EdgeHit struct {
	Edge Edge
	Hop  int
}

// Traverse walks outward from seeds up to maxHops, returning every edge reached
// and how far out it was. Breadth-first, so an edge's Hop is its shortest
// distance from any seed — which is what makes hop count usable as a relevance
// signal.
func (g *Graph) Traverse(seeds []string, maxHops int) []EdgeHit {
	if len(g.adj) == 0 {
		g.reindex()
	}
	type queued struct {
		node string
		hop  int
	}
	visited := make(map[string]bool, len(seeds))
	queue := make([]queued, 0, len(seeds))
	for _, s := range seeds {
		if !visited[s] {
			visited[s] = true
			queue = append(queue, queued{s, 0})
		}
	}

	edgeHop := map[int]int{}
	for len(queue) > 0 {
		cur := queue[0]
		queue = queue[1:]
		if cur.hop >= maxHops {
			continue
		}
		for _, ei := range g.adj[cur.node] {
			if _, seen := edgeHop[ei]; !seen {
				edgeHop[ei] = cur.hop + 1
			}
			e := g.Edges[ei]
			next := e.To
			if next == cur.node {
				next = e.From
			}
			if !visited[next] {
				visited[next] = true
				queue = append(queue, queued{next, cur.hop + 1})
			}
		}
	}

	hits := make([]EdgeHit, 0, len(edgeHop))
	for ei, hop := range edgeHop {
		hits = append(hits, EdgeHit{Edge: g.Edges[ei], Hop: hop})
	}
	// Nearest first, then stable by content so output is reproducible.
	sort.Slice(hits, func(i, j int) bool {
		if hits[i].Hop != hits[j].Hop {
			return hits[i].Hop < hits[j].Hop
		}
		a, b := hits[i].Edge, hits[j].Edge
		if a.From != b.From {
			return a.From < b.From
		}
		if a.Relation != b.Relation {
			return a.Relation < b.Relation
		}
		return a.To < b.To
	})
	return hits
}

// Render describes an edge in the direction it was stated.
func (g *Graph) Render(e Edge) string {
	return g.display(e.From) + " —" + e.Relation + "→ " + g.display(e.To)
}

func (g *Graph) display(key string) string {
	if ent, ok := g.Entities[key]; ok && ent.Name != "" {
		return ent.Name
	}
	return key
}
