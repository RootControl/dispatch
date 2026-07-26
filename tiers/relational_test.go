package tiers

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/RootControl/dispatch/core"
	"github.com/RootControl/dispatch/index"
	"github.com/RootControl/dispatch/internal/fake"
	"github.com/RootControl/dispatch/llm"
)

// --- normalization and seeding ---

func TestNormalizeEntityCollapsesVariants(t *testing.T) {
	cases := []struct{ in, want string }{
		{"Priya Raman", "priya raman"},
		{"  priya   raman  ", "priya raman"},
		{"The VP of Platform", "vp of platform"},
		{"VP of Platform.", "vp of platform"},
		{"Data-Platform", "data platform"},
		{"", ""},

		// Punctuation separates rather than deleting. Deleting welded words
		// together — "packages/data-provider" keyed as "packagesdata provider".
		{"packages/data-provider", "packages data provider"},
		{"@librechat/agents", "librechat agents"},
		{"/client", "client"},
		{"librechat.ai", "librechat ai"},
		{"packages/api", "packages api"},
	}
	for _, c := range cases {
		if got := normalizeEntity(c.in); got != c.want {
			t.Errorf("normalizeEntity(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestSeedsMatchesEntitiesInQuestion(t *testing.T) {
	g := newGraph()
	g.addEntity("Priya Raman", "person")
	g.addEntity("VP of Platform", "person")
	g.addEntity("Payments", "org")

	got := g.Seeds("Who does Priya Raman report to?")
	if !slices.Contains(got, "priya raman") {
		t.Fatalf("expected priya raman seed, got %v", got)
	}
	if slices.Contains(got, "payments") {
		t.Errorf("unrelated entity seeded: %v", got)
	}
	// Punctuation and case in the question must not matter.
	if got := g.Seeds("priya raman!"); !slices.Contains(got, "priya raman") {
		t.Errorf("punctuation broke seeding: %v", got)
	}
	if got := g.Seeds("Who signed the contract?"); len(got) != 0 {
		t.Errorf("expected no seeds, got %v", got)
	}
}

// Seeding both "platform" and "vp of platform" would traverse from a vaguer
// node for no gain.
func TestSeedsPrefersLongerNames(t *testing.T) {
	g := newGraph()
	g.addEntity("Platform", "org")
	g.addEntity("VP of Platform", "person")

	got := g.Seeds("who is the VP of Platform")
	if slices.Contains(got, "platform") {
		t.Errorf("shorter contained name should be dropped: %v", got)
	}
	if !slices.Contains(got, "vp of platform") {
		t.Errorf("expected the longer name, got %v", got)
	}
}

// --- graph mechanics ---

func TestAddEdgeIgnoresSelfLoopsAndDuplicates(t *testing.T) {
	g := newGraph()
	g.addEdge("Priya", "Priya", "reports to", "c1") // self-loop
	g.addEdge("Priya", "VP", "reports to", "c1")
	g.addEdge("priya", "vp", "REPORTS TO", "c1") // same after normalization
	g.addEdge("Priya", "VP", "", "c1")           // empty relation

	if len(g.Edges) != 1 {
		t.Fatalf("expected 1 edge, got %d: %+v", len(g.Edges), g.Edges)
	}
	// An edge referencing unlisted entities must still register them, or
	// traversal could never reach them.
	if _, ok := g.Entities["vp"]; !ok {
		t.Error("edge endpoint not registered as an entity")
	}
}

// The multi-hop claim: an answer two relationships away must be reachable, and
// hop distance must be recorded so nearer facts can rank higher.
func TestTraverseMultiHop(t *testing.T) {
	g := newGraph()
	g.addEdge("Priya Raman", "VP of Platform", "reports to", "c1")
	g.addEdge("VP of Platform", "Vendor Contract", "signed", "c2")
	g.addEdge("Vendor Contract", "Statement Mailer", "covers", "c3")
	g.addEdge("Unrelated Person", "Other Thing", "owns", "c9")
	g.reindex()

	hits := g.Traverse([]string{"priya raman"}, 2)
	byChunk := map[string]int{}
	for _, h := range hits {
		byChunk[h.Edge.ChunkID] = h.Hop
	}
	if byChunk["c1"] != 1 {
		t.Errorf("c1 should be 1 hop, got %d", byChunk["c1"])
	}
	if byChunk["c2"] != 2 {
		t.Errorf("c2 should be 2 hops, got %d", byChunk["c2"])
	}
	if _, ok := byChunk["c3"]; ok {
		t.Error("c3 is 3 hops away and should be beyond maxHops=2")
	}
	if _, ok := byChunk["c9"]; ok {
		t.Error("disconnected edge should not be reachable")
	}
}

// "Priya reports to the VP" must be findable from either end.
func TestTraverseIsUndirected(t *testing.T) {
	g := newGraph()
	g.addEdge("Priya Raman", "VP of Platform", "reports to", "c1")
	g.reindex()

	if hits := g.Traverse([]string{"vp of platform"}, 1); len(hits) != 1 {
		t.Fatalf("expected to reach the edge from its target, got %d hits", len(hits))
	}
}

func TestTraverseIsDeterministic(t *testing.T) {
	g := newGraph()
	for i := range 12 {
		g.addEdge(fmt.Sprintf("E%d", i), fmt.Sprintf("E%d", i+1), "links to", fmt.Sprintf("c%d", i))
	}
	g.reindex()

	first := g.Traverse([]string{"e0"}, 3)
	for range 10 {
		got := g.Traverse([]string{"e0"}, 3)
		if len(got) != len(first) {
			t.Fatalf("hit count varies: %d vs %d", len(got), len(first))
		}
		for i := range first {
			if got[i].Edge != first[i].Edge || got[i].Hop != first[i].Hop {
				t.Fatalf("traversal not deterministic at %d", i)
			}
		}
	}
}

// --- extraction and retrieval ---

func extractingFake(t *testing.T) *fake.LLM {
	t.Helper()
	return &fake.LLM{ChatFunc: func(msgs []llm.Message) (string, error) {
		body := msgs[len(msgs)-1].Content
		switch {
		case strings.Contains(body, "Priya"):
			return `{"entities":[{"name":"Priya Raman","type":"person"},{"name":"VP of Platform","type":"person"}],
			         "relations":[{"from":"Priya Raman","relation":"reports to","to":"VP of Platform"}]}`, nil
		case strings.Contains(body, "renewal"):
			return `{"entities":[{"name":"VP of Platform","type":"person"},{"name":"Vendor Contract","type":"document"}],
			         "relations":[{"from":"VP of Platform","relation":"signed","to":"Vendor Contract"}]}`, nil
		}
		return `{"entities":[],"relations":[]}`, nil
	}}
}

func relationalStore(t *testing.T, f llm.LLM) *index.Store {
	t.Helper()
	s := index.New(index.Config{LLM: f})
	_, err := s.Ingest(context.Background(), []core.Doc{
		{ID: "staffing", Text: "Priya Raman leads the project and reports to the VP of Platform."},
		{ID: "vendor", Text: "The renewal was signed in January by the VP of Platform."},
		{ID: "weather", Text: "Sunny with light coastal wind."},
	})
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func TestBuildGraphExtractsRelations(t *testing.T) {
	f := extractingFake(t)
	rel, stats, err := BuildGraph(context.Background(), f, relationalStore(t, f), GraphOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if stats.Chunks != 3 || stats.LLMCalls != 3 {
		t.Errorf("expected one call per chunk: chunks=%d calls=%d", stats.Chunks, stats.LLMCalls)
	}
	entities, relations := rel.Stats()
	if relations != 2 {
		t.Errorf("relations = %d, want 2", relations)
	}
	if entities < 3 {
		t.Errorf("entities = %d, want at least 3", entities)
	}
}

// Extraction costs one call per chunk, so a rebuild must be free.
func TestBuildGraphUsesCache(t *testing.T) {
	dir := t.TempDir()
	cache := index.NewCache(dir)
	ctx := context.Background()

	f1 := extractingFake(t)
	_, stats1, err := BuildGraph(ctx, f1, relationalStore(t, f1), GraphOptions{Cache: cache, CacheTag: "m1"})
	if err != nil {
		t.Fatal(err)
	}
	if stats1.LLMCalls == 0 {
		t.Fatal("first build should have made calls")
	}

	f2 := extractingFake(t)
	rel2, stats2, err := BuildGraph(ctx, f2, relationalStore(t, f2), GraphOptions{Cache: cache, CacheTag: "m1"})
	if err != nil {
		t.Fatal(err)
	}
	if stats2.LLMCalls != 0 {
		t.Errorf("second build made %d calls, want 0", stats2.LLMCalls)
	}
	if stats2.CacheHits != stats1.Chunks {
		t.Errorf("CacheHits = %d, want %d", stats2.CacheHits, stats1.Chunks)
	}
	// The cached graph must be equivalent, not merely cheap.
	if _, relations := rel2.Stats(); relations != stats1.Relations {
		t.Errorf("cached rebuild produced %d relations, want %d", relations, stats1.Relations)
	}
}

func TestRelationalRetrieveMultiHop(t *testing.T) {
	ctx := context.Background()
	f := extractingFake(t)
	rel, _, err := BuildGraph(ctx, f, relationalStore(t, f), GraphOptions{})
	if err != nil {
		t.Fatal(err)
	}

	var r core.Retriever = rel
	if r.Tier() != core.TierRelational {
		t.Fatalf("Tier() = %s", r.Tier())
	}
	// Two hops: Priya -> VP (chunk staffing), VP -> contract (chunk vendor).
	got, err := r.Retrieve(ctx, core.Query{Text: "Who does Priya Raman report to and what did they sign?", TopK: 5})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) < 2 {
		t.Fatalf("expected both hops, got %d results", len(got))
	}
	if !strings.Contains(got[0].Text, "reports to") {
		t.Errorf("nearest result should hold the direct relationship:\n%s", got[0].Text)
	}
	joined := got[0].Text + got[1].Text
	if !strings.Contains(joined, "signed") {
		t.Errorf("second hop missing:\n%s", joined)
	}
	// Citations must resolve to real chunks, not to graph-internal ids.
	if !strings.HasPrefix(got[0].Cite(), "[relational:staffing#") {
		t.Errorf("Cite() = %q, want a chunk citation", got[0].Cite())
	}
	if got[0].Meta["hops"] != "1" {
		t.Errorf("nearest result hops = %q, want 1", got[0].Meta["hops"])
	}
	// The supporting excerpt must travel with the relationships.
	if !strings.Contains(got[0].Text, "Priya Raman leads") {
		t.Errorf("source excerpt missing:\n%s", got[0].Text)
	}
}

// No known entity in the question is a gap, not an error — the loop refines.
func TestRelationalNoSeedsReturnsNothing(t *testing.T) {
	ctx := context.Background()
	f := extractingFake(t)
	rel, _, err := BuildGraph(ctx, f, relationalStore(t, f), GraphOptions{})
	if err != nil {
		t.Fatal(err)
	}
	got, err := rel.Retrieve(ctx, core.Query{Text: "what is the airspeed of a swallow?", TopK: 5})
	if err != nil {
		t.Fatalf("unknown entities should not error: %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("expected no results, got %d", len(got))
	}
}

func TestBuildGraphRejectsEmptyStore(t *testing.T) {
	f := &fake.LLM{}
	if _, _, err := BuildGraph(context.Background(), f, index.New(index.Config{LLM: f}), GraphOptions{}); err == nil {
		t.Fatal("expected an error building over an empty store")
	}
}

func TestGraphSaveLoad(t *testing.T) {
	ctx := context.Background()
	f := extractingFake(t)
	rel, _, err := BuildGraph(ctx, f, relationalStore(t, f), GraphOptions{})
	if err != nil {
		t.Fatal(err)
	}
	want, err := rel.Retrieve(ctx, core.Query{Text: "Who does Priya Raman report to?", TopK: 5})
	if err != nil {
		t.Fatal(err)
	}

	path := filepath.Join(t.TempDir(), "graph.json")
	if err := rel.Save(path); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadGraph(path, 2)
	if err != nil {
		t.Fatal(err)
	}
	gotEnt, gotRel := loaded.Stats()
	wantEnt, wantRel := rel.Stats()
	if gotEnt != wantEnt || gotRel != wantRel {
		t.Fatalf("loaded graph = %d entities/%d relations, want %d/%d", gotEnt, gotRel, wantEnt, wantRel)
	}
	got, err := loaded.Retrieve(ctx, core.Query{Text: "Who does Priya Raman report to?", TopK: 5})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Fatalf("loaded graph returned %d results, want %d", len(got), len(want))
	}
	// Excerpts must survive, or reloaded results lose their supporting text.
	if !strings.Contains(got[0].Text, "Priya Raman leads") {
		t.Errorf("excerpt lost across save/load:\n%s", got[0].Text)
	}
}

// A corrupt cache entry must not fail the build.
func TestBuildGraphSurvivesCorruptCache(t *testing.T) {
	dir := t.TempDir()
	cache := index.NewCache(dir)
	ctx := context.Background()
	f := extractingFake(t)
	store := relationalStore(t, f)

	chunks, _ := store.Entries()
	if err := cache.Put(index.Key("graph|default", chunks[0].DocID, chunks[0].Embedded()), "{not json"); err != nil {
		t.Fatal(err)
	}
	_, stats, err := BuildGraph(ctx, f, store, GraphOptions{Cache: cache})
	if err != nil {
		t.Fatalf("corrupt cache entry should not fail the build: %v", err)
	}
	if stats.LLMCalls == 0 {
		t.Error("expected re-extraction after the corrupt entry")
	}
}

func TestExtractionJSONShape(t *testing.T) {
	// Guards the contract the prompt asks the model for.
	var ex extraction
	raw := `{"entities":[{"name":"A","type":"person"}],"relations":[{"from":"A","relation":"owns","to":"B"}]}`
	if err := json.Unmarshal([]byte(raw), &ex); err != nil {
		t.Fatal(err)
	}
	if len(ex.Entities) != 1 || ex.Entities[0].Name != "A" {
		t.Errorf("entities decoded wrong: %+v", ex.Entities)
	}
	if len(ex.Relations) != 1 || ex.Relations[0].Relation != "owns" {
		t.Errorf("relations decoded wrong: %+v", ex.Relations)
	}
}

// A long extraction run must not be discarded because one chunk failed. On a
// real corpus this runs for tens of minutes; aborting at chunk 60 of 70 throws
// away every earlier call.
func TestBuildGraphSkipsUnextractableChunks(t *testing.T) {
	var n int
	f := &fake.LLM{ChatFunc: func(msgs []llm.Message) (string, error) {
		n++
		if n == 2 { // the middle chunk fails
			return "", errors.New("model returned reasoning but no answer")
		}
		if strings.Contains(msgs[len(msgs)-1].Content, "Priya") {
			return `{"entities":[{"name":"Priya Raman","type":"person"}],
			         "relations":[{"from":"Priya Raman","relation":"reports to","to":"VP of Platform"}]}`, nil
		}
		return `{"entities":[],"relations":[]}`, nil
	}}

	rel, stats, err := BuildGraph(context.Background(), f, relationalStore(t, f), GraphOptions{})
	if err != nil {
		t.Fatalf("one bad chunk should not fail the build: %v", err)
	}
	if stats.Skipped != 1 {
		t.Errorf("Skipped = %d, want 1", stats.Skipped)
	}
	if stats.FirstError == nil {
		t.Error("a skipped chunk should be explainable, not just counted")
	}
	// The surviving chunks must still have produced a usable graph.
	if _, relations := rel.Stats(); relations == 0 {
		t.Error("expected relations from the chunks that did extract")
	}
}

// Path-like names must not weld into one token, and a question written with
// spaces must reach an entity written with slashes.
func TestNormalizeEntitySeparatesPathSegments(t *testing.T) {
	if got := normalizeEntity("packages/api"); got == normalizeEntity("packagesapi") {
		t.Errorf("packages/api and packagesapi should not collide (both %q)", got)
	}
	g := newGraph()
	g.addEntity("packages/data-provider", "system")
	for _, phrasing := range []string{
		"what depends on packages/data-provider?",
		"what depends on packages data provider?",
		"What Depends On PACKAGES/DATA-PROVIDER?",
	} {
		if seeds := g.Seeds(phrasing); !slices.Contains(seeds, "packages data provider") {
			t.Errorf("Seeds(%q) = %v, want the packages/data-provider entity", phrasing, seeds)
		}
	}
}
