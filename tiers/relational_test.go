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

// --- coreference ---

// alwaysSame isolates the lexical candidate rules from model adjudication.
func alwaysSame(string, string) bool { return true }

func TestCanonicalizeMergesUnambiguousSuffixVariants(t *testing.T) {
	g := newGraph()
	g.addEdge("Project Atlas", "Invoice Generation", "covers", "c1")
	g.addEdge("Atlas", "Legacy Billing Pipeline", "replaces", "c2")
	g.addEdge("Single External Vendor", "Statement Mailer", "serves", "c3")
	g.addEdge("Statement Mailer", "External Vendor", "depends on", "c4")

	merges := g.canonicalize(alwaysSame)
	g.reindex()

	if _, stillThere := g.Entities["atlas"]; stillThere {
		t.Error(`"atlas" should have merged into "project atlas"`)
	}
	if _, ok := g.Entities["project atlas"]; !ok {
		t.Error(`"project atlas" should survive as the canonical form`)
	}
	if _, stillThere := g.Entities["external vendor"]; stillThere {
		t.Error(`"external vendor" should have merged into "single external vendor"`)
	}
	if len(merges) != 2 {
		t.Errorf("expected 2 merges, got %d: %+v", len(merges), merges)
	}

	// The payoff: both variants now reach one node, so a question naming either
	// traverses the union of their edges.
	hits := g.Traverse(g.Seeds("what does Atlas cover and replace?"), 1)
	chunks := map[string]bool{}
	for _, h := range hits {
		chunks[h.Edge.ChunkID] = true
	}
	if !chunks["c1"] || !chunks["c2"] {
		t.Errorf("merged node should reach edges from both variants, got %v", chunks)
	}
}

// The rule must refuse the cases that would fabricate relationships. Every one
// of these is from a real corpus.
func TestCanonicalizeRefusesAmbiguousAndModifierVariants(t *testing.T) {
	g := newGraph()
	// "api" is a suffix of four distinct things — merging would invent edges.
	for _, name := range []string{"packages api", "rag api", "code interpreter api", "openai responses api"} {
		g.addEdge(name, "Something", "used by", "c1")
	}
	g.addEdge("api", "Express Server", "is", "c2")
	// "migration" is a suffix of two different migrations.
	g.addEdge("infrastructure migration", "Q2", "slipped in", "c3")
	g.addEdge("tax engine migration", "Later", "deferred to", "c4")
	g.addEdge("migration", "Something Else", "relates to", "c5")
	// "infrastructure" is a subset of "infrastructure migration" but not a
	// suffix: infrastructure is not the migration of it.
	g.addEdge("infrastructure", "Budget", "allocated from", "c6")

	g.canonicalize(alwaysSame)

	for _, mustSurvive := range []string{"api", "migration", "infrastructure"} {
		if _, ok := g.Entities[mustSurvive]; !ok {
			t.Errorf("%q was merged away; it is ambiguous or a modifier and must stay separate", mustSurvive)
		}
	}
	if _, ok := g.Entities["packages api"]; !ok {
		t.Error(`"packages api" must remain distinct from "api"`)
	}
}

func TestCanonicalizeReportsMergesForAudit(t *testing.T) {
	g := newGraph()
	g.addEdge("Project Atlas", "Thing", "covers", "c1")
	g.addEdge("Atlas", "Other", "replaces", "c2")

	merges := g.canonicalize(alwaysSame)
	if len(merges) != 1 {
		t.Fatalf("expected 1 merge, got %+v", merges)
	}
	// Display names, not normalized keys — the report is for a human.
	if merges[0].From != "Atlas" || merges[0].Into != "Project Atlas" {
		t.Errorf("merge = %+v, want Atlas -> Project Atlas", merges[0])
	}
}

// Merging must not leave an entity pointing at itself.
func TestCanonicalizeDropsSelfLoops(t *testing.T) {
	g := newGraph()
	g.addEdge("Project Atlas", "Atlas", "also known as", "c1")
	g.addEdge("Project Atlas", "Budget", "has", "c2")

	g.canonicalize(alwaysSame)
	for _, e := range g.Edges {
		if e.From == e.To {
			t.Errorf("self-loop survived canonicalization: %+v", e)
		}
	}
}

// The extractor sometimes emits a list as one entity name. Those are useless as
// nodes and dangerous for canonicalize, which would absorb the real "Gemini"
// into the list and inherit every relationship the list had.
func TestEnumerationsAreRejected(t *testing.T) {
	g := newGraph()
	g.addEntity("Claude 3, GPT-4.5, o1, and Gemini", "other")
	g.addEntity("Gemini", "system")
	g.addEdge("Custom Endpoints, OpenAI, and Google", "LibreChat", "supported by", "c1")

	if len(g.Entities) != 1 {
		t.Fatalf("expected only the real entity, got %v", g.Entities)
	}
	if _, ok := g.Entities["gemini"]; !ok {
		t.Error("the genuine entity should survive")
	}
	if len(g.Edges) != 0 {
		t.Errorf("an edge with a list endpoint should be dropped, got %+v", g.Edges)
	}
}

// A merge must be a modest elaboration, not absorption by a long phrase that
// happens to end with the same word.
func TestCanonicalizeBoundsMergeGrowth(t *testing.T) {
	g := newGraph()
	g.addEdge("Gemini", "Vertex", "served by", "c1")
	g.addEdge("a very long unrelated phrase about gemini", "Something", "mentions", "c2")

	g.canonicalize(alwaysSame)
	if _, ok := g.Entities["gemini"]; !ok {
		t.Error("gemini should not be absorbed by a much longer phrase")
	}
}

// The adjudicator is what separates "Atlas"/"Project Atlas" from
// "OpenAI"/"Azure OpenAI" — lexically identical, semantically opposite.
func TestCanonicalizeRespectsAdjudicator(t *testing.T) {
	build := func() *Graph {
		g := newGraph()
		g.addEdge("Project Atlas", "Budget", "has", "c1")
		g.addEdge("Atlas", "Pipeline", "replaces", "c2")
		g.addEdge("Azure OpenAI", "Endpoint", "provides", "c3")
		g.addEdge("OpenAI", "Models", "provides", "c4")
		return g
	}

	// A rejecting adjudicator merges nothing, even though both pairs are
	// lexically identical in shape.
	g := build()
	if merges := g.canonicalize(func(string, string) bool { return false }); len(merges) != 0 {
		t.Errorf("a rejecting adjudicator should merge nothing, got %+v", merges)
	}

	// A selective one merges only what it accepts.
	g = build()
	merges := g.canonicalize(func(short, long string) bool {
		return !strings.Contains(long, "Azure")
	})
	if len(merges) != 1 || merges[0].From != "Atlas" {
		t.Fatalf("expected only the Atlas merge, got %+v", merges)
	}
	if _, ok := g.Entities["openai"]; !ok {
		t.Error("OpenAI must stay distinct from Azure OpenAI")
	}
}

// A failed adjudication must not be read as permission to merge.
func TestConfirmCoreferenceDefaultsToNoOnError(t *testing.T) {
	f := &fake.LLM{ChatFunc: func([]llm.Message) (string, error) {
		return "", errors.New("endpoint down")
	}}
	confirm, fails, firstErr := confirmCoreference(context.Background(), f, index.NewCache(t.TempDir()), "m1")
	if confirm("Atlas", "Project Atlas") {
		t.Error("an unanswered question is not a licence to merge")
	}
	// ...but the failure must be visible, or it looks like a considered "no".
	// One failure per vote attempt.
	if *fails != corefVotes || *firstErr == nil {
		t.Errorf("failure not recorded: fails=%d err=%v", *fails, *firstErr)
	}
}

// Verdicts are cached, so a rebuild costs nothing.
func TestConfirmCoreferenceCachesVerdicts(t *testing.T) {
	var calls int
	cache := index.NewCache(t.TempDir())
	f := &fake.LLM{ChatFunc: func([]llm.Message) (string, error) {
		calls++
		return `{"same": true}`, nil
	}}
	confirm, _, _ := confirmCoreference(context.Background(), f, cache, "m1")
	for range 3 {
		if !confirm("Atlas", "Project Atlas") {
			t.Fatal("expected a merge verdict")
		}
	}
	if calls != corefVotes {
		t.Errorf("made %d calls, want %d (one vote round, rest cached)", calls, corefVotes)
	}
}

// A stochastic judge must not decide by one sample. gemma4 answered both ways
// on the same pair; a minority "yes" must not be enough to merge.
func TestConfirmCoreferenceTakesMajority(t *testing.T) {
	cases := map[string]struct {
		replies []string
		want    bool
	}{
		"unanimous yes": {[]string{"true", "true", "true"}, true},
		"majority yes":  {[]string{"true", "false", "true"}, true},
		"minority yes":  {[]string{"true", "false", "false"}, false},
		"unanimous no":  {[]string{"false", "false", "false"}, false},
	}
	for name, tc := range cases {
		var i int
		f := &fake.LLM{ChatFunc: func([]llm.Message) (string, error) {
			r := tc.replies[i%len(tc.replies)]
			i++
			return `{"same": ` + r + `}`, nil
		}}
		confirm, _, _ := confirmCoreference(context.Background(), f, index.NewCache(t.TempDir()), "m1")
		if got := confirm("Atlas", "Project Atlas"); got != tc.want {
			t.Errorf("%s: got %v, want %v", name, got, tc.want)
		}
	}
}

// --- partial-name seeding ---

// "Who does Priya report to?" must reach "Priya Raman". The corpus only ever
// writes the full name, so there is no "priya" node to merge — the gap is in
// seeding.
func TestSeedsResolvePartialPersonName(t *testing.T) {
	g := newGraph()
	g.addEntity("Priya Raman", "person")
	g.addEntity("VP of Platform", "person")
	g.addEdge("Priya Raman", "VP of Platform", "reports to", "c1")
	g.reindex()

	for _, q := range []string{
		"Who does Priya report to?",
		"who does priya report to",
		"Tell me about PRIYA.",
	} {
		if seeds := g.Seeds(q); !slices.Contains(seeds, "priya raman") {
			t.Errorf("Seeds(%q) = %v, want priya raman", q, seeds)
		}
	}
	// The full name must still work.
	if seeds := g.Seeds("Who does Priya Raman report to?"); !slices.Contains(seeds, "priya raman") {
		t.Errorf("full name broke: %v", g.Seeds("Who does Priya Raman report to?"))
	}
	// And the edge must actually be reachable from the partial-name seed.
	if hits := g.Traverse(g.Seeds("who does priya report to"), 1); len(hits) != 1 {
		t.Errorf("expected to reach the reporting edge, got %d hits", len(hits))
	}
}

// Two people sharing a first name make the reference genuinely ambiguous.
func TestSeedsRefuseAmbiguousPartialNames(t *testing.T) {
	g := newGraph()
	g.addEntity("Priya Raman", "person")
	g.addEntity("Priya Sharma", "person")
	g.addEdge("Priya Raman", "VP of Platform", "reports to", "c1")

	seeds := g.Seeds("Who does Priya report to?")
	for _, s := range seeds {
		if strings.HasPrefix(s, "priya") {
			t.Errorf("ambiguous first name should seed neither, got %v", seeds)
		}
	}
	// Disambiguating by surname still works.
	if got := g.Seeds("Who does Priya Sharma report to?"); !slices.Contains(got, "priya sharma") {
		t.Errorf("full name should still resolve: %v", got)
	}
}

// Partial matching is for people. Allowing it generally would make every path
// prefix a seed — "packages" would pull in "packages/api".
func TestSeedsPartialMatchingIsPeopleOnly(t *testing.T) {
	g := newGraph()
	g.addEntity("packages/api", "system")
	g.addEntity("Azure OpenAI", "org")

	if got := g.Seeds("what does packages depend on?"); len(got) != 0 {
		t.Errorf("a non-person prefix should not seed: %v", got)
	}
	if got := g.Seeds("tell me about azure"); len(got) != 0 {
		t.Errorf("a non-person prefix should not seed: %v", got)
	}
}

// Candidate generation must not be all-pairs: a graph large enough to matter
// would otherwise spend quadratic time before any LLM call is made.
//
// This is the worst case for the bucketing — every entity shares a final token,
// so one bucket holds half of them. Real corpora spread across many buckets.
func BenchmarkCanonicalizeCandidates(b *testing.B) {
	g := newGraph()
	for i := range 5000 {
		g.addEntity(fmt.Sprintf("entity %d alpha", i), "system")
		g.addEntity(fmt.Sprintf("thing %d beta", i), "other")
	}
	b.ResetTimer()
	for b.Loop() {
		g.canonicalize(func(string, string) bool { return false })
	}
}
