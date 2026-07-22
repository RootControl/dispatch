package tiers

import (
	"context"
	"fmt"
	"math"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/RootControl/dispatch/core"
	"github.com/RootControl/dispatch/index"
	"github.com/RootControl/dispatch/internal/fake"
	"github.com/RootControl/dispatch/llm"
)

// --- clustering ---

func unit(v ...float64) []float64 {
	var s float64
	for _, x := range v {
		s += x * x
	}
	n := math.Sqrt(s)
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = x / n
	}
	return out
}

func TestKmeansSeparatesObviousGroups(t *testing.T) {
	// Two tight groups on opposite axes.
	vecs := [][]float64{
		unit(1, 0.02), unit(1, 0.01), unit(1, 0.03),
		unit(0.02, 1), unit(0.01, 1), unit(0.03, 1),
	}
	clusters := kmeans(vecs, 2, 10)
	if len(clusters) != 2 {
		t.Fatalf("expected 2 clusters, got %d", len(clusters))
	}
	for _, c := range clusters {
		if len(c) != 3 {
			t.Fatalf("expected an even 3/3 split, got sizes %v", sizes(clusters))
		}
		// Every member must come from the same group (indices 0-2 or 3-5).
		firstGroup := c[0] < 3
		for _, i := range c {
			if (i < 3) != firstGroup {
				t.Errorf("cluster mixes groups: %v", c)
			}
		}
	}
}

// A tree rebuild that silently reshuffled clusters would change every citation.
func TestKmeansIsDeterministic(t *testing.T) {
	var vecs [][]float64
	for i := range 20 {
		vecs = append(vecs, unit(float64(i%5)+1, float64(i%3)+1, float64(i%7)+1))
	}
	first := kmeans(vecs, 4, 10)
	for range 10 {
		got := kmeans(vecs, 4, 10)
		if len(got) != len(first) {
			t.Fatalf("cluster count varies: %d vs %d", len(got), len(first))
		}
		for i := range first {
			if !slices.Equal(got[i], first[i]) {
				t.Fatalf("clustering not deterministic:\n%v\n%v", got, first)
			}
		}
	}
}

func TestKmeansEdgeCases(t *testing.T) {
	vecs := [][]float64{unit(1, 0), unit(0, 1), unit(1, 1)}
	if got := kmeans(vecs, 1, 5); len(got) != 1 || len(got[0]) != 3 {
		t.Errorf("k=1 should yield one cluster of everything, got %v", got)
	}
	if got := kmeans(vecs, 10, 5); len(got) != 3 {
		t.Errorf("k>n should yield one cluster per vector, got %d", len(got))
	}
	if got := kmeans(nil, 2, 5); got != nil {
		t.Errorf("empty input should yield nil, got %v", got)
	}
	// No cluster may be empty — an empty cluster would summarize nothing.
	for _, c := range kmeans(vecs, 3, 5) {
		if len(c) == 0 {
			t.Error("empty cluster returned")
		}
	}
}

func sizes(clusters [][]int) []int {
	out := make([]int, len(clusters))
	for i, c := range clusters {
		out[i] = len(c)
	}
	return out
}

// --- tree construction ---

// leafStore builds a store of n short docs without contextual chunking.
func leafStore(t *testing.T, f llm.LLM, texts ...string) *index.Store {
	t.Helper()
	s := index.New(index.Config{LLM: f})
	docs := make([]core.Doc, len(texts))
	for i, txt := range texts {
		docs[i] = core.Doc{ID: fmt.Sprintf("d%d", i), Text: txt}
	}
	if _, err := s.Ingest(context.Background(), docs); err != nil {
		t.Fatal(err)
	}
	return s
}

func summarizingFake() *fake.LLM {
	return &fake.LLM{ChatFunc: func(msgs []llm.Message) (string, error) {
		// Echo the excerpt content so summaries stay searchable in tests.
		body := msgs[len(msgs)-1].Content
		body = strings.ReplaceAll(body, "<excerpt", " ")
		body = strings.ReplaceAll(body, "</excerpt", " ")
		return "Summary covering: " + strings.Join(strings.Fields(body), " "), nil
	}}
}

func TestBuildHierarchyReducesToRoot(t *testing.T) {
	f := summarizingFake()
	leaves := leafStore(t, f,
		"Budget four million dollars approved.",
		"Contingency reserve twelve percent allocated.",
		"Vendor contract renews every January.",
		"Second vendor evaluated but not retained.",
		"Schedule slipped one quarter.",
		"Testing compressed against staging replica.",
	)

	h, stats, err := BuildHierarchy(context.Background(), f, leaves, HierarchyOptions{Branching: 3})
	if err != nil {
		t.Fatal(err)
	}
	if stats.Levels < 2 {
		t.Errorf("6 leaves at branching 3 should build at least 2 levels, got %d", stats.Levels)
	}
	if h.Len() != stats.Summaries {
		t.Errorf("indexed %d nodes but reported %d summaries", h.Len(), stats.Summaries)
	}
	// The tree must actually converge, not stall at a level that never shrinks.
	if stats.Summaries >= leaves.Len() {
		t.Errorf("summaries (%d) should be fewer than leaves (%d)", stats.Summaries, leaves.Len())
	}
}

// A degenerate branching factor must still terminate.
func TestBuildHierarchyTerminatesWithBranchingTooSmall(t *testing.T) {
	f := summarizingFake()
	leaves := leafStore(t, f, "alpha one.", "beta two.", "gamma three.", "delta four.")

	_, stats, err := BuildHierarchy(context.Background(), f, leaves, HierarchyOptions{Branching: 2, MaxLevels: 6})
	if err != nil {
		t.Fatal(err)
	}
	if stats.Levels > 6 {
		t.Fatalf("exceeded MaxLevels: %d", stats.Levels)
	}
	if stats.LLMCalls == 0 {
		t.Fatal("expected summarization calls")
	}
}

func TestBuildHierarchyRejectsEmptyStore(t *testing.T) {
	f := &fake.LLM{}
	empty := index.New(index.Config{LLM: f})
	if _, _, err := BuildHierarchy(context.Background(), f, empty, HierarchyOptions{}); err == nil {
		t.Fatal("expected an error building over an empty store")
	}
}

// Reusing leaf embeddings is the point: only summaries should be embedded.
func TestBuildHierarchyDoesNotReEmbedLeaves(t *testing.T) {
	f := summarizingFake()
	leaves := leafStore(t, f, "one.", "two.", "three.", "four.", "five.", "six.")

	_, stats, err := BuildHierarchy(context.Background(), f, leaves, HierarchyOptions{Branching: 3})
	if err != nil {
		t.Fatal(err)
	}
	// Every summary costs one chat call; leaves cost none.
	if stats.LLMCalls != stats.Summaries {
		t.Errorf("LLMCalls=%d should equal Summaries=%d", stats.LLMCalls, stats.Summaries)
	}
	if stats.Summaries > leaves.Len() {
		t.Errorf("summarized more nodes (%d) than there are leaves (%d)", stats.Summaries, leaves.Len())
	}
}

// --- retrieval and persistence ---

func TestHierarchicalRetrieveCites(t *testing.T) {
	f := summarizingFake()
	leaves := leafStore(t, f,
		"Budget four million dollars approved.",
		"Vendor contract renews every January.",
		"Schedule slipped one quarter.",
		"Testing compressed against staging replica.",
	)
	h, _, err := BuildHierarchy(context.Background(), f, leaves, HierarchyOptions{Branching: 2})
	if err != nil {
		t.Fatal(err)
	}

	var r core.Retriever = h
	if r.Tier() != core.TierHierarchical {
		t.Fatalf("Tier() = %s", r.Tier())
	}
	got, err := r.Retrieve(context.Background(), core.Query{Text: "vendor contract", TopK: 3})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) == 0 {
		t.Fatal("expected summary hits")
	}
	if !strings.HasPrefix(got[0].Cite(), "[hierarchical:L") {
		t.Errorf("Cite() = %q, want [hierarchical:L<level>-<n>]", got[0].Cite())
	}
	if got[0].Meta["level"] == "" {
		t.Error("results should record which level they came from")
	}
}

func TestHierarchicalEmptyRetrieve(t *testing.T) {
	h := &Hierarchical{nodes: index.New(index.Config{LLM: &fake.LLM{}})}
	got, err := h.Retrieve(context.Background(), core.Query{Text: "anything", TopK: 3})
	if err != nil {
		t.Fatalf("empty tree should not error: %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("expected no results, got %d", len(got))
	}
}

// A tree costs one LLM call per cluster to build, so it must survive a restart.
func TestHierarchySaveLoad(t *testing.T) {
	ctx := context.Background()
	f := summarizingFake()
	leaves := leafStore(t, f, "alpha budget.", "beta vendor.", "gamma schedule.", "delta testing.")
	h, _, err := BuildHierarchy(ctx, f, leaves, HierarchyOptions{Branching: 2})
	if err != nil {
		t.Fatal(err)
	}
	want, err := h.Retrieve(ctx, core.Query{Text: "vendor", TopK: 3})
	if err != nil {
		t.Fatal(err)
	}

	path := filepath.Join(t.TempDir(), "hierarchy.json")
	if err := h.Save(path); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadHierarchy(f, path)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Len() != h.Len() {
		t.Fatalf("loaded %d nodes, want %d", loaded.Len(), h.Len())
	}
	got, err := loaded.Retrieve(ctx, core.Query{Text: "vendor", TopK: 3})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Fatalf("loaded tree returned %d results, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i].SourceID != want[i].SourceID {
			t.Errorf("result %d = %s, want %s", i, got[i].SourceID, want[i].SourceID)
		}
	}
}
