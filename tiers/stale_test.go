package tiers

import (
	"context"
	"path/filepath"
	"strings"
	"testing"

	"github.com/RootControl/dispatch/core"
	"github.com/RootControl/dispatch/index"
	"github.com/RootControl/dispatch/internal/fake"
	"github.com/RootControl/dispatch/llm"
)

func vendorLLM() *fake.LLM {
	return &fake.LLM{ChatFunc: func(msgs []llm.Message) (string, error) {
		body := msgs[len(msgs)-1].Content
		if strings.Contains(strings.ToLower(body), "mailwright") {
			return `{"entities":[{"name":"Mailwright","type":"org"},{"name":"Atlas","type":"project"}],
			         "relations":[{"from":"Mailwright","to":"Atlas","relation":"supplies"}]}`, nil
		}
		return `{"entities":[],"relations":[]}`, nil
	}}
}

func seededStore(t *testing.T, f llm.LLM, docs ...core.Doc) *index.Store {
	t.Helper()
	s := index.New(index.Config{LLM: f, EmbedTag: "e1"})
	if _, err := s.Ingest(context.Background(), docs); err != nil {
		t.Fatal(err)
	}
	return s
}

// The bug this exists to catch. Prune removes a document's chunks from the
// index; the graph is a separate file, so it kept them and the relational tier
// kept citing them. Citation verification passed those answers, because the
// marker did resolve to retrieved evidence — the evidence was what had gone.
func TestGraphGenerationChangesWhenDocumentsArePruned(t *testing.T) {
	ctx := context.Background()
	f := vendorLLM()
	store := seededStore(t, f,
		core.Doc{ID: "vendor.md", Text: "Mailwright supplies the Atlas statement mailer."},
		core.Doc{ID: "budget.md", Text: "The Atlas budget is four million dollars."},
	)

	rel, _, err := BuildGraph(ctx, f, store, GraphOptions{})
	if err != nil {
		t.Fatal(err)
	}
	built := rel.SourceGeneration()
	if built == "" {
		t.Fatal("the graph recorded no source generation")
	}
	if built != store.Generation() {
		t.Fatalf("graph stamped %s, index is %s", built, store.Generation())
	}

	// The corpus loses a document, exactly as an incremental ingest does.
	store.Prune([]string{"budget.md"})

	// The graph still holds the pruned document's chunks — that is inherent to
	// it being a separate artifact. What must be true is that the mismatch is
	// now detectable rather than silent.
	if rel.SourceGeneration() == store.Generation() {
		t.Fatal("the graph still claims to match an index it no longer describes")
	}
	got, err := rel.Retrieve(ctx, core.Query{Text: "Mailwright", TopK: 5})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("stale graph would still return %d result(s) — caught by the generation stamp", len(got))
}

func TestHierarchyRecordsSourceGeneration(t *testing.T) {
	f := summarizingFake()
	leaves := leafStore(t, f, "alpha one.", "beta two.", "gamma three.", "delta four.")

	h, _, err := BuildHierarchy(context.Background(), f, leaves, HierarchyOptions{Branching: 2, EmbedTag: "e1"})
	if err != nil {
		t.Fatal(err)
	}
	if h.SourceGeneration() != leaves.Generation() {
		t.Fatalf("tree stamped %q, leaves are %q", h.SourceGeneration(), leaves.Generation())
	}

	// It has to survive the round trip, or the check is only ever done in the
	// process that built it — which is the one process that cannot be wrong.
	path := filepath.Join(t.TempDir(), "hierarchy.json")
	if err := h.Save(path); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadHierarchy(f, path)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.SourceGeneration() != leaves.Generation() {
		t.Errorf("after save/load the stamp is %q, want %q", loaded.SourceGeneration(), leaves.Generation())
	}
}

func TestGraphSourceGenerationSurvivesSaveLoad(t *testing.T) {
	ctx := context.Background()
	f := vendorLLM()
	store := seededStore(t, f, core.Doc{ID: "vendor.md", Text: "Mailwright supplies the Atlas statement mailer."})
	rel, _, err := BuildGraph(ctx, f, store, GraphOptions{})
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
	if loaded.SourceGeneration() != store.Generation() {
		t.Errorf("after save/load the stamp is %q, want %q", loaded.SourceGeneration(), store.Generation())
	}
}

// Generation must move when the chunk set moves and hold still when it does
// not, or the check either never fires or fires constantly.
func TestGenerationTracksTheChunkSet(t *testing.T) {
	ctx := context.Background()
	f := &fake.LLM{}
	docs := []core.Doc{
		{ID: "a.md", Text: "The Atlas budget is four million dollars."},
		{ID: "b.md", Text: "Priya Raman leads engineering."},
	}
	store := seededStore(t, f, docs...)
	gen := store.Generation()

	// Re-ingesting unchanged documents must not move it, or every ingest would
	// invalidate artifacts that are still perfectly valid.
	if _, err := store.Ingest(ctx, docs); err != nil {
		t.Fatal(err)
	}
	if store.Generation() != gen {
		t.Error("generation moved on a no-op re-ingest")
	}

	store.Prune([]string{"a.md"})
	if store.Generation() == gen {
		t.Error("generation held still after a document was pruned")
	}
}
