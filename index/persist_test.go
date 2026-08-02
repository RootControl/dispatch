package index

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/RootControl/dispatch/core"
	"github.com/RootControl/dispatch/internal/fake"
)

func TestSaveLoadRoundTrip(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "index.json")

	orig := New(Config{LLM: &fake.LLM{}, EmbedTag: "embed-v1"})
	if _, err := orig.Ingest(ctx, []core.Doc{{ID: "atlas", Text: atlasDoc}}); err != nil {
		t.Fatal(err)
	}
	want, err := orig.Search(ctx, core.Query{Text: "atlas budget", TopK: 4})
	if err != nil {
		t.Fatal(err)
	}
	if err := orig.Save(path); err != nil {
		t.Fatal(err)
	}

	// A fresh store loading that file must rank identically — proving the
	// rebuilt BM25 index and restored vectors match the originals.
	loaded := New(Config{LLM: &fake.LLM{}, EmbedTag: "embed-v1"})
	if err := loaded.Load(path); err != nil {
		t.Fatal(err)
	}
	if loaded.Len() != orig.Len() {
		t.Fatalf("chunk count %d, want %d", loaded.Len(), orig.Len())
	}
	got, err := loaded.Search(ctx, core.Query{Text: "atlas budget", TopK: 4})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Fatalf("result count %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i].Chunk.ID != want[i].Chunk.ID || got[i].Score != want[i].Score {
			t.Fatalf("result %d = (%s, %.6f), want (%s, %.6f)",
				i, got[i].Chunk.ID, got[i].Score, want[i].Chunk.ID, want[i].Score)
		}
	}
	// Context sentences must survive the round trip, or retrieval quality
	// silently degrades on every subsequent load.
	for _, h := range got {
		if h.Chunk.Context == "" && orig.chunks[h.Chunk.ID].Context != "" {
			t.Fatalf("chunk %s lost its context sentence", h.Chunk.ID)
		}
	}
}

// Loading an index built with a different embedding model must fail loudly:
// mixed embedding spaces return plausible nonsense rather than erroring.
func TestLoadRejectsEmbedModelMismatch(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "index.json")

	built := New(Config{LLM: &fake.LLM{}, EmbedTag: "embed-v1"})
	if _, err := built.Ingest(ctx, []core.Doc{{ID: "atlas", Text: atlasDoc}}); err != nil {
		t.Fatal(err)
	}
	if err := built.Save(path); err != nil {
		t.Fatal(err)
	}

	other := New(Config{LLM: &fake.LLM{}, EmbedTag: "embed-v2"})
	if err := other.Load(path); err == nil {
		t.Fatal("expected an error loading an index built with a different embedding model")
	}
}
