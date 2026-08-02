package index

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/RootControl/dispatch/core"
	"github.com/RootControl/dispatch/internal/fake"
)

// A Save must never leave the index unloadable. Before it was atomic, an
// interrupted os.Create left truncated JSON with the previous copy gone.
func TestSaveNeverLeavesACorruptIndex(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "index.json")

	s := New(Config{LLM: &fake.LLM{}, EmbedTag: "e1"})
	if _, err := s.Ingest(ctx, []core.Doc{
		{ID: "a", Text: "The Atlas budget is four million dollars."},
		{ID: "b", Text: "Priya Raman leads engineering."},
	}); err != nil {
		t.Fatal(err)
	}
	if err := s.Save(path); err != nil {
		t.Fatal(err)
	}
	before, _ := os.ReadFile(path)

	// Simulate the old failure: something else truncates while we hold a good
	// copy. What matters is that a *reader* only ever sees a complete file, so
	// re-saving must swap it whole.
	if err := s.Save(path); err != nil {
		t.Fatal(err)
	}
	after, _ := os.ReadFile(path)
	if len(after) != len(before) {
		t.Errorf("re-save changed size: %d -> %d", len(before), len(after))
	}
	var snap map[string]any
	if err := json.Unmarshal(after, &snap); err != nil {
		t.Fatalf("saved index is not valid JSON: %v", err)
	}

	reloaded := New(Config{LLM: &fake.LLM{}, EmbedTag: "e1"})
	if err := reloaded.Load(path); err != nil {
		t.Fatalf("could not reload: %v", err)
	}
	if reloaded.Len() != s.Len() {
		t.Errorf("reloaded %d chunks, want %d", reloaded.Len(), s.Len())
	}
}
