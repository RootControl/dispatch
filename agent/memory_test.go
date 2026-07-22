package agent

import (
	"context"
	"strings"
	"testing"

	"github.com/RootControl/dispatch/core"
	"github.com/RootControl/dispatch/internal/fake"
	"github.com/RootControl/dispatch/llm"
)

func newMemory(t *testing.T, budget int, f llm.LLM) *Memory {
	t.Helper()
	if f == nil {
		f = &fake.LLM{}
	}
	return NewMemory(MemoryConfig{LLM: f, Dir: t.TempDir(), CoreBudget: budget})
}

func TestRememberFillsCoreBlock(t *testing.T) {
	m := newMemory(t, 2000, nil)
	ctx := context.Background()
	if err := m.Remember(ctx, "The Atlas budget is four million dollars."); err != nil {
		t.Fatal(err)
	}
	if err := m.Remember(ctx, "Priya Raman leads Atlas."); err != nil {
		t.Fatal(err)
	}
	block := m.CoreBlock()
	for _, want := range []string{"four million", "Priya Raman"} {
		if !strings.Contains(block, want) {
			t.Errorf("core block missing %q:\n%s", want, block)
		}
	}
	if m.CoreLen() != 2 {
		t.Errorf("CoreLen = %d, want 2", m.CoreLen())
	}
}

func TestRememberIgnoresDuplicatesAndBlanks(t *testing.T) {
	m := newMemory(t, 2000, nil)
	ctx := context.Background()
	for range 3 {
		if err := m.Remember(ctx, "The budget is four million."); err != nil {
			t.Fatal(err)
		}
	}
	if err := m.Remember(ctx, "   "); err != nil {
		t.Fatal(err)
	}
	if m.CoreLen() != 1 {
		t.Fatalf("CoreLen = %d, want 1 (duplicates and blanks ignored)", m.CoreLen())
	}
}

// Overflow must move the oldest entries to archival, not drop them.
func TestCoreOverflowEvictsToArchival(t *testing.T) {
	m := newMemory(t, 80, nil) // room for roughly two short entries
	ctx := context.Background()
	entries := []string{
		"Fact one about the budget.",
		"Fact two about the vendor.",
		"Fact three about the schedule.",
		"Fact four about the staffing.",
	}
	for _, e := range entries {
		if err := m.Remember(ctx, e); err != nil {
			t.Fatal(err)
		}
	}

	if m.CoreLen() >= len(entries) {
		t.Fatalf("core should have overflowed, still holds %d of %d", m.CoreLen(), len(entries))
	}
	if m.ArchivalLen() == 0 {
		t.Fatal("evicted entries should be in archival, not dropped")
	}
	if m.CoreLen()+m.ArchivalLen() != len(entries) {
		t.Errorf("lost entries: core=%d archival=%d, want %d total",
			m.CoreLen(), m.ArchivalLen(), len(entries))
	}
	// The most recent entry must survive in core.
	if !strings.Contains(m.CoreBlock(), "staffing") {
		t.Errorf("newest entry should remain in core:\n%s", m.CoreBlock())
	}
}

// An evicted memory stops being free but must remain findable.
func TestArchivedMemoryIsRetrievable(t *testing.T) {
	m := newMemory(t, 60, nil)
	ctx := context.Background()
	if err := m.Remember(ctx, "The vendor contract renews every January."); err != nil {
		t.Fatal(err)
	}
	if err := m.Remember(ctx, "Testing runs against a staging replica."); err != nil {
		t.Fatal(err)
	}
	if m.ArchivalLen() == 0 {
		t.Skip("nothing evicted; budget tuning made this vacuous")
	}

	got, err := m.Retrieve(ctx, core.Query{Text: "vendor contract renewal", TopK: 3})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) == 0 {
		t.Fatal("archived memory should be retrievable")
	}
	if got[0].Tier != core.TierMemory {
		t.Errorf("Tier = %s, want memory", got[0].Tier)
	}
	if !strings.HasPrefix(got[0].Cite(), "[memory:") {
		t.Errorf("Cite() = %q, want a [memory:...] citation", got[0].Cite())
	}
}

func TestMemorySatisfiesRetrieverWhenEmpty(t *testing.T) {
	var r core.Retriever = newMemory(t, 2000, nil)
	got, err := r.Retrieve(context.Background(), core.Query{Text: "anything", TopK: 3})
	if err != nil {
		t.Fatalf("empty memory should not error: %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("expected no results from empty memory, got %d", len(got))
	}
}

// The point of write-back: memory must survive process restarts.
func TestSaveLoadRoundTrip(t *testing.T) {
	dir := t.TempDir()
	ctx := context.Background()

	first := NewMemory(MemoryConfig{LLM: &fake.LLM{}, Dir: dir, CoreBudget: 60})
	for _, e := range []string{"Budget is four million.", "Priya leads Atlas.", "Vendor renews in January."} {
		if err := first.Remember(ctx, e); err != nil {
			t.Fatal(err)
		}
	}
	if err := first.Save(); err != nil {
		t.Fatal(err)
	}

	second := NewMemory(MemoryConfig{LLM: &fake.LLM{}, Dir: dir, CoreBudget: 60})
	if err := second.Load(); err != nil {
		t.Fatal(err)
	}
	if second.CoreLen() != first.CoreLen() {
		t.Errorf("core entries = %d, want %d", second.CoreLen(), first.CoreLen())
	}
	if second.ArchivalLen() != first.ArchivalLen() {
		t.Errorf("archival entries = %d, want %d", second.ArchivalLen(), first.ArchivalLen())
	}
	if second.CoreBlock() != first.CoreBlock() {
		t.Errorf("core block differs after reload:\n%s\n---\n%s", second.CoreBlock(), first.CoreBlock())
	}
}

// A fresh agent has nothing to remember; that is not an error.
func TestLoadMissingMemoryIsNotAnError(t *testing.T) {
	if err := newMemory(t, 2000, nil).Load(); err != nil {
		t.Fatalf("missing memory should not error: %v", err)
	}
}

func TestWriteBackStoresDurableTakeaway(t *testing.T) {
	f := &fake.LLM{ChatFunc: func([]llm.Message) (string, error) {
		return `{"worth_remembering": true, "takeaway": "The Atlas budget is four million dollars."}`, nil
	}}
	m := newMemory(t, 2000, f)

	got, err := m.WriteBack(context.Background(), "What is the budget?", "Four million [semantic:c#1].")
	if err != nil {
		t.Fatal(err)
	}
	if got == "" {
		t.Fatal("expected a takeaway")
	}
	if !strings.Contains(m.CoreBlock(), "four million") {
		t.Errorf("takeaway not stored:\n%s", m.CoreBlock())
	}
}

// Not every exchange is worth remembering; memory that fills with restatements
// of single answers stops being useful.
func TestWriteBackSkipsUnworthyExchanges(t *testing.T) {
	f := &fake.LLM{ChatFunc: func([]llm.Message) (string, error) {
		return `{"worth_remembering": false, "takeaway": ""}`, nil
	}}
	m := newMemory(t, 2000, f)

	got, err := m.WriteBack(context.Background(), "hi", "hello")
	if err != nil {
		t.Fatal(err)
	}
	if got != "" {
		t.Errorf("expected no takeaway, got %q", got)
	}
	if m.CoreLen() != 0 {
		t.Errorf("nothing should have been stored, core has %d", m.CoreLen())
	}
}
