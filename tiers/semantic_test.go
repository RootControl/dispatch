package tiers

import (
	"context"
	"strings"
	"testing"

	"github.com/RootControl/dispatch/core"
	"github.com/RootControl/dispatch/index"
	"github.com/RootControl/dispatch/internal/fake"
)

func newTestTier(t *testing.T) *Semantic {
	t.Helper()
	s := index.New(index.Config{LLM: &fake.LLM{}})
	_, err := s.Ingest(context.Background(), []core.Doc{
		{ID: "charter", Text: "The approved budget is four million dollars.\n\nPriya Raman leads the project."},
		{ID: "risks", Text: "The statement mailer depends on a single external vendor."},
	})
	if err != nil {
		t.Fatal(err)
	}
	return NewSemantic(s)
}

func TestSemanticSatisfiesRetriever(t *testing.T) {
	var r core.Retriever = newTestTier(t)
	if r.Tier() != core.TierSemantic {
		t.Fatalf("Tier() = %s", r.Tier())
	}
}

func TestSemanticRetrieveCitesCorrectly(t *testing.T) {
	tier := newTestTier(t)
	got, err := tier.Retrieve(context.Background(), core.Query{Text: "budget", TopK: 3})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) == 0 {
		t.Fatal("no results")
	}
	top := got[0]
	if !strings.Contains(top.Text, "four million") {
		t.Errorf("expected the budget chunk first, got %q", top.Text)
	}
	if top.Tier != core.TierSemantic {
		t.Errorf("Tier = %s, want semantic", top.Tier)
	}
	if !strings.HasPrefix(top.Cite(), "[semantic:charter#") {
		t.Errorf("Cite() = %q, want [semantic:charter#N]", top.Cite())
	}
	if top.Meta["doc"] != "charter" {
		t.Errorf("Meta[doc] = %q, want charter", top.Meta["doc"])
	}
}

// An empty store must return no results and no error — the loop reads that as an
// evidence gap, not a failure.
func TestSemanticEmptyStore(t *testing.T) {
	tier := NewSemantic(index.New(index.Config{LLM: &fake.LLM{}}))
	got, err := tier.Retrieve(context.Background(), core.Query{Text: "anything", TopK: 3})
	if err != nil {
		t.Fatalf("empty store should not error: %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("expected no results, got %d", len(got))
	}
}
