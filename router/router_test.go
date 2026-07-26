package router

import (
	"context"
	"slices"
	"testing"

	"github.com/RootControl/dispatch/core"
)

var allTiers = []core.Tier{
	core.TierStructured, core.TierSemantic, core.TierRelational, core.TierHierarchical,
}

// The four example questions from the design doc, each with the tier the doc
// says is right. This is the router's acceptance test.
func TestRouteDocExamples(t *testing.T) {
	cases := []struct {
		question string
		want     core.Tier
	}{
		{"Total on invoice 4471?", core.TierStructured},
		{"What does the corpus say about the migration?", core.TierSemantic},
		{"Who does Bob report to, and what did she sign?", core.TierRelational},
		{"Recurring themes across all docs?", core.TierHierarchical},
	}
	h := Heuristic{Available: allTiers}
	for _, tc := range cases {
		got, err := h.Route(context.Background(), tc.question)
		if err != nil {
			t.Fatal(err)
		}
		if got.Tiers[0] != tc.want {
			t.Errorf("Route(%q) = %v (%s), want %s first", tc.question, got.Tiers, got.Reason, tc.want)
		}
	}
}

func TestRouteAlwaysFallsBackToSemantic(t *testing.T) {
	h := Heuristic{Available: allTiers}
	got, _ := h.Route(context.Background(), "Total on invoice 4471?")
	if !slices.Contains(got.Tiers, core.TierSemantic) {
		t.Fatalf("semantic should always be a fallback, got %v", got.Tiers)
	}
	// ...but not ahead of the tier that actually matched.
	if got.Tiers[0] == core.TierSemantic {
		t.Fatalf("semantic should not outrank a keyword match: %v", got.Tiers)
	}
}

// A tier nobody implements must never be returned as a dead end.
func TestRouteDropsUnavailableTiers(t *testing.T) {
	h := Heuristic{Available: []core.Tier{core.TierSemantic}}
	got, _ := h.Route(context.Background(), "Total on invoice 4471?")
	if len(got.Tiers) != 1 || got.Tiers[0] != core.TierSemantic {
		t.Fatalf("expected only the registered tier, got %v", got.Tiers)
	}
}

// When the only registered tiers are ones the router didn't pick, it should
// still return something retrievable rather than an empty list.
func TestRouteFallsBackToAvailableWhenNoPreferenceRegistered(t *testing.T) {
	h := Heuristic{Available: []core.Tier{core.TierHierarchical}}
	got, _ := h.Route(context.Background(), "Total on invoice 4471?")
	if len(got.Tiers) == 0 {
		t.Fatal("router returned no tiers; nothing could be retrieved")
	}
	if got.Tiers[0] != core.TierHierarchical {
		t.Fatalf("expected fallback to the available tier, got %v", got.Tiers)
	}
}

func TestRouteIsDeterministic(t *testing.T) {
	h := Heuristic{Available: allTiers}
	first, _ := h.Route(context.Background(), "How many invoices did Priya sign overall?")
	for range 20 {
		got, _ := h.Route(context.Background(), "How many invoices did Priya sign overall?")
		if !slices.Equal(got.Tiers, first.Tiers) {
			t.Fatalf("routing not deterministic: %v vs %v", got.Tiers, first.Tiers)
		}
	}
}

// Relational questions come in two flavours. The org-chart vocabulary alone
// misses technical corpora: over a real repository, "what does the client
// workspace depend on?" matched nothing and fell through to semantic.
func TestRouteDependencyQuestionsAreRelational(t *testing.T) {
	h := Heuristic{Available: allTiers}
	for _, q := range []string{
		"What does the client workspace depend on?",
		"Which packages depend on data-provider?",
		"What are the dependencies of the api workspace?",
		"What does packages/api require?",
		"Which module is this built on?",
		"What is this service part of?",
		"What does Aisha own?",
		"Which vendor provides the mailer?",
	} {
		got, err := h.Route(context.Background(), q)
		if err != nil {
			t.Fatal(err)
		}
		if got.Tiers[0] != core.TierRelational {
			t.Errorf("Route(%q) = %v (%s), want relational first", q, got.Tiers, got.Reason)
		}
	}
}

// The added vocabulary must not swallow questions that belong elsewhere.
func TestRouteDependencyVocabularyDoesNotOverreach(t *testing.T) {
	h := Heuristic{Available: allTiers}
	for q, want := range map[string]core.Tier{
		"Recurring themes across all docs?":       core.TierHierarchical,
		"Total on invoice 4471?":                  core.TierStructured,
		"What does the corpus say about caching?": core.TierSemantic,
	} {
		got, _ := h.Route(context.Background(), q)
		if got.Tiers[0] != want {
			t.Errorf("Route(%q) = %v, want %s first", q, got.Tiers, want)
		}
	}
}
