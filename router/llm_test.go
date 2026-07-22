package router

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/RootControl/dispatch/core"
	"github.com/RootControl/dispatch/internal/fake"
	"github.com/RootControl/dispatch/llm"
)

func llmRouter(reply string, err error) LLM {
	return LLM{
		LLM: &fake.LLM{ChatFunc: func([]llm.Message) (string, error) {
			if err != nil {
				return "", err
			}
			return reply, nil
		}},
		Available: allTiers,
	}
}

func TestLLMRouteUsesClassification(t *testing.T) {
	r := llmRouter(`{"tiers":["structured"],"reason":"asks for a total"}`, nil)
	got, err := r.Route(context.Background(), "Total on invoice 4471?")
	if err != nil {
		t.Fatal(err)
	}
	if got.Tiers[0] != core.TierStructured {
		t.Errorf("Tiers = %v, want structured first", got.Tiers)
	}
	if got.Source != "llm" {
		t.Errorf("Source = %q, want llm", got.Source)
	}
	if got.Reason != "asks for a total" {
		t.Errorf("Reason = %q", got.Reason)
	}
}

// A hallucinated tier must be dropped, not propagated as a dead end.
func TestLLMRouteDropsUnknownTiers(t *testing.T) {
	r := llmRouter(`{"tiers":["telepathic","semantic","structured"],"reason":"x"}`, nil)
	got, err := r.Route(context.Background(), "anything")
	if err != nil {
		t.Fatal(err)
	}
	want := []core.Tier{core.TierSemantic, core.TierStructured}
	if len(got.Tiers) != len(want) {
		t.Fatalf("Tiers = %v, want %v", got.Tiers, want)
	}
	for i := range want {
		if got.Tiers[i] != want[i] {
			t.Fatalf("Tiers = %v, want %v", got.Tiers, want)
		}
	}
}

func TestLLMRouteDedupes(t *testing.T) {
	r := llmRouter(`{"tiers":["semantic","Semantic","SEMANTIC"],"reason":"x"}`, nil)
	got, _ := r.Route(context.Background(), "anything")
	if len(got.Tiers) != 1 {
		t.Fatalf("expected deduped tiers, got %v", got.Tiers)
	}
}

// The fallback is the whole safety story: routing confidently to the wrong tier
// retrieves plausible evidence from the wrong place, and nothing downstream can
// tell.
func TestLLMRouteFallsBackOnError(t *testing.T) {
	r := llmRouter("", errors.New("connection refused"))
	got, err := r.Route(context.Background(), "Total on invoice 4471?")
	if err != nil {
		t.Fatal(err)
	}
	if got.Source != "heuristic" {
		t.Errorf("Source = %q, want heuristic", got.Source)
	}
	if got.Tiers[0] != core.TierStructured {
		t.Errorf("fallback should still route correctly, got %v", got.Tiers)
	}
	// The trace must show the route was degraded, not silently normal.
	if !strings.Contains(got.Reason, "llm routing failed") {
		t.Errorf("Reason should record the failure, got %q", got.Reason)
	}
}

func TestLLMRouteFallsBackOnGarbage(t *testing.T) {
	for _, reply := range []string{
		`not json at all`,
		`{"tiers":[],"reason":"dunno"}`,
		`{"tiers":["nonsense"],"reason":"x"}`,
	} {
		r := llmRouter(reply, nil)
		got, err := r.Route(context.Background(), "Recurring themes across all docs?")
		if err != nil {
			t.Fatalf("reply %q: %v", reply, err)
		}
		if got.Source != "heuristic" {
			t.Errorf("reply %q: Source = %q, want heuristic", reply, got.Source)
		}
		if got.Tiers[0] != core.TierHierarchical {
			t.Errorf("reply %q: fallback misrouted: %v", reply, got.Tiers)
		}
	}
}

// Only registered tiers may be returned, whatever the model says.
func TestLLMRouteRespectsAvailable(t *testing.T) {
	r := LLM{
		LLM:       &fake.LLM{ChatFunc: func([]llm.Message) (string, error) { return `{"tiers":["structured"]}`, nil }},
		Available: []core.Tier{core.TierSemantic},
	}
	got, err := r.Route(context.Background(), "Total on invoice 4471?")
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Tiers) != 1 || got.Tiers[0] != core.TierSemantic {
		t.Fatalf("Tiers = %v, want only the registered semantic tier", got.Tiers)
	}
	if got.Source != "heuristic" {
		t.Errorf("Source = %q, want heuristic after the named tier was unavailable", got.Source)
	}
}

// The prompt must carry the available set, or the model cannot honor it.
func TestLLMRoutePromptListsAvailableTiers(t *testing.T) {
	var seen string
	r := LLM{
		LLM: &fake.LLM{ChatFunc: func(msgs []llm.Message) (string, error) {
			seen = msgs[len(msgs)-1].Content
			return `{"tiers":["semantic"]}`, nil
		}},
		Available: []core.Tier{core.TierSemantic, core.TierHierarchical},
	}
	if _, err := r.Route(context.Background(), "what is this about?"); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(seen, "semantic, hierarchical") {
		t.Errorf("prompt missing the available list:\n%s", seen)
	}
	if !strings.Contains(seen, "what is this about?") {
		t.Errorf("prompt missing the question:\n%s", seen)
	}
}
