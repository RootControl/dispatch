package agent

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"

	"github.com/RootControl/dispatch/core"
	"github.com/RootControl/dispatch/internal/fake"
	"github.com/RootControl/dispatch/llm"
	"github.com/RootControl/dispatch/router"
)

// stubRetriever returns canned results and records the queries it was asked.
type stubRetriever struct {
	tier core.Tier
	err  error
	// byQuery lets a test return different evidence for a refined query, which
	// is how we prove refinement actually changed what was retrieved.
	byQuery func(q string) []core.Result

	mu      sync.Mutex
	queries []string
}

func (s *stubRetriever) Tier() core.Tier { return s.tier }

func (s *stubRetriever) Retrieve(ctx context.Context, q core.Query) ([]core.Result, error) {
	s.mu.Lock()
	s.queries = append(s.queries, q.Text)
	s.mu.Unlock()
	if s.err != nil {
		return nil, s.err
	}
	return s.byQuery(q.Text), nil
}

func (s *stubRetriever) seen() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]string(nil), s.queries...)
}

func result(tier core.Tier, id, text string) core.Result {
	return core.Result{Tier: tier, SourceID: id, Text: text}
}

// isJudge distinguishes the judge call from the generate call by system prompt.
func isJudge(msgs []llm.Message) bool {
	return strings.Contains(msgs[0].Content, "sufficient")
}

func newLoop(l llm.LLM, r core.Retriever) *Loop {
	return &Loop{
		LLM:        l,
		Router:     router.Heuristic{Available: []core.Tier{r.Tier()}},
		Retrievers: map[core.Tier]core.Retriever{r.Tier(): r},
	}
}

// The load-bearing test: when the judge says the evidence is insufficient, the
// loop must retrieve a SECOND time using the refined query. Without this, the
// design is a single-pass pipeline wearing a loop's clothes.
func TestLoopRefinesTowardGap(t *testing.T) {
	// Keyed on "contract", a word only the REFINED query contains — so the second
	// round returning different evidence proves refinement changed the retrieval,
	// not just that the question happened to mention vendors.
	ret := &stubRetriever{tier: core.TierSemantic, byQuery: func(q string) []core.Result {
		if strings.Contains(q, "contract") {
			return []core.Result{result(core.TierSemantic, "risks#1", "The mailer vendor contract renews annually.")}
		}
		return []core.Result{result(core.TierSemantic, "charter#1", "The budget is four million dollars.")}
	}}

	var judged int
	f := &fake.LLM{ChatFunc: func(msgs []llm.Message) (string, error) {
		if !isJudge(msgs) {
			return "Budget is four million [semantic:charter#1]; vendor risk noted [semantic:risks#1].", nil
		}
		judged++
		if judged == 1 {
			return `{"sufficient": false, "gap": "no vendor risk information", "refined_query": "vendor contract risk", "next_tier": ""}`, nil
		}
		return `{"sufficient": true, "gap": "", "refined_query": "", "next_tier": ""}`, nil
	}}

	got, err := newLoop(f, ret).Run(context.Background(), "What is the budget and what vendor risk exists?")
	if err != nil {
		t.Fatal(err)
	}

	queries := ret.seen()
	if len(queries) != 2 {
		t.Fatalf("expected 2 retrievals (initial + refined), got %d: %v", len(queries), queries)
	}
	if queries[1] != "vendor contract risk" {
		t.Errorf("second retrieval should use the refined query, got %q", queries[1])
	}
	if !got.Trace.Refined() {
		t.Errorf("Trace.Refined() = false; loop did not actually refine")
	}
	if got.Trace.Rounds != 2 {
		t.Errorf("Rounds = %d, want 2", got.Trace.Rounds)
	}
	// Evidence from both rounds must survive into generation.
	if len(got.Evidence) != 2 {
		t.Fatalf("expected evidence from both rounds, got %d", len(got.Evidence))
	}
	// The trace must show the insufficient verdict, not just the final answer.
	if !strings.Contains(got.Trace.String(), "INSUFFICIENT") {
		t.Errorf("trace should record the insufficient verdict:\n%s", got.Trace)
	}
}

// A question the first retrieval fully answers must not burn extra rounds.
func TestLoopStopsWhenSufficient(t *testing.T) {
	ret := &stubRetriever{tier: core.TierSemantic, byQuery: func(string) []core.Result {
		return []core.Result{result(core.TierSemantic, "charter#1", "The budget is four million dollars.")}
	}}
	f := &fake.LLM{ChatFunc: func(msgs []llm.Message) (string, error) {
		if isJudge(msgs) {
			return `{"sufficient": true}`, nil
		}
		return "Four million [semantic:charter#1].", nil
	}}

	got, err := newLoop(f, ret).Run(context.Background(), "What is the budget?")
	if err != nil {
		t.Fatal(err)
	}
	if n := len(ret.seen()); n != 1 {
		t.Fatalf("expected a single retrieval, got %d", n)
	}
	if got.Trace.Refined() {
		t.Error("should not have refined when the first round sufficed")
	}
	// One judge + one generate.
	if got.Trace.Calls != 2 {
		t.Errorf("Calls = %d, want 2 (judge + generate)", got.Trace.Calls)
	}
}

// A judge that never says "sufficient" must still terminate at MaxSteps.
func TestLoopRespectsMaxSteps(t *testing.T) {
	ret := &stubRetriever{tier: core.TierSemantic, byQuery: func(q string) []core.Result {
		return []core.Result{result(core.TierSemantic, "c#"+q, "text for "+q)}
	}}
	var n int
	f := &fake.LLM{ChatFunc: func(msgs []llm.Message) (string, error) {
		if !isJudge(msgs) {
			return "answer", nil
		}
		n++
		return fmt.Sprintf(`{"sufficient": false, "gap": "more", "refined_query": "q%d"}`, n), nil
	}}
	loop := newLoop(f, ret)
	loop.MaxSteps = 3

	got, err := loop.Run(context.Background(), "unanswerable")
	if err != nil {
		t.Fatal(err)
	}
	if len(ret.seen()) != 3 {
		t.Fatalf("expected exactly MaxSteps retrievals, got %d: %v", len(ret.seen()), ret.seen())
	}
	// The last step must not waste a judge call it cannot act on.
	if n != 2 {
		t.Errorf("judge calls = %d, want 2 (skipped on the final step)", n)
	}
	if got.Trace.Rounds != 3 {
		t.Errorf("Rounds = %d, want 3", got.Trace.Rounds)
	}
}

// A retriever failure must not sink an answer; the round is recorded and skipped.
func TestLoopSurvivesRetrieverError(t *testing.T) {
	bad := &stubRetriever{tier: core.TierSemantic, err: errors.New("backend down")}
	f := &fake.LLM{ChatFunc: func(msgs []llm.Message) (string, error) { return "{}", nil }}

	_, err := newLoop(f, bad).Run(context.Background(), "anything")
	if err == nil {
		t.Fatal("expected an error when no evidence could be retrieved")
	}
	if !strings.Contains(err.Error(), "no evidence") {
		t.Errorf("error should explain the absence of evidence, got %v", err)
	}
}

// A judge that errors or returns garbage must not lose an answerable question.
func TestLoopProceedsWhenJudgeFails(t *testing.T) {
	ret := &stubRetriever{tier: core.TierSemantic, byQuery: func(string) []core.Result {
		return []core.Result{result(core.TierSemantic, "charter#1", "Four million dollars.")}
	}}
	f := &fake.LLM{ChatFunc: func(msgs []llm.Message) (string, error) {
		if isJudge(msgs) {
			return "not json at all", nil
		}
		return "Four million [semantic:charter#1].", nil
	}}

	got, err := newLoop(f, ret).Run(context.Background(), "What is the budget?")
	if err != nil {
		t.Fatalf("a failed judge should not sink an answerable question: %v", err)
	}
	if got.Text == "" {
		t.Error("expected an answer despite the judge failing")
	}
	if !strings.Contains(got.Trace.String(), "judge failed") {
		t.Errorf("trace should record the judge failure:\n%s", got.Trace)
	}
}

func TestMergeDedupesAndCaps(t *testing.T) {
	a := []core.Result{result(core.TierSemantic, "x", "1")}
	b := []core.Result{
		result(core.TierSemantic, "x", "1"), // duplicate citation
		result(core.TierSemantic, "y", "2"),
		result(core.TierSemantic, "z", "3"),
	}
	got := merge(a, b, 2)
	if len(got) != 2 {
		t.Fatalf("expected cap of 2, got %d", len(got))
	}
	if got[0].SourceID != "x" || got[1].SourceID != "y" {
		t.Errorf("expected [x y] preserving arrival order, got %v", []string{got[0].SourceID, got[1].SourceID})
	}
}
