package agent

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/RootControl/dispatch/core"
	"github.com/RootControl/dispatch/internal/fake"
	"github.com/RootControl/dispatch/llm"
)

// neverSufficient is the case MaxSteps alone does not bound well: the judge
// refuses every round, so every question spends the full budget.
func neverSufficient() *fake.LLM {
	return &fake.LLM{ChatFunc: func(msgs []llm.Message) (string, error) {
		if isJudge(msgs) {
			return `{"sufficient": false, "gap": "more", "refined_query": "more detail"}`, nil
		}
		return "An answer.", nil
	}}
}

func budgetLoop(l llm.LLM) *Loop {
	sem := &stubRetriever{
		tier:    core.TierSemantic,
		byQuery: func(string) []core.Result { return []core.Result{result(core.TierSemantic, "a#0", "evidence")} },
	}
	loop := loopOver(l, sem)
	loop.MaxSteps = 5
	return loop
}

func TestMaxCallsBoundsTheQuestion(t *testing.T) {
	f := neverSufficient()
	loop := budgetLoop(f)
	loop.MaxCalls = 3

	answer, err := loop.Run(context.Background(), "what is the budget?")
	if err != nil {
		t.Fatal(err)
	}
	if f.Calls() > 3 {
		t.Errorf("made %d calls under a cap of 3", f.Calls())
	}
	// Reaching the budget is not an error: it is a limit on effort, not a
	// verdict that the question was bad.
	if answer.Text == "" {
		t.Error("hitting the budget produced no answer")
	}
	if !strings.Contains(answer.Trace.String(), "call budget reached") {
		t.Errorf("trace does not say why it stopped:\n%s", answer.Trace)
	}
}

// The budget must always leave room to generate. Spending the last call on a
// judge would leave the loop with evidence it cannot turn into an answer —
// everything paid for and nothing produced.
func TestMaxCallsAlwaysReservesGeneration(t *testing.T) {
	for _, cap := range []int{1, 2, 3, 4} {
		f := neverSufficient()
		loop := budgetLoop(f)
		loop.MaxCalls = cap

		answer, err := loop.Run(context.Background(), "what is the budget?")
		if err != nil {
			t.Fatalf("cap %d: %v", cap, err)
		}
		if answer.Text == "" {
			t.Errorf("cap %d produced no answer", cap)
		}
		if f.Calls() > cap {
			t.Errorf("cap %d: made %d calls", cap, f.Calls())
		}
	}
}

// An expander competes for the same budget as the judge, which is the reason
// the cap is expressed in calls rather than rounds.
func TestMaxCallsCoversExpansion(t *testing.T) {
	var expansions int
	f := neverSufficient()
	loop := budgetLoop(f)
	loop.MaxCalls = 3
	loop.Expander = countingExpander{n: &expansions}

	if _, err := loop.Run(context.Background(), "what is the budget?"); err != nil {
		t.Fatal(err)
	}
	if total := f.Calls() + expansions; total > 3 {
		t.Errorf("judge+generate %d plus %d expansions = %d, over the cap of 3", f.Calls(), expansions, total)
	}
}

type countingExpander struct{ n *int }

func (c countingExpander) Expand(context.Context, string) (string, error) {
	*c.n++
	return "hypothetical", nil
}

func TestNoCapMeansNoCap(t *testing.T) {
	f := neverSufficient()
	loop := budgetLoop(f)

	if _, err := loop.Run(context.Background(), "what is the budget?"); err != nil {
		t.Fatal(err)
	}
	// 5 rounds: 4 judges (the last round skips it) plus one generate.
	if f.Calls() != 5 {
		t.Errorf("made %d calls with no cap, want the full 5 rounds", f.Calls())
	}
}

// A cancelled context must surface as an error rather than an answer built
// from whatever happened to arrive first.
func TestRunHonoursContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	loop := budgetLoop(neverSufficient())
	if _, err := loop.Run(ctx, "what is the budget?"); err == nil {
		t.Fatal("a cancelled context produced an answer")
	}
}

// A deadline that expires mid-retrieval must be reported as a deadline. fanOut
// records a tier's error and carries on, which is right for one tier failing
// and wrong here: every tier fails, evidence is empty, and "no evidence
// retrieved" would blame the corpus for a dead endpoint.
func TestTimeoutIsNotReportedAsAnEmptyCorpus(t *testing.T) {
	slow := &stubRetriever{
		tier: core.TierSemantic,
		byQuery: func(string) []core.Result {
			time.Sleep(30 * time.Millisecond)
			return nil
		},
	}
	loop := loopOver(sufficientJudge(), slow)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()

	_, err := loop.Run(ctx, "what is the budget?")
	if err == nil {
		t.Fatal("an expired deadline produced an answer")
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("error = %v, want it to wrap context.DeadlineExceeded", err)
	}
	if strings.Contains(err.Error(), "no evidence") {
		t.Errorf("a timeout was reported as an empty corpus: %v", err)
	}
}
