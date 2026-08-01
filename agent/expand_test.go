package agent

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/RootControl/dispatch/core"
	"github.com/RootControl/dispatch/internal/fake"
	"github.com/RootControl/dispatch/llm"
)

func hydeFake(passage string) *fake.LLM {
	return &fake.LLM{ChatFunc: func(msgs []llm.Message) (string, error) {
		if isJudge(msgs) {
			return `{"sufficient": true}`, nil
		}
		if strings.Contains(msgs[0].Content, "hypothetical passage") {
			return passage, nil
		}
		return "An answer [semantic:a#0].", nil
	}}
}

// The expansion carries the question with it. Pure HyDE embeds the hypothetical
// alone, staking retrieval on the model having guessed the right subject; the
// question anchors it so a bad guess degrades the query rather than replacing it.
func TestHyDEKeepsTheQuestionInTheEmbeddedText(t *testing.T) {
	h := &HyDE{LLM: hydeFake("The project is a tiered retrieval library written in Go.")}
	got, err := h.Expand(context.Background(), "what is this project?")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(got, "what is this project?") {
		t.Errorf("expansion dropped the question: %q", got)
	}
	if !strings.Contains(got, "tiered retrieval library") {
		t.Errorf("expansion dropped the hypothetical: %q", got)
	}
}

func TestHyDEMinWordsSkipsSpecificQuestions(t *testing.T) {
	f := hydeFake("hypothetical")
	h := &HyDE{LLM: f, MinWords: 8}

	long := "what is the approved contingency reserve for the atlas migration programme this year"
	got, err := h.Expand(context.Background(), long)
	if err != nil {
		t.Fatal(err)
	}
	if got != "" {
		t.Errorf("expanded a question already at the word threshold: %q", got)
	}
	if f.Calls() != 0 {
		t.Errorf("made %d calls for a skipped question", f.Calls())
	}

	if got, err = h.Expand(context.Background(), "what is this?"); err != nil || got == "" {
		t.Errorf("short question was not expanded: %q, %v", got, err)
	}
}

func TestHyDEEmptyReplyIsNoExpansion(t *testing.T) {
	h := &HyDE{LLM: hydeFake("   ")}
	got, err := h.Expand(context.Background(), "what is this?")
	if err != nil {
		t.Fatal(err)
	}
	if got != "" {
		t.Errorf("a blank reply became an expansion: %q", got)
	}
}

// core.Query.Embedding is the contract the stores rely on: the expansion goes
// to the dense half, never to the lexical one.
func TestQueryEmbeddingFallsBackToText(t *testing.T) {
	plain := core.Query{Text: "what is the budget?"}
	if plain.Embedding() != plain.Text {
		t.Errorf("Embedding() = %q, want the query itself", plain.Embedding())
	}
	expanded := core.Query{Text: "what is the budget?", Expanded: "hypothetical passage"}
	if expanded.Embedding() != "hypothetical passage" {
		t.Errorf("Embedding() = %q, want the expansion", expanded.Embedding())
	}
	if expanded.Text != "what is the budget?" {
		t.Error("expansion overwrote the literal query the lexical half needs")
	}
}

// expansionSpy records what the retriever was actually given.
type expansionSpy struct {
	stubRetriever
	gotText, gotExpanded string
}

func (s *expansionSpy) Retrieve(ctx context.Context, q core.Query) ([]core.Result, error) {
	s.mu.Lock()
	s.gotText, s.gotExpanded = q.Text, q.Expanded
	s.mu.Unlock()
	return s.stubRetriever.Retrieve(ctx, q)
}

func TestLoopPassesExpansionToRetrievers(t *testing.T) {
	spy := &expansionSpy{stubRetriever: stubRetriever{
		tier:    core.TierSemantic,
		byQuery: func(string) []core.Result { return []core.Result{result(core.TierSemantic, "a#0", "evidence")} },
	}}
	loop := loopOver(hydeFake("A tiered retrieval library in Go."), spy)
	loop.Expander = &HyDE{LLM: loop.LLM}

	answer, err := loop.Run(context.Background(), "what is this project?")
	if err != nil {
		t.Fatal(err)
	}
	if spy.gotText != "what is this project?" {
		t.Errorf("Text = %q, want the literal question", spy.gotText)
	}
	if !strings.Contains(spy.gotExpanded, "tiered retrieval library") {
		t.Errorf("Expanded = %q, want the hypothetical", spy.gotExpanded)
	}
	if !strings.Contains(answer.Trace.String(), "expand") {
		t.Errorf("trace does not record the expansion:\n%s", answer.Trace)
	}
}

// A broken expander must cost the improvement, not the answer.
func TestLoopSurvivesAFailingExpander(t *testing.T) {
	spy := &expansionSpy{stubRetriever: stubRetriever{
		tier:    core.TierSemantic,
		byQuery: func(string) []core.Result { return []core.Result{result(core.TierSemantic, "a#0", "evidence")} },
	}}
	loop := loopOver(sufficientJudge(), spy)
	loop.Expander = brokenExpander{}

	answer, err := loop.Run(context.Background(), "what is this project?")
	if err != nil {
		t.Fatalf("a failing expander sank the answer: %v", err)
	}
	if spy.gotExpanded != "" {
		t.Errorf("Expanded = %q, want empty after the expander failed", spy.gotExpanded)
	}
	if spy.gotText != "what is this project?" {
		t.Errorf("Text = %q, want the query as written", spy.gotText)
	}
	if !strings.Contains(answer.Trace.String(), "failed, retrieving as written") {
		t.Errorf("trace does not record the failure:\n%s", answer.Trace)
	}
}

type brokenExpander struct{}

func (brokenExpander) Expand(context.Context, string) (string, error) {
	return "", errors.New("endpoint down")
}

// The judge's refined query is a different question and deserves its own
// hypothetical; expanding once would apply round one's guess to round two.
func TestLoopExpandsEachRound(t *testing.T) {
	var expanded []string
	spy := &expansionSpy{stubRetriever: stubRetriever{
		tier:    core.TierSemantic,
		byQuery: func(string) []core.Result { return []core.Result{result(core.TierSemantic, "a#0", "partial")} },
	}}
	f := &fake.LLM{ChatFunc: func(msgs []llm.Message) (string, error) {
		if isJudge(msgs) {
			return `{"sufficient": false, "gap": "the reserve", "refined_query": "contingency reserve"}`, nil
		}
		return "An answer.", nil
	}}
	loop := loopOver(f, spy)
	loop.MaxSteps = 2
	loop.Expander = recordingExpander{seen: &expanded}

	if _, err := loop.Run(context.Background(), "what is the budget?"); err != nil {
		t.Fatal(err)
	}
	if len(expanded) != 2 {
		t.Fatalf("expanded %d time(s) over 2 rounds: %v", len(expanded), expanded)
	}
	if expanded[0] == expanded[1] {
		t.Errorf("both rounds expanded the same text: %v", expanded)
	}
	if expanded[1] != "contingency reserve" {
		t.Errorf("round 2 expanded %q, want the judge's refined query", expanded[1])
	}
}

type recordingExpander struct{ seen *[]string }

func (r recordingExpander) Expand(_ context.Context, q string) (string, error) {
	*r.seen = append(*r.seen, q)
	return "hypothetical for " + q, nil
}
