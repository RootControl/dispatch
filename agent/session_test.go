package agent

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/RootControl/dispatch/core"
	"github.com/RootControl/dispatch/internal/fake"
	"github.com/RootControl/dispatch/llm"
	"github.com/RootControl/dispatch/router"
)

// recordingRetriever remembers every query it was asked, which is the only way
// to see what a follow-up actually searched for.
type recordingRetriever struct {
	tier     core.Tier
	queries  []string
	results  map[string][]core.Result
	fallback []core.Result
}

func (r *recordingRetriever) Tier() core.Tier { return r.tier }

func (r *recordingRetriever) Retrieve(_ context.Context, q core.Query) ([]core.Result, error) {
	r.queries = append(r.queries, q.Text)
	for key, res := range r.results {
		if strings.Contains(strings.ToLower(q.Text), key) {
			return res, nil
		}
	}
	if r.fallback == nil {
		return nil, nil
	}
	return r.fallback, nil
}

// answering is a fake that judges every round sufficient and produces a cited
// answer, so these tests measure what was retrieved rather than re-testing the
// loop.
func answering() *fake.LLM {
	return &fake.LLM{ChatFunc: func(msgs []llm.Message) (string, error) {
		if isJudge(msgs) {
			return `{"sufficient": true}`, nil
		}
		return "An answer [semantic:atlas-charter.md#0].", nil
	}}
}

func sessionLoop(t *testing.T, rec *recordingRetriever, l llm.LLM) *Loop {
	t.Helper()
	return &Loop{
		LLM:        l,
		Router:     router.Heuristic{Available: []core.Tier{core.TierSemantic}},
		Retrievers: map[core.Tier]core.Retriever{core.TierSemantic: rec},
		MaxSteps:   1,
	}
}

// atlasCorpus answers only to questions naming Atlas. That is the whole point:
// a follow-up that drops the subject must miss.
func atlasCorpus() *recordingRetriever {
	return &recordingRetriever{
		tier: core.TierSemantic,
		results: map[string][]core.Result{
			"atlas": {{
				Tier: core.TierSemantic, SourceID: "atlas-charter.md#0",
				Text: "Project Atlas has a budget of four million dollars.", Score: 1,
			}},
		},
	}
}

// anyCorpus answers everything, for the tests about history bookkeeping rather
// than about retrieval.
func anyCorpus() *recordingRetriever {
	r := atlasCorpus()
	r.fallback = []core.Result{{
		Tier: core.TierSemantic, SourceID: "atlas-charter.md#0", Text: "Something.", Score: 1,
	}}
	return r
}

// The failure a Session exists to fix, stated as a test.
//
// "What is its budget?" names nothing. Without a rewrite the retriever is asked
// for exactly those words, matches nothing about Atlas, and the loop answers
// from no evidence — not a worse answer, an answer about the wrong thing.
func TestFollowUpWithoutARewriterRetrievesNothing(t *testing.T) {
	rec := atlasCorpus()
	s := &Session{Loop: sessionLoop(t, rec, answering())}

	ctx := context.Background()
	if _, err := s.Ask(ctx, "What is Project Atlas?"); err != nil {
		t.Fatal(err)
	}
	_, err := s.Ask(ctx, "What is its budget?")
	if err == nil {
		t.Fatal("the follow-up found evidence it should not have")
	}
	if !strings.Contains(err.Error(), "no evidence") {
		t.Errorf("expected a no-evidence failure, got: %v", err)
	}
	if got := rec.queries[len(rec.queries)-1]; got != "What is its budget?" {
		t.Errorf("last query = %q, want the question verbatim", got)
	}
}

// And with one, the same follow-up reaches the same evidence the full question
// would have.
func TestRewriterResolvesAFollowUp(t *testing.T) {
	rec := atlasCorpus()
	condenser := &Condenser{LLM: &fake.LLM{ChatFunc: func([]llm.Message) (string, error) {
		return `{"question": "What is Project Atlas's budget?"}`, nil
	}}}
	s := &Session{Loop: sessionLoop(t, rec, answering()), Rewriter: condenser}

	ctx := context.Background()
	if _, err := s.Ask(ctx, "What is Project Atlas?"); err != nil {
		t.Fatal(err)
	}
	ans, err := s.Ask(ctx, "What is its budget?")
	if err != nil {
		t.Fatal(err)
	}
	if len(ans.Evidence) == 0 {
		t.Fatal("the rewritten follow-up still retrieved nothing")
	}
	if got := rec.queries[len(rec.queries)-1]; !strings.Contains(strings.ToLower(got), "atlas") {
		t.Errorf("last query = %q, want the resolved subject", got)
	}
}

// A rewrite that is not shown is a rewrite nobody can debug: the trace would
// report a sensible retrieval for a question nobody typed.
func TestTheTraceKeepsBothQuestions(t *testing.T) {
	rec := atlasCorpus()
	s := &Session{
		Loop: sessionLoop(t, rec, answering()),
		Rewriter: &Condenser{LLM: &fake.LLM{ChatFunc: func([]llm.Message) (string, error) {
			return `{"question": "What is Project Atlas's budget?"}`, nil
		}}},
	}
	ctx := context.Background()
	s.Ask(ctx, "What is Project Atlas?")
	ans, err := s.Ask(ctx, "What is its budget?")
	if err != nil {
		t.Fatal(err)
	}
	if ans.Trace.Question != "What is its budget?" {
		t.Errorf("trace question = %q, want what the user typed", ans.Trace.Question)
	}
	var found bool
	for _, st := range ans.Trace.Steps {
		if st.Kind == StepRewrite && strings.Contains(st.Query, "Atlas") {
			found = true
		}
	}
	if !found {
		t.Errorf("the rewrite is missing from the trace:\n%s", ans.Trace)
	}
}

// The first question has no history, so it must cost nothing extra.
func TestFirstQuestionIsNeverRewritten(t *testing.T) {
	var calls int
	s := &Session{
		Loop: sessionLoop(t, anyCorpus(), answering()),
		Rewriter: &Condenser{Always: true, LLM: &fake.LLM{ChatFunc: func([]llm.Message) (string, error) {
			calls++
			return `{"question": "rewritten"}`, nil
		}}},
	}
	if _, err := s.Ask(context.Background(), "What is Project Atlas?"); err != nil {
		t.Fatal(err)
	}
	if calls != 0 {
		t.Errorf("the rewriter was called %d time(s) on the first question", calls)
	}
}

// A standalone follow-up must not cost a call either. This is what the lexical
// pre-filter buys, and it is most of the questions in a real conversation.
func TestStandaloneFollowUpSkipsTheRewriteCall(t *testing.T) {
	var calls int
	rec := anyCorpus()
	s := &Session{
		Loop: sessionLoop(t, rec, answering()),
		Rewriter: &Condenser{LLM: &fake.LLM{ChatFunc: func([]llm.Message) (string, error) {
			calls++
			return `{"question": "x"}`, nil
		}}},
	}
	ctx := context.Background()
	s.Ask(ctx, "What is Project Atlas?")
	s.Ask(ctx, "Which vendor signed the renewal contract in January?")
	if calls != 0 {
		t.Errorf("a self-contained question cost %d rewrite call(s)", calls)
	}
}

// A broken rewriter costs the improvement, not the answer.
func TestRewriteFailureFallsBackToTheQuestion(t *testing.T) {
	rec := anyCorpus()
	s := &Session{
		Loop: sessionLoop(t, rec, answering()),
		Rewriter: &Condenser{LLM: &fake.LLM{ChatFunc: func([]llm.Message) (string, error) {
			return "", errors.New("endpoint down")
		}}},
	}
	ctx := context.Background()
	s.Ask(ctx, "What is Project Atlas?")
	if _, err := s.Ask(ctx, "What is its budget?"); err != nil {
		t.Fatalf("a failed rewrite should not fail the answer: %v", err)
	}
	if got := rec.queries[len(rec.queries)-1]; got != "What is its budget?" {
		t.Errorf("last query = %q, want the original question", got)
	}
}

// A rewrite that starts answering must be discarded. Small models do this:
// asked to make a follow-up standalone, they restate the previous answer, and
// retrieval then matches the answer rather than the question.
func TestOverlongRewriteIsDiscarded(t *testing.T) {
	long := strings.Repeat("Project Atlas is a programme with a budget. ", 20)
	c := &Condenser{LLM: &fake.LLM{ChatFunc: func([]llm.Message) (string, error) {
		return `{"question": "` + long + `"}`, nil
	}}}
	got, err := c.Rewrite(context.Background(), "and the budget?",
		[]Turn{{Question: "What is Project Atlas?", Answer: "A programme."}})
	if err != nil {
		t.Fatal(err)
	}
	if got != "" {
		t.Errorf("an answer-shaped rewrite was accepted: %q", got)
	}
}

func TestHistoryIsBoundedByMaxTurns(t *testing.T) {
	var seen string
	s := &Session{
		Loop:     sessionLoop(t, anyCorpus(), answering()),
		MaxTurns: 2,
		Rewriter: &Condenser{Always: true, LLM: &fake.LLM{ChatFunc: func(msgs []llm.Message) (string, error) {
			seen = msgs[len(msgs)-1].Content
			return `{"question": "x"}`, nil
		}}},
	}
	ctx := context.Background()
	for _, q := range []string{"first question", "second question", "third question"} {
		s.Ask(ctx, q)
	}
	s.Ask(ctx, "and it?")
	if strings.Contains(seen, "first question") {
		t.Errorf("history exceeded MaxTurns:\n%s", seen)
	}
	if !strings.Contains(seen, "third question") {
		t.Errorf("the most recent turn is missing from the window:\n%s", seen)
	}
}

func TestResetClearsTheConversation(t *testing.T) {
	var calls int
	s := &Session{
		Loop: sessionLoop(t, anyCorpus(), answering()),
		Rewriter: &Condenser{Always: true, LLM: &fake.LLM{ChatFunc: func([]llm.Message) (string, error) {
			calls++
			return `{"question": "x"}`, nil
		}}},
	}
	ctx := context.Background()
	s.Ask(ctx, "What is Project Atlas?")
	s.Reset()
	if len(s.History()) != 0 {
		t.Fatal("Reset left history behind")
	}
	s.Ask(ctx, "What is its budget?")
	if calls != 0 {
		t.Errorf("a question after Reset was still rewritten against %d turn(s)", calls)
	}
}

func TestDependentDetection(t *testing.T) {
	follow := []string{
		"what about its budget?",
		"and the vendor?",
		"who signed it",
		"is that still true",
		"who else",
		"how about the risks?",
		"the same for Q3?",
		"why?",
	}
	for _, q := range follow {
		if !dependent(q) {
			t.Errorf("%q should be treated as a follow-up", q)
		}
	}
	standalone := []string{
		"What is Project Atlas?",
		"Which vendor signed the renewal contract in January?",
		"How many staff are assigned to the platform team?",
		"Summarize the risks recorded in the March steering minutes.",
	}
	for _, q := range standalone {
		if dependent(q) {
			t.Errorf("%q should not need rewriting", q)
		}
	}
}
