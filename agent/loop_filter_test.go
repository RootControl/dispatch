package agent

import (
	"context"
	"slices"
	"strings"
	"testing"

	"github.com/RootControl/dispatch/core"
	"github.com/RootControl/dispatch/internal/fake"
	"github.com/RootControl/dispatch/llm"
	"github.com/RootControl/dispatch/router"
)

// filteringRetriever honours Query.Filter; stubRetriever (above) does not.
type filteringRetriever struct {
	stubRetriever
	seenFilter core.Filter
}

func (f *filteringRetriever) HonorsFilter() {}

func (f *filteringRetriever) Retrieve(ctx context.Context, q core.Query) ([]core.Result, error) {
	f.mu.Lock()
	f.seenFilter = q.Filter
	f.mu.Unlock()
	return f.stubRetriever.Retrieve(ctx, q)
}

var _ core.Filterable = (*filteringRetriever)(nil)

func sufficientJudge() *fake.LLM {
	return &fake.LLM{ChatFunc: func(msgs []llm.Message) (string, error) {
		if isJudge(msgs) {
			return `{"sufficient": true}`, nil
		}
		return "An answer.", nil
	}}
}

func loopOver(l llm.LLM, rs ...core.Retriever) *Loop {
	reg := map[core.Tier]core.Retriever{}
	avail := make([]core.Tier, 0, len(rs))
	for _, r := range rs {
		reg[r.Tier()] = r
		avail = append(avail, r.Tier())
	}
	slices.Sort(avail)
	return &Loop{
		LLM:        l,
		Router:     router.Heuristic{Available: avail},
		Retrievers: reg,
		MaxSteps:   1,
	}
}

// A filter must reach the retrievers that can apply it.
func TestLoopForwardsFilter(t *testing.T) {
	sem := &filteringRetriever{stubRetriever: stubRetriever{
		tier:    core.TierSemantic,
		byQuery: func(string) []core.Result { return []core.Result{result(core.TierSemantic, "a#0", "evidence")} },
	}}
	loop := loopOver(sufficientJudge(), sem)
	loop.Filter = core.Filter{"dir": "docs"}

	if _, err := loop.Run(context.Background(), "what is the budget?"); err != nil {
		t.Fatal(err)
	}
	if got := sem.seenFilter["dir"]; got != "docs" {
		t.Fatalf("retriever saw filter %v, want dir=docs", sem.seenFilter)
	}
}

// The failure this guards is silent: a tier that ignores the filter returns
// evidence from outside the scope, and nothing in the answer reveals it.
func TestLoopSkipsTiersThatCannotFilter(t *testing.T) {
	sem := &filteringRetriever{stubRetriever: stubRetriever{
		tier:    core.TierSemantic,
		byQuery: func(string) []core.Result { return []core.Result{result(core.TierSemantic, "a#0", "evidence")} },
	}}
	hier := &stubRetriever{
		tier:    core.TierHierarchical,
		byQuery: func(string) []core.Result { return []core.Result{result(core.TierHierarchical, "L1-0", "summary")} },
	}

	loop := loopOver(sufficientJudge(), sem, hier)
	loop.Filter = core.Filter{"dir": "docs"}

	answer, err := loop.Run(context.Background(), "recurring themes and the budget?")
	if err != nil {
		t.Fatal(err)
	}
	if len(hier.seen()) != 0 {
		t.Errorf("the hierarchical tier was queried %d time(s) despite not applying filters", len(hier.seen()))
	}
	if len(sem.seen()) == 0 {
		t.Error("the semantic tier was not queried")
	}
	for _, e := range answer.Evidence {
		if e.Tier == core.TierHierarchical {
			t.Errorf("evidence from a tier that cannot filter reached the answer: %s", e.Cite())
		}
	}
	// The trace has to say so, or the omission is invisible.
	if !strings.Contains(answer.Trace.String(), "cannot filter") {
		t.Errorf("trace does not record the skipped tier:\n%s", answer.Trace)
	}
}

// Without a filter nothing is skipped — the restriction is the filter's, not a
// permanent demotion of tiers that do not implement Filterable.
func TestLoopKeepsAllTiersWithoutFilter(t *testing.T) {
	sem := &filteringRetriever{stubRetriever: stubRetriever{
		tier:    core.TierSemantic,
		byQuery: func(string) []core.Result { return []core.Result{result(core.TierSemantic, "a#0", "evidence")} },
	}}
	hier := &stubRetriever{
		tier:    core.TierHierarchical,
		byQuery: func(string) []core.Result { return []core.Result{result(core.TierHierarchical, "L1-0", "summary")} },
	}

	loop := loopOver(sufficientJudge(), sem, hier)
	if _, err := loop.Run(context.Background(), "recurring themes across all docs?"); err != nil {
		t.Fatal(err)
	}
	if len(hier.seen()) == 0 {
		t.Error("the hierarchical tier was skipped with no filter set")
	}
}

// A filter that leaves nothing usable must fail loudly. Returning an empty
// answer would read as "the corpus does not say", which is a different claim.
func TestLoopErrorsWhenFilterLeavesNoTier(t *testing.T) {
	hier := &stubRetriever{
		tier:    core.TierHierarchical,
		byQuery: func(string) []core.Result { return []core.Result{result(core.TierHierarchical, "L1-0", "summary")} },
	}
	loop := loopOver(sufficientJudge(), hier)
	loop.Filter = core.Filter{"dir": "docs"}

	_, err := loop.Run(context.Background(), "recurring themes across all docs?")
	if err == nil {
		t.Fatal("expected an error when no registered tier can apply the filter")
	}
	if !strings.Contains(err.Error(), "filter") {
		t.Errorf("error does not mention the filter: %v", err)
	}
	if len(hier.seen()) != 0 {
		t.Error("a tier that cannot filter was queried anyway")
	}
}

// The judge may promote a tier the router did not pick — that is how it
// corrects a misroute — but it must not readmit one the filter ruled out.
func TestJudgeCannotReadmitAFilteredOutTier(t *testing.T) {
	sem := &filteringRetriever{stubRetriever: stubRetriever{
		tier:    core.TierSemantic,
		byQuery: func(string) []core.Result { return []core.Result{result(core.TierSemantic, "a#0", "partial")} },
	}}
	hier := &stubRetriever{
		tier:    core.TierHierarchical,
		byQuery: func(string) []core.Result { return []core.Result{result(core.TierHierarchical, "L1-0", "summary")} },
	}

	f := &fake.LLM{ChatFunc: func(msgs []llm.Message) (string, error) {
		if isJudge(msgs) {
			return `{"sufficient": false, "gap": "themes", "refined_query": "themes", "next_tier": "hierarchical"}`, nil
		}
		return "An answer.", nil
	}}
	loop := loopOver(f, sem, hier)
	loop.MaxSteps = 3
	loop.Filter = core.Filter{"dir": "docs"}

	if _, err := loop.Run(context.Background(), "what is the budget?"); err != nil {
		t.Fatal(err)
	}
	if len(hier.seen()) != 0 {
		t.Fatalf("the judge readmitted a tier the filter excluded: %v", hier.seen())
	}
}

// citingLoop answers with the given text over one fixed evidence item.
func citingLoop(answer string) (*Loop, *stubRetriever) {
	sem := &stubRetriever{
		tier:    core.TierSemantic,
		byQuery: func(string) []core.Result { return []core.Result{result(core.TierSemantic, "a.md#0", "evidence")} },
	}
	f := &fake.LLM{ChatFunc: func(msgs []llm.Message) (string, error) {
		if isJudge(msgs) {
			return `{"sufficient": true}`, nil
		}
		return answer, nil
	}}
	return loopOver(f, sem), sem
}

// A fabricated citation reaching the caller with nothing to distinguish it from
// a real one is the failure a grounded system exists to prevent. The eval has
// always checked this; until now the library did not.
func TestLoopReportsFabricatedCitations(t *testing.T) {
	loop, _ := citingLoop("Budget is 4M [semantic:a.md#0] and 2M [semantic:invented.md#9].")

	answer, err := loop.Run(context.Background(), "what is the budget?")
	if err != nil {
		t.Fatal(err)
	}
	if answer.Citations.OK() {
		t.Fatal("a citation pointing at nothing was not reported")
	}
	if len(answer.Citations.Unresolved) != 1 || answer.Citations.Unresolved[0] != "[semantic:invented.md#9]" {
		t.Errorf("unresolved = %v", answer.Citations.Unresolved)
	}
	// The default policy leaves the text alone: the caller is told and decides.
	if !strings.Contains(answer.Text, "invented.md#9") {
		t.Error("CiteReport altered the answer text")
	}
	if !strings.Contains(answer.Trace.String(), "FABRICATED") {
		t.Errorf("trace does not flag the fabrication:\n%s", answer.Trace)
	}
}

func TestCitationPolicyStrip(t *testing.T) {
	loop, _ := citingLoop("Budget is 4M [semantic:a.md#0] and 2M [semantic:invented.md#9].")
	loop.CitationPolicy = CiteStrip

	answer, err := loop.Run(context.Background(), "what is the budget?")
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(answer.Text, "invented.md#9") {
		t.Errorf("CiteStrip left the fabricated marker in: %q", answer.Text)
	}
	if !strings.Contains(answer.Text, "[semantic:a.md#0]") {
		t.Errorf("CiteStrip removed a real citation too: %q", answer.Text)
	}
	// The report still names what was stripped, or the removal is itself silent.
	if len(answer.Citations.Unresolved) != 1 {
		t.Errorf("unresolved = %v, want the stripped marker still reported", answer.Citations.Unresolved)
	}
}

func TestCitationPolicyError(t *testing.T) {
	loop, _ := citingLoop("Budget is 4M [semantic:invented.md#9].")
	loop.CitationPolicy = CiteError

	answer, err := loop.Run(context.Background(), "what is the budget?")
	if err == nil {
		t.Fatal("CiteError returned a fabricated citation without an error")
	}
	if !strings.Contains(err.Error(), "invented.md#9") {
		t.Errorf("error does not name the bad marker: %v", err)
	}
	// The answer and its report still come back, so a caller can log what the
	// model actually said rather than only that something was wrong.
	if answer.Text == "" || answer.Citations.OK() {
		t.Error("CiteError discarded the evidence of what went wrong")
	}
}

// A clean answer must not be disturbed by any of this.
func TestCitationVerificationLeavesGoodAnswersAlone(t *testing.T) {
	const good = "Budget is 4M [semantic:a.md#0]."
	for _, policy := range []CitationPolicy{CiteReport, CiteStrip, CiteError} {
		loop, _ := citingLoop(good)
		loop.CitationPolicy = policy
		answer, err := loop.Run(context.Background(), "what is the budget?")
		if err != nil {
			t.Fatalf("policy %d: %v", policy, err)
		}
		if answer.Text != good {
			t.Errorf("policy %d rewrote a clean answer: %q", policy, answer.Text)
		}
		if !answer.Citations.OK() || len(answer.Citations.Resolved) != 1 {
			t.Errorf("policy %d: report = %+v", policy, answer.Citations)
		}
	}
}

// streamingFake implements llm.Streamer, splitting the answer into fragments.
type streamingFake struct {
	*fake.LLM
	streamed int
}

func (s *streamingFake) ChatStream(ctx context.Context, msgs []llm.Message, onDelta func(string)) (string, error) {
	text, err := s.Chat(ctx, msgs)
	if err != nil {
		return "", err
	}
	s.streamed++
	for _, word := range strings.SplitAfter(text, " ") {
		if onDelta != nil && word != "" {
			onDelta(word)
		}
	}
	return text, nil
}

func TestLoopStreamsGeneration(t *testing.T) {
	f := &streamingFake{LLM: &fake.LLM{ChatFunc: func(msgs []llm.Message) (string, error) {
		if isJudge(msgs) {
			return `{"sufficient": true}`, nil
		}
		return "The budget is four million dollars.", nil
	}}}
	sem := &stubRetriever{
		tier:    core.TierSemantic,
		byQuery: func(string) []core.Result { return []core.Result{result(core.TierSemantic, "a#0", "evidence")} },
	}

	loop := loopOver(f, sem)
	var got []string
	loop.OnDelta = func(s string) { got = append(got, s) }

	answer, err := loop.Run(context.Background(), "what is the budget?")
	if err != nil {
		t.Fatal(err)
	}
	if len(got) < 2 {
		t.Fatalf("received %d fragment(s), want the answer in pieces: %v", len(got), got)
	}
	// Answer.Text still holds the whole answer, so a streaming caller and a
	// non-streaming one get the same value back.
	if strings.Join(got, "") != answer.Text {
		t.Errorf("fragments %q do not reassemble to %q", strings.Join(got, ""), answer.Text)
	}
	if f.streamed != 1 {
		t.Errorf("streamed %d time(s), want only generation", f.streamed)
	}
}

// Only generation streams. The judge produces JSON nobody wants to watch
// assemble, and streaming it would emit it into the answer.
func TestLoopDoesNotStreamTheJudge(t *testing.T) {
	f := &streamingFake{LLM: &fake.LLM{ChatFunc: func(msgs []llm.Message) (string, error) {
		if isJudge(msgs) {
			return `{"sufficient": false, "gap": "x", "refined_query": "y"}`, nil
		}
		return "An answer.", nil
	}}}
	sem := &stubRetriever{
		tier:    core.TierSemantic,
		byQuery: func(string) []core.Result { return []core.Result{result(core.TierSemantic, "a#0", "evidence")} },
	}
	loop := loopOver(f, sem)
	loop.MaxSteps = 3
	var got strings.Builder
	loop.OnDelta = func(s string) { got.WriteString(s) }

	if _, err := loop.Run(context.Background(), "what is the budget?"); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(got.String(), "sufficient") {
		t.Errorf("judge JSON was streamed to the caller: %q", got.String())
	}
	if f.streamed != 1 {
		t.Errorf("streamed %d time(s), want 1", f.streamed)
	}
}

// A non-streaming LLM must still work with OnDelta set: the interface is
// optional and the loop falls back to Chat.
func TestLoopFallsBackWhenTheModelCannotStream(t *testing.T) {
	loop, _ := citingLoop("An answer [semantic:a.md#0].")
	loop.OnDelta = func(string) { t.Error("OnDelta fired on a non-streaming model") }

	answer, err := loop.Run(context.Background(), "what is the budget?")
	if err != nil {
		t.Fatal(err)
	}
	if answer.Text == "" {
		t.Error("no answer from the non-streaming fallback")
	}
}
