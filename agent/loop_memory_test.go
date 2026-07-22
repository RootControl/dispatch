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

// The keyword router has no pattern for the memory tier, so if the loop did not
// add it, archival memory would only ever be searched by luck.
func TestLoopAlwaysConsultsRegisteredMemory(t *testing.T) {
	ctx := context.Background()
	f := &fake.LLM{ChatFunc: func(msgs []llm.Message) (string, error) {
		if isJudge(msgs) {
			return `{"sufficient": true}`, nil
		}
		return "answer [semantic:c#1]", nil
	}}
	mem := NewMemory(MemoryConfig{LLM: f, Dir: t.TempDir(), CoreBudget: 40})
	// Force eviction so something actually lives in archival.
	for _, e := range []string{"Vendor contract renews each January.", "Budget is four million dollars."} {
		if err := mem.Remember(ctx, e); err != nil {
			t.Fatal(err)
		}
	}
	if mem.ArchivalLen() == 0 {
		t.Fatal("expected an evicted entry in archival")
	}

	ret := &stubRetriever{tier: core.TierSemantic, byQuery: func(string) []core.Result {
		return []core.Result{result(core.TierSemantic, "c#1", "corpus text")}
	}}
	loop := &Loop{
		LLM:    f,
		Router: router.Heuristic{Available: []core.Tier{core.TierSemantic}},
		Retrievers: map[core.Tier]core.Retriever{
			core.TierSemantic: ret,
			core.TierMemory:   mem,
		},
		Memory: mem,
	}

	got, err := loop.Run(ctx, "anything at all")
	if err != nil {
		t.Fatal(err)
	}
	routed := got.Trace.Steps[0].Tiers
	if !slices.Contains(routed, core.TierMemory) {
		t.Fatalf("memory should be in the fan-out, got %v", routed)
	}
	if !strings.Contains(got.Trace.String(), "retrieve [memory]") {
		t.Errorf("trace should show a memory retrieval:\n%s", got.Trace)
	}
}

// The core block must reach the generation prompt, since that is what makes it
// "always in context" rather than merely stored.
func TestLoopPutsCoreMemoryInGenerationPrompt(t *testing.T) {
	ctx := context.Background()
	var generatePrompt string
	f := &fake.LLM{ChatFunc: func(msgs []llm.Message) (string, error) {
		if isJudge(msgs) {
			return `{"sufficient": true}`, nil
		}
		generatePrompt = msgs[len(msgs)-1].Content
		return "answer", nil
	}}
	mem := NewMemory(MemoryConfig{LLM: f, Dir: t.TempDir()})
	if err := mem.Remember(ctx, "Priya Raman leads Project Atlas."); err != nil {
		t.Fatal(err)
	}

	ret := &stubRetriever{tier: core.TierSemantic, byQuery: func(string) []core.Result {
		return []core.Result{result(core.TierSemantic, "c#1", "corpus text")}
	}}
	loop := &Loop{
		LLM:        f,
		Router:     router.Heuristic{Available: []core.Tier{core.TierSemantic}},
		Retrievers: map[core.Tier]core.Retriever{core.TierSemantic: ret},
		Memory:     mem,
	}
	if _, err := loop.Run(ctx, "who leads it?"); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(generatePrompt, "Priya Raman leads Project Atlas.") {
		t.Errorf("core memory missing from generation prompt:\n%s", generatePrompt)
	}
	// Memory must be marked uncitable, or the model will invent [memory:...]
	// citations that resolve to nothing in Answer.Evidence.
	if !strings.Contains(generatePrompt, "NOT citable") {
		t.Errorf("core memory should be marked uncitable:\n%s", generatePrompt)
	}
}

// Write-back is best-effort: losing a takeaway must not lose the answer.
func TestLoopSurvivesWriteBackFailure(t *testing.T) {
	ctx := context.Background()
	f := &fake.LLM{ChatFunc: func(msgs []llm.Message) (string, error) {
		switch {
		case isJudge(msgs):
			return `{"sufficient": true}`, nil
		case strings.Contains(msgs[0].Content, "worth remembering"):
			return "definitely not json", nil
		}
		return "the answer [semantic:c#1]", nil
	}}
	mem := NewMemory(MemoryConfig{LLM: f, Dir: t.TempDir()})
	ret := &stubRetriever{tier: core.TierSemantic, byQuery: func(string) []core.Result {
		return []core.Result{result(core.TierSemantic, "c#1", "corpus text")}
	}}
	loop := &Loop{
		LLM:        f,
		Router:     router.Heuristic{Available: []core.Tier{core.TierSemantic}},
		Retrievers: map[core.Tier]core.Retriever{core.TierSemantic: ret},
		Memory:     mem,
		WriteBack:  true,
	}

	got, err := loop.Run(ctx, "a question")
	if err != nil {
		t.Fatalf("write-back failure should not sink the answer: %v", err)
	}
	if got.Text == "" {
		t.Error("expected an answer despite write-back failing")
	}
	if !strings.Contains(got.Trace.String(), "remember") {
		t.Errorf("trace should record the write-back attempt:\n%s", got.Trace)
	}
}
