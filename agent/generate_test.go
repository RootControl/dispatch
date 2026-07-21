package agent

import (
	"context"
	"strings"
	"testing"

	"github.com/RootControl/dispatch/core"
	"github.com/RootControl/dispatch/internal/fake"
	"github.com/RootControl/dispatch/llm"
)

var evidence = []core.Result{
	{Tier: core.TierSemantic, SourceID: "charter#2", Text: "The approved figure is four million dollars."},
	{Tier: core.TierSemantic, SourceID: "risks#1", Text: "Renewal was signed in January."},
}

func TestFormatEvidenceLabelsEveryResult(t *testing.T) {
	got := FormatEvidence(evidence)
	for _, want := range []string{"[semantic:charter#2]", "[semantic:risks#1]", "four million"} {
		if !strings.Contains(got, want) {
			t.Errorf("formatted evidence missing %q:\n%s", want, got)
		}
	}
}

func TestGeneratePassesCitableEvidence(t *testing.T) {
	var seen string
	f := &fake.LLM{ChatFunc: func(msgs []llm.Message) (string, error) {
		seen = msgs[len(msgs)-1].Content
		return "The budget is four million dollars [semantic:charter#2].", nil
	}}
	got, err := Generate(context.Background(), f, "What is the budget?", evidence)
	if err != nil {
		t.Fatal(err)
	}
	// The prompt must carry the exact citation markers, or the model cannot
	// produce citations that resolve back to results.
	if !strings.Contains(seen, "[semantic:charter#2]") {
		t.Errorf("prompt lacked citation marker:\n%s", seen)
	}
	if !strings.Contains(seen, "What is the budget?") {
		t.Errorf("prompt lacked the question:\n%s", seen)
	}
	if !strings.Contains(got, "[semantic:charter#2]") {
		t.Errorf("answer lacked a citation: %q", got)
	}
}

// Generating with no evidence would invite the model to answer from parametric
// memory — exactly the ungrounded output this design exists to prevent.
func TestGenerateRefusesWithoutEvidence(t *testing.T) {
	f := &fake.LLM{ChatFunc: func([]llm.Message) (string, error) {
		t.Fatal("should not have called the LLM with zero evidence")
		return "", nil
	}}
	if _, err := Generate(context.Background(), f, "What is the budget?", nil); err == nil {
		t.Fatal("expected an error when generating with no evidence")
	}
}
