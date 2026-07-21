package fake

import (
	"context"
	"math"
	"testing"

	"github.com/RootControl/dispatch/llm"
)

func dot(a, b []float64) float64 {
	var s float64
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

func TestEmbedDeterministicAndUnit(t *testing.T) {
	f := &LLM{}
	got, err := f.Embed(context.Background(), []string{"invoice total due", "invoice total due"})
	if err != nil {
		t.Fatal(err)
	}
	// Identical inputs -> identical vectors.
	for i := range got[0] {
		if got[0][i] != got[1][i] {
			t.Fatalf("identical text embedded differently at dim %d", i)
		}
	}
	// Unit length.
	if n := math.Sqrt(dot(got[0], got[0])); math.Abs(n-1) > 1e-9 {
		t.Fatalf("embedding not unit length: %v", n)
	}
}

func TestEmbedSharedTokensAreCloser(t *testing.T) {
	f := &LLM{}
	v, err := f.Embed(context.Background(), []string{
		"the cat sat on the mat",
		"the cat sat on the rug", // shares most tokens
		"quarterly revenue projections",
	})
	if err != nil {
		t.Fatal(err)
	}
	near := dot(v[0], v[1])
	far := dot(v[0], v[2])
	if near <= far {
		t.Fatalf("expected shared-token pair closer: near=%.3f far=%.3f", near, far)
	}
}

func TestChatJSONScripted(t *testing.T) {
	f := &LLM{ChatFunc: func([]llm.Message) (string, error) {
		return `{"sufficient": true, "gap": ""}`, nil
	}}
	var out struct {
		Sufficient bool   `json:"sufficient"`
		Gap        string `json:"gap"`
	}
	if err := f.ChatJSON(context.Background(), []llm.Message{llm.User("q")}, &out); err != nil {
		t.Fatal(err)
	}
	if !out.Sufficient || out.Gap != "" {
		t.Fatalf("unexpected decode: %+v", out)
	}
	if f.Calls() != 1 {
		t.Fatalf("expected 1 call, got %d", f.Calls())
	}
}
