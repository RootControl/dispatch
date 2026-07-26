package index

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/RootControl/dispatch/core"
	"github.com/RootControl/dispatch/internal/fake"
	"github.com/RootControl/dispatch/llm"
)

func hits(ids ...string) []Hit {
	out := make([]Hit, len(ids))
	for i, id := range ids {
		out[i] = Hit{Chunk: core.Chunk{ID: id, Text: "text of " + id}, Score: float64(len(ids) - i)}
	}
	return out
}

func hitIDList(h []Hit) []string {
	out := make([]string, len(h))
	for i, x := range h {
		out[i] = x.Chunk.ID
	}
	return out
}

func TestRerankReordersByScore(t *testing.T) {
	f := &fake.LLM{ChatFunc: func([]llm.Message) (string, error) {
		// Third candidate is the best; first is worst.
		return `{"scores":[{"id":0,"score":1},{"id":1,"score":5},{"id":2,"score":9}]}`, nil
	}}
	r := &LLMReranker{LLM: f}
	got, err := r.Rerank(context.Background(), "q", hits("a", "b", "c"), 3)
	if err != nil {
		t.Fatal(err)
	}
	if ids := hitIDList(got); ids[0] != "c" || ids[2] != "a" {
		t.Errorf("order = %v, want c first and a last", ids)
	}
	if got[0].Score != 9 {
		t.Errorf("score not carried through: %v", got[0].Score)
	}
}

func TestRerankTruncatesToTopK(t *testing.T) {
	f := &fake.LLM{ChatFunc: func([]llm.Message) (string, error) {
		return `{"scores":[{"id":0,"score":1},{"id":1,"score":9},{"id":2,"score":5}]}`, nil
	}}
	got, _ := (&LLMReranker{LLM: f}).Rerank(context.Background(), "q", hits("a", "b", "c"), 2)
	if len(got) != 2 {
		t.Fatalf("expected 2 hits, got %d", len(got))
	}
	if hitIDList(got)[0] != "b" {
		t.Errorf("expected the best hit kept, got %v", hitIDList(got))
	}
}

// Reranking refines an already-relevant shortlist. Failing the query would lose
// results entirely; returning them unreranked loses only the refinement.
func TestRerankFallsBackToRetrievalOrder(t *testing.T) {
	for name, f := range map[string]*fake.LLM{
		"error":     {ChatFunc: func([]llm.Message) (string, error) { return "", errors.New("down") }},
		"garbage":   {ChatFunc: func([]llm.Message) (string, error) { return "not json", nil }},
		"no scores": {ChatFunc: func([]llm.Message) (string, error) { return `{"scores":[]}`, nil }},
	} {
		got, err := (&LLMReranker{LLM: f}).Rerank(context.Background(), "q", hits("a", "b", "c"), 3)
		if err != nil {
			t.Errorf("%s: should not error: %v", name, err)
		}
		if ids := hitIDList(got); len(ids) != 3 || ids[0] != "a" {
			t.Errorf("%s: expected retrieval order preserved, got %v", name, ids)
		}
	}
}

// A partial reply must not drop the candidates it failed to mention.
func TestRerankKeepsUnscoredCandidates(t *testing.T) {
	f := &fake.LLM{ChatFunc: func([]llm.Message) (string, error) {
		return `{"scores":[{"id":1,"score":9}]}`, nil
	}}
	got, _ := (&LLMReranker{LLM: f}).Rerank(context.Background(), "q", hits("a", "b", "c"), 3)
	if len(got) != 3 {
		t.Fatalf("unscored candidates were dropped: %v", hitIDList(got))
	}
	if hitIDList(got)[0] != "b" {
		t.Errorf("scored candidate should lead: %v", hitIDList(got))
	}
}

// Out-of-range ids in the reply must not panic or reorder wrongly.
func TestRerankIgnoresBogusIDs(t *testing.T) {
	f := &fake.LLM{ChatFunc: func([]llm.Message) (string, error) {
		return `{"scores":[{"id":99,"score":10},{"id":-1,"score":10},{"id":0,"score":7}]}`, nil
	}}
	got, err := (&LLMReranker{LLM: f}).Rerank(context.Background(), "q", hits("a", "b"), 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || hitIDList(got)[0] != "a" {
		t.Errorf("got %v", hitIDList(got))
	}
}

func TestRerankPromptCarriesQueryAndPassages(t *testing.T) {
	var seen string
	f := &fake.LLM{ChatFunc: func(m []llm.Message) (string, error) {
		seen = m[len(m)-1].Content
		return `{"scores":[{"id":0,"score":5}]}`, nil
	}}
	if _, err := (&LLMReranker{LLM: f}).Rerank(context.Background(), "what is the budget?", hits("a"), 1); err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"what is the budget?", "[0]", "text of a"} {
		if !strings.Contains(seen, want) {
			t.Errorf("prompt missing %q:\n%s", want, seen)
		}
	}
}

// A reranker can only reorder what it is given, so Search must over-fetch when
// one is configured — otherwise it could improve the order but never the
// membership, which is most of the available gain.
func TestSearchOverFetchesForReranker(t *testing.T) {
	var candidates int
	f := &fake.LLM{}
	s := New(Config{LLM: f, Rerank: rerankFunc(func(_ context.Context, _ string, h []Hit, topK int) ([]Hit, error) {
		candidates = len(h)
		return truncate(h, topK), nil
	})})
	docs := make([]core.Doc, 30)
	for i := range docs {
		docs[i] = core.Doc{ID: string(rune('a' + i)), Text: "budget figures and vendor details"}
	}
	if _, err := s.Ingest(context.Background(), docs); err != nil {
		t.Fatal(err)
	}
	got, err := s.Search(context.Background(), "budget", 3)
	if err != nil {
		t.Fatal(err)
	}
	if candidates <= 3 {
		t.Errorf("reranker saw %d candidates for topK=3; Search did not over-fetch", candidates)
	}
	if len(got) != 3 {
		t.Errorf("Search returned %d hits, want topK=3", len(got))
	}
}

type rerankFunc func(context.Context, string, []Hit, int) ([]Hit, error)

func (f rerankFunc) Rerank(ctx context.Context, q string, h []Hit, k int) ([]Hit, error) {
	return f(ctx, q, h, k)
}
