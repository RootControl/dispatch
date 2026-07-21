package index

import (
	"context"
	"strings"
	"testing"

	"github.com/RootControl/dispatch/core"
	"github.com/RootControl/dispatch/internal/fake"
	"github.com/RootControl/dispatch/llm"
)

// A document whose budget figure lives in a chunk that never names the project.
// A query naming the project + "budget" should struggle to find that chunk by
// raw text but succeed once contextual chunking injects the project name.
const atlasDoc = "Atlas kickoff roster.\n\n" +
	"Budget four million dollars.\n\n" +
	"Lunch parking logistics.\n\n" +
	"Sunny coastal weather."

func buildStore(t *testing.T, contextual bool) *Store {
	t.Helper()
	f := &fake.LLM{ChatFunc: func(msgs []llm.Message) (string, error) {
		// A real contextualizer writes a chunk-specific situating sentence: it
		// names the source project (injecting "atlas" into chunks that omit it)
		// and describes what the chunk is about. The budget chunk's context
		// therefore mentions the budget — which is exactly the signal that makes
		// it findable by "atlas budget".
		chunk := between(msgs[len(msgs)-1].Content, "<chunk>", "</chunk>")
		if strings.Contains(strings.ToLower(chunk), "budget") {
			return "This part of the Atlas project report gives the approved budget.", nil
		}
		return "This part of the Atlas project report covers logistics.", nil
	}}
	s := New(Config{
		LLM:           f,
		Contextualize: contextual,
		Chunk:         ChunkOptions{TargetTokens: 6, NoOverlap: true},
	})
	if _, err := s.Ingest(context.Background(), []core.Doc{{ID: "atlas", Text: atlasDoc}}); err != nil {
		t.Fatalf("ingest: %v", err)
	}
	if s.Len() != 4 {
		t.Fatalf("expected 4 chunks, got %d", s.Len())
	}
	return s
}

func between(s, open, close string) string {
	_, rest, ok := strings.Cut(s, open)
	if !ok {
		return ""
	}
	inner, _, ok := strings.Cut(rest, close)
	if !ok {
		return rest
	}
	return inner
}

func rankOf(hits []Hit, id string) (int, float64, bool) {
	for i, h := range hits {
		if h.Chunk.ID == id {
			return i, h.Score, true
		}
	}
	return -1, 0, false
}

func TestContextualChunkingImprovesRetrieval(t *testing.T) {
	const target = "atlas#1" // the budget chunk, which never says "atlas"
	ctx := context.Background()

	raw, err := buildStore(t, false).Search(ctx, "atlas budget", 4)
	if err != nil {
		t.Fatal(err)
	}
	ctxual, err := buildStore(t, true).Search(ctx, "atlas budget", 4)
	if err != nil {
		t.Fatal(err)
	}

	rawRank, rawScore, ok := rankOf(raw, target)
	if !ok {
		t.Fatalf("target missing from raw results: %v", hitIDs(raw))
	}
	ctxRank, ctxScore, ok := rankOf(ctxual, target)
	if !ok {
		t.Fatalf("target missing from contextual results: %v", hitIDs(ctxual))
	}

	// Contextual chunking should rank the budget chunk first...
	if ctxRank != 0 {
		t.Fatalf("contextual: target rank = %d, want 0 (results %v)", ctxRank, hitIDs(ctxual))
	}
	// ...and strictly better than the raw pipeline did.
	if !(ctxRank < rawRank || ctxScore > rawScore) {
		t.Fatalf("contextual did not improve target: rawRank=%d rawScore=%.4f ctxRank=%d ctxScore=%.4f",
			rawRank, rawScore, ctxRank, ctxScore)
	}
	t.Logf("target rank raw=%d ctx=%d (score raw=%.4f ctx=%.4f)", rawRank, ctxRank, rawScore, ctxScore)
}

func TestContextCacheAvoidsRepeatCalls(t *testing.T) {
	dir := t.TempDir()
	cache := NewCache(dir)
	ctx := context.Background()
	docs := []core.Doc{{ID: "atlas", Text: atlasDoc}}

	newStore := func() (*Store, *fake.LLM) {
		f := &fake.LLM{ChatFunc: func([]llm.Message) (string, error) {
			return "context sentence.", nil
		}}
		return New(Config{
			LLM:           f,
			Cache:         cache,
			Contextualize: true,
			Chunk:         ChunkOptions{TargetTokens: 6, NoOverlap: true},
		}), f
	}

	s1, f1 := newStore()
	st1, err := s1.Ingest(ctx, docs)
	if err != nil {
		t.Fatal(err)
	}
	// Embeddings aren't chat calls; every context sentence is one chat call.
	if f1.Calls() != st1.Chunks || st1.LLMCalls != st1.Chunks {
		t.Fatalf("first ingest: calls=%d LLMCalls=%d chunks=%d", f1.Calls(), st1.LLMCalls, st1.Chunks)
	}

	// Second ingest with the same cache dir should make zero chat calls.
	s2, f2 := newStore()
	st2, err := s2.Ingest(ctx, docs)
	if err != nil {
		t.Fatal(err)
	}
	if f2.Calls() != 0 {
		t.Fatalf("second ingest made %d chat calls, want 0 (cache miss)", f2.Calls())
	}
	if st2.CacheHits != st2.Chunks || st2.LLMCalls != 0 {
		t.Fatalf("second ingest: CacheHits=%d LLMCalls=%d chunks=%d", st2.CacheHits, st2.LLMCalls, st2.Chunks)
	}

	// And Plan should now predict zero calls.
	if plan := s2.Plan(docs); plan.LLMCalls != 0 || plan.CacheHits != plan.Chunks {
		t.Fatalf("plan after cache warm: LLMCalls=%d CacheHits=%d chunks=%d", plan.LLMCalls, plan.CacheHits, plan.Chunks)
	}
}

func hitIDs(hits []Hit) []string {
	out := make([]string, len(hits))
	for i, h := range hits {
		out[i] = h.Chunk.ID
	}
	return out
}
