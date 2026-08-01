package index

import (
	"context"
	"os"
	"testing"

	"github.com/RootControl/dispatch/core"
	"github.com/RootControl/dispatch/internal/fake"
	"github.com/RootControl/dispatch/llm"
)

// Embeddings were the one ingest cost the cache did not cover: context
// sentences, graph extraction and coreference were all content-addressed, so a
// re-ingest reported "0 calls" while still paying to embed every chunk.
func TestEmbeddingCacheAvoidsRepeatEmbedding(t *testing.T) {
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
			EmbedTag:      "test-embed",
			Chunk:         ChunkOptions{TargetTokens: 6, NoOverlap: true},
		}), f
	}

	s1, f1 := newStore()
	st1, err := s1.Ingest(ctx, docs)
	if err != nil {
		t.Fatal(err)
	}
	if f1.Embeds() != st1.Chunks || st1.EmbedCalls != st1.Chunks || st1.EmbedHits != 0 {
		t.Fatalf("first ingest: embedded=%d EmbedCalls=%d EmbedHits=%d chunks=%d",
			f1.Embeds(), st1.EmbedCalls, st1.EmbedHits, st1.Chunks)
	}

	// A fresh store sharing the cache dir — a re-ingest in a new process, with
	// no index to skip from — must not embed anything at all.
	s2, f2 := newStore()
	st2, err := s2.Ingest(ctx, docs)
	if err != nil {
		t.Fatal(err)
	}
	if f2.Embeds() != 0 {
		t.Fatalf("second ingest embedded %d texts, want 0", f2.Embeds())
	}
	if st2.EmbedHits != st2.Chunks || st2.EmbedCalls != 0 {
		t.Fatalf("second ingest: EmbedHits=%d EmbedCalls=%d chunks=%d", st2.EmbedHits, st2.EmbedCalls, st2.Chunks)
	}

	// The cached vectors must be the vectors, not merely the right count: a
	// cache that round-trips a subtly different float ranks differently from a
	// live run, which is exactly the kind of drift the eval numbers cannot see.
	q := "What is the Atlas budget?"
	h1, err := s1.Search(ctx, core.Query{Text: q, TopK: 5})
	if err != nil {
		t.Fatal(err)
	}
	h2, err := s2.Search(ctx, core.Query{Text: q, TopK: 5})
	if err != nil {
		t.Fatal(err)
	}
	if len(h1) != len(h2) {
		t.Fatalf("cached store returned %d hits, live store %d", len(h2), len(h1))
	}
	for i := range h1 {
		if h1[i].Chunk.ID != h2[i].Chunk.ID || h1[i].Score != h2[i].Score {
			t.Fatalf("rank %d: live %s/%.17g, cached %s/%.17g",
				i, h1[i].Chunk.ID, h1[i].Score, h2[i].Chunk.ID, h2[i].Score)
		}
	}
}

// The embedding cache is keyed on content alone, so the same passage under two
// document IDs shares one entry rather than paying twice.
func TestEmbeddingCacheSharesIdenticalText(t *testing.T) {
	ctx := context.Background()
	f := &fake.LLM{}
	s := New(Config{LLM: f, Cache: NewCache(t.TempDir()), EmbedTag: "e1"})

	same := "The Atlas migration budget is one million dollars."
	st, err := s.Ingest(ctx, []core.Doc{{ID: "a", Text: same}, {ID: "b", Text: same}})
	if err != nil {
		t.Fatal(err)
	}
	if st.Chunks != 2 {
		t.Fatalf("expected 2 chunks, got %d", st.Chunks)
	}
	if st.EmbedCalls != 1 || st.EmbedHits != 1 {
		t.Fatalf("identical text under two doc IDs: EmbedCalls=%d EmbedHits=%d, want 1 and 1", st.EmbedCalls, st.EmbedHits)
	}
}

// A different embedding model must not be served another model's vectors.
func TestEmbeddingCacheIsScopedToModel(t *testing.T) {
	ctx := context.Background()
	cache := NewCache(t.TempDir())
	docs := []core.Doc{{ID: "a", Text: "Expenses over five hundred dollars need approval."}}

	s1 := New(Config{LLM: &fake.LLM{}, Cache: cache, EmbedTag: "model-one"})
	if _, err := s1.Ingest(ctx, docs); err != nil {
		t.Fatal(err)
	}

	f2 := &fake.LLM{}
	s2 := New(Config{LLM: f2, Cache: cache, EmbedTag: "model-two"})
	st, err := s2.Ingest(ctx, docs)
	if err != nil {
		t.Fatal(err)
	}
	if st.EmbedHits != 0 || f2.Embeds() == 0 {
		t.Fatalf("a second embedding model reused the first's vectors: EmbedHits=%d embedded=%d", st.EmbedHits, f2.Embeds())
	}
}

// A truncated or corrupt cache entry must read as a miss. Returning a short
// vector would silently corrupt every ranking it takes part in, where a miss
// costs one embedding call.
func TestCorruptVectorCacheEntryIsAMiss(t *testing.T) {
	cache := NewCache(t.TempDir())
	key := VecKey("e1", "some text")
	if err := cache.PutVector(key, []float64{0.1, 0.2}); err != nil {
		t.Fatal(err)
	}
	if _, ok := cache.GetVector(key); !ok {
		t.Fatal("a vector just written did not read back")
	}

	for name, bad := range map[string]string{
		"truncated JSON": `[0.1, 0.2`,
		"not an array":   `{"v":[0.1]}`,
		"empty array":    `[]`,
	} {
		if err := writeRaw(cache, key, bad); err != nil {
			t.Fatal(err)
		}
		if v, ok := cache.GetVector(key); ok {
			t.Errorf("%s read back as a hit: %v", name, v)
		}
	}
}

// writeRaw overwrites a cached vector with arbitrary bytes, to exercise the
// decode path against entries the encoder would never produce.
func writeRaw(c *Cache, key, content string) error {
	return os.WriteFile(c.vecPath(key), []byte(content), 0o644)
}
