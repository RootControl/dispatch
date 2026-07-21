package index

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/RootControl/dispatch/core"
	"github.com/RootControl/dispatch/llm"
)

// Store is the reference hybrid index: contextual chunking on ingest, cosine +
// BM25 fused by RRF on search, all in memory and stdlib-only. It backs the
// semantic tier and the agent's archival memory. Everything above the Store
// depends only on its methods, so a production engine can replace it wholesale.
type Store struct {
	llm        llm.LLM
	contextLLM llm.LLM // optional cheaper model for context sentences
	cache      *Cache
	opts       Config

	mu     sync.RWMutex
	chunks map[string]core.Chunk
	vec    *vectorIndex
	bm     *bm25Index
}

// Config configures a Store.
type Config struct {
	LLM           llm.LLM      // required: embeddings (+ context sentences if ContextLLM unset)
	ContextLLM    llm.LLM      // optional: cheaper model just for context sentences
	Cache         *Cache       // optional: content-hash cache for context sentences
	Chunk         ChunkOptions // splitter tuning
	Contextualize bool         // run the contextual-chunking LLM pass on ingest
	CacheTag      string       // model identity folded into cache keys; default "default"
	EmbedTag      string       // embedding-model identity recorded in saved indexes
	Parallelism   int          // concurrent context-sentence calls; default 4
}

// New builds a Store from cfg.
func New(cfg Config) *Store {
	if cfg.CacheTag == "" {
		cfg.CacheTag = "default"
	}
	if cfg.Parallelism <= 0 {
		cfg.Parallelism = 4
	}
	cl := cfg.ContextLLM
	if cl == nil {
		cl = cfg.LLM
	}
	return &Store{
		llm:        cfg.LLM,
		contextLLM: cl,
		cache:      cfg.Cache,
		opts:       cfg,
		chunks:     map[string]core.Chunk{},
		vec:        &vectorIndex{},
		bm:         newBM25(),
	}
}

// IngestStats reports what an ingest did (or, from Plan, would do).
type IngestStats struct {
	Docs      int
	Chunks    int
	LLMCalls  int // context-sentence calls actually made (cache misses)
	CacheHits int // context sentences served from cache
}

// Plan estimates an ingest without calling the LLM or mutating the store, so
// `ingest --dry-run` can report chunk count and expected LLM calls — consulting
// the cache so the estimate reflects work already done.
func (s *Store) Plan(docs []core.Doc) IngestStats {
	var st IngestStats
	for _, doc := range docs {
		texts := Split(doc.Text, s.opts.Chunk)
		st.Docs++
		st.Chunks += len(texts)
		if !s.opts.Contextualize {
			continue
		}
		for _, t := range texts {
			if s.cache.Has(Key(s.opts.CacheTag, doc.ID, t)) {
				st.CacheHits++
			} else {
				st.LLMCalls++
			}
		}
	}
	return st
}

// Ingest chunks each document, optionally writes a situating context sentence
// per chunk (contextual chunking), embeds context+chunk, and indexes both the
// vector and BM25 representations. Safe to call repeatedly to add documents.
func (s *Store) Ingest(ctx context.Context, docs []core.Doc) (IngestStats, error) {
	var st IngestStats
	for _, doc := range docs {
		texts := Split(doc.Text, s.opts.Chunk)
		chunks := make([]core.Chunk, len(texts))
		for i, t := range texts {
			chunks[i] = core.Chunk{
				ID:       fmt.Sprintf("%s#%d", doc.ID, i),
				DocID:    doc.ID,
				Text:     t,
				Position: i,
				Meta:     doc.Meta,
			}
		}
		if s.opts.Contextualize {
			if err := s.addContext(ctx, doc, chunks, &st); err != nil {
				return st, err
			}
		}

		inputs := make([]string, len(chunks))
		for i, c := range chunks {
			inputs[i] = c.Embedded()
		}
		vecs, err := s.llm.Embed(ctx, inputs)
		if err != nil {
			return st, fmt.Errorf("index: embed doc %q: %w", doc.ID, err)
		}
		if len(vecs) != len(chunks) {
			return st, fmt.Errorf("index: embed doc %q: got %d vectors for %d chunks", doc.ID, len(vecs), len(chunks))
		}

		s.mu.Lock()
		for i, c := range chunks {
			s.chunks[c.ID] = c
			s.vec.add(c.ID, vecs[i])
			// Index context+chunk for BM25 too, so the situating sentence's terms
			// aid lexical search — matching Anthropic's contextual-retrieval recipe.
			s.bm.add(c.ID, c.Embedded())
		}
		s.mu.Unlock()

		st.Docs++
		st.Chunks += len(chunks)
	}
	return st, nil
}

// addContext fills each chunk's Context field, cache-first, with bounded
// concurrency. Goroutines write disjoint chunk indices, so no per-chunk locking
// is needed; only the shared counters are synchronized.
func (s *Store) addContext(ctx context.Context, doc core.Doc, chunks []core.Chunk, st *IngestStats) error {
	var calls, hits atomic.Int64
	err := forEachLimited(ctx, len(chunks), s.opts.Parallelism, func(ctx context.Context, i int) error {
		key := Key(s.opts.CacheTag, doc.ID, chunks[i].Text)
		if v, ok := s.cache.Get(key); ok {
			chunks[i].Context = v
			hits.Add(1)
			return nil
		}
		reply, err := s.contextLLM.Chat(ctx, contextMessages(doc, chunks[i].Text))
		if err != nil {
			return fmt.Errorf("index: context sentence for %s#%d: %w", doc.ID, i, err)
		}
		chunks[i].Context = strings.TrimSpace(reply)
		calls.Add(1)
		return s.cache.Put(key, chunks[i].Context)
	})
	st.LLMCalls += int(calls.Load())
	st.CacheHits += int(hits.Load())
	return err
}

// contextMessages builds the contextual-chunking prompt: given the whole
// document and one chunk, ask for a single sentence that situates the chunk for
// retrieval. This is the ~49% failure-reduction lever from the design doc.
func contextMessages(doc core.Doc, chunk string) []llm.Message {
	return []llm.Message{
		llm.System("You situate a chunk of text within its source document to improve search retrieval. " +
			"Reply with a single short sentence giving the context needed to understand what the chunk is about. " +
			"Do not add preamble or quotation marks."),
		llm.User(fmt.Sprintf("<document>\n%s\n</document>\n\n<chunk>\n%s\n</chunk>\n\n"+
			"Give the one-sentence context that situates this chunk within the document.", doc.Text, chunk)),
	}
}

// Hit is one search result: the matched chunk and its fused RRF score.
type Hit struct {
	Chunk core.Chunk
	Score float64
}

// Search runs the hybrid query: embed, take a candidate pool from each of the
// cosine and BM25 indexes, fuse by RRF, and return the top topK chunks.
func (s *Store) Search(ctx context.Context, query string, topK int) ([]Hit, error) {
	if topK <= 0 {
		topK = 10
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if len(s.chunks) == 0 {
		return nil, nil
	}

	qvec, err := s.llm.Embed(ctx, []string{query})
	if err != nil {
		return nil, fmt.Errorf("index: embed query: %w", err)
	}
	pool := max(topK*5, 20)
	vecHits := s.vec.search(qvec[0], pool)
	bmHits := s.bm.search(query, pool)

	fused := fuseRRF([][]scored{vecHits, bmHits}, 60, topK)
	out := make([]Hit, 0, len(fused))
	for _, f := range fused {
		out = append(out, Hit{Chunk: s.chunks[f.id], Score: f.score})
	}
	return out, nil
}

// Len reports the number of indexed chunks.
func (s *Store) Len() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.chunks)
}

// forEachLimited runs fn for indices 0..n-1 with at most limit concurrent
// goroutines, returning the first error and cancelling the rest.
func forEachLimited(ctx context.Context, n, limit int, fn func(ctx context.Context, i int) error) error {
	if limit <= 0 {
		limit = 1
	}
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	sem := make(chan struct{}, limit)
	var wg sync.WaitGroup
	var mu sync.Mutex
	var firstErr error

	for i := range n {
		select {
		case <-ctx.Done():
			mu.Lock()
			if firstErr == nil {
				firstErr = ctx.Err()
			}
			mu.Unlock()
			wg.Wait()
			return firstErr
		case sem <- struct{}{}:
		}
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			defer func() { <-sem }()
			if err := fn(ctx, i); err != nil {
				mu.Lock()
				if firstErr == nil {
					firstErr = err
					cancel()
				}
				mu.Unlock()
			}
		}(i)
	}
	wg.Wait()
	return firstErr
}
