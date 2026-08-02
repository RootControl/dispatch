package index

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"maps"
	"slices"
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
	// docHash records the fingerprint each document was last ingested under, so
	// a re-ingest can skip documents nothing about which has changed. It is
	// persisted with the index, because the whole point is to survive a restart.
	docHash map[string]string
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
	// Rerank, when set, reorders the fused shortlist before Search returns.
	// Search over-fetches to give it something to work with.
	Rerank      Reranker
	Parallelism int // concurrent context-sentence calls; default 4
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
		docHash:    map[string]string{},
	}
}

// fingerprint hashes everything that determines a document's chunks and their
// embeddings: the text, the metadata that rides along on each chunk, the
// splitter settings, and the two model identities that decide what text is
// actually embedded. Ingest skips a document whose fingerprint is unchanged, so
// anything left out of this would be a setting you could change with no effect.
func (s *Store) fingerprint(doc core.Doc) string {
	h := sha256.New()
	write := func(parts ...string) {
		for _, p := range parts {
			h.Write([]byte(p))
			h.Write([]byte{0})
		}
	}
	write(doc.ID, doc.Text)
	for _, k := range slices.Sorted(maps.Keys(doc.Meta)) {
		write(k, doc.Meta[k])
	}
	c := s.opts.Chunk.withDefaults()
	write(fmt.Sprint(c.TargetTokens, c.OverlapTokens, c.MinWords, c.Headings, s.opts.Contextualize),
		s.opts.CacheTag, s.opts.EmbedTag)
	return hex.EncodeToString(h.Sum(nil))
}

// IngestStats reports what an ingest did (or, from Plan, would do).
type IngestStats struct {
	Docs       int
	Chunks     int
	LLMCalls   int // context-sentence calls actually made (cache misses)
	CacheHits  int // context sentences served from cache
	Unchanged  int // docs skipped: fingerprint matched what is already indexed
	EmbedCalls int // chunks sent to the embedder (cache misses)
	EmbedHits  int // chunk vectors served from cache
}

// Plan estimates an ingest without calling the LLM or mutating the store, so
// `ingest --dry-run` can report chunk count and expected LLM calls — consulting
// the cache and what is already indexed so the estimate reflects work already
// done. It cannot predict embedding cache hits for chunks whose context
// sentence is not yet written, since the cached key covers context+chunk; those
// are counted as calls, which makes the estimate an upper bound rather than an
// optimistic one.
func (s *Store) Plan(docs []core.Doc) IngestStats {
	var st IngestStats
	s.mu.RLock()
	defer s.mu.RUnlock()
	for _, doc := range docs {
		st.Docs++
		if s.docHash[doc.ID] == s.fingerprint(doc) {
			st.Unchanged++
			for _, c := range s.chunks {
				if c.DocID == doc.ID {
					st.Chunks++
				}
			}
			continue
		}
		texts := Split(doc.Text, s.opts.Chunk)
		st.Chunks += len(texts)
		for _, t := range texts {
			cs, cached := "", true
			if s.opts.Contextualize {
				if v, ok := s.cache.Get(Key(s.opts.CacheTag, doc.ID, t)); ok {
					cs, st.CacheHits = v, st.CacheHits+1
				} else {
					cached, st.LLMCalls = false, st.LLMCalls+1
				}
			}
			if cached && s.cache.HasVector(VecKey(s.opts.EmbedTag, core.Chunk{Text: t, Context: cs}.Embedded())) {
				st.EmbedHits++
			} else {
				st.EmbedCalls++
			}
		}
	}
	return st
}

// Delete removes every chunk belonging to the named documents and reports how
// many chunks went. Unknown document IDs are ignored.
func (s *Store) Delete(docIDs ...string) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.deleteLocked(docIDs)
}

// Prune removes every document not named in keep and reports how many chunks
// went. It is how a corpus that lost a file stops answering from it: an index
// built incrementally has no other way to learn that a document is gone.
func (s *Store) Prune(keep []string) int {
	live := make(map[string]bool, len(keep))
	for _, id := range keep {
		live[id] = true
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	var stale []string
	for _, c := range s.chunks {
		if !live[c.DocID] {
			stale = append(stale, c.DocID)
		}
	}
	return s.deleteLocked(stale)
}

// deleteLocked drops the documents' chunks from all three structures at once.
// The caller holds the write lock.
func (s *Store) deleteLocked(docIDs []string) int {
	if len(docIDs) == 0 {
		return 0
	}
	want := make(map[string]bool, len(docIDs))
	for _, d := range docIDs {
		want[d] = true
	}
	drop := make(map[string]bool)
	for id, c := range s.chunks {
		if want[c.DocID] {
			drop[id] = true
		}
	}
	for id := range drop {
		delete(s.chunks, id)
	}
	for d := range want {
		delete(s.docHash, d)
	}
	s.vec.remove(drop)
	s.bm.remove(drop)
	return len(drop)
}

// Ingest chunks each document, optionally writes a situating context sentence
// per chunk (contextual chunking), embeds context+chunk, and indexes both the
// vector and BM25 representations.
//
// Ingest is an upsert: re-ingesting a document replaces its chunks rather than
// adding a second copy. Before this was explicit, the chunk map deduplicated by
// ID while the vector and BM25 indexes appended, so a second ingest of unchanged
// documents left one entry per chunk and two vectors per chunk — inflating that
// chunk's rank credit under RRF, and persisting through Save, which walks the
// vector index. Len reported the map size, so nothing showed.
func (s *Store) Ingest(ctx context.Context, docs []core.Doc) (IngestStats, error) {
	var st IngestStats
	for _, doc := range docs {
		st.Docs++
		fp := s.fingerprint(doc)

		// Nothing that determines this document's chunks or their vectors has
		// changed, so re-deriving them would produce byte-identical results at
		// full embedding cost. Skipping is what makes `ingest` over a corpus
		// where one file moved cost one file rather than the corpus.
		s.mu.RLock()
		unchanged, indexed := s.docHash[doc.ID] == fp, 0
		if unchanged {
			for _, c := range s.chunks {
				if c.DocID == doc.ID {
					indexed++
				}
			}
		}
		s.mu.RUnlock()
		if unchanged {
			st.Unchanged++
			st.Chunks += indexed
			continue
		}

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

		vecs, err := s.embed(ctx, doc.ID, chunks, &st)
		if err != nil {
			return st, err
		}

		s.mu.Lock()
		// Upsert: drop whatever this document previously contributed before
		// adding its new chunks. A document that now splits into fewer chunks
		// would otherwise leave its old tail behind, indexed and unreachable
		// from the document it no longer belongs to.
		s.deleteLocked([]string{doc.ID})
		s.docHash[doc.ID] = fp
		for i, c := range chunks {
			s.chunks[c.ID] = c
			s.vec.add(c.ID, vecs[i])
			// Index context+chunk for BM25 too, so the situating sentence's terms
			// aid lexical search — matching Anthropic's contextual-retrieval recipe.
			s.bm.add(c.ID, c.Embedded())
		}
		s.mu.Unlock()

		st.Chunks += len(chunks)
	}
	return st, nil
}

// embed returns one vector per chunk, serving what it can from the cache and
// sending only the misses to the embedder.
//
// Embeddings were the one ingest cost the cache did not cover: context
// sentences, graph extraction and coreference were all content-addressed, so a
// re-ingest reported "0 calls" while still paying to embed every chunk. The key
// is the embedded text — context sentence included — under the embedding
// model's tag, because that is exactly what determines the vector.
func (s *Store) embed(ctx context.Context, docID string, chunks []core.Chunk, st *IngestStats) ([][]float64, error) {
	vecs := make([][]float64, len(chunks))
	var missIdx []int
	var missText []string
	for i, c := range chunks {
		text := c.Embedded()
		if v, ok := s.cache.GetVector(VecKey(s.opts.EmbedTag, text)); ok {
			vecs[i] = v
			st.EmbedHits++
			continue
		}
		missIdx = append(missIdx, i)
		missText = append(missText, text)
	}
	if len(missText) == 0 {
		return vecs, nil
	}

	got, err := s.llm.Embed(ctx, missText)
	if err != nil {
		return nil, fmt.Errorf("index: embed doc %q: %w", docID, err)
	}
	if len(got) != len(missText) {
		return nil, fmt.Errorf("index: embed doc %q: got %d vectors for %d chunks", docID, len(got), len(missText))
	}
	for j, i := range missIdx {
		vecs[i] = got[j]
		if err := s.cache.PutVector(VecKey(s.opts.EmbedTag, missText[j]), got[j]); err != nil {
			return nil, fmt.Errorf("index: cache vector for %s: %w", chunks[i].ID, err)
		}
	}
	st.EmbedCalls += len(missText)
	return vecs, nil
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

// Hit is one search result: the matched chunk, its fused RRF score, and how it
// got there.
type Hit struct {
	Chunk core.Chunk
	Score float64
	// VectorRank and TextRank are the 1-based positions this chunk held in the
	// cosine and BM25 candidate lists, or 0 when that half did not return it.
	//
	// They answer the first question anyone asks of a hybrid index: did this
	// come back because the embedding matched, because the terms matched, or
	// because both did? The fused score cannot say — two chunks scoring almost
	// identically can have arrived for entirely different reasons. A chunk with
	// TextRank 1 and VectorRank 0 is a lexical hit on a rare term; one ranked
	// mid-list by both is a weak agreement, and the difference matters when
	// deciding whether a corpus needs better chunking or a reranker.
	VectorRank, TextRank int
}

// Sources renders the provenance for a trace line: "vec#3 text#1", or "vec#3"
// when only one half found it.
func (h Hit) Sources() string {
	var parts []string
	if h.VectorRank > 0 {
		parts = append(parts, fmt.Sprintf("vec#%d", h.VectorRank))
	}
	if h.TextRank > 0 {
		parts = append(parts, fmt.Sprintf("text#%d", h.TextRank))
	}
	if len(parts) == 0 {
		return "reranked"
	}
	return strings.Join(parts, " ")
}

// Searcher is the retrieval half of a store: the surface the semantic tier and
// archival memory actually depend on. Keeping it separate from ingestion is
// what lets a production engine — pgvector, DuckDB — back the same tiers
// without reimplementing chunking, contextualisation or persistence.
type Searcher interface {
	Search(ctx context.Context, q core.Query) ([]Hit, error)
	Len() int
}

var _ Searcher = (*Store)(nil)

// Search runs the hybrid query: embed, take a candidate pool from each of the
// cosine and BM25 indexes, fuse by RRF, and return the top q.TopK chunks.
// q.Filter, when set, restricts the scan to chunks whose Meta matches.
func (s *Store) Search(ctx context.Context, q core.Query) ([]Hit, error) {
	topK := q.TopK
	if topK <= 0 {
		topK = 10
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if len(s.chunks) == 0 {
		return nil, nil
	}

	// Resolving the filter to a predicate over chunk IDs keeps both indexes
	// filtering on the same rule, and keeps the rule itself in one place.
	var allow func(string) bool
	if !q.Filter.Empty() {
		allow = func(id string) bool { return q.Filter.Match(s.chunks[id].Meta) }
	}

	// Dense retrieval embeds q.Embedding(), which is the HyDE expansion when one
	// was written; BM25 always searches the literal query. See core.Query.Embedding.
	qvec, err := s.llm.Embed(ctx, []string{q.Embedding()})
	if err != nil {
		return nil, fmt.Errorf("index: embed query: %w", err)
	}
	pool := max(topK*5, 20)
	vecHits := s.vec.search(qvec[0], pool, allow)
	bmHits := s.bm.search(q.Text, pool, allow)

	// Over-fetch when reranking: a reranker can only reorder what it is given,
	// so handing it exactly topK would let it improve the order and never the
	// membership — most of the available gain.
	fuseTo := topK
	if s.opts.Rerank != nil {
		fuseTo = max(topK*4, 20)
	}
	// Order matters: index 0 is the cosine list and index 1 the BM25 one, which
	// is what Hit.VectorRank and Hit.TextRank read back out.
	ranked := fuseRRF([][]scored{vecHits, bmHits}, 60, fuseTo)
	out := make([]Hit, 0, len(ranked))
	for _, f := range ranked {
		out = append(out, Hit{
			Chunk:      s.chunks[f.id],
			Score:      f.score,
			VectorRank: f.ranks[0],
			TextRank:   f.ranks[1],
		})
	}
	if s.opts.Rerank != nil {
		return s.opts.Rerank.Rerank(ctx, q.Text, out, topK)
	}
	return out, nil
}

// SetReranker enables reranking on an already-built store, so a loaded index
// can be searched with or without it.
func (s *Store) SetReranker(r Reranker) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.opts.Rerank = r
}

// DocIDs reports the documents currently indexed, sorted. Callers use it to
// work out what a corpus has lost since the last ingest.
func (s *Store) DocIDs() []string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	seen := make(map[string]bool, len(s.chunks))
	for _, c := range s.chunks {
		seen[c.DocID] = true
	}
	return slices.Sorted(maps.Keys(seen))
}

// Len reports the number of indexed chunks.
func (s *Store) Len() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.chunks)
}

// Entries returns the indexed chunks and their vectors, in index order and
// aligned by position. It lets callers reuse embeddings that were already paid
// for — the hierarchical tier clusters leaf vectors rather than re-embedding
// the whole corpus.
func (s *Store) Entries() ([]core.Chunk, [][]float64) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	chunks := make([]core.Chunk, 0, len(s.vec.ids))
	vectors := make([][]float64, 0, len(s.vec.ids))
	for i, id := range s.vec.ids {
		chunks = append(chunks, s.chunks[id])
		vectors = append(vectors, s.vec.vecs[i])
	}
	return chunks, vectors
}

// Add indexes a chunk whose embedding the caller already has. The vector must be
// unit-normalized and from the same embedding model as the rest of the store;
// nothing here can check that, so callers own it.
func (s *Store) Add(c core.Chunk, vector []float64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.chunks[c.ID] = c
	s.vec.add(c.ID, vector)
	s.bm.add(c.ID, c.Embedded())
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
