package index

import (
	"context"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/RootControl/dispatch/core"
	"github.com/RootControl/dispatch/internal/fake"
)

// consistent asserts the three structures agree. The duplicate-vector bug was
// invisible precisely because Len() reads the map while search and Save read the
// vector index, so every test here checks all three rather than the count.
func consistent(t *testing.T, s *Store, wantChunks int) {
	t.Helper()
	s.mu.RLock()
	defer s.mu.RUnlock()
	if len(s.chunks) != wantChunks {
		t.Errorf("chunks map = %d, want %d", len(s.chunks), wantChunks)
	}
	if len(s.vec.ids) != wantChunks {
		t.Errorf("vector index = %d entries, want %d", len(s.vec.ids), wantChunks)
	}
	if len(s.vec.vecs) != len(s.vec.ids) {
		t.Errorf("vector index: %d ids but %d vectors", len(s.vec.ids), len(s.vec.vecs))
	}
	if len(s.bm.docs) != wantChunks {
		t.Errorf("bm25 index = %d docs, want %d", len(s.bm.docs), wantChunks)
	}
	seen := map[string]bool{}
	for _, id := range s.vec.ids {
		if seen[id] {
			t.Errorf("vector index holds %q twice", id)
		}
		seen[id] = true
		if _, ok := s.chunks[id]; !ok {
			t.Errorf("vector index holds %q, which is not in the chunk map", id)
		}
	}
	var total int
	for _, d := range s.bm.docs {
		total += d.len
	}
	if total != s.bm.totalLen {
		t.Errorf("bm25 totalLen = %d, want %d from the surviving docs", s.bm.totalLen, total)
	}
}

func testDocs() []core.Doc {
	return []core.Doc{
		{ID: "a", Text: "Expenses over five hundred dollars need director approval."},
		{ID: "b", Text: "The Atlas migration budget is one million dollars."},
	}
}

// TestReingestDoesNotDuplicate is the regression test for the duplicate-vector
// bug: a second ingest of unchanged documents used to leave one chunk-map entry
// and two vector-index entries per chunk, which inflated that chunk's RRF rank
// credit and persisted through Save.
func TestReingestDoesNotDuplicate(t *testing.T) {
	ctx := context.Background()
	s := New(Config{LLM: &fake.LLM{}})
	docs := testDocs()

	first, err := s.Ingest(ctx, docs)
	if err != nil {
		t.Fatal(err)
	}
	consistent(t, s, first.Chunks)

	for i := range 3 {
		if _, err := s.Ingest(ctx, docs); err != nil {
			t.Fatalf("re-ingest %d: %v", i, err)
		}
		consistent(t, s, first.Chunks)
	}
	if s.Len() != first.Chunks {
		t.Fatalf("Len() = %d after four ingests, want %d", s.Len(), first.Chunks)
	}
}

// TestReingestChangedDocReplacesChunks covers the case a plain "skip if present"
// check would miss: the document changed, and it now splits into fewer chunks
// than before. The surplus tail must go, not linger unreachable.
func TestReingestChangedDocReplacesChunks(t *testing.T) {
	ctx := context.Background()
	s := New(Config{LLM: &fake.LLM{}, Chunk: ChunkOptions{TargetTokens: 6, NoOverlap: true}})

	long := core.Doc{ID: "a", Text: "one two three four five. six seven eight nine ten. eleven twelve thirteen fourteen."}
	if _, err := s.Ingest(ctx, []core.Doc{long}); err != nil {
		t.Fatal(err)
	}
	before := s.Len()
	if before < 2 {
		t.Fatalf("setup: wanted a doc that splits into several chunks, got %d", before)
	}

	short := core.Doc{ID: "a", Text: "one two three."}
	if _, err := s.Ingest(ctx, []core.Doc{short}); err != nil {
		t.Fatal(err)
	}
	if s.Len() >= before {
		t.Fatalf("Len() = %d after shrinking the doc, want fewer than %d", s.Len(), before)
	}
	consistent(t, s, s.Len())

	for id := range s.chunks {
		if _, err := s.Search(ctx, core.Query{Text: "one two three", TopK: 10}); err != nil {
			t.Fatal(err)
		}
		if s.chunks[id].DocID != "a" {
			t.Fatalf("chunk %q survived with DocID %q", id, s.chunks[id].DocID)
		}
	}
}

func TestDeleteAndPrune(t *testing.T) {
	ctx := context.Background()
	s := New(Config{LLM: &fake.LLM{}})
	docs := testDocs()
	st, err := s.Ingest(ctx, docs)
	if err != nil {
		t.Fatal(err)
	}

	if n := s.Delete("nope"); n != 0 {
		t.Fatalf("deleting an unknown doc removed %d chunks", n)
	}
	consistent(t, s, st.Chunks)

	removed := s.Delete("a")
	if removed == 0 {
		t.Fatal("deleting doc a removed nothing")
	}
	consistent(t, s, st.Chunks-removed)

	// Deleted content must stop being retrievable, not merely stop being counted.
	hits, err := s.Search(ctx, core.Query{Text: "director approval expenses", TopK: 10})
	if err != nil {
		t.Fatal(err)
	}
	for _, h := range hits {
		if h.Chunk.DocID == "a" {
			t.Fatalf("search still returns chunk %q from the deleted doc", h.Chunk.ID)
		}
	}

	// Deleting clears the fingerprint, so a re-ingest genuinely rebuilds.
	again, err := s.Ingest(ctx, docs)
	if err != nil {
		t.Fatal(err)
	}
	if again.Unchanged != 1 {
		t.Fatalf("after deleting a, re-ingest reported %d unchanged, want 1 (b only)", again.Unchanged)
	}
	consistent(t, s, st.Chunks)

	if n := s.Prune([]string{"a"}); n == 0 {
		t.Fatal("prune kept doc b, which is no longer in the corpus")
	}
	consistent(t, s, s.Len())
	for _, c := range s.chunks {
		if c.DocID != "a" {
			t.Fatalf("prune left chunk from doc %q", c.DocID)
		}
	}
}

func TestIngestSkipsUnchangedDocs(t *testing.T) {
	ctx := context.Background()
	cache := NewCache(t.TempDir())
	f := &fake.LLM{}
	s := New(Config{LLM: f, Cache: cache, EmbedTag: "test-embed"})
	docs := testDocs()

	first, err := s.Ingest(ctx, docs)
	if err != nil {
		t.Fatal(err)
	}
	if first.Unchanged != 0 || first.EmbedCalls != first.Chunks {
		t.Fatalf("first ingest: Unchanged=%d EmbedCalls=%d chunks=%d", first.Unchanged, first.EmbedCalls, first.Chunks)
	}

	second, err := s.Ingest(ctx, docs)
	if err != nil {
		t.Fatal(err)
	}
	if second.Unchanged != len(docs) {
		t.Fatalf("second ingest: Unchanged=%d, want %d", second.Unchanged, len(docs))
	}
	if second.EmbedCalls != 0 || second.EmbedHits != 0 {
		t.Fatalf("second ingest embedded anything at all: calls=%d hits=%d", second.EmbedCalls, second.EmbedHits)
	}
	if second.Chunks != first.Chunks {
		t.Fatalf("second ingest reported %d chunks, want the %d already indexed", second.Chunks, first.Chunks)
	}

	// Editing one document must re-derive that one and leave the other alone.
	docs[0].Text += " Approvals expire after ninety days."
	third, err := s.Ingest(ctx, docs)
	if err != nil {
		t.Fatal(err)
	}
	if third.Unchanged != 1 {
		t.Fatalf("after editing one doc: Unchanged=%d, want 1", third.Unchanged)
	}
	consistent(t, s, s.Len())
}

// TestFingerprintCoversSettings guards the failure mode where a setting can be
// changed with no effect because the skip check does not know about it.
func TestFingerprintCoversSettings(t *testing.T) {
	doc := core.Doc{ID: "a", Text: "some text", Meta: map[string]string{"path": "x.md"}}
	base := Config{LLM: &fake.LLM{}, EmbedTag: "e1", CacheTag: "c1"}

	fp := func(mut func(*Config), mutDoc func(*core.Doc)) string {
		cfg := base
		mut(&cfg)
		d := doc
		mutDoc(&d)
		return New(cfg).fingerprint(d)
	}
	nop, nopDoc := func(*Config) {}, func(*core.Doc) {}
	want := fp(nop, nopDoc)

	cases := map[string]string{
		"chunk size":    fp(func(c *Config) { c.Chunk.TargetTokens = 42 }, nopDoc),
		"overlap":       fp(func(c *Config) { c.Chunk.OverlapTokens = 42 }, nopDoc),
		"min words":     fp(func(c *Config) { c.Chunk.MinWords = 5 }, nopDoc),
		"headings":      fp(func(c *Config) { c.Chunk.Headings = true }, nopDoc),
		"contextualize": fp(func(c *Config) { c.Contextualize = true }, nopDoc),
		"context model": fp(func(c *Config) { c.CacheTag = "c2" }, nopDoc),
		"embed model":   fp(func(c *Config) { c.EmbedTag = "e2" }, nopDoc),
		"doc text":      fp(nop, func(d *core.Doc) { d.Text += "!" }),
		"doc meta":      fp(nop, func(d *core.Doc) { d.Meta = map[string]string{"path": "y.md"} }),
		"doc id":        fp(nop, func(d *core.Doc) { d.ID = "b" }),
	}
	for name, got := range cases {
		if got == want {
			t.Errorf("changing %s did not change the fingerprint, so an ingest would skip the doc", name)
		}
	}

	// Metadata is hashed in sorted key order, so an identical map fingerprints
	// identically however Go happens to iterate it.
	multi := core.Doc{ID: "a", Text: "t", Meta: map[string]string{"a": "1", "b": "2", "c": "3"}}
	s := New(base)
	stable := s.fingerprint(multi)
	for range 50 {
		if got := s.fingerprint(multi); got != stable {
			t.Fatalf("fingerprint varies with map iteration order: %s then %s", stable, got)
		}
	}
}

func TestSaveLoadRoundTripsDocHashes(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "index.json")
	docs := testDocs()

	s1 := New(Config{LLM: &fake.LLM{}, EmbedTag: "e1"})
	st, err := s1.Ingest(ctx, docs)
	if err != nil {
		t.Fatal(err)
	}
	if err := s1.Save(path); err != nil {
		t.Fatal(err)
	}

	s2 := New(Config{LLM: &fake.LLM{}, EmbedTag: "e1"})
	if err := s2.Load(path); err != nil {
		t.Fatal(err)
	}
	consistent(t, s2, st.Chunks)

	// The whole point of persisting fingerprints: ingest in a new process skips
	// what the previous one already did.
	after, err := s2.Ingest(ctx, docs)
	if err != nil {
		t.Fatal(err)
	}
	if after.Unchanged != len(docs) {
		t.Fatalf("ingest over a loaded index: Unchanged=%d, want %d", after.Unchanged, len(docs))
	}
}

// TestLoadRepairsDuplicatedSnapshot covers indexes written by the version that
// duplicated vectors: loading one must produce a clean store rather than
// faithfully reconstructing the corruption.
func TestLoadRepairsDuplicatedSnapshot(t *testing.T) {
	path := filepath.Join(t.TempDir(), "index.json")
	dup := `{"embed_model":"e1","chunks":[
	  {"ID":"a#0","DocID":"a","Text":"hello world","Position":0},
	  {"ID":"a#0","DocID":"a","Text":"hello world","Position":0}],
	 "vectors":[[1,0],[1,0]]}`
	if err := os.WriteFile(path, []byte(dup), 0o644); err != nil {
		t.Fatal(err)
	}

	s := New(Config{LLM: &fake.LLM{}, EmbedTag: "e1"})
	if err := s.Load(path); err != nil {
		t.Fatal(err)
	}
	consistent(t, s, 1)
}

func TestVectorIndexRemovePreservesOrder(t *testing.T) {
	vi := &vectorIndex{}
	for _, id := range []string{"a", "b", "c", "d", "e"} {
		vi.add(id, []float64{1})
	}
	if got := vi.remove(map[string]bool{"b": true, "d": true}); got != 2 {
		t.Fatalf("remove reported %d, want 2", got)
	}
	if want := []string{"a", "c", "e"}; !slices.Equal(vi.ids, want) {
		t.Fatalf("ids = %v, want %v (order must survive removal)", vi.ids, want)
	}
	if len(vi.vecs) != len(vi.ids) {
		t.Fatalf("%d ids but %d vectors after removal", len(vi.ids), len(vi.vecs))
	}
}

// TestBM25RemoveRestoresCorpusStats checks the part removal is easy to get
// wrong: idf and length normalisation are computed from corpus-wide totals, so
// a removal that forgets them scores survivors against a corpus that is gone.
func TestBM25RemoveRestoresCorpusStats(t *testing.T) {
	build := func(docs map[string]string) *bm25Index {
		bm := newBM25()
		for _, id := range slices.Sorted(maps.Keys(docs)) {
			bm.add(id, docs[id])
		}
		return bm
	}
	keep := map[string]string{"a": "alpha beta gamma", "c": "gamma delta"}
	all := map[string]string{"a": "alpha beta gamma", "b": "beta beta epsilon", "c": "gamma delta"}

	want := build(keep)
	got := build(all)
	got.remove(map[string]bool{"b": true})

	if got.totalLen != want.totalLen {
		t.Errorf("totalLen = %d, want %d", got.totalLen, want.totalLen)
	}
	if !maps.Equal(got.df, want.df) {
		t.Errorf("df = %v, want %v", got.df, want.df)
	}
	// "epsilon" appeared only in the removed doc; leaving a zero entry behind
	// would skew idf for every other term through N/df arithmetic.
	if _, ok := got.df["epsilon"]; ok {
		t.Error("df still holds a term only the removed document contained")
	}
	if a, b := got.search("gamma", 10, nil), want.search("gamma", 10, nil); !slices.Equal(a, b) {
		t.Errorf("scores after removal = %v, want %v (identical to never having indexed b)", a, b)
	}
}
