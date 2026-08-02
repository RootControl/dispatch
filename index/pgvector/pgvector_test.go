package pgvector

import (
	"context"
	"database/sql/driver"
	"strings"
	"testing"

	"github.com/RootControl/dispatch/core"
	"github.com/RootControl/dispatch/index"
	"github.com/RootControl/dispatch/internal/fake"
)

func newTestStore(t *testing.T, db *fakeDB) *Store {
	t.Helper()
	s, err := New(Config{DB: openFake(db), LLM: &fake.LLM{Dims: 4}, Dims: 4})
	if err != nil {
		t.Fatal(err)
	}
	return s
}

// searchDB canned-answers both halves of the hybrid and the row load.
func searchDB() *fakeDB {
	return &fakeDB{
		rows: map[string][][]driver.Value{
			"embedding <=>": {{"a#0"}, {"b#0"}},
			"ts_rank_cd":    {{"b#0"}, {"c#0"}},
			"SELECT id, doc_id": {
				{"a#0", "a", "text a", "ctx a", int64(0), []byte(`{"dir":"docs"}`)},
				{"b#0", "b", "text b", "", int64(0), []byte(`{}`)},
				{"c#0", "c", "text c", "", int64(0), nil},
			},
		},
		cols: map[string][]string{
			"embedding <=>":     {"id"},
			"ts_rank_cd":        {"id"},
			"SELECT id, doc_id": {"id", "doc_id", "text", "context", "position", "meta"},
		},
	}
}

func TestNewValidatesConfig(t *testing.T) {
	good := Config{DB: openFake(&fakeDB{}), LLM: &fake.LLM{}, Dims: 4}
	if _, err := New(good); err != nil {
		t.Fatalf("valid config rejected: %v", err)
	}
	for name, mut := range map[string]func(*Config){
		"no DB":       func(c *Config) { c.DB = nil },
		"no LLM":      func(c *Config) { c.LLM = nil },
		"no dims":     func(c *Config) { c.Dims = 0 },
		"bad table":   func(c *Config) { c.Table = "chunks; DROP TABLE users" },
		"quoted":      func(c *Config) { c.Table = `"chunks"` },
		"leading dig": func(c *Config) { c.Table = "1chunks" },
		"bad ts cfg":  func(c *Config) { c.TextSearchConfig = "english'; --" },
	} {
		cfg := good
		mut(&cfg)
		if _, err := New(cfg); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
}

func TestSearchRunsBothHalvesAndFusesThem(t *testing.T) {
	db := searchDB()
	s := newTestStore(t, db)

	hits, err := s.Search(context.Background(), core.Query{Text: "atlas budget", TopK: 3})
	if err != nil {
		t.Fatal(err)
	}

	// b#0 is the only id in both lists, so RRF must rank it first.
	if len(hits) == 0 || hits[0].Chunk.ID != "b#0" {
		t.Fatalf("hits = %v, want b#0 first (it appears in both rankings)", hitIDs(hits))
	}
	if len(hits) != 3 {
		t.Fatalf("got %d hits, want the 3 fused ids", len(hits))
	}

	vec, ok := db.find("embedding <=>")
	if !ok {
		t.Fatal("no vector-search query was run")
	}
	if got, want := vec.Args[0].(string), "["; !strings.HasPrefix(got, want) {
		t.Errorf("vector query bound %q, want a pgvector literal", got)
	}
	fts, ok := db.find("plainto_tsquery")
	if !ok {
		t.Fatal("no full-text query was run")
	}
	if fts.Args[0] != "atlas budget" {
		t.Errorf("text query bound %v, want the raw query text", fts.Args[0])
	}
}

// Rows come back from Postgres in whatever order it likes; the fused ranking
// has to survive that, or the ranking is silently discarded.
func TestSearchPreservesFusedOrderNotScanOrder(t *testing.T) {
	db := searchDB()
	// Return the rows in an order that contradicts the fusion.
	db.rows["SELECT id, doc_id"] = [][]driver.Value{
		{"c#0", "c", "text c", "", int64(0), nil},
		{"a#0", "a", "text a", "ctx a", int64(0), []byte(`{}`)},
		{"b#0", "b", "text b", "", int64(0), []byte(`{}`)},
	}
	s := newTestStore(t, db)

	hits, err := s.Search(context.Background(), core.Query{Text: "q", TopK: 3})
	if err != nil {
		t.Fatal(err)
	}
	if hits[0].Chunk.ID != "b#0" {
		t.Fatalf("hits = %v, want b#0 first; scan order won over the ranking", hitIDs(hits))
	}
	// Scores must descend — they are the RRF scores, not row order.
	for i := 1; i < len(hits); i++ {
		if hits[i].Score > hits[i-1].Score {
			t.Fatalf("scores not descending: %v", hits)
		}
	}
}

// A row deleted between the ranking query and the load must drop out rather
// than leave a zero-valued chunk in the evidence.
func TestSearchSkipsRowsThatVanished(t *testing.T) {
	db := searchDB()
	db.rows["SELECT id, doc_id"] = [][]driver.Value{
		{"b#0", "b", "text b", "", int64(0), []byte(`{}`)},
	}
	s := newTestStore(t, db)

	hits, err := s.Search(context.Background(), core.Query{Text: "q", TopK: 3})
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 1 || hits[0].Chunk.ID != "b#0" {
		t.Fatalf("hits = %v, want only the row that still exists", hitIDs(hits))
	}
}

func TestFilterClause(t *testing.T) {
	s := newTestStore(t, &fakeDB{})

	cases := []struct {
		name     string
		filter   core.Filter
		wantSQL  []string
		wantArgs []any
	}{
		{"empty", nil, nil, nil},
		{"exact", core.Filter{"dir": "docs"},
			[]string{`meta ->> $2 = $3`}, []any{"dir", "docs"}},
		{"prefix", core.Filter{"dir": "docs*"},
			[]string{`meta ->> $2 LIKE $3`}, []any{"dir", "docs%"}},
		{"two clauses sorted", core.Filter{"team": "finance", "dir": "docs"},
			[]string{`meta ->> $2 = $3`, `meta ->> $4 = $5`},
			[]any{"dir", "docs", "team", "finance"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			where, args := s.filterClause(tc.filter)
			for _, want := range tc.wantSQL {
				if !strings.Contains(where, want) {
					t.Errorf("where = %q, want it to contain %q", where, want)
				}
			}
			if len(args) != len(tc.wantArgs) {
				t.Fatalf("args = %v, want %v", args, tc.wantArgs)
			}
			for i := range args {
				if args[i] != tc.wantArgs[i] {
					t.Errorf("arg %d = %v, want %v", i, args[i], tc.wantArgs[i])
				}
			}
			// Nothing from the filter may reach the SQL text itself.
			for k, v := range tc.filter {
				if strings.Contains(where, k) || strings.Contains(where, strings.TrimSuffix(v, "*")) {
					t.Errorf("filter content was interpolated into SQL: %q", where)
				}
			}
		})
	}
}

// A filter value containing LIKE metacharacters must not widen the prefix
// match — "%" would otherwise match everything under a filter meant to narrow.
func TestFilterEscapesLikeWildcards(t *testing.T) {
	s := newTestStore(t, &fakeDB{})
	_, args := s.filterClause(core.Filter{"dir": "100%_secret*"})
	if got := args[1].(string); got != `100\%\_secret%` {
		t.Fatalf("prefix arg = %q, want the literal's own wildcards escaped", got)
	}
}

func TestSearchAppliesFilterToBothHalves(t *testing.T) {
	db := searchDB()
	s := newTestStore(t, db)

	_, err := s.Search(context.Background(), core.Query{
		Text: "q", TopK: 3, Filter: core.Filter{"dir": "docs"}})
	if err != nil {
		t.Fatal(err)
	}
	for _, marker := range []string{"embedding <=>", "plainto_tsquery"} {
		q, ok := db.find(marker)
		if !ok {
			t.Fatalf("no query containing %q", marker)
		}
		if !strings.Contains(q.SQL, "meta ->>") {
			t.Errorf("%s query is unfiltered: %s", marker, q.SQL)
		}
		if len(q.Args) != 3 {
			t.Errorf("%s query bound %d args, want query + key + value", marker, len(q.Args))
		}
	}
}

func TestUpsertRejectsMismatchedInput(t *testing.T) {
	s := newTestStore(t, &fakeDB{})
	ctx := context.Background()

	if err := s.Upsert(ctx, []core.Chunk{{ID: "a"}}, nil); err == nil {
		t.Error("expected an error when chunks and vectors disagree in length")
	}
	if err := s.Upsert(ctx, []core.Chunk{{ID: "a"}}, [][]float64{{1, 2}}); err == nil {
		t.Error("expected an error for a vector of the wrong width")
	}
	if err := s.Upsert(ctx, nil, nil); err != nil {
		t.Errorf("empty upsert should be a no-op, got %v", err)
	}
}

func TestUpsertBindsChunkFields(t *testing.T) {
	db := &fakeDB{}
	s := newTestStore(t, db)

	chunk := core.Chunk{ID: "a#0", DocID: "a", Text: "body", Context: "situating sentence",
		Position: 2, Meta: map[string]string{"dir": "docs"}}
	if err := s.Upsert(context.Background(), []core.Chunk{chunk}, [][]float64{{0.5, 0.5, 0.5, 0.5}}); err != nil {
		t.Fatal(err)
	}
	q, ok := db.find("INSERT INTO")
	if !ok {
		t.Fatal("no insert was issued")
	}
	if !strings.Contains(q.SQL, "ON CONFLICT (id) DO UPDATE") {
		t.Error("insert is not an upsert; re-ingesting a doc would fail on the primary key")
	}
	want := []any{"a#0", "a", "body", "situating sentence", int64(2), `{"dir":"docs"}`, "[0.5,0.5,0.5,0.5]"}
	for i, w := range want {
		if q.Args[i] != w {
			t.Errorf("arg %d = %v (%T), want %v", i, q.Args[i], q.Args[i], w)
		}
	}
	// The last column is the embedded text: context + body, which is what the
	// full-text half searches. Indexing the body alone would drop the
	// situating sentence from lexical search and silently halve the benefit of
	// contextual chunking on this backend.
	if got := q.Args[7].(string); !strings.Contains(got, "situating sentence") || !strings.Contains(got, "body") {
		t.Errorf("embedded column = %q, want context and body", got)
	}
}

func TestDelete(t *testing.T) {
	db := &fakeDB{}
	s := newTestStore(t, db)

	if n, err := s.Delete(context.Background()); n != 0 || err != nil {
		t.Errorf("deleting nothing: %d, %v", n, err)
	}
	if _, err := s.Delete(context.Background(), "a", "b"); err != nil {
		t.Fatal(err)
	}
	q, ok := db.find("DELETE FROM")
	if !ok {
		t.Fatal("no delete was issued")
	}
	if !strings.Contains(q.SQL, "$1,$2") {
		t.Errorf("delete does not bind each id: %s", q.SQL)
	}
	if len(q.Args) != 2 || q.Args[0] != "a" || q.Args[1] != "b" {
		t.Errorf("delete args = %v", q.Args)
	}
}

func TestMigrateIsIdempotentSQL(t *testing.T) {
	db := &fakeDB{}
	s := newTestStore(t, db)
	if err := s.Migrate(context.Background()); err != nil {
		t.Fatal(err)
	}
	seen := db.seen()
	if len(seen) < 5 {
		t.Fatalf("migrate issued %d statements, want the table and its indexes", len(seen))
	}
	for _, q := range seen {
		if !strings.Contains(q.SQL, "IF NOT EXISTS") {
			t.Errorf("statement is not idempotent: %s", q.SQL)
		}
	}
}

// The store must satisfy the interface the semantic tier depends on, which is
// the whole point of the package.
func TestImplementsSearcher(t *testing.T) {
	var _ index.Searcher = (*Store)(nil)
}

func hitIDs(hits []index.Hit) []string {
	out := make([]string, len(hits))
	for i, h := range hits {
		out[i] = h.Chunk.ID
	}
	return out
}
