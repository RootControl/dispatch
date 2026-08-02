// Package integration runs index/pgvector against a real PostgreSQL server.
//
// It is a separate module so the pgx driver never enters dispatch's go.mod:
// the parent module's ./... skips any directory holding its own go.mod, so
// `go vet ./... && go test ./...` at the repo root stays hermetic and
// dependency-free. This is the counterpart to the unit tests, which use a
// stdlib database/sql fake driver and can verify query construction, binding
// and scan order but not that Postgres accepts any of it.
//
//	docker run -d --name pgtest -e POSTGRES_PASSWORD=dispatch \
//	    -e POSTGRES_DB=dispatch -p 55432:5432 pgvector/pgvector:pg17
//
//	cd index/pgvector/integration
//	DISPATCH_PG_DSN='postgres://postgres:dispatch@localhost:55432/dispatch?sslmode=disable' \
//	    go test -v ./...
package integration

import (
	"context"
	"database/sql"
	"hash/fnv"
	"math"
	"os"
	"strings"
	"testing"
	"unicode"

	"github.com/RootControl/dispatch/core"
	"github.com/RootControl/dispatch/index"
	"github.com/RootControl/dispatch/index/pgvector"
	"github.com/RootControl/dispatch/llm"
	"github.com/RootControl/dispatch/tiers"

	_ "github.com/jackc/pgx/v5/stdlib"
)

const dims = 64

// embedder is a deterministic bag-of-token-hashes embedder, the same idea as
// internal/fake. It is reimplemented here rather than imported because
// internal/ is not reachable from another module — and because the point of
// this test is the storage layer, not the embeddings.
type embedder struct{}

func (embedder) Embed(_ context.Context, inputs []string) ([][]float64, error) {
	out := make([][]float64, len(inputs))
	for i, in := range inputs {
		v := make([]float64, dims)
		for _, tok := range strings.FieldsFunc(strings.ToLower(in), func(r rune) bool {
			return !unicode.IsLetter(r) && !unicode.IsNumber(r)
		}) {
			h := fnv.New32a()
			h.Write([]byte(tok))
			v[h.Sum32()%dims] += 1
		}
		var norm float64
		for _, x := range v {
			norm += x * x
		}
		if norm = math.Sqrt(norm); norm > 0 {
			for j := range v {
				v[j] /= norm
			}
		}
		out[i] = v
	}
	return out, nil
}

// adapter satisfies llm.LLM. pgvector only ever calls Embed — it embeds the
// query at search time and does nothing else with the model — so the chat
// methods panic rather than returning a plausible zero value that would let a
// wiring mistake pass unnoticed.
type adapter struct{ embedder }

var _ llm.LLM = adapter{}

func (adapter) Chat(context.Context, []llm.Message) (string, error) {
	panic("pgvector called Chat; it should only embed")
}

func (adapter) ChatJSON(context.Context, []llm.Message, any) error {
	panic("pgvector called ChatJSON; it should only embed")
}

// store opens a connection, migrates a uniquely-named table, and drops it after.
func store(t *testing.T) *pgvector.Store {
	t.Helper()
	dsn := os.Getenv("DISPATCH_PG_DSN")
	if dsn == "" {
		t.Skip("set DISPATCH_PG_DSN to run the pgvector integration tests")
	}
	db, err := sql.Open("pgx", dsn)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	if err := db.Ping(); err != nil {
		t.Fatalf("cannot reach %s: %v", dsn, err)
	}

	// One table per test, so tests do not see each other's rows and a failure
	// leaves nothing behind for the next run to trip over.
	table := "t_" + strings.Map(func(r rune) rune {
		if unicode.IsLetter(r) || unicode.IsNumber(r) {
			return unicode.ToLower(r)
		}
		return '_'
	}, t.Name())

	s, err := pgvector.New(pgvector.Config{DB: db, LLM: adapter{}, Dims: dims, Table: table})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Migrate(context.Background()); err != nil {
		t.Fatalf("migrate: %v", err)
	}
	t.Cleanup(func() { db.Exec("DROP TABLE IF EXISTS " + table) })
	return s
}

func chunk(id, doc, text, context_ string, meta map[string]string) core.Chunk {
	return core.Chunk{ID: id, DocID: doc, Text: text, Context: context_, Meta: meta}
}

func seed(t *testing.T, s *pgvector.Store) []core.Chunk {
	t.Helper()
	chunks := []core.Chunk{
		chunk("docs/budget.md#0", "docs/budget.md",
			"The approved Atlas migration budget is four million dollars.",
			"This excerpt from the Atlas charter states the budget.",
			map[string]string{"dir": "docs", "team": "finance"}),
		chunk("docs/vendor.md#0", "docs/vendor.md",
			"Mailwright supplies the customer statement mailer under contract.",
			"This excerpt covers vendor arrangements.",
			map[string]string{"dir": "docs", "team": "ops"}),
		chunk("notes/staffing.md#0", "notes/staffing.md",
			"Priya Raman leads engineering and reports to the programme director.",
			"", // deliberately no context sentence
			map[string]string{"dir": "notes", "team": "people"}),
		chunk("notes/risk.md#0", "notes/risk.md",
			"The contingency reserve stands at twelve percent of the budget.",
			"This excerpt lists project risks.",
			nil), // deliberately no metadata
	}
	texts := make([]string, len(chunks))
	for i, c := range chunks {
		texts[i] = c.Embedded()
	}
	vecs, err := embedder{}.Embed(context.Background(), texts)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Upsert(context.Background(), chunks, vecs); err != nil {
		t.Fatalf("upsert: %v", err)
	}
	return chunks
}

func hitIDs(hits []index.Hit) []string {
	out := make([]string, len(hits))
	for i, h := range hits {
		out[i] = h.Chunk.ID
	}
	return out
}

// The first thing that has to work: Postgres has to accept the DDL. The unit
// tests can check the SQL is idempotent and parameterised; only a server can
// say whether `vector(64)`, the HNSW opclass and the GIN expression index are
// real.
func TestMigrateIsAcceptedAndRepeatable(t *testing.T) {
	s := store(t)
	// Twice: CREATE ... IF NOT EXISTS has to mean it against a live catalog.
	if err := s.Migrate(context.Background()); err != nil {
		t.Fatalf("second migrate: %v", err)
	}
	if n := s.Len(); n != 0 {
		t.Errorf("Len() = %d on a fresh table", n)
	}
}

func TestUpsertAndSearch(t *testing.T) {
	s := store(t)
	chunks := seed(t, s)

	if n := s.Len(); n != len(chunks) {
		t.Fatalf("Len() = %d, want %d", n, len(chunks))
	}

	hits, err := s.Search(context.Background(), core.Query{Text: "atlas migration budget", TopK: 3})
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(hits) == 0 {
		t.Fatal("search returned nothing")
	}
	if hits[0].Chunk.ID != "docs/budget.md#0" {
		t.Errorf("top hit = %s, want docs/budget.md#0 (hits: %v)", hits[0].Chunk.ID, hitIDs(hits))
	}
	// Every column has to survive the round trip, not just the ID.
	top := hits[0].Chunk
	if top.DocID != "docs/budget.md" || !strings.Contains(top.Text, "four million") {
		t.Errorf("round-tripped chunk = %+v", top)
	}
	if top.Context == "" {
		t.Error("the situating context sentence was lost")
	}
	if top.Meta["team"] != "finance" {
		t.Errorf("meta = %v, want team=finance", top.Meta)
	}
	// RRF scores descend.
	for i := 1; i < len(hits); i++ {
		if hits[i].Score > hits[i-1].Score {
			t.Errorf("scores not descending: %v", hits)
		}
	}
}

// seedAgainstTheVectorHalf stores the fillers with a vector identical to the
// query's own, and the target with one orthogonal to it. The dense half
// therefore ranks every filler at distance 0 and the target dead last — so if
// the target still comes back first, only the lexical half can have put it
// there.
//
// Isolating this took three attempts, each of which passed while the full-text
// column was deliberately broken:
//
//  1. Seeding normally and searching for a term in the text. The vector
//     contained that term too, so either half could have produced the hit.
//  2. Giving the target a body-only vector. It was the only row in the table,
//     so every query returned it.
//  3. Giving every row one identical vector, on the theory that a tie carries
//     no ranking signal. Postgres returns tied rows in reverse insertion
//     order, which handed the last-inserted target the top slot for free.
//
// Hence this: the dense half is not neutralised but actively pointed the wrong
// way, which is the only version that cannot pass by accident.
func seedAgainstTheVectorHalf(t *testing.T, s *pgvector.Store, query string, target core.Chunk, fillers ...core.Chunk) {
	t.Helper()
	ctx := context.Background()

	qvec, err := embedder{}.Embed(ctx, []string{query})
	if err != nil {
		t.Fatal(err)
	}
	// A unit vector on some dimension the query does not occupy: cosine
	// distance 1 from the query, against the fillers' 0.
	orthogonal := make([]float64, dims)
	for i, x := range qvec[0] {
		if x == 0 {
			orthogonal[i] = 1
			break
		}
	}
	var sum float64
	for _, x := range orthogonal {
		sum += x
	}
	if sum == 0 {
		t.Fatal("setup: the query occupies every dimension, so no orthogonal vector exists")
	}

	chunks := append([]core.Chunk{target}, fillers...)
	vecs := make([][]float64, len(chunks))
	vecs[0] = orthogonal
	for i := 1; i < len(vecs); i++ {
		vecs[i] = qvec[0]
	}
	if err := s.Upsert(ctx, chunks, vecs); err != nil {
		t.Fatalf("upsert: %v", err)
	}
}

// The lexical half is the part the fake driver could not exercise: whether
// to_tsvector/plainto_tsquery actually match against a live server, and whether
// a hit can come from that half alone.
func TestFullTextHalfContributes(t *testing.T) {
	s := store(t)
	const query = "Mailwright reconciliation"
	seedAgainstTheVectorHalf(t, s, query,
		chunk("target#0", "target", "Mailwright supplies the quarterly reconciliation feed.", "", nil),
		chunk("filler-a#0", "filler-a", "Unrelated notes about scheduling and staffing.", "", nil),
		chunk("filler-b#0", "filler-b", "More unrelated notes about locale strings.", "", nil),
	)

	hits, err := s.Search(context.Background(), core.Query{Text: query, TopK: 3})
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) == 0 || hits[0].Chunk.ID != "target#0" {
		t.Errorf("hits = %v, want target#0 first — the dense half ranks it last, "+
			"so only the lexical half can lift it", hitIDs(hits))
	}
}

// A chunk stores its body and its context sentence separately but must index
// them together, so a term appearing only in the context is still searchable.
// Getting this wrong halves the benefit of contextual chunking on this backend.
//
// Isolated the same way, and for the same reason — see
// seedAgainstTheVectorHalf for the three weaker versions this replaced.
func TestFullTextSearchesTheContextSentence(t *testing.T) {
	s := store(t)
	const query = "Zephyrine addendum"
	seedAgainstTheVectorHalf(t, s, query,
		// "Zephyrine" appears only in the context sentence, never in the body.
		chunk("target#0", "target",
			"The figure was revised upward at the quarterly review.",
			"This excerpt from the Zephyrine addendum states the revised figure.", nil),
		chunk("filler-a#0", "filler-a", "Unrelated notes about scheduling.", "", nil),
		chunk("filler-b#0", "filler-b", "More unrelated notes about locales.", "", nil),
	)

	hits, err := s.Search(context.Background(), core.Query{Text: query, TopK: 3})
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) == 0 || hits[0].Chunk.ID != "target#0" {
		t.Errorf("hits = %v, want target#0 first — full-text is indexing the body "+
			"without its context sentence", hitIDs(hits))
	}
}

func TestFilters(t *testing.T) {
	s := store(t)
	seed(t, s)
	ctx := context.Background()

	cases := []struct {
		name   string
		filter core.Filter
		want   []string
	}{
		{"exact", core.Filter{"dir": "notes"}, []string{"notes/staffing.md#0"}},
		{"prefix", core.Filter{"dir": "doc*"}, []string{"docs/budget.md#0", "docs/vendor.md#0"}},
		{"two clauses", core.Filter{"dir": "docs", "team": "ops"}, []string{"docs/vendor.md#0"}},
		{"matches nothing", core.Filter{"dir": "nowhere"}, nil},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			hits, err := s.Search(ctx, core.Query{Text: "budget vendor staffing", TopK: 10, Filter: tc.filter})
			if err != nil {
				t.Fatal(err)
			}
			got := hitIDs(hits)
			if len(got) != len(tc.want) {
				t.Fatalf("hits = %v, want %v", got, tc.want)
			}
			allowed := map[string]bool{}
			for _, id := range tc.want {
				allowed[id] = true
			}
			for _, id := range got {
				if !allowed[id] {
					t.Errorf("filter %v returned %s", tc.filter, id)
				}
			}
		})
	}

	// A chunk with no metadata at all must be excluded by any filter rather
	// than admitted — the permissive reading would leak unlabelled content past
	// a tenant filter. notes/risk.md#0 has nil Meta, stored as '{}'::jsonb.
	hits, err := s.Search(ctx, core.Query{Text: "contingency reserve", TopK: 10, Filter: core.Filter{"team": "finance"}})
	if err != nil {
		t.Fatal(err)
	}
	for _, h := range hits {
		if h.Chunk.ID == "notes/risk.md#0" {
			t.Error("a chunk with no metadata passed a metadata filter")
		}
	}
}

// LIKE metacharacters in a filter value must not widen the match. A stored '%'
// would otherwise let "100%*" match everything.
func TestFilterEscapesLikeWildcards(t *testing.T) {
	s := store(t)
	ctx := context.Background()

	chunks := []core.Chunk{
		chunk("a#0", "a", "first document about budgets", "", map[string]string{"tag": "100%_secret"}),
		chunk("b#0", "b", "second document about budgets", "", map[string]string{"tag": "100XYsecret"}),
	}
	vecs, _ := embedder{}.Embed(ctx, []string{chunks[0].Embedded(), chunks[1].Embedded()})
	if err := s.Upsert(ctx, chunks, vecs); err != nil {
		t.Fatal(err)
	}

	// Unescaped, "100%_secret*" becomes LIKE '100%_secret%' where % and _ are
	// wildcards, matching b#0 as well.
	hits, err := s.Search(ctx, core.Query{Text: "document budgets", TopK: 10, Filter: core.Filter{"tag": "100%_secret*"}})
	if err != nil {
		t.Fatal(err)
	}
	if got := hitIDs(hits); len(got) != 1 || got[0] != "a#0" {
		t.Errorf("hits = %v, want only a#0 — the literal's wildcards widened the filter", got)
	}
}

// Upsert means upsert: re-storing a document replaces it rather than failing on
// the primary key or duplicating the row. This is the pgvector counterpart of
// the duplicate-vector bug the in-memory store had.
func TestUpsertReplacesRatherThanDuplicating(t *testing.T) {
	s := store(t)
	ctx := context.Background()
	seed(t, s)
	before := s.Len()

	edited := chunk("docs/budget.md#0", "docs/budget.md",
		"The approved Atlas migration budget was raised to six million dollars.",
		"This excerpt from the Atlas charter states the revised budget.",
		map[string]string{"dir": "docs", "team": "finance"})
	vecs, _ := embedder{}.Embed(ctx, []string{edited.Embedded()})
	if err := s.Upsert(ctx, []core.Chunk{edited}, vecs); err != nil {
		t.Fatalf("re-upsert: %v", err)
	}

	if after := s.Len(); after != before {
		t.Fatalf("Len() = %d after re-upserting one chunk, want %d", after, before)
	}
	hits, err := s.Search(ctx, core.Query{Text: "atlas migration budget", TopK: 1})
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 1 || !strings.Contains(hits[0].Chunk.Text, "six million") {
		t.Errorf("top hit = %q, want the edited text", hits[0].Chunk.Text)
	}
}

func TestDelete(t *testing.T) {
	s := store(t)
	ctx := context.Background()
	chunks := seed(t, s)

	n, err := s.Delete(ctx, "nope")
	if err != nil {
		t.Fatal(err)
	}
	if n != 0 {
		t.Errorf("deleting an unknown doc removed %d rows", n)
	}

	if n, err = s.Delete(ctx, "docs/budget.md", "docs/vendor.md"); err != nil {
		t.Fatal(err)
	}
	if n != 2 {
		t.Errorf("deleted %d rows, want 2", n)
	}
	if got := s.Len(); got != len(chunks)-2 {
		t.Errorf("Len() = %d, want %d", got, len(chunks)-2)
	}

	// Deleted content must stop being retrievable, not merely stop being counted.
	hits, err := s.Search(ctx, core.Query{Text: "atlas migration budget", TopK: 10})
	if err != nil {
		t.Fatal(err)
	}
	for _, h := range hits {
		if strings.HasPrefix(h.Chunk.ID, "docs/") {
			t.Errorf("search still returns %s from a deleted document", h.Chunk.ID)
		}
	}
}

func TestSearchEmptyTable(t *testing.T) {
	s := store(t)
	hits, err := s.Search(context.Background(), core.Query{Text: "anything", TopK: 5})
	if err != nil {
		t.Fatalf("searching an empty table: %v", err)
	}
	if len(hits) != 0 {
		t.Errorf("hits = %v, want none", hitIDs(hits))
	}
}

// A vector of the wrong width must be refused before it reaches the server,
// where it would be a confusing type error rather than a clear one.
func TestWrongDimensionsRejected(t *testing.T) {
	s := store(t)
	err := s.Upsert(context.Background(),
		[]core.Chunk{chunk("x#0", "x", "text", "", nil)},
		[][]float64{{1, 2, 3}})
	if err == nil {
		t.Fatal("a 3-dimensional vector was accepted into a 64-dimensional column")
	}
	if !strings.Contains(err.Error(), "dimension") {
		t.Errorf("error = %v, want it to name the dimension mismatch", err)
	}
}

// The whole point of the package: it satisfies index.Searcher, so the semantic
// tier runs on it unchanged.
func TestSemanticTierOverPostgres(t *testing.T) {
	s := store(t)
	seed(t, s)

	semantic := tiers.NewSemantic(s)
	results, err := semantic.Retrieve(context.Background(), core.Query{Text: "atlas migration budget", TopK: 2})
	if err != nil {
		t.Fatal(err)
	}
	if len(results) == 0 {
		t.Fatal("the semantic tier retrieved nothing")
	}
	if results[0].SourceID != "docs/budget.md#0" {
		t.Errorf("top result = %s", results[0].SourceID)
	}
	if results[0].Tier != core.TierSemantic {
		t.Errorf("tier = %s", results[0].Tier)
	}
	if results[0].Meta["doc"] != "docs/budget.md" {
		t.Errorf("meta = %v, want the doc for per-document diversity capping", results[0].Meta)
	}
	// Result.Text carries context+body, which is what the generator reads.
	if !strings.Contains(results[0].Text, "charter") {
		t.Errorf("result text lost the context sentence: %q", results[0].Text)
	}
}

// The tier must also honour filters through the Postgres path, or a filtered
// query would silently widen when the backend is swapped.
func TestSemanticTierFiltersOverPostgres(t *testing.T) {
	s := store(t)
	seed(t, s)

	semantic := tiers.NewSemantic(s)
	results, err := semantic.Retrieve(context.Background(), core.Query{
		Text: "budget vendor staffing", TopK: 10, Filter: core.Filter{"dir": "notes"}})
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 || results[0].SourceID != "notes/staffing.md#0" {
		ids := make([]string, len(results))
		for i, r := range results {
			ids[i] = r.SourceID
		}
		t.Errorf("results = %v, want only the notes chunk", ids)
	}
}
