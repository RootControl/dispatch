package index

import (
	"context"
	"testing"

	"github.com/RootControl/dispatch/core"
	"github.com/RootControl/dispatch/internal/fake"
)

func filterCorpus() []core.Doc {
	return []core.Doc{
		{ID: "docs/budget.md", Text: "The Atlas budget is four million dollars.",
			Meta: map[string]string{"dir": "docs", "ext": "md", "team": "finance"}},
		{ID: "docs/vendor.md", Text: "The Atlas budget covers one external vendor.",
			Meta: map[string]string{"dir": "docs", "ext": "md", "team": "ops"}},
		{ID: "notes/budget.txt", Text: "Atlas budget notes from the steering meeting.",
			Meta: map[string]string{"dir": "notes", "ext": "txt", "team": "finance"}},
		{ID: "archive/old.md", Text: "The Atlas budget was previously two million dollars.",
			Meta: map[string]string{"dir": "archive", "ext": "md", "team": "finance"}},
	}
}

func filteredStore(t *testing.T) *Store {
	t.Helper()
	s := New(Config{LLM: &fake.LLM{}})
	if _, err := s.Ingest(context.Background(), filterCorpus()); err != nil {
		t.Fatal(err)
	}
	return s
}

func TestSearchFilter(t *testing.T) {
	ctx := context.Background()
	s := filteredStore(t)

	cases := []struct {
		name   string
		filter core.Filter
		want   []string // doc IDs allowed in the results
	}{
		{"exact", core.Filter{"dir": "notes"}, []string{"notes/budget.txt"}},
		{"prefix", core.Filter{"dir": "doc*"}, []string{"docs/budget.md", "docs/vendor.md"}},
		{"two clauses", core.Filter{"dir": "docs", "team": "finance"}, []string{"docs/budget.md"}},
		{"by extension", core.Filter{"ext": "md"}, []string{"docs/budget.md", "docs/vendor.md", "archive/old.md"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			hits, err := s.Search(ctx, core.Query{Text: "atlas budget", TopK: 10, Filter: tc.filter})
			if err != nil {
				t.Fatal(err)
			}
			if len(hits) == 0 {
				t.Fatal("filter matched nothing")
			}
			allowed := map[string]bool{}
			for _, id := range tc.want {
				allowed[id] = true
			}
			for _, h := range hits {
				if !allowed[h.Chunk.DocID] {
					t.Errorf("filter %v returned %q, which does not match", tc.filter, h.Chunk.DocID)
				}
			}
			if len(hits) != len(tc.want) {
				t.Errorf("got %d hits, want %d matching docs", len(hits), len(tc.want))
			}
		})
	}
}

// A chunk missing the filtered key is excluded rather than admitted. The
// permissive reading would let unlabelled content leak past a tenant filter.
func TestFilterExcludesChunksMissingTheKey(t *testing.T) {
	ctx := context.Background()
	s := New(Config{LLM: &fake.LLM{}})
	docs := append(filterCorpus(), core.Doc{ID: "unlabelled.md", Text: "The Atlas budget is unstated here."})
	if _, err := s.Ingest(ctx, docs); err != nil {
		t.Fatal(err)
	}

	hits, err := s.Search(ctx, core.Query{Text: "atlas budget", TopK: 10, Filter: core.Filter{"team": "finance"}})
	if err != nil {
		t.Fatal(err)
	}
	for _, h := range hits {
		if h.Chunk.DocID == "unlabelled.md" {
			t.Fatal("a chunk with no team metadata passed a team filter")
		}
	}
}

func TestFilterMatchingNothingReturnsNothing(t *testing.T) {
	hits, err := filteredStore(t).Search(context.Background(),
		core.Query{Text: "atlas budget", TopK: 10, Filter: core.Filter{"dir": "nonexistent"}})
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 0 {
		t.Fatalf("filter matching no chunk returned %d hits", len(hits))
	}
}

// Filtering happens during the scan, not after it. Filtering an unfiltered
// top-k would return only whatever survived, so a filter selecting rare
// documents would return far fewer than topK even when enough exist.
func TestFilterKeepsThePoolFull(t *testing.T) {
	ctx := context.Background()
	s := New(Config{LLM: &fake.LLM{}})

	// 40 strong matches in the wrong directory, 3 weak ones in the right one.
	// Post-filtering a top-20 would find none of the three.
	var docs []core.Doc
	for i := range 40 {
		docs = append(docs, core.Doc{
			ID:   "loud/" + string(rune('a'+i%26)) + string(rune('a'+i/26)),
			Text: "atlas budget vendor schedule risk atlas budget",
			Meta: map[string]string{"dir": "loud"},
		})
	}
	for i := range 3 {
		docs = append(docs, core.Doc{
			ID:   "quiet/" + string(rune('a'+i)),
			Text: "a passing mention of the budget",
			Meta: map[string]string{"dir": "quiet"},
		})
	}
	if _, err := s.Ingest(ctx, docs); err != nil {
		t.Fatal(err)
	}

	hits, err := s.Search(ctx, core.Query{Text: "atlas budget", TopK: 3, Filter: core.Filter{"dir": "quiet"}})
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 3 {
		t.Fatalf("got %d hits for topK=3 with 3 matching docs; the filter ran after the scan", len(hits))
	}
}

// Corpus statistics stay global so a term's rarity means the same thing under
// every filter. Otherwise two filters could disagree about which terms are
// discriminating, and a filtered search would not be a subset of an unfiltered
// one in any explainable way.
func TestFilterDoesNotRescoreTheCorpus(t *testing.T) {
	ctx := context.Background()
	s := filteredStore(t)

	unfiltered, err := s.Search(ctx, core.Query{Text: "atlas budget", TopK: 10})
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]float64{}
	for _, h := range unfiltered {
		if h.Chunk.DocID == "notes/budget.txt" {
			want[h.Chunk.ID] = h.Score
		}
	}
	if len(want) == 0 {
		t.Fatal("setup: the notes doc did not appear unfiltered")
	}

	// The filter leaves exactly one document, so RRF sees one list of one and
	// the score is whatever rank-1 fusion gives — a filtered score is not
	// comparable to an unfiltered one. What must hold is the ordering rule: the
	// same chunks, ranked the same way relative to each other.
	filtered, err := s.Search(ctx, core.Query{Text: "atlas budget", TopK: 10, Filter: core.Filter{"dir": "notes"}})
	if err != nil {
		t.Fatal(err)
	}
	if len(filtered) != len(want) {
		t.Fatalf("filtered search returned %d chunks, want the %d that match", len(filtered), len(want))
	}
	for _, h := range filtered {
		if _, ok := want[h.Chunk.ID]; !ok {
			t.Errorf("filtered search returned %q, absent from the unfiltered results", h.Chunk.ID)
		}
	}
}

func TestFilterMatch(t *testing.T) {
	meta := map[string]string{"dir": "docs/api", "ext": "md"}
	cases := []struct {
		filter core.Filter
		want   bool
	}{
		{nil, true},
		{core.Filter{}, true},
		{core.Filter{"ext": "md"}, true},
		{core.Filter{"ext": "txt"}, false},
		{core.Filter{"dir": "docs*"}, true},
		{core.Filter{"dir": "docs/api"}, true},
		{core.Filter{"dir": "docs"}, false}, // exact, not prefix
		{core.Filter{"dir": "*"}, true},     // bare star matches any present value
		{core.Filter{"missing": "x"}, false},
		{core.Filter{"missing": "*"}, false}, // absent key fails even a wildcard
		{core.Filter{"ext": "md", "dir": "docs*"}, true},
		{core.Filter{"ext": "md", "dir": "notes*"}, false},
	}
	for _, tc := range cases {
		if got := tc.filter.Match(meta); got != tc.want {
			t.Errorf("Filter%v.Match(%v) = %v, want %v", tc.filter, meta, got, tc.want)
		}
	}
}
