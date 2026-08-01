package index

import (
	"math"
	"strings"
	"testing"
)

func TestFuseRRF(t *testing.T) {
	// a: ranked [x, y, z]; b: ranked [y, x]. k=60.
	// x: 1/61 + 1/62; y: 1/62 + 1/61; z: 1/63. x and y tie; tie-break by id -> x first.
	a := []scored{{"x", 9}, {"y", 8}, {"z", 7}}
	b := []scored{{"y", 5}, {"x", 4}}
	got := fuseRRF([][]scored{a, b}, 60, 0)

	wantX := 1.0/61 + 1.0/62
	if math.Abs(got[0].score-wantX) > 1e-12 {
		t.Fatalf("top score = %v, want %v", got[0].score, wantX)
	}
	if got[0].id != "x" || got[1].id != "y" || got[2].id != "z" {
		t.Fatalf("order = %v, want [x y z]", ids(got))
	}
}

func TestBM25RanksExactTermMatch(t *testing.T) {
	bm := newBM25()
	bm.add("d0", "the quarterly revenue report for the north region")
	bm.add("d1", "invoice 4471 total amount due")
	bm.add("d2", "weather notes and parking logistics")

	got := bm.search("invoice 4471", 0, nil)
	if len(got) == 0 || got[0].id != "d1" {
		t.Fatalf("expected d1 first for exact term match, got %v", ids(got))
	}
}

func TestBM25RareTermScoresHigher(t *testing.T) {
	bm := newBM25()
	// "report" is common (idf low); "4471" is rare (idf high).
	bm.add("d0", "annual report summary")
	bm.add("d1", "quarterly report summary")
	bm.add("d2", "monthly report invoice 4471")

	rare := bm.search("4471", 0, nil)
	common := bm.search("report", 0, nil)
	if len(rare) == 0 || len(common) == 0 {
		t.Fatal("expected matches for both queries")
	}
	if rare[0].score <= common[0].score {
		t.Fatalf("rare term should score higher: rare=%.3f common=%.3f", rare[0].score, common[0].score)
	}
}

func TestSplitParagraphBoundariesAndOverlap(t *testing.T) {
	text := "Alpha beta gamma delta.\n\nEpsilon zeta eta theta.\n\nIota kappa lambda mu."
	// Tiny target forces one chunk per short paragraph.
	chunks := Split(text, ChunkOptions{TargetTokens: 6, NoOverlap: true})
	if len(chunks) != 3 {
		t.Fatalf("expected 3 chunks, got %d: %q", len(chunks), chunks)
	}

	withOverlap := Split(text, ChunkOptions{TargetTokens: 6, OverlapTokens: 4})
	// Overlap should cause the second chunk to begin with the first's tail.
	if !strings.Contains(withOverlap[1], "Alpha") {
		t.Fatalf("expected overlap to carry prior text into chunk 2, got %q", withOverlap[1])
	}
}

func TestSplitOversizedParagraph(t *testing.T) {
	long := strings.Repeat("word ", 100)
	chunks := Split(long, ChunkOptions{TargetTokens: 12, NoOverlap: true})
	if len(chunks) < 2 {
		t.Fatalf("expected an oversized paragraph to split, got %d chunks", len(chunks))
	}
}

func ids(s []scored) []string {
	out := make([]string, len(s))
	for i, x := range s {
		out[i] = x.id
	}
	return out
}

// Real corpora contain fragments that carry no answer. Observed on a repository
// checkout: a file whose entire content was another filename, and a block of
// badge markup — both indexed, both unretrievable, each costing an LLM call.
func TestSplitDropsChunksWithoutProse(t *testing.T) {
	cases := map[string]struct {
		text string
		want int
	}{
		"filename only": {"CLAUDE.md", 0},
		"badge markup":  {`<p align="center"><a href="https://x.ai"><img src="/logo.svg" height="256"></a></p>`, 0},
		"heading only":  {"## Configuration", 0},
		"real prose":    {"The approved budget figure was four million dollars across three teams.", 1},
	}
	for name, tc := range cases {
		got := Split(tc.text, ChunkOptions{TargetTokens: 200, NoOverlap: true, MinWords: 8})
		if len(got) != tc.want {
			t.Errorf("%s: got %d chunks, want %d (%q)", name, len(got), tc.want, got)
		}
	}
	// Off by default: discarding a user's content silently is worse than
	// indexing noise, so everything is kept unless asked.
	if got := Split("CLAUDE.md", ChunkOptions{TargetTokens: 200, NoOverlap: true}); len(got) != 1 {
		t.Errorf("filtering should be opt-in, got %v", got)
	}
}
