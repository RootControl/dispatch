package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func run(label string, k int, ranks map[string]int) retrievalRun {
	return retrievalRun{Label: label, TopK: k, Ranks: ranks}
}

func TestRecallAt(t *testing.T) {
	r := run("x", 5, map[string]int{"a": 0, "b": 2, "c": 4, "d": -1})
	if hits, total := r.recallAt(1); hits != 1 || total != 4 {
		t.Errorf("recall@1 = %d/%d, want 1/4", hits, total)
	}
	if hits, total := r.recallAt(5); hits != 3 || total != 4 {
		t.Errorf("recall@5 = %d/%d, want 3/4 (the miss does not count)", hits, total)
	}
}

// -1 means "not retrieved at all", which must sort worse than any real
// position — otherwise a chunk falling out of the results reads as an
// improvement, which is the one direction the comparison must never get wrong.
func TestRankBetterTreatsAMissAsWorst(t *testing.T) {
	cases := []struct {
		a, b int
		want bool
	}{
		{0, 1, true},   // #1 beats #2
		{1, 0, false},  // #2 does not beat #1
		{0, -1, true},  // any hit beats a miss
		{-1, 0, false}, // a miss never beats a hit
		{-1, -1, false},
		{4, 4, false},
	}
	for _, tc := range cases {
		if got := rankBetter(tc.a, tc.b); got != tc.want {
			t.Errorf("rankBetter(%d, %d) = %v, want %v", tc.a, tc.b, got, tc.want)
		}
	}
}

func TestSaveLoadRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "base.json")
	want := run("bge-reranker-base", 5, map[string]int{"a#0": 0, "b#0": -1})
	if err := saveRun(path, want); err != nil {
		t.Fatal(err)
	}
	got, err := loadRun(path)
	if err != nil {
		t.Fatal(err)
	}
	if got.Label != want.Label || got.TopK != want.TopK || len(got.Ranks) != len(want.Ranks) {
		t.Fatalf("round trip = %+v, want %+v", got, want)
	}
	if got.Ranks["b#0"] != -1 {
		t.Errorf("a miss did not survive the round trip: %v", got.Ranks)
	}
}

func TestLoadRunRejectsEmptyAndMissing(t *testing.T) {
	dir := t.TempDir()
	if _, err := loadRun(filepath.Join(dir, "nope.json")); err == nil {
		t.Error("expected an error for a missing file")
	}
	empty := filepath.Join(dir, "empty.json")
	os.WriteFile(empty, []byte(`{"label":"x","k":5}`), 0o644)
	if _, err := loadRun(empty); err == nil {
		t.Error("expected an error for a run with no per-chunk ranks")
	}
	bad := filepath.Join(dir, "bad.json")
	os.WriteFile(bad, []byte(`not json`), 0o644)
	if _, err := loadRun(bad); err == nil {
		t.Error("expected an error for malformed JSON")
	}
}

// capture runs f with stdout redirected.
func capture(t *testing.T, f func()) string {
	t.Helper()
	old := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w
	done := make(chan string)
	go func() {
		var b strings.Builder
		buf := make([]byte, 4096)
		for {
			n, err := r.Read(buf)
			b.Write(buf[:n])
			if err != nil {
				break
			}
		}
		done <- b.String()
	}()
	f()
	w.Close()
	os.Stdout = old
	return <-done
}

// The case that motivated the feature: the cross-encoder gained thirteen
// first-place hits and lost two top-five ones. Two recall figures cannot show
// that; a per-chunk diff can.
func TestCompareShowsBothDirections(t *testing.T) {
	base := run("none", 5, map[string]int{
		"gained#0": 3, "gained#1": 2, "lost#0": 4, "same#0": 0,
	})
	now := run("cross-encoder", 5, map[string]int{
		"gained#0": 0, "gained#1": 0, "lost#0": -1, "same#0": 0,
	})

	out := capture(t, func() { compareRuns(base, now) })

	for _, want := range []string{"vs none", "improved", "regressed", "gained#0", "lost#0"} {
		if !strings.Contains(out, want) {
			t.Errorf("output missing %q:\n%s", want, out)
		}
	}
	if !strings.Contains(out, "chunks improved: 2, regressed: 1") {
		t.Errorf("counts wrong:\n%s", out)
	}
	// A chunk that fell out of the results entirely must read as a miss.
	if !strings.Contains(out, "-> miss") {
		t.Errorf("a chunk falling out of the top-k is not shown as a miss:\n%s", out)
	}
}

// Comparing runs over different chunk sets silently divides by different
// denominators, which makes the aggregate meaningless. Say so rather than
// printing a confident delta.
func TestCompareWarnsOnDifferentChunkSets(t *testing.T) {
	base := run("old", 5, map[string]int{"a#0": 0, "gone#0": 1})
	now := run("new", 5, map[string]int{"a#0": 0, "added#0": 1})

	out := capture(t, func() { compareRuns(base, now) })
	if !strings.Contains(out, "do not cover the same chunks") {
		t.Errorf("no warning about mismatched chunk sets:\n%s", out)
	}
	if !strings.Contains(out, "gone#0") || !strings.Contains(out, "added#0") {
		t.Errorf("warning does not name the differing chunks:\n%s", out)
	}
}

// Two runs at different k produce recall figures that are not comparable.
func TestCompareWarnsOnDifferentK(t *testing.T) {
	base := run("old", 5, map[string]int{"a#0": 3})
	now := run("new", 1, map[string]int{"a#0": 0})

	out := capture(t, func() { compareRuns(base, now) })
	if !strings.Contains(out, "not comparable") {
		t.Errorf("no warning about differing k:\n%s", out)
	}
}

func TestCompareIdenticalRuns(t *testing.T) {
	r := run("x", 5, map[string]int{"a#0": 0, "b#0": 2})
	out := capture(t, func() { compareRuns(r, r) })
	if !strings.Contains(out, "chunks improved: 0, regressed: 0, unchanged: 2") {
		t.Errorf("identical runs should show no movement:\n%s", out)
	}
	if strings.Contains(out, "do not cover the same chunks") {
		t.Errorf("spurious chunk-set warning:\n%s", out)
	}
}

// The soundness hole this tool shipped with, found on its first real use.
// Chunk IDs are "doc#position", so a splitter change keeps every ID and swaps
// the text behind it — and comparing those ranks produces a clean table of
// deltas about two things that were never the same chunk.
func TestCompareDetectsRechunkedIDs(t *testing.T) {
	base := retrievalRun{Label: "flat", TopK: 5,
		Ranks: map[string]int{"doc.md#0": 2, "doc.md#1": 0},
		Texts: map[string]string{"doc.md#0": "aaa", "doc.md#1": "bbb"}}
	now := retrievalRun{Label: "headings", TopK: 5,
		Ranks: map[string]int{"doc.md#0": 0, "doc.md#1": 0},
		Texts: map[string]string{"doc.md#0": "CHANGED", "doc.md#1": "bbb"}}

	out := capture(t, func() { compareRuns(base, now) })

	if !strings.Contains(out, "hold different text") {
		t.Errorf("no warning about re-chunked IDs:\n%s", out)
	}
	// doc.md#0 apparently went #3 -> #1, which would be a headline improvement
	// and is meaningless. It must not appear in the improved section at all.
	if strings.Contains(out, "\nimproved:\n") || strings.Contains(out, "doc.md#0 ") {
		t.Errorf("listed a re-chunked ID as an improvement:\n%s", out)
	}
	if !strings.Contains(out, "chunks improved: 0, regressed: 0") {
		t.Errorf("re-chunked ID leaked into the counts:\n%s", out)
	}
}

// A run saved before digests existed must still compare, just without the
// re-chunk check — an older baseline should not become unusable.
func TestCompareToleratesRunsWithoutDigests(t *testing.T) {
	base := retrievalRun{Label: "old", TopK: 5, Ranks: map[string]int{"a#0": 2}}
	now := retrievalRun{Label: "new", TopK: 5,
		Ranks: map[string]int{"a#0": 0}, Texts: map[string]string{"a#0": "x"}}

	out := capture(t, func() { compareRuns(base, now) })
	if !strings.Contains(out, "chunks improved: 1") {
		t.Errorf("a digest-less baseline should still compare:\n%s", out)
	}
	if strings.Contains(out, "hold different text") {
		t.Errorf("warned about re-chunking with nothing to compare against:\n%s", out)
	}
}
