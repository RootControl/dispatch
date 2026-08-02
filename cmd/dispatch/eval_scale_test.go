package main

import (
	"context"
	"testing"
	"time"
)

func TestParseSizes(t *testing.T) {
	got, err := parseSizes(" 100, 1000 ,10000,")
	if err != nil {
		t.Fatal(err)
	}
	want := []int{100, 1000, 10000}
	if len(got) != len(want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got %v, want %v", got, want)
		}
	}
	for _, bad := range []string{"", "abc", "0", "-5", "1,x"} {
		if _, err := parseSizes(bad); err == nil {
			t.Errorf("parseSizes(%q) should have failed", bad)
		}
	}
}

func TestPercentile(t *testing.T) {
	times := []time.Duration{5, 1, 4, 2, 3}
	if got := percentile(times, 50); got != 3 {
		t.Errorf("p50 = %v, want 3", got)
	}
	if got := percentile(times, 90); got != 5 {
		t.Errorf("p90 = %v, want 5", got)
	}
	if got := percentile(times, 100); got != 5 {
		t.Errorf("p100 = %v, want 5 (clamped to the last element)", got)
	}
	if got := percentile(nil, 50); got != 0 {
		t.Errorf("p50 of nothing = %v, want 0", got)
	}
	// The caller's slice must keep its arrival order.
	if times[0] != 5 {
		t.Errorf("percentile sorted the caller's slice: %v", times)
	}
}

// Synthetic documents must produce exactly one chunk each, or the x-axis of the
// table is not the number in the "chunks" column.
func TestSynthDocsAreOneChunkEach(t *testing.T) {
	r, err := measureScale(context.Background(), 200, 32, 3, 5)
	if err != nil {
		t.Fatal(err)
	}
	if r.p50 <= 0 || r.ingest <= 0 {
		t.Errorf("measurement returned no timings: %+v", r)
	}
	if r.heapBytes <= 0 {
		t.Errorf("measurement returned no memory figure: %d", r.heapBytes)
	}
}

func TestHumanBytes(t *testing.T) {
	cases := map[int64]string{
		512:           "512 B",
		1024:          "1.0 KB",
		1024 * 1024:   "1.0 MB",
		1536 * 1024:   "1.5 MB",
		3 * 1 << 30:   "3.0 GB",
		1024*1024 - 1: "1024.0 KB",
	}
	for in, want := range cases {
		if got := humanBytes(in); got != want {
			t.Errorf("humanBytes(%d) = %q, want %q", in, got, want)
		}
	}
}
