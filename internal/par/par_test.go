package par

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestForEachRunsEveryIndexOnce(t *testing.T) {
	const n = 50
	var mu sync.Mutex
	seen := map[int]int{}
	err := ForEach(context.Background(), n, 8, func(_ context.Context, i int) error {
		mu.Lock()
		seen[i]++
		mu.Unlock()
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	for i := range n {
		if seen[i] != 1 {
			t.Fatalf("index %d ran %d times, want 1", i, seen[i])
		}
	}
}

func TestForEachRespectsTheLimit(t *testing.T) {
	var inFlight, peak atomic.Int64
	err := ForEach(context.Background(), 40, 3, func(_ context.Context, _ int) error {
		cur := inFlight.Add(1)
		for {
			p := peak.Load()
			if cur <= p || peak.CompareAndSwap(p, cur) {
				break
			}
		}
		time.Sleep(2 * time.Millisecond)
		inFlight.Add(-1)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if got := peak.Load(); got > 3 {
		t.Errorf("peak concurrency %d exceeded the limit of 3", got)
	}
	if got := peak.Load(); got < 2 {
		t.Errorf("peak concurrency %d: nothing actually ran in parallel", got)
	}
}

// A limit of zero or less must serialize rather than deadlock or run unbounded.
func TestForEachZeroLimitSerializes(t *testing.T) {
	var inFlight, peak atomic.Int64
	if err := ForEach(context.Background(), 10, 0, func(_ context.Context, _ int) error {
		cur := inFlight.Add(1)
		if cur > peak.Load() {
			peak.Store(cur)
		}
		time.Sleep(time.Millisecond)
		inFlight.Add(-1)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if peak.Load() != 1 {
		t.Errorf("peak concurrency %d with limit 0, want 1", peak.Load())
	}
}

// The first error wins and stops the rest, so a failing pass does not keep
// paying an API for work whose result is already being discarded.
func TestForEachReturnsFirstErrorAndCancelsTheRest(t *testing.T) {
	boom := errors.New("boom")
	var started atomic.Int64
	var cancelled atomic.Int64

	err := ForEach(context.Background(), 100, 4, func(ctx context.Context, i int) error {
		started.Add(1)
		if i == 0 {
			return boom
		}
		select {
		case <-ctx.Done():
			cancelled.Add(1)
		case <-time.After(50 * time.Millisecond):
		}
		return nil
	})
	if !errors.Is(err, boom) {
		t.Fatalf("err = %v, want boom", err)
	}
	if started.Load() == 100 {
		t.Error("every index started: the error did not stop the loop")
	}
	if cancelled.Load() == 0 {
		t.Error("in-flight calls were not cancelled")
	}
}

// A caller's cancellation must surface as an error rather than as a quietly
// short run — the whole point of the check in BuildGraph.
func TestForEachReportsCallerCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	err := ForEach(ctx, 100, 2, func(_ context.Context, i int) error {
		if i == 1 {
			cancel()
		}
		return nil
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
}

func TestForEachHandlesZeroItems(t *testing.T) {
	if err := ForEach(context.Background(), 0, 4, func(context.Context, int) error {
		t.Fatal("fn ran with no items")
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}
