// Package par runs bounded-concurrency work over a range of indices.
//
// Every expensive pass in this project has the same shape: N independent LLM
// calls, each latency-bound rather than CPU-bound, against a server that will
// serve a handful at once. Running them one at a time leaves the machine idle
// for essentially the whole of a 40-minute ingest.
//
// It lives here rather than in index because two packages need it, and
// exporting it from index would put an internal utility in the public API of a
// package whose surface is meant to be a Store.
package par

import (
	"context"
	"sync"
)

// ForEach runs fn for indices 0..n-1 with at most limit concurrent goroutines.
// It returns the first error, having cancelled the context passed to the
// remaining calls.
//
// Use this when any failure should abort the run. For work where one failed
// item should be skipped rather than sink the batch — graph extraction, where
// aborting at chunk 60 of 70 discards an hour of calls — have fn record its own
// error and return nil, so only cancellation stops the loop.
func ForEach(ctx context.Context, n, limit int, fn func(ctx context.Context, i int) error) error {
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
