package main

import (
	"fmt"
	"os"
	"sync"
	"sync/atomic"
)

// defaultJobs is how many evaluation items run at once. It matches the LLM
// client's own concurrency cap, so the limit that actually binds is the one the
// client enforces rather than a second, different number here.
const defaultJobs = 4

// mapIndexed runs fn over 0..n-1 with at most jobs concurrent, and returns the
// results in INPUT order.
//
// Order matters more than it looks: an evaluation whose report reshuffles itself
// run to run cannot be diffed against the previous run, which is most of what a
// benchmark is for. Concurrency is a latency change, so it must not be visible
// in the output.
//
// Progress goes to stderr as items finish, since with the work spread across
// goroutines the ordered report cannot print incrementally and a long silence
// is indistinguishable from a hang.
func mapIndexed[T any](n, jobs int, label string, fn func(i int) T) []T {
	if jobs <= 0 {
		jobs = defaultJobs
	}
	out := make([]T, n)
	sem := make(chan struct{}, jobs)
	var wg sync.WaitGroup
	var done atomic.Int64

	showProgress := n > 1 && isTerminal()
	for i := range n {
		wg.Add(1)
		sem <- struct{}{}
		go func(i int) {
			defer wg.Done()
			defer func() { <-sem }()
			out[i] = fn(i) // disjoint indices, so no lock is needed
			if showProgress {
				fmt.Fprintf(os.Stderr, "\r%s %d/%d", label, done.Add(1), n)
			}
		}(i)
	}
	wg.Wait()
	if showProgress {
		fmt.Fprintf(os.Stderr, "\r%s\r", spaces(len(label)+12))
	}
	return out
}

func spaces(n int) string {
	b := make([]byte, n)
	for i := range b {
		b[i] = ' '
	}
	return string(b)
}

// isTerminal reports whether stderr is a terminal, so progress is not written
// into a redirected log.
func isTerminal() bool {
	fi, err := os.Stderr.Stat()
	return err == nil && fi.Mode()&os.ModeCharDevice != 0
}
