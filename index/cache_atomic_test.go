package index

import (
	"fmt"
	"os"
	"strings"
	"sync"
	"testing"
)

// A cache entry must be all-or-nothing.
//
// Put used os.WriteFile, which opens with O_TRUNC and then writes: a reader
// arriving mid-write sees a prefix, and a crash mid-write leaves that prefix on
// disk permanently. For the vector cache a truncated entry is caught by the
// JSON decode and reported as a miss, which costs one embedding call. For the
// text cache there is nothing to catch it — half a context sentence is a
// perfectly valid string, so it is returned as the situating sentence, embedded
// into the chunk, and indexed for BM25. The corruption then survives every
// later ingest, because the cache is exactly what a re-ingest trusts instead of
// calling the model.
//
// This became reachable rather than theoretical when the graph and hierarchy
// passes started running concurrently: several goroutines now write the cache
// at once, and two of them can want the same key.
func TestCacheEntriesAreNeverTorn(t *testing.T) {
	c := NewCache(t.TempDir())
	const key = "contended"
	// Large enough that a write is several syscalls, so a racing reader has a
	// window to land in. A short string would make this pass by luck.
	a := strings.Repeat("a", 512*1024)
	b := strings.Repeat("b", 512*1024)

	var wg sync.WaitGroup
	stop := make(chan struct{})
	torn := make(chan string, 1)

	for i := range 4 {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			val := a
			if i%2 == 1 {
				val = b
			}
			for {
				select {
				case <-stop:
					return
				default:
				}
				if err := c.Put(key, val); err != nil {
					return
				}
			}
		}(i)
	}

	for range 4 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-stop:
					return
				default:
				}
				got, ok := c.Get(key)
				switch {
				case !ok: // not yet written: an honest miss
				case got == a || got == b:
				default:
					select {
					case torn <- fmt.Sprintf("%d bytes, want %d", len(got), len(a)):
					default:
					}
					return
				}
			}
		}()
	}

	// Long enough to lose the race many times over, if it can be lost at all.
	for range 20000 {
		if _, ok := c.Get(key); !ok {
			continue
		}
	}
	close(stop)
	wg.Wait()

	select {
	case d := <-torn:
		t.Fatalf("read a partial cache entry: %s", d)
	default:
	}
}

// The vector cache degrades safely rather than silently, and must keep doing
// so: a truncated entry is a miss, never a short vector.
func TestTruncatedVectorEntryIsAMiss(t *testing.T) {
	c := NewCache(t.TempDir())
	if err := c.PutVector("k", []float64{1, 2, 3}); err != nil {
		t.Fatal(err)
	}
	// The tear the old writer allowed.
	if err := os.WriteFile(c.vecPath("k"), []byte("[1,2,"), 0o644); err != nil {
		t.Fatal(err)
	}
	if v, ok := c.GetVector("k"); ok {
		t.Fatalf("a truncated vector was served as a hit: %v", v)
	}
}
