package main

import (
	"context"
	"flag"
	"fmt"
	"math/rand/v2"
	"os"
	"runtime"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/RootControl/dispatch/core"
	"github.com/RootControl/dispatch/index"
	"github.com/RootControl/dispatch/internal/fake"
)

// evalScale measures how the reference store behaves as a corpus grows.
//
// READ THIS BEFORE READING THE NUMBERS. This measures the index structure and
// nothing else. It uses a deterministic local embedder and synthetic documents,
// so it needs no API key and costs nothing — and so it says NOTHING about
// retrieval quality at scale. Recall on synthetic text is an artifact of the
// generator. What it does measure is the thing the README could previously only
// assert: that a flat cosine scan and an in-memory BM25 index stop being
// reasonable somewhere, and roughly where.
//
// For quality at scale you need a real corpus and real embeddings:
//
//	dispatch eval retrieval --index .dispatch/big.json -k 5
func evalScale(args []string) error {
	fs := flag.NewFlagSet("eval scale", flag.ExitOnError)
	sizesArg := fs.String("sizes", "100,1000,10000", "comma-separated corpus sizes in chunks")
	queries := fs.Int("queries", 50, "queries to time at each size")
	dims := fs.Int("dims", 768, "embedding width (nomic-embed-text is 768)")
	topK := fs.Int("k", 5, "results per query")
	if err := fs.Parse(args); err != nil {
		return err
	}

	sizes, err := parseSizes(*sizesArg)
	if err != nil {
		return err
	}

	fmt.Printf("=== index scaling: %d dims, k=%d, %d queries per size ===\n", *dims, *topK, *queries)
	fmt.Println("Synthetic documents and a local deterministic embedder: this measures the")
	fmt.Println("index structure, not retrieval quality. No API key, no cost, no recall number.")
	fmt.Println()

	// Rows are measured first and written after, because tabwriter computes
	// column widths from what it has buffered: flushing per row to show
	// progress would align each row against itself and produce a ragged table.
	rows := make([]string, 0, len(sizes))
	for _, n := range sizes {
		fmt.Fprintf(os.Stderr, "measuring %d chunks...\r", n)
		r, err := measureScale(context.Background(), n, *dims, *queries, *topK)
		if err != nil {
			return err
		}
		rows = append(rows, fmt.Sprintf("%d\t%s\t%s\t%s\t%s\t%s", n,
			r.ingest.Round(time.Millisecond),
			r.p50.Round(time.Microsecond), r.p90.Round(time.Microsecond),
			humanBytes(r.heapBytes), humanBytes(r.heapBytes/int64(max(n, 1)))))
	}
	fmt.Fprint(os.Stderr, "                          \r")

	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "chunks\tingest\tsearch p50\tsearch p90\tresident\tper chunk")
	fmt.Fprintln(w, "------\t------\t----------\t----------\t--------\t---------")
	for _, row := range rows {
		fmt.Fprintln(w, row)
	}
	w.Flush()

	fmt.Println()
	fmt.Println("Search is a full scan of every vector plus a full scan of the BM25 postings,")
	fmt.Println("so latency grows linearly and memory holds every vector at 8 bytes per")
	fmt.Println("dimension. Both are fine while they are fine and then abruptly are not.")
	fmt.Println("Past the point where the p90 or the resident size stops suiting you, swap in")
	fmt.Println("index/pgvector — an HNSW index searches in sublinear time and keeps the")
	fmt.Println("corpus out of process memory. Everything above index.Searcher is unchanged.")
	return nil
}

type scaleResult struct {
	ingest    time.Duration
	p50, p90  time.Duration
	heapBytes int64
}

// measureScale builds a store of n synthetic chunks and times searches over it.
func measureScale(ctx context.Context, n, dims, queries, topK int) (scaleResult, error) {
	var r scaleResult
	f := &fake.LLM{Dims: dims}

	// Baseline the heap before building, so the reported figure is the index
	// rather than whatever the process happened to be holding.
	runtime.GC()
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)

	s := index.New(index.Config{LLM: f})
	docs := synthDocs(n)

	start := time.Now()
	if _, err := s.Ingest(ctx, docs); err != nil {
		return r, err
	}
	r.ingest = time.Since(start)

	runtime.GC()
	runtime.ReadMemStats(&after)
	r.heapBytes = int64(after.HeapAlloc) - int64(before.HeapAlloc)
	if r.heapBytes < 0 {
		r.heapBytes = int64(after.HeapAlloc)
	}

	// A fixed seed keeps the query set identical across sizes, so the columns
	// compare with each other rather than with a different set of questions.
	rng := rand.New(rand.NewPCG(1, 2))
	times := make([]time.Duration, 0, queries)
	for range queries {
		q := synthQuery(rng)
		start := time.Now()
		if _, err := s.Search(ctx, core.Query{Text: q, TopK: topK}); err != nil {
			return r, err
		}
		times = append(times, time.Since(start))
	}
	r.p50, r.p90 = percentile(times, 50), percentile(times, 90)
	return r, nil
}

// Vocabulary for synthetic documents. Real enough that BM25 has varied term
// frequencies to work with, which is what makes the timing representative;
// nothing here makes the *rankings* meaningful.
var scaleVocab = strings.Fields(`budget schedule vendor contract risk migration staffing
	steering charter approval reserve contingency quarter replica staging invoice
	renewal assessment threshold escalation dependency rollout owner deadline`)

// synthDocs builds n one-chunk documents. One chunk per document keeps the
// chunk count exactly n, so the x-axis is the number the table reports.
func synthDocs(n int) []core.Doc {
	rng := rand.New(rand.NewPCG(42, 7))
	docs := make([]core.Doc, n)
	for i := range n {
		words := make([]string, 30)
		for j := range words {
			words[j] = scaleVocab[rng.IntN(len(scaleVocab))]
		}
		docs[i] = core.Doc{
			ID:   fmt.Sprintf("synth/%06d.md", i),
			Text: strings.Join(words, " ") + ".",
			Meta: map[string]string{"dir": fmt.Sprintf("shard%02d", i%16)},
		}
	}
	return docs
}

func synthQuery(rng *rand.Rand) string {
	words := make([]string, 4)
	for i := range words {
		words[i] = scaleVocab[rng.IntN(len(scaleVocab))]
	}
	return strings.Join(words, " ")
}

func parseSizes(s string) ([]int, error) {
	var out []int
	for _, part := range strings.Split(s, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		var n int
		if _, err := fmt.Sscanf(part, "%d", &n); err != nil || n <= 0 {
			return nil, fmt.Errorf("bad size %q: want a positive integer", part)
		}
		out = append(out, n)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no sizes given")
	}
	return out, nil
}

// percentile returns the p-th percentile of times, sorting a copy so the
// caller's slice keeps its arrival order.
func percentile(times []time.Duration, p int) time.Duration {
	if len(times) == 0 {
		return 0
	}
	sorted := make([]time.Duration, len(times))
	copy(sorted, times)
	for i := 1; i < len(sorted); i++ {
		for j := i; j > 0 && sorted[j] < sorted[j-1]; j-- {
			sorted[j], sorted[j-1] = sorted[j-1], sorted[j]
		}
	}
	i := (p * len(sorted)) / 100
	if i >= len(sorted) {
		i = len(sorted) - 1
	}
	return sorted[i]
}

func humanBytes(b int64) string {
	const unit = 1024
	if b < unit {
		return fmt.Sprintf("%d B", b)
	}
	div, exp := int64(unit), 0
	for n := b / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(b)/float64(div), "KMGT"[exp])
}
