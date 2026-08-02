package tiers

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/RootControl/dispatch/core"
	"github.com/RootControl/dispatch/index"
	"github.com/RootControl/dispatch/internal/fake"
	"github.com/RootControl/dispatch/llm"
)

// orderedStore builds a store whose chunks name the same entity in two
// different surface forms, one per document. Which form ends up as the entity's
// display name — and therefore what every rendered relationship in an answer
// says — depends on which document is applied to the graph first.
func orderedStore(t *testing.T, f llm.LLM) *index.Store {
	t.Helper()
	s := index.New(index.Config{LLM: f})
	_, err := s.Ingest(context.Background(), []core.Doc{
		{ID: "a-first", Text: "Priya Raman leads Atlas."},
		{ID: "b-second", Text: "The renewal was signed by PRIYA RAMAN."},
		{ID: "c-third", Text: "Atlas depends on the mailer."},
	})
	if err != nil {
		t.Fatal(err)
	}
	return s
}

// slowFake answers extraction prompts, sleeping longer for the documents that
// come EARLIER in chunk order. Under a parallel build the calls therefore
// complete in roughly reverse order, so anything that depends on completion
// order rather than chunk order comes out backwards.
func slowFake(delays map[string]time.Duration, reply func(body string) string) *fake.LLM {
	return &fake.LLM{ChatFunc: func(msgs []llm.Message) (string, error) {
		body := msgs[len(msgs)-1].Content
		for marker, d := range delays {
			if strings.Contains(body, marker) {
				time.Sleep(d)
				break
			}
		}
		return reply(body), nil
	}}
}

// The graph must not depend on which extraction call returns first.
//
// addEntity keeps the display name as first seen and edges are appended, so a
// build that applied results in completion order would produce a different
// Entity.Name — and a different edge order — every run, over an unchanged
// corpus. Extraction runs concurrently now; applying the results in chunk order
// is what keeps the output identical to the sequential build.
func TestGraphIsIdenticalWhateverOrderExtractionsReturn(t *testing.T) {
	reply := func(body string) string {
		switch {
		case strings.Contains(body, "leads Atlas"):
			return `{"entities":[{"name":"Priya Raman","type":"person"}],
			         "relations":[{"from":"Priya Raman","relation":"leads","to":"Atlas"}]}`
		case strings.Contains(body, "PRIYA RAMAN"):
			return `{"entities":[{"name":"PRIYA RAMAN","type":"person"}],
			         "relations":[{"from":"PRIYA RAMAN","relation":"signed","to":"renewal"}]}`
		default:
			return `{"entities":[{"name":"Atlas","type":"system"}],
			         "relations":[{"from":"Atlas","relation":"depends on","to":"mailer"}]}`
		}
	}

	// Serial: the reference result, and what every earlier test measured.
	serialFake := slowFake(nil, reply)
	serial, _, err := BuildGraph(context.Background(), serialFake, orderedStore(t, serialFake),
		GraphOptions{NoCoreference: true, Parallelism: 1})
	if err != nil {
		t.Fatal(err)
	}

	// Parallel, with the first chunk deliberately the slowest.
	skewed := slowFake(map[string]time.Duration{
		"leads Atlas":  120 * time.Millisecond,
		"PRIYA RAMAN":  60 * time.Millisecond,
		"depends on t": 0,
	}, reply)
	parallel, _, err := BuildGraph(context.Background(), skewed, orderedStore(t, skewed),
		GraphOptions{NoCoreference: true, Parallelism: 4})
	if err != nil {
		t.Fatal(err)
	}

	if got, want := render(t, parallel.graph), render(t, serial.graph); got != want {
		t.Errorf("the parallel build produced a different graph:\n--- parallel ---\n%s\n--- serial ---\n%s", got, want)
	}
	// This absolute check is load-bearing, not decoration. Both builds above run
	// the same apply loop, so a change to the ORDER that loop applies in moves
	// them together and the comparison keeps passing: reversing it was tried,
	// and only this line failed. A differential test cannot see a bug that is
	// common to both sides.
	if name := parallel.graph.Entities["priya raman"].Name; name != "Priya Raman" {
		t.Errorf("display name = %q, want %q (the form in the first chunk, not the first call to return)", name, "Priya Raman")
	}
}

func render(t *testing.T, g *Graph) string {
	t.Helper()
	b, err := json.MarshalIndent(struct {
		Entities map[string]Entity
		Edges    []Edge
	}{g.Entities, g.Edges}, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

// Concurrency has to be observable, or "it is parallel now" is a claim about
// code shape rather than about behaviour. This blocks every extraction call
// until several are in flight at once: it can only finish if they overlap.
func TestGraphExtractionCallsOverlap(t *testing.T) {
	const want = 3 // one per chunk in orderedStore
	var mu sync.Mutex
	var inFlight, peak int
	all := make(chan struct{})

	f := &fake.LLM{ChatFunc: func([]llm.Message) (string, error) {
		mu.Lock()
		inFlight++
		if inFlight > peak {
			peak = inFlight
		}
		if inFlight == want {
			close(all)
		}
		mu.Unlock()

		select {
		case <-all:
		case <-time.After(3 * time.Second):
			// Fall through rather than hang: the assertion below reports the
			// real peak, which is a far better failure message than a timeout.
		}

		mu.Lock()
		inFlight--
		mu.Unlock()
		return `{"entities":[],"relations":[]}`, nil
	}}

	if _, _, err := BuildGraph(context.Background(), f, orderedStore(t, f),
		GraphOptions{NoCoreference: true, Parallelism: 4}); err != nil {
		t.Fatal(err)
	}
	if peak < want {
		t.Errorf("peak concurrent extractions = %d, want %d: the calls are still serialized", peak, want)
	}
}

// Parallelism 1 must genuinely serialize, so the setting is a real escape hatch
// for a server that cannot take concurrent requests.
func TestGraphParallelismOneSerializes(t *testing.T) {
	var mu sync.Mutex
	var inFlight, peak int
	f := &fake.LLM{ChatFunc: func([]llm.Message) (string, error) {
		mu.Lock()
		inFlight++
		if inFlight > peak {
			peak = inFlight
		}
		mu.Unlock()
		time.Sleep(20 * time.Millisecond)
		mu.Lock()
		inFlight--
		mu.Unlock()
		return `{"entities":[],"relations":[]}`, nil
	}}

	if _, _, err := BuildGraph(context.Background(), f, orderedStore(t, f),
		GraphOptions{NoCoreference: true, Parallelism: 1}); err != nil {
		t.Fatal(err)
	}
	if peak != 1 {
		t.Errorf("peak concurrent extractions = %d under Parallelism 1", peak)
	}
}

// An interrupted build must fail rather than save a smaller graph.
//
// Every outstanding chunk fails at once with the same context error, so the old
// skip-and-continue path turned a cancelled ingest into a graph holding
// whatever finished first — stamped with the current index generation, and so
// indistinguishable from a complete one. The staleness check cannot catch this:
// the generation really is current.
func TestCancelledGraphBuildFailsRatherThanTruncating(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	var mu sync.Mutex
	var seen int
	f := &fake.LLM{ChatFunc: func([]llm.Message) (string, error) {
		mu.Lock()
		seen++
		n := seen
		mu.Unlock()
		if n >= 2 {
			cancel()
		}
		if ctx.Err() != nil {
			return "", ctx.Err()
		}
		return `{"entities":[{"name":"Atlas","type":"system"}],"relations":[]}`, nil
	}}

	_, stats, err := BuildGraph(ctx, f, orderedStore(t, f), GraphOptions{NoCoreference: true, Parallelism: 1})
	if err == nil {
		t.Fatal("a cancelled build reported success")
	}
	if !strings.Contains(err.Error(), "interrupted") {
		t.Errorf("error should say the build was interrupted, got: %v", err)
	}
	if stats.Skipped == 0 {
		t.Error("expected the cancelled chunks to be counted as skipped")
	}
}

// A build where every chunk failed is a failed build, not an empty graph.
//
// Found by running it rather than by reading it: a slow endpoint timed out on
// both chunks of a two-document corpus, and `ingest --graph` cheerfully wrote a
// graph with zero entities, stamped with the current index generation — so the
// staleness check called it current, and it had already overwritten the
// previous good graph. Skipping SOME chunks stays correct; skipping all of them
// means the endpoint was down.
func TestGraphBuildFailsWhenEveryChunkFails(t *testing.T) {
	f := &fake.LLM{ChatFunc: func([]llm.Message) (string, error) {
		return "", errors.New("context deadline exceeded")
	}}
	_, stats, err := BuildGraph(context.Background(), f, orderedStore(t, f),
		GraphOptions{NoCoreference: true, Parallelism: 4})
	if err == nil {
		t.Fatal("a build where nothing extracted reported success")
	}
	if !strings.Contains(err.Error(), "failed build") {
		t.Errorf("the error should distinguish a failed build from an empty graph, got: %v", err)
	}
	if stats.Skipped != stats.Chunks {
		t.Errorf("Skipped = %d of %d chunks", stats.Skipped, stats.Chunks)
	}
}

// ...but a build where only some chunks failed must still succeed, because on a
// real corpus extraction runs for tens of minutes and aborting at chunk 60 of
// 70 throws away every earlier call.
func TestGraphBuildSurvivesPartialFailure(t *testing.T) {
	f := &fake.LLM{ChatFunc: func(msgs []llm.Message) (string, error) {
		if strings.Contains(msgs[len(msgs)-1].Content, "PRIYA RAMAN") {
			return "", errors.New("model returned reasoning but no answer")
		}
		return `{"entities":[{"name":"Atlas","type":"system"}],
		         "relations":[{"from":"Atlas","relation":"depends on","to":"mailer"}]}`, nil
	}}
	rel, stats, err := BuildGraph(context.Background(), f, orderedStore(t, f),
		GraphOptions{NoCoreference: true, Parallelism: 4})
	if err != nil {
		t.Fatalf("one bad chunk should not fail the build: %v", err)
	}
	if stats.Skipped != 1 {
		t.Errorf("Skipped = %d, want 1", stats.Skipped)
	}
	if _, relations := rel.Stats(); relations == 0 {
		t.Error("the chunks that did extract should still have produced a graph")
	}
}

// The tree must not renumber itself between runs: node IDs are positional
// (L1-0, L1-1, ...) and citations quote them, so a summary written into
// whichever slot finished first would make a saved citation point at different
// text after a rebuild.
func TestHierarchyNodeIDsDoNotDependOnCompletionOrder(t *testing.T) {
	texts := []string{
		"Alpha document about the budget and its four million dollars.",
		"Beta document about the vendor contract renewal in January.",
		"Gamma document about staffing and who reports to whom.",
		"Delta document about the testing strategy and the staging replica.",
	}

	build := func(parallelism int, delays map[string]time.Duration) *Hierarchical {
		t.Helper()
		f := slowFake(delays, func(body string) string {
			// Echo something identifying so each summary is traceable to its
			// members regardless of when it was produced.
			for _, name := range []string{"Alpha", "Beta", "Gamma", "Delta"} {
				if strings.Contains(body, name) {
					return "Summary mentioning " + name
				}
			}
			return "Summary"
		})
		store := index.New(index.Config{LLM: f})
		docs := make([]core.Doc, len(texts))
		for i, tx := range texts {
			docs[i] = core.Doc{ID: fmt.Sprintf("doc%d", i), Text: tx}
		}
		if _, err := store.Ingest(context.Background(), docs); err != nil {
			t.Fatal(err)
		}
		h, _, err := BuildHierarchy(context.Background(), f, store, HierarchyOptions{
			Branching: 2, Parallelism: parallelism,
			Cache: index.NewCache(t.TempDir()),
		})
		if err != nil {
			t.Fatal(err)
		}
		return h
	}

	serial := build(1, nil)
	parallel := build(4, map[string]time.Duration{"Alpha": 100 * time.Millisecond, "Beta": 50 * time.Millisecond})

	got, want := nodeMap(serial), nodeMap(parallel)
	for id, text := range want {
		if got[id] != text {
			t.Errorf("node %s: parallel build has %q, serial build has %q", id, text, got[id])
		}
	}
	if len(got) != len(want) {
		t.Errorf("node count differs: serial %d, parallel %d", len(got), len(want))
	}
}

func nodeMap(h *Hierarchical) map[string]string {
	chunks, _ := h.nodes.Entries()
	out := map[string]string{}
	for _, c := range chunks {
		out[c.ID] = c.Text
	}
	return out
}
