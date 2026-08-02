package main

import (
	"context"
	"flag"
	"fmt"

	"github.com/RootControl/dispatch/agent"
)

// evalSelfCheck runs the answer eval's negative controls.
//
// The README has always claimed the harness was checked against negative
// controls before its numbers were believed: a case with a deliberately wrong
// expect_sources must report a retrieval failure while still scoring its facts,
// proving the two metrics are independent; a case demanding a fact absent from
// the corpus must report a facts failure and name what is missing.
//
// That was done once, by hand. This runs it, because a check that cannot fail
// looks exactly like a check that passes — and twice in this repo's history a
// test has been found passing while the thing it tested was broken.
//
// Each control asserts a scorer OUTCOME, not an answer. It needs the model to
// produce something plausible, not something correct, so it stays meaningful on
// a small local model.
func evalSelfCheck(args []string) error {
	fs := flag.NewFlagSet("eval self-check", flag.ExitOnError)
	indexPath := fs.String("index", defaultIndexPath, "index to probe")
	topK := fs.Int("k", 4, "results per tier")
	if err := fs.Parse(args); err != nil {
		return err
	}

	st, err := buildStack(stackOptions{IndexPath: *indexPath})
	if err != nil {
		return err
	}
	chunks, _ := st.store.Entries()
	if len(chunks) == 0 {
		return fmt.Errorf("index %s is empty", *indexPath)
	}
	// Ground the controls in a document the index actually holds, so the
	// self-check runs against any corpus rather than only the bundled one.
	realDoc := chunks[0].DocID

	loop := &agent.Loop{
		LLM: st.client, Router: st.router(false), Retrievers: st.registry,
		MaxSteps: 1, TopK: *topK,
	}

	controls := []struct {
		name string
		why  string
		c    answerCase
		// check reports why the outcome is wrong, or "" when the control held.
		check func(answerScore) string
	}{
		{
			name: "wrong expect_sources reports RETRIEVAL",
			why:  "retrieval and facts must be independent: a wrong source must not be masked by a right answer",
			c: answerCase{
				Question:      "What is this corpus about?",
				ExpectSources: []string{"a-document-that-does-not-exist.md"},
			},
			check: func(s answerScore) string {
				if s.err != nil {
					return "the case errored instead of scoring: " + s.err.Error()
				}
				if s.retrieved {
					return "scored retrieval OK against a source that is not in the corpus"
				}
				return ""
			},
		},
		{
			name: "absent fact reports FACTS",
			why:  "a required fact the corpus cannot support must fail, and be named",
			c: answerCase{
				Question:      "What is this corpus about?",
				ExpectSources: []string{realDoc},
				MustInclude:   []Fact{{"zzqx-sentinel-value-9471"}},
			},
			check: func(s answerScore) string {
				if s.err != nil {
					return "the case errored instead of scoring: " + s.err.Error()
				}
				if s.factsFound != 0 {
					return "found a fact that appears nowhere in the corpus"
				}
				if len(s.missing) == 0 {
					return "failed the fact but did not name what was missing"
				}
				return ""
			},
		},
		{
			name: "a real source reports retrieval OK",
			why:  "the controls above only mean something if the positive case passes",
			c: answerCase{
				Question:      "What is this corpus about?",
				ExpectSources: []string{realDoc},
			},
			check: func(s answerScore) string {
				if s.err != nil {
					return "the case errored instead of scoring: " + s.err.Error()
				}
				if !s.retrieved {
					return fmt.Sprintf("did not retrieve %s, which is in the index — "+
						"the negative controls above prove nothing if this fails", realDoc)
				}
				return ""
			},
		},
	}

	fmt.Printf("=== eval self-check: %d controls against %s ===\n\n", len(controls), *indexPath)
	ctx := context.Background()
	failed := 0
	for _, ctl := range controls {
		score := scoreAnswer(ctx, loop, ctl.c, false)
		if why := ctl.check(score); why != "" {
			failed++
			fmt.Printf("FAIL  %s\n      %s\n      why it matters: %s\n\n", ctl.name, why, ctl.why)
			continue
		}
		fmt.Printf("ok    %s\n", ctl.name)
	}

	fmt.Printf("\n%d/%d controls held\n", len(controls)-failed, len(controls))
	if failed > 0 {
		return fmt.Errorf("%d negative control(s) failed: the answer eval's numbers cannot be trusted "+
			"until these pass, because a check that cannot fail looks exactly like a check that passes", failed)
	}
	fmt.Println("The answer eval can distinguish a retrieval failure from a generation one,")
	fmt.Println("and does not score facts the corpus cannot support.")
	return nil
}
