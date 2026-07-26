package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"text/tabwriter"

	"github.com/RootControl/dispatch/core"
	"github.com/RootControl/dispatch/llm"
	"github.com/RootControl/dispatch/router"
)

// evalCase is one routing expectation.
type evalCase struct {
	Question string `json:"question"`
	Expect   string `json:"expect"`
	Note     string `json:"note,omitempty"`
}

// runEval scores routers against a labeled case file. Routing is the cheapest
// thing to get wrong and the hardest to notice: a misrouted question retrieves
// plausible evidence from the wrong tier, and the answer looks fine. Scoring it
// separately from retrieval keeps that failure visible.
func runEval(args []string) error {
	// Two evaluations, because routing and answering fail for different reasons
	// and a combined score would hide which one broke.
	if len(args) > 0 && args[0] == "answers" {
		return evalAnswers(args[1:])
	}
	if len(args) > 0 && args[0] == "retrieval" {
		return evalRetrieval(args[1:])
	}

	fs := flag.NewFlagSet("eval", flag.ExitOnError)
	casesPath := fs.String("cases", "./testdata/eval/routing.json", "labeled routing cases")
	which := fs.String("router", "heuristic", "which router to score: heuristic, llm, or both")
	verbose := fs.Bool("v", false, "show every case, not just failures")
	if err := fs.Parse(args); err != nil {
		return err
	}

	data, err := os.ReadFile(*casesPath)
	if err != nil {
		return err
	}
	var cases []evalCase
	if err := json.Unmarshal(data, &cases); err != nil {
		return fmt.Errorf("parse %s: %w", *casesPath, err)
	}
	if len(cases) == 0 {
		return fmt.Errorf("no cases in %s", *casesPath)
	}

	// Score against all four tiers regardless of what is currently indexed:
	// this measures classification, not deployment.
	available := []core.Tier{
		core.TierStructured, core.TierSemantic, core.TierRelational, core.TierHierarchical,
	}

	routers := map[string]router.Router{}
	switch *which {
	case "heuristic":
		routers["heuristic"] = router.Heuristic{Available: available}
	case "llm", "both":
		client, err := llm.New(llm.Config{})
		if err != nil {
			return err
		}
		routers["llm"] = router.LLM{LLM: client, Available: available}
		if *which == "both" {
			routers["heuristic"] = router.Heuristic{Available: available}
		}
	default:
		return fmt.Errorf("unknown --router %q (want heuristic, llm, or both)", *which)
	}

	ctx := context.Background()
	for _, name := range []string{"heuristic", "llm"} {
		r, ok := routers[name]
		if !ok {
			continue
		}
		if err := scoreRouter(ctx, name, r, cases, *verbose); err != nil {
			return err
		}
	}
	return nil
}

func scoreRouter(ctx context.Context, name string, r router.Router, cases []evalCase, verbose bool) error {
	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Printf("\n=== %s ===\n", name)

	var correct, top2 int
	for _, c := range cases {
		d, err := r.Route(ctx, c.Question)
		if err != nil {
			return err
		}
		got := ""
		if len(d.Tiers) > 0 {
			got = string(d.Tiers[0])
		}
		hit := got == c.Expect
		if hit {
			correct++
		}
		// Top-2 matters because the loop fans out: a correct tier ranked second
		// is still searched in the same round, so it is a near-miss, not a miss.
		inTop2 := false
		for i, tier := range d.Tiers {
			if i < 2 && string(tier) == c.Expect {
				inTop2 = true
			}
		}
		if inTop2 {
			top2++
		}

		if !hit || verbose {
			mark := "FAIL"
			if hit {
				mark = "ok"
			} else if inTop2 {
				mark = "~top2"
			}
			fmt.Fprintf(w, "%s\twant %s\tgot %s\t%s\n", mark, c.Expect, got, truncate(c.Question, 52))
		}
	}
	w.Flush()

	n := len(cases)
	fmt.Printf("top-1: %d/%d (%.0f%%)   top-2: %d/%d (%.0f%%)\n",
		correct, n, 100*float64(correct)/float64(n), top2, n, 100*float64(top2)/float64(n))
	return nil
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
