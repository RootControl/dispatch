package agent

import (
	"fmt"
	"strings"
	"time"

	"github.com/RootControl/dispatch/core"
)

// StepKind labels what happened at one point in the loop.
type StepKind string

const (
	StepRoute    StepKind = "route"
	StepRetrieve StepKind = "retrieve"
	StepJudge    StepKind = "judge"
	StepGenerate StepKind = "generate"
	StepRemember StepKind = "remember"
	StepVerify   StepKind = "verify"
	StepDiverse  StepKind = "diverse"
	StepExpand   StepKind = "expand"
)

// Step is one recorded action. Not every field applies to every kind; the
// zero values are simply not printed.
type Step struct {
	N          int
	Kind       StepKind
	Tier       core.Tier
	Tiers      []core.Tier
	Query      string
	Results    int
	Sufficient bool
	Gap        string
	Detail     string
	Elapsed    time.Duration
}

// Trace is the record of how an answer was reached. It exists so the claim that
// retrieval is "a loop the agent drives" is inspectable rather than asserted: if
// the loop never refines, the trace shows a single retrieve and says so.
type Trace struct {
	Question string
	Steps    []Step
	Rounds   int // retrieve/judge cycles actually executed
	Calls    int // LLM calls made by the loop (judge + generate)
}

func (t *Trace) add(s Step) {
	s.N = len(t.Steps) + 1
	t.Steps = append(t.Steps, s)
}

// String renders the trace for --trace output.
func (t *Trace) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "question: %s\n", t.Question)
	for _, s := range t.Steps {
		fmt.Fprintf(&b, "%2d %-8s", s.N, s.Kind)
		switch s.Kind {
		case StepRoute:
			fmt.Fprintf(&b, " %v", s.Tiers)
		case StepRetrieve:
			fmt.Fprintf(&b, " [%s] %d result(s) for %q", s.Tier, s.Results, truncate(s.Query, 60))
		case StepJudge:
			if s.Sufficient {
				fmt.Fprint(&b, " sufficient")
			} else {
				fmt.Fprintf(&b, " INSUFFICIENT gap=%q", truncate(s.Gap, 70))
			}
		case StepGenerate:
			fmt.Fprintf(&b, " %d evidence item(s)", s.Results)
		case StepVerify:
			fmt.Fprintf(&b, " %d citation(s) resolved", s.Results)
		case StepDiverse:
			fmt.Fprintf(&b, " %d evidence item(s) kept", s.Results)
		case StepExpand:
			fmt.Fprint(&b, " hypothetical passage")
		}
		if s.Detail != "" {
			fmt.Fprintf(&b, " — %s", s.Detail)
		}
		if s.Elapsed > 0 {
			fmt.Fprintf(&b, " (%s)", s.Elapsed.Round(time.Millisecond))
		}
		b.WriteByte('\n')
	}
	fmt.Fprintf(&b, "rounds: %d, llm calls: %d", t.Rounds, t.Calls)
	return b.String()
}

// Refined reports whether the loop actually refined its query — the property
// that distinguishes an agentic loop from a single-pass pipeline.
func (t *Trace) Refined() bool { return t.Rounds > 1 }

func truncate(s string, n int) string {
	s = strings.ReplaceAll(strings.TrimSpace(s), "\n", " ")
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
