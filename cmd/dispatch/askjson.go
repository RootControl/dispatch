package main

import (
	"encoding/json"
	"io"
	"time"

	"github.com/RootControl/dispatch/agent"
	"github.com/RootControl/dispatch/core"
	"github.com/RootControl/dispatch/llm"
)

// askResult is the machine-readable form of one answered question.
//
// The loop now produces a lot of structured data — a citation report, a step
// trace, token counts — and until this existed all of it was reachable only by
// scraping prose off stdout. The field that earns this most is Citations: a CI
// job can assert `citations.ok` and fail a build on a fabricated source, which
// is not something a human is going to catch by reading answers.
type askResult struct {
	Question  string         `json:"question"`
	Answer    string         `json:"answer"`
	Citations citationsJSON  `json:"citations"`
	Evidence  []evidenceJSON `json:"evidence"`
	Trace     traceJSON      `json:"trace"`
	Usage     usageJSON      `json:"usage"`
}

type citationsJSON struct {
	// OK is false when the answer cites anything that is not in the evidence.
	OK         bool     `json:"ok"`
	Resolved   []string `json:"resolved"`
	Unresolved []string `json:"unresolved"`
	Uncited    []string `json:"uncited"`
}

type evidenceJSON struct {
	Cite   string  `json:"cite"`
	Tier   string  `json:"tier"`
	Source string  `json:"source"`
	Score  float64 `json:"score"`
	// Sources says which half of the hybrid found this chunk ("vec#3 text#1"),
	// empty for tiers that have no such notion.
	Sources string `json:"sources,omitempty"`
	Doc     string `json:"doc,omitempty"`
	Text    string `json:"text"`
}

type traceJSON struct {
	Rounds int         `json:"rounds"`
	Calls  int         `json:"calls"`
	Steps  []stepJSON  `json:"steps"`
	Tiers  []core.Tier `json:"tiers_routed"`
}

type stepJSON struct {
	N          int    `json:"n"`
	Kind       string `json:"kind"`
	Tier       string `json:"tier,omitempty"`
	Query      string `json:"query,omitempty"`
	Results    int    `json:"results,omitempty"`
	Sufficient *bool  `json:"sufficient,omitempty"`
	Gap        string `json:"gap,omitempty"`
	Detail     string `json:"detail,omitempty"`
	ElapsedMS  int64  `json:"elapsed_ms,omitempty"`
}

type usageJSON struct {
	Calls int `json:"calls"`
	// Prompt, Completion and Total come from the endpoint's own usage block.
	// Reported is false when the server sends none, so a consumer can tell "no
	// tokens" from "not measured" — a zero would read as free.
	Prompt     int  `json:"prompt_tokens"`
	Completion int  `json:"completion_tokens"`
	Total      int  `json:"total_tokens"`
	Reported   bool `json:"reported"`
}

// buildAskResult converts an answer into its JSON shape.
func buildAskResult(question string, a agent.Answer, u llm.Usage) askResult {
	res := askResult{
		Question: question,
		Answer:   a.Text,
		Citations: citationsJSON{
			OK:         a.Citations.OK(),
			Resolved:   nonNil(a.Citations.Resolved),
			Unresolved: nonNil(a.Citations.Unresolved),
			Uncited:    nonNil(a.Citations.Uncited),
		},
		Evidence: make([]evidenceJSON, 0, len(a.Evidence)),
		Usage: usageJSON{
			Calls:      u.Calls,
			Prompt:     u.PromptTokens,
			Completion: u.CompletionTokens,
			Total:      u.Total(),
			Reported:   u.Total() > 0,
		},
	}
	for _, e := range a.Evidence {
		res.Evidence = append(res.Evidence, evidenceJSON{
			Cite:    e.Cite(),
			Tier:    string(e.Tier),
			Source:  e.SourceID,
			Score:   e.Score,
			Sources: e.Meta["sources"],
			Doc:     e.Meta["doc"],
			Text:    e.Text,
		})
	}
	if a.Trace != nil {
		res.Trace = traceJSON{
			Rounds: a.Trace.Rounds,
			Calls:  a.Trace.Calls,
			Steps:  make([]stepJSON, 0, len(a.Trace.Steps)),
		}
		for _, s := range a.Trace.Steps {
			step := stepJSON{
				N: s.N, Kind: string(s.Kind), Tier: string(s.Tier),
				Query: s.Query, Results: s.Results, Gap: s.Gap, Detail: s.Detail,
				ElapsedMS: s.Elapsed.Round(time.Millisecond).Milliseconds(),
			}
			// Sufficient is only meaningful on a judge step; a plain bool would
			// report every retrieve step as "insufficient".
			if s.Kind == agent.StepJudge {
				v := s.Sufficient
				step.Sufficient = &v
			}
			if s.Kind == agent.StepRoute {
				res.Trace.Tiers = s.Tiers
			}
			res.Trace.Steps = append(res.Trace.Steps, step)
		}
	}
	return res
}

// nonNil renders an empty list as [] rather than null, so a consumer can index
// it without a nil check.
func nonNil(s []string) []string {
	if s == nil {
		return []string{}
	}
	return s
}

func writeJSON(w io.Writer, v any) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(v)
}
