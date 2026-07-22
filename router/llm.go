package router

import (
	"context"
	"fmt"
	"slices"
	"strings"

	"github.com/RootControl/dispatch/core"
	"github.com/RootControl/dispatch/llm"
)

// LLM classifies with a model and falls back to the keyword heuristic whenever
// that fails or returns nothing usable.
//
// Routing is the one place where being confidently wrong is worse than being
// vague: a question sent to the wrong tier retrieves plausible evidence from the
// wrong place, and nothing downstream can tell. So the fallback is never
// skipped, and an unusable classification is treated as no classification.
type LLM struct {
	LLM       llm.LLM
	Available []core.Tier
	// Fallback runs when classification fails. Defaults to the keyword
	// Heuristic over the same Available set.
	Fallback Router
}

var _ Router = LLM{}

const routeSystem = `You route a question to the retrieval tier(s) best able to answer it.

Tiers:
- structured: exact values from a relational database — totals, counts, averages, lookups
  by id. Choose it when the answer is arithmetic over records.
- semantic: what the document corpus says about a topic. The default for open-ended
  questions about content.
- relational: connections between named entities — who reports to whom, what depends on
  what, chains of relationships across several hops.
- hierarchical: corpus-wide synthesis — recurring themes, overall patterns, what the
  collection says in aggregate. Choose it when no single passage could hold the answer.
- memory: facts recorded earlier from previous work.

Return the tiers worth searching, best first. Usually one, at most two. Add a second only
when the question genuinely has two parts needing different kinds of lookup.

Use ONLY tiers from the available list you are given.

Reply with a JSON object only:
{"tiers": ["<tier>", ...], "reason": "<short justification>"}`

// Route classifies the question, validates the answer against the available
// tiers, and falls back on any problem.
func (r LLM) Route(ctx context.Context, question string) (Decision, error) {
	fallback := r.Fallback
	if fallback == nil {
		fallback = Heuristic{Available: r.Available}
	}

	available := make([]string, len(r.Available))
	for i, t := range r.Available {
		available[i] = string(t)
	}
	user := fmt.Sprintf("Available tiers: %s\n\nQuestion: %s", strings.Join(available, ", "), question)

	var out struct {
		Tiers  []string `json:"tiers"`
		Reason string   `json:"reason"`
	}
	err := r.LLM.ChatJSON(ctx, []llm.Message{llm.System(routeSystem), llm.User(user)}, &out)
	if err != nil {
		return fallbackWith(ctx, fallback, question, "llm routing failed: "+err.Error())
	}

	// Keep only tiers that exist and are registered. A hallucinated tier name is
	// dropped rather than propagated as a dead end.
	ranked := make([]core.Tier, 0, len(out.Tiers))
	for _, name := range out.Tiers {
		t := core.Tier(strings.ToLower(strings.TrimSpace(name)))
		if slices.Contains(r.Available, t) && !slices.Contains(ranked, t) {
			ranked = append(ranked, t)
		}
	}
	if len(ranked) == 0 {
		return fallbackWith(ctx, fallback, question, "llm named no available tier")
	}

	reason := strings.TrimSpace(out.Reason)
	if reason == "" {
		reason = "classified by model"
	}
	return Decision{Tiers: ranked, Reason: reason, Source: "llm"}, nil
}

// fallbackWith runs the fallback router and records why it was needed, so
// --trace shows a degraded route rather than silently looking normal.
func fallbackWith(ctx context.Context, fallback Router, question, why string) (Decision, error) {
	d, err := fallback.Route(ctx, question)
	if err != nil {
		return d, err
	}
	d.Source = "heuristic"
	d.Reason = why + "; " + d.Reason
	return d, nil
}
