// Package router classifies a question into the tier(s) most likely to answer
// it. It returns a ranked list rather than a single winner: the agent loop fans
// out, and a confident-but-wrong single choice is the failure mode that makes
// routed retrieval worse than plain RAG.
package router

import (
	"context"
	"fmt"
	"regexp"
	"slices"
	"strings"

	"github.com/RootControl/dispatch/core"
)

// Decision is a routing outcome.
type Decision struct {
	Tiers  []core.Tier // ranked, best first; always non-empty
	Reason string      // human-readable justification, shown by --trace
	Source string      // which router produced this: "heuristic" or "llm"
}

// Router classifies questions. The LLM-backed implementation lands later and
// falls back to Heuristic; keeping this an interface means the agent loop never
// changes when it does.
type Router interface {
	Route(ctx context.Context, question string) (Decision, error)
}

// Heuristic routes by keyword match. It is the zero-cost, zero-latency baseline
// and the fallback whenever LLM classification fails.
type Heuristic struct {
	// Available limits routing to registered tiers. A tier the router likes but
	// nobody implements is dropped rather than returned as a dead end.
	Available []core.Tier
}

var _ Router = Heuristic{}

// Keyword patterns per tier. Word boundaries matter: without \b, "sign" matches
// "design" and the relational tier starts eating design questions.
var patterns = map[core.Tier]*regexp.Regexp{
	core.TierStructured: regexp.MustCompile(`\b(total|totals|sum|count|how many|how much|average|avg|invoice|invoices|amount|balance|revenue|price|paid|owed|due|per (month|year|quarter|day))\b`),
	// Two vocabularies, because relational questions come in two flavours and
	// the org-chart one alone misses technical corpora entirely: asked "what
	// does the client workspace depend on?" over a real repository, this matched
	// nothing and fell through to semantic.
	core.TierRelational: regexp.MustCompile(`\b(` +
		// people and documents
		`who|whom|whose|reports? to|reported to|manager|managers|related to|connected to|` +
		`relationship|signed|sign|approved by|works for|belongs to|` +
		// systems and components
		`depends? on|depended on|dependency|dependencies|requires?|imports?|` +
		`consumed by|provided by|maintained by|owns|owned by|part of|consists of|built on` +
		`)\b`),
	core.TierHierarchical: regexp.MustCompile(`\b(themes?|recurring|across all|across the|overall|in general|common|trends?|summar(y|ize|ise)|patterns?|generally)\b`),
}

// Route scores each tier by how many distinct keywords it matches, ranks by
// score, and always appends the semantic tier as a fallback — semantic search is
// the sensible default when nothing else clearly fits.
func (h Heuristic) Route(ctx context.Context, question string) (Decision, error) {
	q := strings.ToLower(question)

	type hit struct {
		tier  core.Tier
		count int
		terms []string
	}
	var hits []hit
	for tier, re := range patterns {
		if m := re.FindAllString(q, -1); len(m) > 0 {
			hits = append(hits, hit{tier: tier, count: len(m), terms: dedupe(m)})
		}
	}
	// Rank by match count, tie-broken by tier name for determinism.
	slices.SortFunc(hits, func(a, b hit) int {
		if a.count != b.count {
			return b.count - a.count
		}
		return strings.Compare(string(a.tier), string(b.tier))
	})

	ranked := make([]core.Tier, 0, len(hits)+1)
	var why []string
	for _, hi := range hits {
		ranked = append(ranked, hi.tier)
		why = append(why, fmt.Sprintf("%s(%s)", hi.tier, strings.Join(hi.terms, ",")))
	}
	ranked = append(ranked, core.TierSemantic) // always a fallback

	reason := "no tier keywords matched; defaulting to semantic"
	if len(why) > 0 {
		reason = "matched " + strings.Join(why, " ")
	}

	ranked = dedupeTiers(ranked)
	if len(h.Available) > 0 {
		ranked = slices.DeleteFunc(ranked, func(t core.Tier) bool {
			return !slices.Contains(h.Available, t)
		})
		// Everything the router wanted is unimplemented — fall back to whatever
		// tiers do exist rather than returning nothing to retrieve from.
		if len(ranked) == 0 {
			ranked = slices.Clone(h.Available)
			reason += "; no preferred tier registered, using all available"
		}
	}
	return Decision{Tiers: ranked, Reason: reason, Source: "heuristic"}, nil
}

func dedupe(s []string) []string {
	seen := map[string]bool{}
	out := s[:0]
	for _, x := range s {
		if !seen[x] {
			seen[x] = true
			out = append(out, x)
		}
	}
	return out
}

func dedupeTiers(s []core.Tier) []core.Tier {
	seen := map[core.Tier]bool{}
	out := make([]core.Tier, 0, len(s))
	for _, x := range s {
		if !seen[x] {
			seen[x] = true
			out = append(out, x)
		}
	}
	return out
}
