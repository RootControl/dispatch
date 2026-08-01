package agent

import (
	"context"
	"strings"
	"testing"

	"github.com/RootControl/dispatch/core"
)

func semResult(id, text string) core.Result {
	doc, _, _ := strings.Cut(id, "#")
	return core.Result{
		Tier: core.TierSemantic, SourceID: id, Text: text,
		Meta: map[string]string{"doc": doc},
	}
}

func citesOf(rs []core.Result) []string {
	out := make([]string, len(rs))
	for i, r := range rs {
		out[i] = r.Cite()
	}
	return out
}

// The measured case. Against a real 70-chunk index, "who is staffed on the
// project?" returned CLAUDE.md #0, #1 and #6 in a four-item evidence set: one
// document said three ways, with #0 and #1 adjacent and sharing the default
// 100-token chunk overlap verbatim.
func TestDiversityBreaksUpASingleDocumentDominating(t *testing.T) {
	shared := "the deployment guide covers redis sentinel failover and the mongo replica set"
	candidates := []core.Result{
		semResult("CLAUDE.md#0", shared+" plus notes on the api gateway"),
		semResult("CLAUDE.md#1", shared+" plus notes on the worker queue"),
		semResult("CLAUDE.md#6", "unrelated content about translation files and locale strings"),
		semResult("other.md#0", "staffing roster naming priya raman as the engineering lead"),
	}

	kept, drops := selectDiverse(candidates, Diversity{PerDoc: 2}, 4)

	// #0 and #1 restate each other, so #1 goes as a near-duplicate before the
	// per-doc cap is even reached — which is the better outcome: the cap is a
	// quota, the overlap check is a reason.
	if len(drops) != 1 || drops[0].cite != "[semantic:CLAUDE.md#1]" || drops[0].reason != "near-duplicate" {
		t.Fatalf("drops = %+v, want CLAUDE.md#1 as a near-duplicate", drops)
	}
	if len(kept) != 3 {
		t.Fatalf("kept %v, want 3", citesOf(kept))
	}
	// The point of the exercise: the other document now reaches the evidence
	// instead of being crowded out by a restatement.
	if kept[len(kept)-1].SourceID != "other.md#0" {
		t.Errorf("kept %v, want the other document to take the freed slot", citesOf(kept))
	}
}

// With the near-duplicate check unable to fire, the per-doc cap is what stops
// one file taking every slot. Kept separate from the test above so a change
// that disabled one mechanism cannot be masked by the other.
func TestDiversityPerDocCapAlone(t *testing.T) {
	candidates := []core.Result{
		semResult("CLAUDE.md#0", "redis sentinel failover configuration and port assignments"),
		semResult("CLAUDE.md#1", "translation files and locale string extraction workflow"),
		semResult("CLAUDE.md#6", "docker compose overrides for the development environment"),
		semResult("other.md#0", "staffing roster naming priya raman as the engineering lead"),
	}
	kept, drops := selectDiverse(candidates, Diversity{PerDoc: 2}, 4)

	if len(kept) != 3 {
		t.Fatalf("kept %v, want 3 with a per-doc cap of 2", citesOf(kept))
	}
	if len(drops) != 1 || drops[0].cite != "[semantic:CLAUDE.md#6]" || drops[0].reason != "doc cap" {
		t.Fatalf("drops = %+v, want the third CLAUDE.md chunk over the cap", drops)
	}
	if kept[len(kept)-1].SourceID != "other.md#0" {
		t.Errorf("kept %v, want the other document in", citesOf(kept))
	}
}

func TestDiversityDropsNearDuplicates(t *testing.T) {
	// Two chunks sharing an overlapping boundary: most tokens in common.
	a := "the contingency reserve stands at twelve percent of the approved budget figure"
	b := "the contingency reserve stands at twelve percent of the approved budget total"
	c := "vendor mailwright renews the support contract every january without review"

	kept, drops := selectDiverse([]core.Result{
		semResult("a.md#0", a), semResult("a.md#1", b), semResult("b.md#0", c),
	}, Diversity{}, 5)

	if len(kept) != 2 {
		t.Fatalf("kept %v, want the duplicate pair collapsed to one", citesOf(kept))
	}
	if kept[0].SourceID != "a.md#0" || kept[1].SourceID != "b.md#0" {
		t.Errorf("kept %v, want the first arrival and the distinct item", citesOf(kept))
	}
	if len(drops) != 1 || drops[0].reason != "near-duplicate" {
		t.Errorf("drops = %+v", drops)
	}
}

// Redundancy crosses tiers: a summary built from a passage restates it under a
// different citation, so both look like independent sources and are not.
func TestDiversityCatchesSummaryRestatingItsSource(t *testing.T) {
	passage := "atlas migration budget four million dollars approved by the steering committee in march"
	summary := "the steering committee approved the atlas migration budget of four million dollars in march"

	kept, _ := selectDiverse([]core.Result{
		semResult("charter.md#2", passage),
		{Tier: core.TierHierarchical, SourceID: "L1-3", Text: summary},
	}, Diversity{Threshold: 0.5}, 5)

	if len(kept) != 1 {
		t.Fatalf("kept %v, want the summary suppressed as a restatement", citesOf(kept))
	}
}

// Distinct content must survive: suppression that is too eager is worse than
// none, because it discards evidence the answer needed.
func TestDiversityKeepsGenuinelyDifferentEvidence(t *testing.T) {
	candidates := []core.Result{
		semResult("a.md#0", "the approved budget is four million dollars for the fiscal year"),
		semResult("b.md#0", "priya raman leads engineering and reports to the programme director"),
		semResult("c.md#0", "the vendor contract renews every january unless cancelled in december"),
		{Tier: core.TierRelational, SourceID: "graph", Text: "mailwright supplies the statement mailer component"},
	}
	kept, drops := selectDiverse(candidates, Diversity{}, 10)
	if len(kept) != len(candidates) {
		t.Fatalf("kept %v of %d, dropped %+v — none of these restate each other",
			citesOf(kept), len(candidates), drops)
	}
}

// Items with no document metadata and no "#" — summaries, graph nodes — are
// each their own document, or a per-doc cap would collapse the whole tier.
func TestDiversityPerDocDoesNotCollapseDerivedTiers(t *testing.T) {
	candidates := []core.Result{
		{Tier: core.TierHierarchical, SourceID: "L1-0", Text: "themes of scheduling and vendor risk"},
		{Tier: core.TierHierarchical, SourceID: "L1-1", Text: "staffing levels and hiring plans by quarter"},
		{Tier: core.TierHierarchical, SourceID: "L2-0", Text: "budget approvals and reserve policy overall"},
	}
	kept, _ := selectDiverse(candidates, Diversity{PerDoc: 1}, 10)
	if len(kept) != 3 {
		t.Fatalf("kept %v, want all three: distinct summaries are not one document", citesOf(kept))
	}
}

func TestSourceDoc(t *testing.T) {
	cases := map[string]core.Result{
		"semantic:a.md":       semResult("a.md#3", "text"),
		"semantic:b.md":       {Tier: core.TierSemantic, SourceID: "b.md#0", Text: "no meta"},
		"[hierarchical:L1-0]": {Tier: core.TierHierarchical, SourceID: "L1-0", Text: "summary"},
	}
	for want, r := range cases {
		if got := sourceDoc(r); got != want {
			t.Errorf("sourceDoc(%s) = %q, want %q", r.Cite(), got, want)
		}
	}
	// Two tiers citing the same document ID are different sources, so the cap
	// applies to each separately.
	sem := core.Result{Tier: core.TierSemantic, SourceID: "a.md#0"}
	mem := core.Result{Tier: core.TierMemory, SourceID: "a.md#0"}
	if sourceDoc(sem) == sourceDoc(mem) {
		t.Error("the same source ID under two tiers collapsed to one document")
	}
}

func TestJaccard(t *testing.T) {
	ab := contentTokens("alpha beta gamma delta")
	if got := jaccard(ab, ab); got != 1 {
		t.Errorf("identical sets = %v, want 1", got)
	}
	if got := jaccard(ab, contentTokens("epsilon zeta eta theta")); got != 0 {
		t.Errorf("disjoint sets = %v, want 0", got)
	}
	if got := jaccard(ab, nil); got != 0 {
		t.Errorf("empty set = %v, want 0", got)
	}
	// Stopwords must not make unrelated passages look similar.
	x := contentTokens("the budget was approved and it will be there for them")
	y := contentTokens("the vendor was selected and it will be there for them")
	if got := jaccard(x, y); got >= 0.6 {
		t.Errorf("stopword-heavy unrelated passages scored %v; stopwords are leaking in", got)
	}
}

// Suppression is off unless asked for: it discards retrieved evidence, and
// doing that to someone who did not ask is the wrong default.
func TestLoopDiversityIsOptIn(t *testing.T) {
	dup := "the contingency reserve stands at twelve percent of the approved budget"
	sem := &stubRetriever{
		tier: core.TierSemantic,
		byQuery: func(string) []core.Result {
			return []core.Result{
				semResult("a.md#0", dup+" figure"),
				semResult("a.md#1", dup+" total"),
			}
		},
	}

	off := loopOver(sufficientJudge(), sem)
	answer, err := off.Run(context.Background(), "what is the reserve?")
	if err != nil {
		t.Fatal(err)
	}
	if len(answer.Evidence) != 2 {
		t.Fatalf("evidence = %v, want both items with diversity off", citesOf(answer.Evidence))
	}

	on := loopOver(sufficientJudge(), sem)
	on.Diversity = Diversity{Enabled: true}
	answer, err = on.Run(context.Background(), "what is the reserve?")
	if err != nil {
		t.Fatal(err)
	}
	if len(answer.Evidence) != 1 {
		t.Fatalf("evidence = %v, want the duplicate suppressed", citesOf(answer.Evidence))
	}
	if !strings.Contains(answer.Trace.String(), "dropped 1") {
		t.Errorf("trace does not record the drop:\n%s", answer.Trace)
	}
}

// Suppression must choose which items fill the budget, not inherit whatever
// arrived first: a duplicate that consumed a slot before suppression ran would
// leave the budget short.
func TestDiversityFreesBudgetForDistinctEvidence(t *testing.T) {
	dup := "the contingency reserve stands at twelve percent of the approved budget"
	sem := &stubRetriever{
		tier: core.TierSemantic,
		byQuery: func(string) []core.Result {
			return []core.Result{
				semResult("a.md#0", dup+" figure"),
				semResult("a.md#1", dup+" total"),
				semResult("b.md#0", "priya raman leads engineering and reports to the director"),
			}
		},
	}
	loop := loopOver(sufficientJudge(), sem)
	loop.Diversity = Diversity{Enabled: true}
	loop.MaxEvidence = 2

	answer, err := loop.Run(context.Background(), "reserve and staffing?")
	if err != nil {
		t.Fatal(err)
	}
	got := citesOf(answer.Evidence)
	if len(got) != 2 {
		t.Fatalf("evidence = %v, want 2", got)
	}
	if got[1] != "[semantic:b.md#0]" {
		t.Errorf("evidence = %v, want the distinct item to take the freed slot", got)
	}
}
