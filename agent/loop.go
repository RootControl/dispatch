package agent

import (
	"context"
	"fmt"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/RootControl/dispatch/core"
	"github.com/RootControl/dispatch/llm"
	"github.com/RootControl/dispatch/router"
)

// Loop drives retrieval as a cycle rather than a pipeline stage:
//
//	route -> fan-out retrieve -> judge sufficiency -> refine toward the gap -> repeat
//
// It stops when the judge calls the evidence sufficient or the step budget runs
// out, then generates a cited answer. The judge is the load-bearing part: if it
// is lenient, every question one-shots and this degenerates into plain RAG with
// extra latency. Trace.Refined() reports whether refinement actually happened.
type Loop struct {
	LLM        llm.LLM
	Router     router.Router
	Retrievers map[core.Tier]core.Retriever

	MaxSteps    int // retrieve/judge rounds; default 3
	TopK        int // results per tier per round; default 5
	MaxEvidence int // evidence items carried into generation; default 12

	// Memory, when set, puts the core block in the generation prompt. Register
	// it in Retrievers as well to let the loop search archival memory.
	Memory *Memory
	// WriteBack extracts a durable takeaway after answering and stores it.
	// Costs one extra LLM call per question.
	WriteBack bool

	// OnDelta, when set and when the LLM implements llm.Streamer, receives the
	// answer in fragments as the model produces them. Only generation streams:
	// the judge and the expander produce JSON nobody wants to watch assemble.
	//
	// Answer.Text still holds the whole answer, so a caller that streams to a
	// terminal and a caller that does not get the same value back. Note the
	// ordering that follows from that: citations are verified after the last
	// fragment has already been printed, so a streamed answer can show a
	// fabricated citation before the warning about it arrives.
	OnDelta func(string)

	// MaxCalls caps the LLM calls one Run may make — judge, expand, generate and
	// write-back together. Zero means no cap.
	//
	// MaxSteps bounds rounds, which is not the same thing: enabling an expander
	// adds a call per round, write-back adds one at the end, and a judge that
	// never returns "sufficient" spends the full budget every time. This is the
	// bound on what a single question can cost, expressed in the unit that is
	// actually billed.
	//
	// Hitting it is not an error. The loop stops retrieving and answers from
	// what it has, because a budget is a limit on effort rather than a
	// declaration that the question was bad.
	MaxCalls int

	// Expander, when set, rewrites each query into text better suited to dense
	// retrieval before the fan-out. Costs one LLM call per round. See HyDE.
	Expander Expander

	// Diversity, when enabled, drops evidence that restates evidence already
	// held — overlapping adjacent chunks, or a summary of a passage alongside
	// the passage. Off by default: it discards retrieved evidence, and doing
	// that silently to someone who did not ask is the wrong default.
	Diversity Diversity

	// CitationPolicy decides what happens when the model cites a source that is
	// not in the evidence. Default CiteReport: recorded on the Answer and in
	// the trace, answer returned unchanged.
	CitationPolicy CitationPolicy

	// Filter scopes every retrieval to chunks whose metadata matches. Tiers
	// that cannot apply it — anything not implementing core.Filterable — are
	// dropped from the fan-out rather than allowed to answer from outside the
	// scope, and the trace records which. A filter that leaves no tier standing
	// is an error, not a silent empty answer.
	Filter core.Filter
}

func (l *Loop) defaults() (maxSteps, topK, maxEvidence int) {
	maxSteps, topK, maxEvidence = l.MaxSteps, l.TopK, l.MaxEvidence
	if maxSteps <= 0 {
		maxSteps = 3
	}
	if topK <= 0 {
		topK = 5
	}
	if maxEvidence <= 0 {
		maxEvidence = 12
	}
	return
}

// Run executes the loop and returns a cited answer with its trace.
func (l *Loop) Run(ctx context.Context, question string) (Answer, error) {
	maxSteps, topK, maxEvidence := l.defaults()
	tr := &Trace{Question: question}

	decision, err := l.Router.Route(ctx, question)
	if err != nil {
		return Answer{}, fmt.Errorf("agent: route: %w", err)
	}
	tierSet := decision.Tiers
	reason := decision.Reason

	// Memory is always worth consulting when registered. A keyword router cannot
	// know what is in memory, so leaving this to routing means archival memory is
	// searched only by luck — and the whole point of write-back is that earlier
	// work informs later questions.
	if l.Memory != nil {
		if _, ok := l.Retrievers[core.TierMemory]; ok && !slices.Contains(tierSet, core.TierMemory) {
			tierSet = append(tierSet, core.TierMemory)
			reason += "; memory always consulted"
		}
	}
	// Drop tiers that cannot honour the filter. Doing this once, before the
	// first round, keeps the judge's view of "available tiers" honest too — it
	// should not be told to refine into a tier the filter has ruled out.
	if !l.Filter.Empty() {
		kept, dropped := l.filterable(tierSet)
		if len(kept) == 0 {
			return Answer{Trace: tr}, fmt.Errorf(
				"agent: filter %v leaves no usable tier (none of %v applies metadata filters)", l.Filter, tierSet)
		}
		if len(dropped) > 0 {
			reason += fmt.Sprintf("; %v skipped: cannot filter", dropped)
		}
		tierSet = kept
	}
	tr.add(Step{Kind: StepRoute, Tiers: tierSet, Detail: reason})

	query := question
	var evidence []core.Result

	for step := 1; step <= maxSteps; step++ {
		// Check the deadline explicitly. fanOut records a tier's error and
		// carries on, which is right for one tier failing and wrong for a
		// cancelled context: every tier would fail, evidence would be empty,
		// and the loop would report "no evidence retrieved" — a timeout
		// misattributed as an empty corpus.
		if err := ctx.Err(); err != nil {
			return Answer{Trace: tr}, fmt.Errorf("agent: %w", err)
		}
		tr.Rounds = step

		// Expand per round, not once: the judge's refined query is a different
		// question and deserves its own hypothetical. A failure costs the
		// improvement, not the answer.
		expanded := ""
		if l.Expander != nil && l.canSpend(tr) {
			var err error
			expanded, err = l.Expander.Expand(ctx, query)
			tr.Calls++
			switch {
			case err != nil:
				tr.add(Step{Kind: StepExpand, Detail: "failed, retrieving as written: " + err.Error()})
			case expanded == "":
				tr.add(Step{Kind: StepExpand, Detail: "no expansion made"})
			default:
				tr.add(Step{Kind: StepExpand, Detail: truncate(expanded, 70)})
			}
		}

		got := l.fanOut(ctx, tierSet, query, expanded, topK, tr)
		evidence = l.mergeEvidence(evidence, got, maxEvidence, tr)

		// Judging on the final step would spend a call on a verdict we cannot
		// act on, so skip it and go straight to generation.
		if step == maxSteps {
			break
		}
		if !l.canSpend(tr) {
			tr.add(Step{Kind: StepJudge, Sufficient: true,
				Detail: fmt.Sprintf("call budget reached (%d), answering from what was retrieved", l.MaxCalls)})
			break
		}
		j, err := l.judge(ctx, question, query, evidence)
		tr.Calls++
		if err != nil {
			// A failed judge should not sink an answer we can already give.
			tr.add(Step{Kind: StepJudge, Sufficient: true, Detail: "judge failed, proceeding: " + err.Error()})
			break
		}
		tr.add(Step{Kind: StepJudge, Sufficient: j.Sufficient, Gap: j.Gap, Query: j.RefinedQuery})
		if j.Sufficient {
			break
		}

		// Refine toward the gap. Both fields are advisory: keep the current
		// query/tiers when the judge leaves them blank or names a tier that is
		// not registered.
		if q := strings.TrimSpace(j.RefinedQuery); q != "" {
			query = q
		}
		// Promote the judge's suggested tier to the front, but keep the rest.
		// Replacing the set outright narrows the search permanently: observed on
		// a two-part question where the judge named the relational tier, the
		// loop dropped semantic — which held the missing fact — and then spent
		// its remaining rounds re-querying the same graph edges and learning
		// nothing. A suggestion is a reordering, not an exclusion.
		//
		// The judge may name a registered tier the router did not pick, which is
		// how it corrects a misroute — so this admits tiers outside tierSet. The
		// one thing it must not readmit is a tier the filter ruled out.
		if t := core.Tier(strings.TrimSpace(j.NextTier)); t != "" {
			if ok := l.canUse(t); ok {
				promoted := []core.Tier{t}
				for _, existing := range tierSet {
					if existing != t {
						promoted = append(promoted, existing)
					}
				}
				tierSet = promoted
			}
		}
	}

	if err := ctx.Err(); err != nil {
		return Answer{Trace: tr}, fmt.Errorf("agent: %w", err)
	}
	if len(evidence) == 0 {
		return Answer{Trace: tr}, fmt.Errorf("agent: no evidence retrieved for %q", question)
	}

	var memoryBlock string
	if l.Memory != nil {
		memoryBlock = l.Memory.CoreBlock()
	}

	start := time.Now()
	text, err := generate(ctx, l.LLM, question, memoryBlock, evidence, l.OnDelta)
	tr.Calls++
	tr.add(Step{Kind: StepGenerate, Results: len(evidence), Elapsed: time.Since(start)})
	if err != nil {
		return Answer{Trace: tr}, err
	}

	// Resolve the answer's citations against the evidence it was given. The
	// eval has always checked this; the library did not, so a fabricated
	// citation reached the caller with nothing to distinguish it from a real
	// one. It costs no LLM call — the answer and the evidence are both in hand.
	report := VerifyCitations(text, evidence)
	verify := Step{Kind: StepVerify, Results: len(report.Resolved)}
	if !report.OK() {
		verify.Detail = "FABRICATED: " + strings.Join(report.Unresolved, " ")
	}
	tr.add(verify)
	switch {
	case report.OK():
	case l.CitationPolicy == CiteError:
		return Answer{Text: text, Evidence: evidence, Trace: tr, Citations: report},
			fmt.Errorf("agent: answer cites %d source(s) not in evidence: %s",
				len(report.Unresolved), strings.Join(report.Unresolved, " "))
	case l.CitationPolicy == CiteStrip:
		text = stripCitations(text, report.Unresolved)
	}

	// Write-back is best-effort: a failure here means the next session starts
	// colder, not that this answer is wrong, so it is recorded and not returned.
	if l.Memory != nil && l.WriteBack {
		takeaway, err := l.Memory.WriteBack(ctx, question, text)
		tr.Calls++
		switch {
		case err != nil:
			tr.add(Step{Kind: StepRemember, Detail: "failed: " + err.Error()})
		case takeaway == "":
			tr.add(Step{Kind: StepRemember, Detail: "nothing worth remembering"})
		default:
			tr.add(Step{Kind: StepRemember, Detail: truncate(takeaway, 70)})
		}
	}

	return Answer{Text: text, Evidence: evidence, Trace: tr, Citations: report}, nil
}

// canSpend reports whether an optional call — an expansion or a judge verdict —
// still fits in the budget.
//
// It reserves one call for generation. Spending the last of the budget on a
// judge would leave the loop with evidence it cannot turn into an answer, which
// is the worst possible place to run out: everything was paid for and nothing
// was produced.
func (l *Loop) canSpend(tr *Trace) bool {
	if l.MaxCalls <= 0 {
		return true
	}
	return tr.Calls+1 < l.MaxCalls
}

// canUse reports whether a tier is registered and, when a filter is set, able
// to apply it.
func (l *Loop) canUse(t core.Tier) bool {
	r, ok := l.Retrievers[t]
	if !ok {
		return false
	}
	if l.Filter.Empty() {
		return true
	}
	_, filterable := r.(core.Filterable)
	return filterable
}

// filterable splits a tier set into those whose retriever applies Query.Filter
// and those that do not. An unregistered tier stays in the kept set so it
// produces its usual "no retriever registered" trace entry rather than being
// silently reclassified as a filtering problem.
func (l *Loop) filterable(tierSet []core.Tier) (kept, dropped []core.Tier) {
	for _, t := range tierSet {
		r, ok := l.Retrievers[t]
		if _, isFilterable := r.(core.Filterable); !ok || isFilterable {
			kept = append(kept, t)
		} else {
			dropped = append(dropped, t)
		}
	}
	return kept, dropped
}

// fanOut retrieves from every named tier concurrently. A tier that errors is
// recorded and skipped rather than failing the whole round: partial evidence
// still beats none, and the judge will see the gap.
func (l *Loop) fanOut(ctx context.Context, tierSet []core.Tier, query, expanded string, topK int, tr *Trace) []core.Result {
	type outcome struct {
		tier    core.Tier
		results []core.Result
		err     error
		elapsed time.Duration
	}
	outcomes := make([]outcome, len(tierSet))
	var wg sync.WaitGroup
	for i, t := range tierSet {
		r, ok := l.Retrievers[t]
		if !ok {
			outcomes[i] = outcome{tier: t, err: fmt.Errorf("no retriever registered")}
			continue
		}
		wg.Add(1)
		go func(i int, t core.Tier, r core.Retriever) {
			defer wg.Done()
			start := time.Now()
			got, err := r.Retrieve(ctx, core.Query{
				Text: query, Expanded: expanded, TopK: topK, Filter: l.Filter})
			outcomes[i] = outcome{tier: t, results: got, err: err, elapsed: time.Since(start)}
		}(i, t, r)
	}
	wg.Wait()

	var all []core.Result
	for _, o := range outcomes {
		st := Step{Kind: StepRetrieve, Tier: o.tier, Query: query, Results: len(o.results), Elapsed: o.elapsed}
		if o.err != nil {
			st.Detail = "error: " + o.err.Error()
		}
		tr.add(st)
		all = append(all, o.results...)
	}
	return all
}

// mergeEvidence folds a round's results into the evidence set, applying
// near-duplicate suppression when it is enabled.
//
// Suppression runs over the whole set rather than only the new arrivals,
// because redundancy is a property of the pair: a chunk that restates one held
// since round one is just as wasteful as two arriving together. Re-selecting
// each round costs a Jaccard over at most MaxEvidence items and keeps the rule
// "no two items in the evidence say the same thing" true rather than
// approximately true.
func (l *Loop) mergeEvidence(existing, incoming []core.Result, maxEvidence int, tr *Trace) []core.Result {
	if !l.Diversity.Enabled {
		return merge(existing, incoming, maxEvidence)
	}
	// Merge with no cap first, so suppression chooses which items fill the
	// budget rather than inheriting whatever arrived before it was full.
	all := merge(existing, incoming, len(existing)+len(incoming))
	kept, drops := selectDiverse(all, l.Diversity, maxEvidence)
	if len(drops) > 0 {
		var b strings.Builder
		for i, d := range drops {
			if i == 4 {
				fmt.Fprintf(&b, " ... +%d more", len(drops)-4)
				break
			}
			if i > 0 {
				b.WriteByte(' ')
			}
			fmt.Fprintf(&b, "%s (%s)", d.cite, d.reason)
		}
		tr.add(Step{Kind: StepDiverse, Results: len(kept),
			Detail: fmt.Sprintf("dropped %d: %s", len(drops), b.String())})
	}
	return kept
}

// merge appends new results to existing evidence, dropping duplicates by
// citation identity and capping the total.
//
// Order is preserved rather than sorted by score: Result.Score is tier-local and
// explicitly not comparable across tiers, so sorting would rank a BM25 score
// against a graph hop count. Arrival order follows the router's tier ranking,
// which is a meaningful ordering.
func merge(existing, incoming []core.Result, maxEvidence int) []core.Result {
	seen := make(map[string]bool, len(existing))
	for _, r := range existing {
		seen[r.Cite()] = true
	}
	for _, r := range incoming {
		if len(existing) >= maxEvidence {
			break
		}
		if seen[r.Cite()] {
			continue
		}
		seen[r.Cite()] = true
		existing = append(existing, r)
	}
	return existing
}

// judgment is the judge's structured verdict.
type judgment struct {
	Sufficient   bool   `json:"sufficient"`
	Gap          string `json:"gap"`
	RefinedQuery string `json:"refined_query"`
	NextTier     string `json:"next_tier"`
}

const judgeSystem = `You audit whether retrieved evidence is sufficient to answer a question COMPLETELY.

Be strict. Set "sufficient" to true ONLY if every part of the question can be answered
from the evidence alone, with specifics. If the question has multiple parts, all of them
must be covered. If answering would require any fact not present in the evidence, it is
NOT sufficient.

When it is not sufficient, name the single most important missing fact in "gap", and write
a search query targeting exactly that gap in "refined_query". The refined query should use
different wording than the query that just ran — repeating it will retrieve the same thing.

Reply with a JSON object only:
{"sufficient": <bool>, "gap": "<missing fact, or empty>", "refined_query": "<query, or empty>", "next_tier": "<tier name, or empty>"}`

// judge asks whether the evidence answers the question, and if not, what is
// missing and how to search for it.
func (l *Loop) judge(ctx context.Context, question, lastQuery string, evidence []core.Result) (judgment, error) {
	tiers := make([]string, 0, len(l.Retrievers))
	for t := range l.Retrievers {
		tiers = append(tiers, string(t))
	}
	slices.Sort(tiers)

	user := fmt.Sprintf(
		"Question: %s\n\nThe query just run was: %s\n\nAvailable tiers for next_tier: %s\n\n<evidence>\n%s\n</evidence>",
		question, lastQuery, strings.Join(tiers, ", "), FormatEvidence(evidence))

	var j judgment
	err := l.LLM.ChatJSON(ctx, []llm.Message{llm.System(judgeSystem), llm.User(user)}, &j)
	return j, err
}
