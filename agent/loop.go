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
	tr.add(Step{Kind: StepRoute, Tiers: tierSet, Detail: reason})

	query := question
	var evidence []core.Result

	for step := 1; step <= maxSteps; step++ {
		tr.Rounds = step
		got := l.fanOut(ctx, tierSet, query, topK, tr)
		evidence = merge(evidence, got, maxEvidence)

		// Judging on the final step would spend a call on a verdict we cannot
		// act on, so skip it and go straight to generation.
		if step == maxSteps {
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
		if t := core.Tier(strings.TrimSpace(j.NextTier)); t != "" {
			if _, ok := l.Retrievers[t]; ok {
				tierSet = []core.Tier{t}
			}
		}
	}

	if len(evidence) == 0 {
		return Answer{Trace: tr}, fmt.Errorf("agent: no evidence retrieved for %q", question)
	}

	var memoryBlock string
	if l.Memory != nil {
		memoryBlock = l.Memory.CoreBlock()
	}

	start := time.Now()
	text, err := generate(ctx, l.LLM, question, memoryBlock, evidence)
	tr.Calls++
	tr.add(Step{Kind: StepGenerate, Results: len(evidence), Elapsed: time.Since(start)})
	if err != nil {
		return Answer{Trace: tr}, err
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

	return Answer{Text: text, Evidence: evidence, Trace: tr}, nil
}

// fanOut retrieves from every named tier concurrently. A tier that errors is
// recorded and skipped rather than failing the whole round: partial evidence
// still beats none, and the judge will see the gap.
func (l *Loop) fanOut(ctx context.Context, tierSet []core.Tier, query string, topK int, tr *Trace) []core.Result {
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
			got, err := r.Retrieve(ctx, core.Query{Text: query, TopK: topK})
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
