package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"regexp"
	"strings"

	"github.com/RootControl/dispatch/agent"
	"github.com/RootControl/dispatch/core"
)

// Fact is one required fact, expressed as accepted surface forms. A model may
// write "12%", "twelve percent", or "12 percent" for the same correct fact, and
// grading that as a miss would measure phrasing rather than correctness.
type Fact []string

// UnmarshalJSON accepts either a bare string or a list of alternatives, so
// simple cases stay readable in the case file.
func (f *Fact) UnmarshalJSON(b []byte) error {
	var one string
	if err := json.Unmarshal(b, &one); err == nil {
		*f = Fact{one}
		return nil
	}
	var many []string
	if err := json.Unmarshal(b, &many); err != nil {
		return fmt.Errorf("a fact must be a string or a list of alternatives: %w", err)
	}
	*f = many
	return nil
}

// found reports whether any accepted form appears in the answer.
func (f Fact) found(answer string) bool {
	for _, alt := range f {
		if strings.Contains(answer, strings.ToLower(alt)) {
			return true
		}
	}
	return false
}

func (f Fact) String() string { return strings.Join(f, "|") }

// answerCase is one end-to-end expectation.
type answerCase struct {
	Question string `json:"question"`
	// MustInclude are facts the answer has to state to be correct.
	MustInclude []Fact `json:"must_include"`
	// ExpectSources are document IDs at least one of which must appear in the
	// retrieved evidence. This is what separates a retrieval failure from a
	// generation failure.
	ExpectSources []string `json:"expect_sources"`
	Note          string   `json:"note,omitempty"`
}

// bracketRE finds any bracketed span; pairRE finds tier:source pairs inside one.
//
// Two patterns rather than one because models group citations: asked for
// [tier:source] markers, gemma4 emitted
// "[semantic:a.md#0, semantic:a.md#1, semantic:a.md#6]" — three citations in a
// single bracket. A single-marker regex finds none of them and the citation
// check then passes vacuously, which is worse than failing.
var (
	bracketRE = regexp.MustCompile(`\[[^\]]+\]`)
	pairRE    = regexp.MustCompile(`([a-z]+):([^\s,\]]+)`)
)

// extractCitations returns every citation in text, normalized to the canonical
// [tier:source] form so it can be compared against Result.Cite().
func extractCitations(text string) []string {
	var out []string
	for _, bracket := range bracketRE.FindAllString(text, -1) {
		for _, p := range pairRE.FindAllStringSubmatch(bracket, -1) {
			out = append(out, "["+p[1]+":"+p[2]+"]")
		}
	}
	return out
}

// scored is one case's outcome, decomposed so a failure points at its cause.
type answerScore struct {
	retrieved    bool // a expected source reached the evidence
	factsFound   int
	factsTotal   int
	missing      []string
	badCitations []string // cited markers absent from the evidence
	tiers        []core.Tier
	rounds       int
	err          error
}

func (s answerScore) pass() bool {
	return s.err == nil && s.retrieved && s.factsFound == s.factsTotal && len(s.badCitations) == 0
}

// evalAnswers runs the full loop over labeled cases and reports three numbers
// that fail for different reasons and need different fixes:
//
//	retrieval  — did the right document reach the evidence at all?
//	facts      — given the evidence, did the answer state the required facts?
//	citations  — did every marker in the answer resolve to real evidence?
//
// A single pass/fail rate would hide which half of the system is broken.
func evalAnswers(args []string) error {
	fs := flag.NewFlagSet("eval-answers", flag.ExitOnError)
	casesPath := fs.String("cases", "./testdata/eval/answers.json", "labeled answer cases")
	indexPath := fs.String("index", defaultIndexPath, "index to read")
	topK := fs.Int("k", 5, "results per tier per round")
	maxSteps := fs.Int("max-steps", 3, "maximum retrieve/judge rounds")
	sqlDir := fs.String("sql-dir", "", "CSV tables enabling the structured tier")
	maxHops := fs.Int("max-hops", 2, "relational traversal depth")
	llmRoute := fs.Bool("llm-router", false, "classify with the model instead of keywords")
	rerank := fs.Bool("rerank", false, "rescore the retrieved shortlist with the model")
	verbose := fs.Bool("v", false, "show the answer text for every case")
	if err := fs.Parse(args); err != nil {
		return err
	}

	data, err := os.ReadFile(*casesPath)
	if err != nil {
		return err
	}
	var cases []answerCase
	if err := json.Unmarshal(data, &cases); err != nil {
		return fmt.Errorf("parse %s: %w", *casesPath, err)
	}
	if len(cases) == 0 {
		return fmt.Errorf("no cases in %s", *casesPath)
	}

	st, err := buildStack(stackOptions{
		IndexPath: *indexPath, SQLDir: *sqlDir, MaxHops: *maxHops, Rerank: *rerank,
	})
	if err != nil {
		return err
	}
	loop := &agent.Loop{
		LLM:        st.client,
		Router:     st.router(*llmRoute),
		Retrievers: st.registry,
		MaxSteps:   *maxSteps,
		TopK:       *topK,
	}

	fmt.Printf("=== answer quality: %d cases over %d chunks, tiers %v ===\n\n",
		len(cases), st.store.Len(), st.available)

	ctx := context.Background()
	var retrievedN, factsN, factsTotalN, citeOK, passN int
	for i, c := range cases {
		s := scoreAnswer(ctx, loop, c, *verbose)
		if s.retrieved {
			retrievedN++
		}
		factsN += s.factsFound
		factsTotalN += s.factsTotal
		if len(s.badCitations) == 0 && s.err == nil {
			citeOK++
		}
		if s.pass() {
			passN++
		}
		printAnswerCase(i+1, c, s)
	}

	fmt.Printf("\nretrieval:  %d/%d  (expected document reached the evidence)\n", retrievedN, len(cases))
	fmt.Printf("facts:      %d/%d  (required facts stated in the answer)\n", factsN, factsTotalN)
	fmt.Printf("citations:  %d/%d  (no markers pointing at absent evidence)\n", citeOK, len(cases))
	fmt.Printf("end-to-end: %d/%d\n", passN, len(cases))
	return nil
}

func scoreAnswer(ctx context.Context, loop *agent.Loop, c answerCase, verbose bool) answerScore {
	s := answerScore{factsTotal: len(c.MustInclude)}

	ans, err := loop.Run(ctx, c.Question)
	if err != nil {
		s.err = err
		return s
	}
	if ans.Trace != nil {
		s.rounds = ans.Trace.Rounds
		if len(ans.Trace.Steps) > 0 {
			s.tiers = ans.Trace.Steps[0].Tiers
		}
	}

	// Retrieval: did an expected document reach the evidence? Chunk IDs are
	// "<doc>#<n>", so a prefix match identifies the document.
	cited := map[string]bool{}
	for _, r := range ans.Evidence {
		cited[r.Cite()] = true
		for _, want := range c.ExpectSources {
			if strings.HasPrefix(r.SourceID, want) {
				s.retrieved = true
			}
		}
	}

	lower := strings.ToLower(ans.Text)
	for _, f := range c.MustInclude {
		if f.found(lower) {
			s.factsFound++
		} else {
			s.missing = append(s.missing, f.String())
		}
	}

	// Every marker in the answer must resolve to evidence that was actually
	// retrieved. A marker that does not is a fabricated citation — the failure
	// mode a grounded system exists to prevent, and invisible without this check.
	for _, m := range extractCitations(ans.Text) {
		if !cited[m] {
			s.badCitations = append(s.badCitations, m)
		}
	}

	if verbose {
		fmt.Printf("--- %s\n%s\n\n", c.Question, strings.TrimSpace(ans.Text))
	}
	return s
}

func printAnswerCase(n int, c answerCase, s answerScore) {
	status := "PASS"
	switch {
	case s.err != nil:
		status = "ERROR"
	case !s.retrieved:
		status = "RETRIEVAL"
	case s.factsFound < s.factsTotal:
		status = "FACTS"
	case len(s.badCitations) > 0:
		status = "CITATION"
	}

	fmt.Printf("%-9s %d. %s\n", status, n, truncate(c.Question, 66))
	fmt.Printf("          facts %d/%d  rounds %d  route %v\n", s.factsFound, s.factsTotal, s.rounds, s.tiers)
	if s.err != nil {
		fmt.Printf("          error: %v\n", s.err)
	}
	if !s.retrieved {
		fmt.Printf("          expected one of %v in the evidence\n", c.ExpectSources)
	}
	if len(s.missing) > 0 {
		fmt.Printf("          missing: %s\n", strings.Join(s.missing, ", "))
	}
	if len(s.badCitations) > 0 {
		fmt.Printf("          fabricated citations: %s\n", strings.Join(s.badCitations, " "))
	}
}
