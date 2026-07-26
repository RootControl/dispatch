package main

import (
	"encoding/json"
	"os"
	"slices"
	"testing"
)

func TestFactAcceptsStringOrList(t *testing.T) {
	var single Fact
	if err := json.Unmarshal([]byte(`"Mailwright"`), &single); err != nil {
		t.Fatal(err)
	}
	if len(single) != 1 || single[0] != "Mailwright" {
		t.Errorf("bare string decoded as %v", single)
	}

	var many Fact
	if err := json.Unmarshal([]byte(`["12%","twelve percent"]`), &many); err != nil {
		t.Fatal(err)
	}
	if len(many) != 2 {
		t.Errorf("list decoded as %v", many)
	}

	var bad Fact
	if err := json.Unmarshal([]byte(`42`), &bad); err == nil {
		t.Error("a number should not decode as a fact")
	}
}

// Grading must measure correctness, not phrasing: any accepted surface form of
// the same fact counts.
func TestFactMatchesAnyAlternative(t *testing.T) {
	f := Fact{"twelve percent", "12%", "12 percent"}
	for _, answer := range []string{
		"a reserve of twelve percent remains",
		"roughly 12% held back",
		"about 12 percent",
	} {
		if !f.found(answer) {
			t.Errorf("Fact.found(%q) = false, want true", answer)
		}
	}
	if f.found("a reserve of six percent remains") {
		t.Error("matched a different figure")
	}
}

func TestExtractCitationsSingleMarkers(t *testing.T) {
	answer := "Budget is 4M [semantic:atlas-charter.md#2] and vendor is Mailwright [relational:atlas-vendor-assessment.md#0]."
	got := extractCitations(answer)
	want := []string{"[semantic:atlas-charter.md#2]", "[relational:atlas-vendor-assessment.md#0]"}
	if !slices.Equal(got, want) {
		t.Errorf("markers = %v, want %v", got, want)
	}
}

// Models group citations. gemma4 emitted three in one bracket against a real
// corpus; a single-marker regex finds none and the check passes vacuously.
func TestExtractCitationsGroupedInOneBracket(t *testing.T) {
	answer := "Redis uses 6380 [semantic:redis-config/README.md#0, semantic:redis-config/README.md#1, semantic:redis-config/README.md#6]."
	got := extractCitations(answer)
	want := []string{
		"[semantic:redis-config/README.md#0]",
		"[semantic:redis-config/README.md#1]",
		"[semantic:redis-config/README.md#6]",
	}
	if !slices.Equal(got, want) {
		t.Errorf("grouped markers = %v, want %v", got, want)
	}
}

func TestExtractCitationsIgnoresProse(t *testing.T) {
	for _, s := range []string{"see [the appendix]", "[Note: revised]", "an array[0] index", "[]"} {
		if m := extractCitations(s); len(m) != 0 {
			t.Errorf("extractCitations(%q) = %v, want none", s, m)
		}
	}
}

// A failure must be attributable: pass() requires all three checks, and each
// one failing alone is enough to fail the case.
func TestAnswerScorePass(t *testing.T) {
	good := answerScore{retrieved: true, factsFound: 2, factsTotal: 2}
	if !good.pass() {
		t.Error("a fully correct case should pass")
	}
	for name, s := range map[string]answerScore{
		"retrieval miss": {retrieved: false, factsFound: 2, factsTotal: 2},
		"missing fact":   {retrieved: true, factsFound: 1, factsTotal: 2},
		"bad citation":   {retrieved: true, factsFound: 2, factsTotal: 2, badCitations: []string{"[semantic:nope]"}},
	} {
		if s.pass() {
			t.Errorf("%s should fail", name)
		}
	}
}

// The bundled case file must stay loadable and non-trivial, or the eval
// silently measures nothing.
func TestBundledAnswerCasesAreValid(t *testing.T) {
	data, err := os.ReadFile("../../testdata/eval/answers.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases []answerCase
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	if len(cases) < 5 {
		t.Fatalf("only %d cases", len(cases))
	}
	for i, c := range cases {
		if c.Question == "" {
			t.Errorf("case %d has no question", i)
		}
		if len(c.MustInclude) == 0 {
			t.Errorf("case %d (%q) requires no facts, so it cannot fail", i, c.Question)
		}
		if len(c.ExpectSources) == 0 {
			t.Errorf("case %d (%q) names no expected source, so retrieval is ungraded", i, c.Question)
		}
	}
}
