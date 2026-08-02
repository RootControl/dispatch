package agent

import (
	"slices"
	"strings"
	"testing"

	"github.com/RootControl/dispatch/core"
)

// These moved here with ExtractCitations: the eval and the runtime now share
// one implementation, so they share its tests. Two copies would drift, and the
// copy that drifts is the one that stops catching fabricated citations.

func TestExtractCitationsSingleMarkers(t *testing.T) {
	answer := "Budget is 4M [semantic:atlas-charter.md#2] and vendor is Mailwright [relational:atlas-vendor-assessment.md#0]."
	got := ExtractCitations(answer)
	want := []string{"[semantic:atlas-charter.md#2]", "[relational:atlas-vendor-assessment.md#0]"}
	if !slices.Equal(got, want) {
		t.Errorf("markers = %v, want %v", got, want)
	}
}

// Models group citations. gemma4 emitted three in one bracket against a real
// corpus; a single-marker regex finds none and the check passes vacuously.
func TestExtractCitationsGroupedInOneBracket(t *testing.T) {
	answer := "Redis uses 6380 [semantic:redis-config/README.md#0, semantic:redis-config/README.md#1, semantic:redis-config/README.md#6]."
	got := ExtractCitations(answer)
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
		if m := ExtractCitations(s); len(m) != 0 {
			t.Errorf("ExtractCitations(%q) = %v, want none", s, m)
		}
	}
}

func ev(tier core.Tier, id string) core.Result {
	return core.Result{Tier: tier, SourceID: id, Text: "evidence " + id}
}

func TestVerifyCitations(t *testing.T) {
	evidence := []core.Result{
		ev(core.TierSemantic, "a.md#0"),
		ev(core.TierSemantic, "b.md#3"),
		ev(core.TierRelational, "graph"),
	}

	t.Run("all resolve", func(t *testing.T) {
		rep := VerifyCitations("Budget is 4M [semantic:a.md#0] per [relational:graph].", evidence)
		if !rep.OK() {
			t.Fatalf("unresolved = %v, want none", rep.Unresolved)
		}
		if !slices.Equal(rep.Resolved, []string{"[relational:graph]", "[semantic:a.md#0]"}) {
			t.Errorf("resolved = %v", rep.Resolved)
		}
		if !slices.Equal(rep.Uncited, []string{"[semantic:b.md#3]"}) {
			t.Errorf("uncited = %v, want the evidence the answer never used", rep.Uncited)
		}
	})

	t.Run("fabricated source", func(t *testing.T) {
		rep := VerifyCitations("Budget is 4M [semantic:invented.md#9].", evidence)
		if rep.OK() {
			t.Fatal("a citation pointing at nothing was reported as fine")
		}
		if !slices.Equal(rep.Unresolved, []string{"[semantic:invented.md#9]"}) {
			t.Errorf("unresolved = %v", rep.Unresolved)
		}
	})

	// The right source under the wrong tier resolves to nothing: citations are
	// resolved by the whole marker, because that is what a reader would follow.
	t.Run("right source wrong tier", func(t *testing.T) {
		rep := VerifyCitations("See [hierarchical:a.md#0].", evidence)
		if rep.OK() {
			t.Fatal("a marker naming a tier that did not produce it should not resolve")
		}
	})

	t.Run("same source cited twice counts once", func(t *testing.T) {
		rep := VerifyCitations("[semantic:a.md#0] and again [semantic:a.md#0].", evidence)
		if len(rep.Resolved) != 1 {
			t.Errorf("resolved = %v, want one entry", rep.Resolved)
		}
	})

	t.Run("no citations at all", func(t *testing.T) {
		rep := VerifyCitations("The corpus does not say.", evidence)
		if !rep.OK() {
			t.Error("an answer with no markers has nothing to fabricate")
		}
		if len(rep.Uncited) != len(evidence) {
			t.Errorf("uncited = %v, want every item", rep.Uncited)
		}
	})
}

func TestStripCitations(t *testing.T) {
	cases := []struct {
		name       string
		text       string
		unresolved []string
		want       string
	}{
		{"lone marker", "Budget is 4M [semantic:fake#0].", []string{"[semantic:fake#0]"}, "Budget is 4M."},
		{"keeps the good one", "4M [semantic:real#0] and 2M [semantic:fake#0].",
			[]string{"[semantic:fake#0]"}, "4M [semantic:real#0] and 2M."},
		// The grouped case is why this is not a string replace: removing one
		// marker from inside a shared bracket has to rewrite the bracket.
		{"one of a group", "Sources [semantic:real#0, semantic:fake#0, semantic:real#1].",
			[]string{"[semantic:fake#0]"}, "Sources [semantic:real#0, semantic:real#1]."},
		{"whole group goes", "Sources [semantic:fake#0, semantic:fake#1].",
			[]string{"[semantic:fake#0]", "[semantic:fake#1]"}, "Sources."},
		{"prose brackets survive", "See [the appendix] and [semantic:fake#0].",
			[]string{"[semantic:fake#0]"}, "See [the appendix] and."},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := stripCitations(tc.text, tc.unresolved); got != tc.want {
				t.Errorf("got  %q\nwant %q", got, tc.want)
			}
		})
	}
}

func TestTidySpacingKeepsLineStructure(t *testing.T) {
	got := tidySpacing("line one  here .\n\nline two ,  yes")
	if !strings.Contains(got, "\n\n") {
		t.Errorf("paragraph break lost: %q", got)
	}
	if strings.Contains(got, "  ") || strings.Contains(got, " .") {
		t.Errorf("spacing not tidied: %q", got)
	}
}
