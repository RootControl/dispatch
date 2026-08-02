package index

import (
	"slices"
	"strings"
	"testing"
)

const headingDoc = "Preamble before any heading.\n\n" +
	"# Atlas\n\nThe programme replaces the legacy mailer.\n\n" +
	"## Budget\n\nFour million dollars approved.\n\n" +
	"### Contingency\n\nTwelve percent held in reserve.\n\n" +
	"## Staffing\n\nPriya Raman leads engineering.\n\n" +
	"# Risks\n\nSchedule slipped one quarter.\n"

func paths(secs []Section) []string {
	out := make([]string, len(secs))
	for i, s := range secs {
		out[i] = s.Path
	}
	return out
}

func TestSectionsBuildHeadingPaths(t *testing.T) {
	got := paths(Sections(headingDoc))
	want := []string{"", "Atlas", "Atlas > Budget", "Atlas > Budget > Contingency", "Atlas > Staffing", "Risks"}
	if !slices.Equal(got, want) {
		t.Fatalf("paths = %v\nwant %v", got, want)
	}
	// A deeper heading must not leak into a later sibling's path.
	for _, s := range Sections(headingDoc) {
		if s.Path == "Atlas > Staffing" && strings.Contains(s.Body, "Twelve percent") {
			t.Error("contingency text leaked into the staffing section")
		}
	}
}

// A '#' inside a fenced block is a shell or Python comment, not a heading.
// Technical documentation is full of these, and splitting a code block in half
// is worse than not splitting at all.
func TestSectionsIgnoreHeadingsInsideCodeFences(t *testing.T) {
	doc := "# Setup\n\nRun it:\n\n```bash\n# install the thing\nnpm install\n### not a heading\n```\n\nDone.\n"
	secs := Sections(doc)
	if len(secs) != 1 {
		t.Fatalf("got %d sections, want 1: %v", len(secs), paths(secs))
	}
	if !strings.Contains(secs[0].Body, "npm install") || !strings.Contains(secs[0].Body, "### not a heading") {
		t.Errorf("code block was broken up: %q", secs[0].Body)
	}
	if !HasHeadings(doc) {
		t.Error("HasHeadings missed the real heading")
	}
	if HasHeadings("```\n# only a comment\n```\n") {
		t.Error("HasHeadings treated a fenced comment as a heading")
	}
}

// Real documents skip levels constantly. An h1 followed by an h3 should leave
// the middle slot empty, not be treated as malformed.
func TestSectionsToleratesSkippedLevels(t *testing.T) {
	got := paths(Sections("# One\n\na\n\n### Three\n\nb\n\n## Two\n\nc\n"))
	want := []string{"One", "One > Three", "One > Two"}
	if !slices.Equal(got, want) {
		t.Fatalf("paths = %v, want %v", got, want)
	}
}

// Setext headings are deliberately not matched: a line of dashes is also a
// table separator and a horizontal rule, and guessing wrong splits a table.
func TestSectionsLeavesTablesAlone(t *testing.T) {
	doc := "# Data\n\n| a | b |\n|---|---|\n| 1 | 2 |\n"
	secs := Sections(doc)
	if len(secs) != 1 {
		t.Fatalf("got %d sections, want the table intact: %v", len(secs), paths(secs))
	}
	if !strings.Contains(secs[0].Body, "|---|---|") {
		t.Errorf("table separator was consumed: %q", secs[0].Body)
	}
}

func TestSectionsOnPlainProse(t *testing.T) {
	text := "Just a paragraph.\n\nAnd another one.\n"
	secs := Sections(text)
	if len(secs) != 1 || secs[0].Path != "" {
		t.Fatalf("sections = %v, want one unnamed section", paths(secs))
	}
	if HasHeadings(text) {
		t.Error("HasHeadings on prose with no headings")
	}
}

// The point of the feature: a chunk never spans two sections, and carries the
// path of the one it came from.
func TestSplitHeadingsKeepsSectionsApart(t *testing.T) {
	opts := ChunkOptions{TargetTokens: 400, NoOverlap: true, Headings: true}
	chunks := Split(headingDoc, opts)

	for _, c := range chunks {
		if strings.Contains(c, "Four million") && strings.Contains(c, "Priya Raman") {
			t.Errorf("a chunk spans the Budget and Staffing sections:\n%s", c)
		}
	}
	var budget string
	for _, c := range chunks {
		if strings.Contains(c, "Four million") {
			budget = c
		}
	}
	if budget == "" {
		t.Fatal("the budget text was lost")
	}
	if !strings.HasPrefix(budget, "Atlas > Budget") {
		t.Errorf("chunk does not carry its heading path:\n%s", budget)
	}
}

// Turning the flag off must reproduce the previous behaviour exactly, so the
// feature is opt-in in fact and not just in name.
func TestSplitWithoutHeadingsIsUnchanged(t *testing.T) {
	opts := ChunkOptions{TargetTokens: 30, NoOverlap: true}
	flat := Split(headingDoc, opts)
	for _, c := range flat {
		if strings.HasPrefix(c, "Atlas > ") {
			t.Errorf("heading path appeared with Headings off:\n%s", c)
		}
	}
	// And a document with no headings is identical either way.
	prose := "One paragraph here.\n\nAnother paragraph there.\n\nA third for luck.\n"
	on := Split(prose, ChunkOptions{TargetTokens: 10, NoOverlap: true, Headings: true})
	off := Split(prose, ChunkOptions{TargetTokens: 10, NoOverlap: true})
	if !slices.Equal(on, off) {
		t.Errorf("headings changed a document that has none:\n on=%v\noff=%v", on, off)
	}
}

// The prefix costs tokens in every chunk, so it must come out of the budget
// rather than silently pushing chunks over it.
func TestHeadingPrefixComesOutOfTheBudget(t *testing.T) {
	long := "# A Fairly Long Heading Path Indeed\n\n" + strings.Repeat("word ", 200)
	chunks := Split(long, ChunkOptions{TargetTokens: 60, NoOverlap: true, Headings: true})
	if len(chunks) < 2 {
		t.Fatalf("expected the body to split, got %d chunk(s)", len(chunks))
	}
	for _, c := range chunks {
		if got := estimateTokens(c); got > 60+20 {
			t.Errorf("chunk is %d tokens against a 60 target; the prefix was not budgeted", got)
		}
	}
}

// A section whose prose is only a heading must not survive --min-words: the
// path is structure, not content, so it cannot rescue an empty chunk.
func TestHeadingPathDoesNotDefeatMinWords(t *testing.T) {
	doc := "# Contents\n\n# Real Section\n\nThis section has several genuine words of prose in it.\n"
	chunks := Split(doc, ChunkOptions{TargetTokens: 400, NoOverlap: true, Headings: true, MinWords: 5})
	if len(chunks) != 1 {
		t.Fatalf("got %d chunks, want only the one with prose: %q", len(chunks), chunks)
	}
	if !strings.Contains(chunks[0], "genuine words") {
		t.Errorf("kept the wrong chunk: %q", chunks[0])
	}
}
