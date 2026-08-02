package agent

import (
	"regexp"
	"slices"
	"strings"

	"github.com/RootControl/dispatch/core"
)

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

// ExtractCitations returns every [tier:source] marker in text, normalized to the
// canonical form Result.Cite produces so it can be compared against evidence.
//
// This lives in the library rather than the eval because both need it and two
// copies would drift — and the copy that drifts is the one that stops catching
// fabricated citations, silently.
func ExtractCitations(text string) []string {
	var out []string
	for _, bracket := range bracketRE.FindAllString(text, -1) {
		for _, p := range pairRE.FindAllStringSubmatch(bracket, -1) {
			out = append(out, "["+p[1]+":"+p[2]+"]")
		}
	}
	return out
}

// CitationReport is the result of resolving an answer's citations against the
// evidence it was generated from.
type CitationReport struct {
	// Resolved are markers that point at evidence actually retrieved.
	Resolved []string
	// Unresolved are markers that point at nothing. These are fabricated: the
	// model invented a source, and a grounded answer with an invented source is
	// worse than an ungrounded one, because it looks checkable.
	Unresolved []string
	// Uncited reports evidence the answer never cited. Not a fault — evidence
	// is over-fetched on purpose — but a high count alongside a confident
	// answer is worth seeing.
	Uncited []string
}

// OK reports whether every marker resolved.
func (r CitationReport) OK() bool { return len(r.Unresolved) == 0 }

// VerifyCitations resolves the markers in text against evidence.
func VerifyCitations(text string, evidence []core.Result) CitationReport {
	known := make(map[string]bool, len(evidence))
	for _, e := range evidence {
		known[e.Cite()] = true
	}

	var rep CitationReport
	cited := make(map[string]bool)
	for _, m := range ExtractCitations(text) {
		if cited[m] {
			continue // the same source cited twice is one citation
		}
		cited[m] = true
		if known[m] {
			rep.Resolved = append(rep.Resolved, m)
		} else {
			rep.Unresolved = append(rep.Unresolved, m)
		}
	}
	for _, e := range evidence {
		if !cited[e.Cite()] {
			rep.Uncited = append(rep.Uncited, e.Cite())
		}
	}
	slices.Sort(rep.Resolved)
	slices.Sort(rep.Unresolved)
	slices.Sort(rep.Uncited)
	return rep
}

// CitationPolicy decides what Loop.Run does about a fabricated citation.
type CitationPolicy int

const (
	// CiteReport records unresolved markers on the Answer and in the trace and
	// returns the answer unchanged. The default: the caller is told, and
	// decides. Silently altering a model's answer is its own kind of dishonesty.
	CiteReport CitationPolicy = iota
	// CiteStrip removes unresolved markers from the answer text. Use when the
	// answer is shown to a person and a marker resolving to nothing is worse
	// than a sentence with no marker. The report still lists what was stripped.
	CiteStrip
	// CiteError fails the answer outright. Use where an unverifiable claim must
	// not leave the process at all.
	CiteError
)

// stripCitations removes the given markers from text and tidies the whitespace
// their removal leaves behind.
//
// Grouped citations are why this is not a plain string replace: a marker may be
// one of several inside a single bracket, so removing "[semantic:b#0]" from
// "[semantic:a#0, semantic:b#0]" has to rewrite the bracket rather than fail to
// match it. A bracket left with nothing in it is removed entirely.
func stripCitations(text string, unresolved []string) string {
	drop := make(map[string]bool, len(unresolved))
	for _, m := range unresolved {
		drop[m] = true
	}
	out := bracketRE.ReplaceAllStringFunc(text, func(bracket string) string {
		pairs := pairRE.FindAllStringSubmatch(bracket, -1)
		if len(pairs) == 0 {
			return bracket // not a citation bracket; leave it alone
		}
		kept := make([]string, 0, len(pairs))
		for _, p := range pairs {
			if m := "[" + p[1] + ":" + p[2] + "]"; !drop[m] {
				kept = append(kept, p[1]+":"+p[2])
			}
		}
		if len(kept) == 0 {
			return ""
		}
		return "[" + strings.Join(kept, ", ") + "]"
	})
	return tidySpacing(out)
}

// tidySpacing repairs the gaps a removed marker leaves: a doubled space, or a
// space stranded before a full stop.
func tidySpacing(s string) string {
	for _, fix := range []struct{ from, to string }{
		{"  ", " "}, {" .", "."}, {" ,", ","}, {" ;", ";"}, {" )", ")"}, {"( ", "("},
	} {
		for strings.Contains(s, fix.from) {
			s = strings.ReplaceAll(s, fix.from, fix.to)
		}
	}
	var lines []string
	for line := range strings.SplitSeq(s, "\n") {
		lines = append(lines, strings.TrimRight(line, " \t"))
	}
	return strings.TrimSpace(strings.Join(lines, "\n"))
}
