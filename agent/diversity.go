package agent

import (
	"strings"
	"unicode"

	"github.com/RootControl/dispatch/core"
)

// Diversity controls how hard the loop works to keep the evidence set from
// filling up with the same content twice.
//
// The problem is concrete. merge dedups by citation identity, which catches
// nothing but literally the same chunk. Measured against a real 70-chunk index,
// "who is staffed on the project?" returned CLAUDE.md#0, #1 and #6 in a
// four-item evidence set — three chunks from one document, two of them
// adjacent and therefore sharing the default 100-token overlap verbatim. The
// evidence budget bought one document said three ways.
//
// It compounds across tiers: a passage can arrive as a semantic chunk and again
// inside a hierarchical summary that was built from it, under two citations
// that resolve to two different sources saying the same thing.
type Diversity struct {
	// Threshold is the Jaccard token overlap above which two evidence items are
	// treated as the same content, 0 to 1. Zero disables suppression entirely.
	// 0.6 is the default the loop uses when Enabled is set: adjacent chunks
	// under the default 100-token overlap score well below it, so this catches
	// genuine restatement rather than mere adjacency.
	Threshold float64
	// PerDoc caps how many items one source document may contribute. Zero means
	// no cap. This is the blunter instrument and the more effective one on the
	// case above, where the issue was one document winning four slots.
	PerDoc int
	// Enabled turns suppression on. It is a separate field so a zero Threshold
	// means "use the default" rather than "silently do nothing".
	Enabled bool
}

func (d Diversity) withDefaults() Diversity {
	if d.Threshold <= 0 {
		d.Threshold = 0.6
	}
	if d.Threshold > 1 {
		d.Threshold = 1
	}
	return d
}

// dropped records why an evidence item was excluded, so the trace can say what
// the evidence set would have held.
type dropped struct {
	cite   string
	reason string
}

// selectDiverse filters candidates in arrival order, skipping any item that
// restates one already kept.
//
// Arrival order is preserved rather than re-ranked: Result.Score is tier-local
// and not comparable across tiers, so the first item of a near-duplicate pair
// wins by virtue of its tier's position in the router's ranking. That is the
// same ordering rule merge already relies on, and picking "the better one"
// would require comparing a BM25 score against a graph hop count.
func selectDiverse(candidates []core.Result, d Diversity, limit int) ([]core.Result, []dropped) {
	d = d.withDefaults()

	var kept []core.Result
	var keptTokens []map[string]bool
	var drops []dropped
	perDoc := map[string]int{}

	for _, c := range candidates {
		if len(kept) >= limit {
			break
		}
		doc := sourceDoc(c)
		if d.PerDoc > 0 && perDoc[doc] >= d.PerDoc {
			drops = append(drops, dropped{c.Cite(), "doc cap"})
			continue
		}

		toks := contentTokens(c.Text)
		redundant := false
		for _, prev := range keptTokens {
			if jaccard(toks, prev) >= d.Threshold {
				redundant = true
				break
			}
		}
		if redundant {
			drops = append(drops, dropped{c.Cite(), "near-duplicate"})
			continue
		}

		kept = append(kept, c)
		keptTokens = append(keptTokens, toks)
		perDoc[doc]++
	}
	return kept, drops
}

// sourceDoc identifies the document an item came from, for the per-document cap.
//
// Meta["doc"] is set by the semantic tier; otherwise the source ID is split at
// the "#" that separates a document from its chunk position. A summary or a
// graph node has neither, and is its own document — which is right, since the
// cap exists to stop one file dominating and those are not files.
func sourceDoc(r core.Result) string {
	if doc, ok := r.Meta["doc"]; ok && doc != "" {
		return string(r.Tier) + ":" + doc
	}
	if doc, _, found := strings.Cut(r.SourceID, "#"); found {
		return string(r.Tier) + ":" + doc
	}
	return r.Cite()
}

// contentTokens reduces text to a set of lowercase word tokens, dropping the
// stopwords that would otherwise make any two English passages look similar.
func contentTokens(s string) map[string]bool {
	out := map[string]bool{}
	for _, f := range strings.FieldsFunc(strings.ToLower(s), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsNumber(r)
	}) {
		if len(f) > 2 && !evidenceStopwords[f] {
			out[f] = true
		}
	}
	return out
}

// jaccard is the intersection over union of two token sets.
//
// Set overlap rather than a sequence measure, because the duplication that
// matters here is restatement: an overlapping chunk boundary and a summary of a
// passage both repeat its distinctive words while sharing little word order.
func jaccard(a, b map[string]bool) float64 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}
	small, large := a, b
	if len(large) < len(small) {
		small, large = large, small
	}
	inter := 0
	for t := range small {
		if large[t] {
			inter++
		}
	}
	union := len(a) + len(b) - inter
	if union == 0 {
		return 0
	}
	return float64(inter) / float64(union)
}

var evidenceStopwords = map[string]bool{
	"the": true, "and": true, "for": true, "are": true, "but": true, "not": true,
	"you": true, "all": true, "can": true, "has": true, "have": true, "was": true,
	"were": true, "will": true, "with": true, "this": true, "that": true,
	"from": true, "they": true, "been": true, "than": true, "them": true,
	"which": true, "would": true, "there": true, "their": true,
	"when": true, "what": true, "each": true, "does": true, "into": true,
	"more": true, "some": true, "such": true, "only": true, "other": true,
	"also": true, "any": true, "its": true, "our": true, "out": true, "use": true,
}
