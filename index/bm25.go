package index

import (
	"math"
	"strings"
	"unicode"
)

// bm25Index is a classic Okapi BM25 lexical index. It complements the vector
// index: BM25 catches exact terms (invoice numbers, names, error codes) that
// embeddings blur together. The two are fused by RRF in the store.
type bm25Index struct {
	k1, b    float64
	docs     []bm25Doc
	df       map[string]int // document frequency per term
	totalLen int
}

type bm25Doc struct {
	id  string
	tf  map[string]int
	len int
}

func newBM25() *bm25Index {
	return &bm25Index{k1: 1.2, b: 0.75, df: map[string]int{}}
}

func (bm *bm25Index) add(id, text string) {
	toks := tokenize(text)
	tf := make(map[string]int, len(toks))
	for _, t := range toks {
		tf[t]++
	}
	for t := range tf {
		bm.df[t]++
	}
	bm.docs = append(bm.docs, bm25Doc{id: id, tf: tf, len: len(toks)})
	bm.totalLen += len(toks)
}

// remove drops every document whose id is in drop, decrementing the term
// document-frequencies and corpus length it contributed. Those are what the idf
// and length-normalisation terms are computed from, so leaving them behind would
// score the remaining documents against a corpus that no longer exists.
func (bm *bm25Index) remove(drop map[string]bool) {
	if len(drop) == 0 {
		return
	}
	kept := bm.docs[:0]
	for _, d := range bm.docs {
		if !drop[d.id] {
			kept = append(kept, d)
			continue
		}
		for t := range d.tf {
			if bm.df[t]--; bm.df[t] <= 0 {
				delete(bm.df, t)
			}
		}
		bm.totalLen -= d.len
	}
	for i := len(kept); i < len(bm.docs); i++ {
		bm.docs[i] = bm25Doc{}
	}
	bm.docs = kept
}

// search returns the n best-scoring documents that allow accepts. The corpus
// statistics — average length and N for idf — stay over the whole index rather
// than the filtered subset, so a term's rarity means the same thing whatever
// filter is applied and two filters cannot disagree about what "rare" is.
func (bm *bm25Index) search(query string, n int, allow func(string) bool) []scored {
	if len(bm.docs) == 0 {
		return nil
	}
	qterms := tokenize(query)
	avg := float64(bm.totalLen) / float64(len(bm.docs))
	N := float64(len(bm.docs))

	out := make([]scored, 0, len(bm.docs))
	for _, d := range bm.docs {
		if allow != nil && !allow(d.id) {
			continue
		}
		var score float64
		for _, q := range qterms {
			tf, ok := d.tf[q]
			if !ok {
				continue
			}
			df := bm.df[q]
			// BM25 idf with the +1 inside the log to keep it non-negative.
			idf := math.Log(1 + (N-float64(df)+0.5)/(float64(df)+0.5))
			num := float64(tf) * (bm.k1 + 1)
			den := float64(tf) + bm.k1*(1-bm.b+bm.b*float64(d.len)/avg)
			score += idf * num / den
		}
		if score > 0 {
			out = append(out, scored{id: d.id, score: score})
		}
	}
	sortScored(out)
	if n > 0 && len(out) > n {
		out = out[:n]
	}
	return out
}

// tokenize lowercases, splits on non-alphanumeric runes, and drops a small
// stopword set. No stemming — keeping it dependency-free and predictable.
func tokenize(s string) []string {
	fields := strings.FieldsFunc(strings.ToLower(s), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsNumber(r)
	})
	out := fields[:0]
	for _, f := range fields {
		if !stopwords[f] {
			out = append(out, f)
		}
	}
	return out
}

var stopwords = map[string]bool{
	"a": true, "an": true, "and": true, "are": true, "as": true, "at": true,
	"be": true, "by": true, "for": true, "from": true, "in": true, "is": true,
	"it": true, "of": true, "on": true, "or": true, "that": true, "the": true,
	"to": true, "was": true, "were": true, "will": true, "with": true,
}
