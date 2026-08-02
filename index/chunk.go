package index

import (
	"regexp"
	"strings"
)

// ChunkOptions controls the splitter. Sizes are in estimated tokens; the
// estimator (estimateTokens) approximates an OpenAI tokenizer from whitespace
// words, which is close enough for chunk sizing without a tokenizer dependency.
type ChunkOptions struct {
	TargetTokens  int  // soft max per chunk; default 800
	OverlapTokens int  // carried between adjacent chunks; default 100
	NoOverlap     bool // force zero overlap (used by tests for clean boundaries)
	// MinWords drops chunks carrying less prose than this. Real corpora contain
	// fragments that answer nothing — a file whose whole content is another
	// filename, a block of badge markup — and each costs an LLM call to
	// contextualise, occupies the index, and can surface as a false positive.
	//
	// OFF by default, because silently discarding a user's content is worse
	// than indexing some noise: a corpus of short entries — a glossary, one-line
	// FAQ answers — would lose them with no error and no way to notice. Opt in
	// with `ingest --min-words`, which reports how many chunks it dropped.
	MinWords int
	// Headings splits on markdown headings before paragraphs, so a chunk never
	// spans two sections, and prefixes each chunk with its heading path.
	//
	// The prefix is the point. "Atlas > Budget > Contingency" in front of a
	// chunk gives the embedder and BM25 the same kind of situating signal that
	// contextual chunking pays an LLM call per chunk to write — except this one
	// is free, exact, and already in the document. It composes with contextual
	// chunking rather than replacing it.
	//
	// Off by default: it changes chunk boundaries and therefore every vector,
	// so turning it on is a re-ingest. Documents with no headings are unaffected
	// either way.
	Headings bool
}

func (o ChunkOptions) withDefaults() ChunkOptions {
	if o.TargetTokens <= 0 {
		o.TargetTokens = 800
	}
	switch {
	case o.NoOverlap:
		o.OverlapTokens = 0
	case o.OverlapTokens <= 0:
		o.OverlapTokens = 100
	}
	if o.OverlapTokens >= o.TargetTokens {
		o.OverlapTokens = o.TargetTokens / 4
	}
	return o
}

// hasProse reports whether a chunk carries enough ordinary words to answer
// anything. Markup tokens are not counted: a block of image badges is long by
// word count and empty of content.
func hasProse(s string, minWords int) bool {
	if minWords <= 0 {
		return true
	}
	n := 0
	for _, f := range strings.Fields(s) {
		if strings.ContainsAny(f, "<>=\"/") || strings.HasPrefix(f, "!") || strings.HasPrefix(f, "[!") {
			continue // markup, an attribute, or a badge
		}
		if len(strings.Trim(f, "#*`|-_()[]")) >= 2 {
			n++
		}
	}
	return n >= minWords
}

// estimateTokens approximates token count as words × 4/3, the usual rough ratio
// of BPE tokens to whitespace words for English prose.
func estimateTokens(s string) int {
	return len(strings.Fields(s)) * 4 / 3
}

var (
	paragraphSplit = regexp.MustCompile(`\n\s*\n`)
	sentenceSplit  = regexp.MustCompile(`(?m)([.!?])\s+`)
)

// Split breaks text into retrievable chunks, preferring paragraph boundaries,
// falling back to sentences and then word windows for oversized units, and
// carrying OverlapTokens of trailing context into each subsequent chunk.
func Split(text string, opts ChunkOptions) []string {
	opts = opts.withDefaults()
	if opts.Headings && HasHeadings(text) {
		return splitSections(text, opts)
	}
	return splitFlat(text, opts, "")
}

// splitSections chunks each heading-delimited section independently, so a chunk
// never spans a section boundary, and prefixes each with its heading path.
//
// Sections are not packed together even when two small ones would fit in one
// chunk. That would put two unrelated topics behind one embedding and one
// citation, which is the merge the section boundary exists to prevent — the
// cost is some chunks well under target, which is the cheaper mistake.
func splitSections(text string, opts ChunkOptions) []string {
	var chunks []string
	for _, sec := range Sections(text) {
		chunks = append(chunks, splitFlat(sec.Body, opts, sec.Path)...)
	}
	return chunks
}

// splitFlat is the paragraph/sentence/word splitter. prefix, when set, is the
// heading path prepended to every chunk it produces.
func splitFlat(text string, opts ChunkOptions, prefix string) []string {
	// The prefix costs tokens in every chunk, so it comes out of the budget
	// rather than silently pushing chunks over it.
	target := opts.TargetTokens
	if prefix != "" {
		if t := target - estimateTokens(prefix); t > 0 {
			target = t
		}
	}
	units := segmentize(text, target)

	var chunks []string
	var cur []string
	curTok := 0
	emit := func(text string) {
		// MinWords is judged on the prose alone: the heading path is structure,
		// not content, so counting it would let a prefix rescue a chunk that
		// says nothing.
		if !hasProse(text, opts.MinWords) {
			return
		}
		if prefix != "" {
			text = prefix + "\n\n" + text
		}
		chunks = append(chunks, text)
	}
	flush := func() {
		if len(cur) == 0 {
			return
		}
		emit(strings.Join(cur, "\n\n"))
		cur, curTok = overlapTail(cur, opts.OverlapTokens)
	}
	for _, u := range units {
		ut := estimateTokens(u)
		if curTok+ut > target {
			flush()
		}
		cur = append(cur, u)
		curTok += ut
	}
	if len(cur) > 0 {
		emit(strings.Join(cur, "\n\n"))
	}
	return chunks
}

// segmentize returns text pieces each no larger than target tokens, splitting
// oversized paragraphs by sentence and oversized sentences by word window.
func segmentize(text string, target int) []string {
	var out []string
	for _, p := range paragraphSplit.Split(text, -1) {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		if estimateTokens(p) <= target {
			out = append(out, p)
			continue
		}
		for _, s := range splitSentences(p) {
			if estimateTokens(s) <= target {
				out = append(out, s)
			} else {
				out = append(out, splitWordWindows(s, target)...)
			}
		}
	}
	return out
}

func splitSentences(p string) []string {
	// Re-attach the terminal punctuation the split consumed.
	marked := sentenceSplit.ReplaceAllString(p, "$1\x00")
	parts := strings.Split(marked, "\x00")
	out := make([]string, 0, len(parts))
	for _, s := range parts {
		if s = strings.TrimSpace(s); s != "" {
			out = append(out, s)
		}
	}
	return out
}

func splitWordWindows(s string, target int) []string {
	words := strings.Fields(s)
	// target tokens ≈ target*3/4 words.
	win := max(target*3/4, 1)
	var out []string
	for i := 0; i < len(words); i += win {
		end := min(i+win, len(words))
		out = append(out, strings.Join(words[i:end], " "))
	}
	return out
}

// overlapTail returns the trailing units whose cumulative tokens stay within
// overlapTokens, to seed the next chunk. Returns empty when overlap is zero.
func overlapTail(units []string, overlapTokens int) ([]string, int) {
	if overlapTokens <= 0 {
		return nil, 0
	}
	var out []string
	tok := 0
	for i := len(units) - 1; i >= 0; i-- {
		t := estimateTokens(units[i])
		if tok+t > overlapTokens && len(out) > 0 {
			break
		}
		out = append([]string{units[i]}, out...)
		tok += t
		if tok >= overlapTokens {
			break
		}
	}
	return out, tok
}
