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
	units := segmentize(text, opts.TargetTokens)

	var chunks []string
	var cur []string
	curTok := 0
	flush := func() {
		if len(cur) == 0 {
			return
		}
		chunks = append(chunks, strings.Join(cur, "\n\n"))
		cur, curTok = overlapTail(cur, opts.OverlapTokens)
	}
	for _, u := range units {
		ut := estimateTokens(u)
		if curTok+ut > opts.TargetTokens {
			flush()
		}
		cur = append(cur, u)
		curTok += ut
	}
	if len(cur) > 0 {
		chunks = append(chunks, strings.Join(cur, "\n\n"))
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
