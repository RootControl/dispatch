package index

import (
	"regexp"
	"strings"
)

// Section is a heading-delimited region of a markdown document, carrying the
// full path of headings above it.
type Section struct {
	Path string // "Atlas > Budget > Contingency"; empty above the first heading
	Body string
}

// atxHeading matches a markdown ATX heading: one to six '#', a space, the text,
// and optional trailing '#'. Setext headings (underlined with === or ---) are
// deliberately not matched: a line of dashes is also a table separator and a
// horizontal rule, and guessing wrong splits a table down the middle.
var atxHeading = regexp.MustCompile(`^(#{1,6})[ \t]+(.+?)[ \t]*#*$`)

// fenceLine matches the start or end of a fenced code block.
var fenceLine = regexp.MustCompile("^\\s{0,3}(```|~~~)")

// Sections splits markdown into heading-delimited regions.
//
// The reason this exists: the paragraph splitter treats a heading as just
// another line, so "## Budget" ends up glued to whichever paragraph it lands
// next to, and a section longer than one chunk is cut mid-topic with nothing
// saying what it was about. Sections give a chunk a boundary that means
// something and a Path that says where it came from.
//
// Headings inside fenced code blocks are ignored. A shell comment or a Python
// comment starts with '#', so a corpus of technical documentation is full of
// lines that look exactly like headings and are not — and splitting a code
// block in half is worse than not splitting at all.
func Sections(text string) []Section {
	var out []Section
	var stack []string // heading text by level-1 depth
	var body strings.Builder
	path := ""
	inFence := false

	flush := func() {
		if s := strings.TrimSpace(body.String()); s != "" {
			out = append(out, Section{Path: path, Body: s})
		}
		body.Reset()
	}

	for line := range strings.SplitSeq(text, "\n") {
		if fenceLine.MatchString(line) {
			inFence = !inFence
			body.WriteString(line)
			body.WriteByte('\n')
			continue
		}
		m := atxHeading.FindStringSubmatch(line)
		if inFence || m == nil {
			body.WriteString(line)
			body.WriteByte('\n')
			continue
		}

		// A heading closes the previous section and opens a new one.
		flush()
		level, title := len(m[1]), strings.TrimSpace(m[2])
		// Truncate the stack to this heading's depth, then push. A jump from
		// h1 straight to h3 simply leaves the intermediate slot empty rather
		// than being treated as malformed; real documents do this constantly.
		if level-1 < len(stack) {
			stack = stack[:level-1]
		}
		for len(stack) < level-1 {
			stack = append(stack, "")
		}
		stack = append(stack, title)

		parts := make([]string, 0, len(stack))
		for _, s := range stack {
			if s != "" {
				parts = append(parts, s)
			}
		}
		path = strings.Join(parts, " > ")
	}
	flush()
	return out
}

// HasHeadings reports whether text contains at least one markdown heading
// outside a code fence. Callers use it to skip section splitting for plain
// prose, where it would do nothing but cost a pass over the text.
func HasHeadings(text string) bool {
	inFence := false
	for line := range strings.SplitSeq(text, "\n") {
		if fenceLine.MatchString(line) {
			inFence = !inFence
			continue
		}
		if !inFence && atxHeading.MatchString(line) {
			return true
		}
	}
	return false
}
