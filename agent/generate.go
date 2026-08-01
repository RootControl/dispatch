// Package agent drives retrieval as a loop and generates cited answers.
package agent

import (
	"context"
	"fmt"
	"strings"

	"github.com/RootControl/dispatch/core"
	"github.com/RootControl/dispatch/llm"
)

// Answer is a generated response plus the evidence it was grounded in, so a
// caller can resolve every [tier:source] citation back to a retrieved Result.
type Answer struct {
	Text     string
	Evidence []core.Result
	// Trace records how the answer was reached. Set by Loop.Run; nil for a bare
	// Generate call.
	Trace *Trace
	// Citations resolves every [tier:source] marker in Text against Evidence.
	// Set by Loop.Run. Citations.Unresolved is the one field worth checking
	// before showing an answer to anyone: a marker that resolves to nothing is
	// a fabricated source, and it looks exactly like a real one.
	Citations CitationReport
}

const generateSystem = `You answer questions strictly from the evidence provided.

Rules:
- Use only the evidence. Do not add outside knowledge.
- Cite every factual claim with the exact [tier:source] marker shown on the evidence you used.
- If the evidence does not answer the question, say so plainly and state what is missing.
- Be concise.`

// Generate produces a cited answer from evidence. It is a single pass; the agent
// loop calls it once retrieval is judged sufficient.
func Generate(ctx context.Context, l llm.LLM, question string, evidence []core.Result) (string, error) {
	return generate(ctx, l, question, "", evidence, nil)
}

// generate is Generate plus an optional core-memory block. Memory is presented
// separately from evidence and explicitly marked uncited: it is the agent's own
// prior conclusion, not a retrieved source, and letting the model cite it would
// manufacture citations that resolve to nothing.
func generate(ctx context.Context, l llm.LLM, question, memoryBlock string, evidence []core.Result, onDelta func(string)) (string, error) {
	if len(evidence) == 0 {
		return "", fmt.Errorf("agent: no evidence to answer from")
	}
	var prompt strings.Builder
	if memoryBlock != "" {
		fmt.Fprintf(&prompt, "<memory>\nRemembered from earlier work. Useful for orientation, but NOT citable — cite only evidence.\n%s\n</memory>\n\n", memoryBlock)
	}
	fmt.Fprintf(&prompt, "<evidence>\n%s\n</evidence>\n\nQuestion: %s", FormatEvidence(evidence), question)

	msgs := []llm.Message{llm.System(generateSystem), llm.User(prompt.String())}

	// Stream when the caller asked for it and the model supports it. onDelta
	// nil means nobody is watching, so there is no reason to stream.
	var text string
	var err error
	if s, ok := l.(llm.Streamer); ok && onDelta != nil {
		text, err = s.ChatStream(ctx, msgs, onDelta)
	} else {
		text, err = l.Chat(ctx, msgs)
	}
	if err != nil {
		return "", err
	}
	// A blank answer is a failure, not an answer. Returning it would surface as
	// a content problem — "the corpus didn't say" — when the real cause is the
	// model producing nothing.
	if strings.TrimSpace(text) == "" {
		return "", fmt.Errorf("agent: model returned an empty answer for %q", question)
	}
	return text, nil
}

// FormatEvidence renders results as a citation-labeled block. The [tier:source]
// marker is the same string Result.Cite produces, so the model can copy it
// verbatim and the caller can match it back.
func FormatEvidence(evidence []core.Result) string {
	var b strings.Builder
	for _, r := range evidence {
		fmt.Fprintf(&b, "%s\n%s\n\n", r.Cite(), strings.TrimSpace(r.Text))
	}
	return strings.TrimSpace(b.String())
}
