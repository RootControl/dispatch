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
	if len(evidence) == 0 {
		return "", fmt.Errorf("agent: no evidence to answer from")
	}
	msgs := []llm.Message{
		llm.System(generateSystem),
		llm.User(fmt.Sprintf("<evidence>\n%s\n</evidence>\n\nQuestion: %s", FormatEvidence(evidence), question)),
	}
	return l.Chat(ctx, msgs)
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
