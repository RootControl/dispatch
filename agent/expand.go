package agent

import (
	"context"
	"strings"

	"github.com/RootControl/dispatch/llm"
)

// Expander rewrites a question into text better suited to dense retrieval.
type Expander interface {
	// Expand returns a passage to embed in place of the question. Returning an
	// empty string with a nil error means "no expansion worth making" and the
	// query is used as written.
	Expand(ctx context.Context, question string) (string, error)
}

// HyDE writes a hypothetical answer to the question and retrieves with that
// instead of the question itself.
//
// This targets a specific measured failure. "What is this project and what is
// it for?" has no distinctive content words, so both halves of the hybrid latch
// onto incidental matches — the top hit on the real corpus was a translations
// how-to that merely says "in the project". Contextual chunking helps and does
// not rescue it, because the query has nothing to match on.
//
// A hypothetical answer does. Asked that question, a model writes a paragraph
// full of the vocabulary a real answer would contain — architecture, purpose,
// components — and a passage that answers the question resembles that paragraph
// far more than it resembles the question. Questions and answers do not look
// alike in embedding space; answers and answers do.
//
// The cost is one LLM call per question and a real failure mode: the model
// invents specifics, and a confident hypothetical about the wrong subject
// retrieves confidently wrong passages. That is why this is off by default, why
// the expansion reaches only the dense half (core.Query.Embedding explains
// why), and why the honest thing is to measure it:
//
//	dispatch eval retrieval -k 5            # baseline
//	dispatch eval retrieval -k 5 --hyde     # with expansion
type HyDE struct {
	LLM llm.LLM
	// MinWords skips expansion for questions already specific enough not to
	// need it, since each one costs a call and can only add noise to a query
	// that was working. Default 0: expand everything. Set it to something like
	// 8 to spend the call only on short, vague questions.
	MinWords int
}

var _ Expander = (*HyDE)(nil)

const hydeSystem = `You write a short hypothetical passage that would answer the user's question,
as if excerpted from the document that contains the answer.

Write it as a factual document excerpt, not as a reply. No preamble, no hedging, no
"the answer is". Two or three sentences.

You do not know the real answer and that is fine — invent plausible specifics. This text
is never shown to anyone and is never used as an answer. It is used only to find real
documents that resemble it, so what matters is that it uses the vocabulary a genuine
answer would use.`

// Expand writes the hypothetical passage.
//
// A failure returns an empty string and the error; the loop treats that as "no
// expansion" and retrieves with the question as written, because a broken
// expander should cost the improvement rather than the answer.
func (h *HyDE) Expand(ctx context.Context, question string) (string, error) {
	if h.MinWords > 0 && len(strings.Fields(question)) >= h.MinWords {
		return "", nil
	}
	reply, err := h.LLM.Chat(ctx, []llm.Message{
		llm.System(hydeSystem),
		llm.User(question),
	})
	if err != nil {
		return "", err
	}
	reply = strings.TrimSpace(reply)
	if reply == "" {
		return "", nil
	}
	// Prepend the question. Pure HyDE embeds the hypothetical alone, which
	// stakes the whole retrieval on the model having guessed the right subject;
	// keeping the question in the embedded text anchors it to what was actually
	// asked, so a bad hypothetical degrades the query rather than replacing it.
	return question + "\n\n" + reply, nil
}
