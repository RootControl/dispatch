package agent

import (
	"context"
	"fmt"
	"strings"

	"github.com/RootControl/dispatch/llm"
)

// Turn is one exchange in a conversation.
type Turn struct {
	Question string
	Answer   string
}

// Rewriter turns a follow-up question into one that stands on its own.
type Rewriter interface {
	// Rewrite returns a standalone question. Returning an empty string with a
	// nil error means "this question already stands alone" and it is used as
	// written — which is the common case and must stay free.
	Rewrite(ctx context.Context, question string, history []Turn) (string, error)
}

// Session is a multi-turn conversation over one Loop.
//
// Every retrieval path in this project takes a question and matches it against
// the corpus. That is exactly wrong for the second question in a conversation:
// "what about its budget?" contains one content word, and the thing it is about
// is in the PREVIOUS turn. BM25 matches "budget" against every document that
// mentions one, the embedding has almost nothing to work with, and the graph
// tier finds no seed because the question names no entity. The result is not a
// worse answer, it is evidence about the wrong subject.
//
// A Session fixes that where the problem is — in the query — rather than by
// stuffing history into the generation prompt, which would leave retrieval
// still fetching the wrong chunks and only give the model a better chance of
// noticing.
//
// History lives here rather than on Loop because a Loop is per-question and
// safe to share; a conversation is not.
type Session struct {
	Loop *Loop
	// Rewriter condenses a follow-up into a standalone question. Without one a
	// Session is just a Loop that remembers what was said, which is still
	// useful for `WriteBack` but does nothing for retrieval.
	Rewriter Rewriter
	// MaxTurns bounds the history handed to the rewriter; default 4. Older
	// turns are dropped rather than summarized: the referent of a follow-up is
	// nearly always in the last turn or two, and a longer window mostly adds
	// tokens and chances to resolve a pronoun to the wrong thing.
	MaxTurns int

	turns []Turn
}

// History returns the turns so far.
func (s *Session) History() []Turn { return append([]Turn(nil), s.turns...) }

// Reset clears the conversation. Ask on a fresh Session and Ask after Reset
// must behave identically, so a user who says "forget that" gets what they
// asked for rather than a quieter version of the same context.
func (s *Session) Reset() { s.turns = nil }

// Ask answers a question in the context of the conversation so far.
func (s *Session) Ask(ctx context.Context, question string) (Answer, error) {
	asked := question
	var rewritten string

	if s.Rewriter != nil && len(s.turns) > 0 {
		got, err := s.Rewriter.Rewrite(ctx, question, s.window())
		switch {
		case err != nil:
			// A failed rewrite costs the improvement, not the answer. The
			// original question retrieves what it would have retrieved without
			// a Session at all, which is a worse result and not a broken one.
			rewritten = ""
		case strings.TrimSpace(got) != "" && !strings.EqualFold(strings.TrimSpace(got), strings.TrimSpace(question)):
			rewritten = strings.TrimSpace(got)
			asked = rewritten
		}
	}

	ans, err := s.Loop.Run(ctx, asked)
	if err != nil {
		return ans, err
	}
	// Record what was actually asked of the corpus. A question silently
	// rewritten into something that retrieves the wrong subject is otherwise
	// inexplicable from the outside: the trace would show a sensible retrieval
	// for a question nobody typed.
	if ans.Trace != nil {
		ans.Trace.Question = question
		if rewritten != "" {
			ans.Trace.add(Step{Kind: StepRewrite, Query: rewritten,
				Detail: "follow-up resolved against the conversation"})
		}
	}

	// The stored answer is what a later rewrite reads to resolve a pronoun, so
	// citations stay in: "[semantic:atlas-staffing.md#0]" is noise to a human
	// and a useful hint about the subject to the model.
	s.turns = append(s.turns, Turn{Question: question, Answer: ans.Text})
	return ans, nil
}

func (s *Session) window() []Turn {
	max := s.MaxTurns
	if max <= 0 {
		max = 4
	}
	if len(s.turns) <= max {
		return s.turns
	}
	return s.turns[len(s.turns)-max:]
}

// Condenser is the reference Rewriter: it asks the model to restate a follow-up
// as a standalone question.
type Condenser struct {
	LLM llm.LLM
	// Always rewrites every question once there is history, instead of only
	// those that look like follow-ups. It costs one LLM call per turn in
	// exchange for catching elliptical questions no lexical rule spots.
	Always bool
	// MaxAnswerChars truncates each remembered answer before it reaches the
	// prompt; default 400. Full answers make the rewrite prompt larger than the
	// question by an order of magnitude, and the referent is nearly always in
	// the first sentence or two.
	MaxAnswerChars int
}

var _ Rewriter = (*Condenser)(nil)

// condenseSystem is the rewrite prompt, and its shape was measured rather than
// guessed. A first version described the task and closed with "if the question
// already stands on its own, reply with it unchanged"; against a real endpoint
// it resolved one follow-up in three, leaving "What is its budget?" untouched.
// llama3.2:3b and qwen2.5:7b failed IDENTICALLY, which is what said the prompt
// was at fault rather than the model — a 2.3x larger model reproducing a
// failure exactly is not a capacity problem.
//
// Making the pronoun rule imperative and showing four worked examples took it
// to five in five. The examples name a subject deliberately unrelated to any
// corpus here, and the check was re-run against a different subject again, so
// the gain is the instruction generalizing rather than the model copying a name
// out of the prompt.
const condenseSystem = `You rewrite a follow-up question so that it stands on its own.

The user is mid-conversation. Rewrite their new question so that someone who has NOT seen
the conversation could answer it.

RULES:
- Every pronoun or bare reference that points at an earlier turn ("it", "its", "they",
  "that one", "there") MUST be replaced with the actual name from the conversation.
- A question that omits its subject entirely ("and the budget?") MUST have the subject
  put back.
- Change nothing else: keep their wording, their scope, and what they are asking for.
  Do not answer. Do not add detail they did not ask about.
- If, and only if, the question already names its own subject, reply with it unchanged.

Examples, given a conversation about the Meridian Compact:
  "What is its budget?"  -> "What is the Meridian Compact's budget?"
  "Who leads it?"        -> "Who leads the Meridian Compact?"
  "and the vendor?"      -> "Which vendor is involved in the Meridian Compact?"
  "Who signed the renewal in January?" -> "Who signed the renewal in January?"

Reply with a JSON object only: {"question": "<the standalone question>"}`

// Rewrite condenses a follow-up against the conversation.
func (c *Condenser) Rewrite(ctx context.Context, question string, history []Turn) (string, error) {
	if len(history) == 0 {
		return "", nil
	}
	if !c.Always && !dependent(question) {
		return "", nil
	}

	limit := c.MaxAnswerChars
	if limit <= 0 {
		limit = 400
	}
	var b strings.Builder
	for _, t := range history {
		fmt.Fprintf(&b, "Q: %s\nA: %s\n\n", t.Question, clip(t.Answer, limit))
	}
	fmt.Fprintf(&b, "Follow-up question: %s", question)

	var out struct {
		Question string `json:"question"`
	}
	if err := c.LLM.ChatJSON(ctx, []llm.Message{
		llm.System(condenseSystem),
		llm.User(b.String()),
	}, &out); err != nil {
		return "", err
	}

	got := strings.TrimSpace(out.Question)
	if got == "" {
		return "", nil
	}
	// A rewrite that balloons is a rewrite that has started answering. The
	// failure is real on a small model: asked to make "what about its budget?"
	// standalone, it can return a paragraph restating the previous answer,
	// which then retrieves against the answer rather than the question.
	if len(got) > 4*len(question)+120 {
		return "", nil
	}
	return got, nil
}

// dependent reports whether a question appears to lean on earlier turns.
//
// This is a lexical pre-filter, and it is the right shape for the trade: a
// false negative costs nothing that a Session-less run would not also cost —
// the question retrieves as written — while calling the model on every
// standalone question costs a call per turn forever. It errs toward not
// rewriting, because rewriting a question that did not need it is how a clear
// question becomes a narrowed one.
func dependent(q string) bool {
	lower := strings.ToLower(strings.TrimSpace(q))
	words := strings.FieldsFunc(lower, func(r rune) bool {
		return !('a' <= r && r <= 'z') && !('0' <= r && r <= '9')
	})
	if len(words) == 0 {
		return false
	}
	// A leading conjunction is the plainest signal of a continuation:
	// "and the vendor?", "so who signed it".
	switch words[0] {
	case "and", "so", "but", "then", "what", "how", "why":
		if words[0] != "what" && words[0] != "how" && words[0] != "why" {
			return true
		}
	}
	// "what about ...", "how about ..." are elliptical by construction.
	if len(words) > 1 && words[1] == "about" {
		return true
	}
	for _, w := range words {
		if referring[w] {
			return true
		}
	}
	// A very short question rarely names its own subject. Four words is where
	// "who signed the contract" (self-contained) gives way to "who signed it".
	return len(words) <= 3
}

// referring lists the words that point outside the question. Deliberately not
// "the": "the budget" is ambiguous in a way "its budget" is not, and treating
// every definite article as a follow-up would rewrite nearly everything.
var referring = map[string]bool{
	"it": true, "its": true, "it's": true, "they": true, "them": true, "their": true,
	"he": true, "him": true, "his": true, "she": true, "her": true, "hers": true,
	"this": true, "that": true, "these": true, "those": true, "there": true,
	"same": true, "one": true, "ones": true, "another": true, "else": true,
	"instead": true, "above": true, "previous": true, "earlier": true,
}

func clip(s string, n int) string {
	s = strings.TrimSpace(s)
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
