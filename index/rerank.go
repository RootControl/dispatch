package index

import (
	"context"
	"fmt"
	"sort"
	"strings"

	"github.com/RootControl/dispatch/llm"
)

// Reranker reorders candidates by relevance to a query. It runs after hybrid
// retrieval has narrowed the corpus to a shortlist.
//
// This is the second half of Anthropic's contextual-retrieval result: contextual
// chunking alone cut retrieval failures ~49%, and adding reranking took it to
// ~67%. Retrieval and reranking optimise different things — retrieval has to
// scan everything cheaply, a reranker only has to order a shortlist, so it can
// afford to actually read each candidate against the query.
type Reranker interface {
	// Rerank returns hits ordered best-first, at most topK. It may return fewer
	// than it was given but must never invent one.
	Rerank(ctx context.Context, query string, hits []Hit, topK int) ([]Hit, error)
}

// LLMReranker scores each candidate with the model. Scoring is one call per
// batch, not per candidate: a shortlist fits in a single prompt, and asking once
// lets the model compare candidates against each other rather than rate them in
// isolation.
type LLMReranker struct {
	LLM llm.LLM
	// BatchSize caps how many candidates go into one scoring call. Default 20 —
	// large enough that the whole shortlist usually fits in one request, small
	// enough that the prompt stays inside a modest context window.
	BatchSize int
}

var _ Reranker = (*LLMReranker)(nil)

const rerankSystem = `You score how well each passage answers a question.

Score each passage 0-10:
  0-2   unrelated
  3-5   same topic, does not answer the question
  6-8   contains part of the answer
  9-10  directly answers the question

Judge only whether the passage answers THIS question. A well-written passage about
something else scores low. Score every passage you are given, by its number.

Reply with a JSON object only: {"scores": [{"id": <number>, "score": <number>}, ...]}`

// Rerank scores the shortlist and returns the best topK.
//
// A failure returns the input order rather than an error: reranking is a
// refinement over an already-relevant shortlist, so losing it degrades results
// slightly, while failing the query loses them entirely.
func (r *LLMReranker) Rerank(ctx context.Context, query string, hits []Hit, topK int) ([]Hit, error) {
	if len(hits) == 0 {
		return hits, nil
	}
	batch := r.BatchSize
	if batch <= 0 {
		batch = 20
	}
	if len(hits) > batch {
		hits = hits[:batch]
	}

	var b strings.Builder
	for i, h := range hits {
		fmt.Fprintf(&b, "[%d] %s\n\n", i, strings.TrimSpace(h.Chunk.Embedded()))
	}
	var out struct {
		Scores []struct {
			ID    int     `json:"id"`
			Score float64 `json:"score"`
		} `json:"scores"`
	}
	msgs := []llm.Message{
		llm.System(rerankSystem),
		llm.User(fmt.Sprintf("Question: %s\n\n<passages>\n%s</passages>", query, b.String())),
	}
	if err := r.LLM.ChatJSON(ctx, msgs, &out); err != nil || len(out.Scores) == 0 {
		return truncate(hits, topK), nil
	}

	// Keep the retrieval order as the tie-break and the fallback for anything
	// the model declined to score, so an incomplete reply degrades to partial
	// reranking rather than dropping candidates.
	score := make([]float64, len(hits))
	for i := range score {
		score[i] = -1
	}
	for _, s := range out.Scores {
		if s.ID >= 0 && s.ID < len(hits) {
			score[s.ID] = s.Score
		}
	}
	idx := make([]int, len(hits))
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool { return score[idx[a]] > score[idx[b]] })

	out2 := make([]Hit, 0, len(hits))
	for _, i := range idx {
		h := hits[i]
		if score[i] >= 0 {
			h.Score = score[i]
		}
		out2 = append(out2, h)
	}
	return truncate(out2, topK), nil
}

func truncate(hits []Hit, topK int) []Hit {
	if topK > 0 && len(hits) > topK {
		return hits[:topK]
	}
	return hits
}
