package index

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"sort"
	"strings"
	"time"
)

// CrossEncoder reranks with a purpose-built reranking model over an HTTP
// /rerank endpoint — Cohere Rerank, Jina, Voyage, or a locally-served
// bge-reranker behind text-embeddings-inference.
//
// This exists because reranking with a general chat model was measured worse
// than no reranking at all, twice, at up to 13x the wall-clock: recall@5 fell
// from 94% to 39% with a 3B model and to 88% with a 7B one. That is not a
// too-small-model result — the 7B matches a 9.6 GB reasoning model on routing
// and answer quality. The mechanism is that RRF's own top-5 is already 94%
// correct and Search over-fetches 20 candidates, so a reranker that does not
// order better than RRF is handed 15 extra chances to evict a right answer.
//
// A cross-encoder is a different kind of model, not a bigger one: it reads the
// query and passage jointly and is trained on exactly this ordering task, which
// is the assumption behind the 49% -> 67% figure the LLM reranker was built
// from and never met. Whether it beats RRF on your corpus is an empirical
// question this repo can answer:
//
//	dispatch eval retrieval -k 5              # baseline
//	dispatch eval retrieval -k 5 --rerank     # with the cross-encoder
//
// If it does not win, leave it off. That is the same standard the LLM reranker
// was held to and failed.
type CrossEncoder struct {
	BaseURL string // e.g. https://api.cohere.com/v2 or http://localhost:8080
	APIKey  string
	Model   string // optional; TEI serves a single model and ignores it

	// TopN caps what the endpoint is asked to return. Zero means topK.
	TopN int
	// HTTPClient defaults to a client with a 60s timeout.
	HTTPClient *http.Client
}

var _ Reranker = (*CrossEncoder)(nil)

// NewCrossEncoderFromEnv builds a CrossEncoder from RERANK_BASE_URL,
// RERANK_API_KEY and RERANK_MODEL. It returns an error when no base URL is
// configured, so a caller asking for reranking is told the endpoint is missing
// rather than quietly getting none.
func NewCrossEncoderFromEnv() (*CrossEncoder, error) {
	base := strings.TrimRight(os.Getenv("RERANK_BASE_URL"), "/")
	if base == "" {
		return nil, errors.New("index: no reranker endpoint (set RERANK_BASE_URL)")
	}
	return &CrossEncoder{
		BaseURL: base,
		APIKey:  os.Getenv("RERANK_API_KEY"),
		Model:   os.Getenv("RERANK_MODEL"),
	}, nil
}

// rerankResult covers the response shapes the common servers return. Cohere and
// Jina wrap the list in {"results": [...]}; text-embeddings-inference returns a
// bare array. The score field is "relevance_score" for the first two and
// "score" for the third, so both are accepted.
type rerankResult struct {
	Index          int      `json:"index"`
	RelevanceScore *float64 `json:"relevance_score"`
	Score          *float64 `json:"score"`
}

func (r rerankResult) score() (float64, bool) {
	switch {
	case r.RelevanceScore != nil:
		return *r.RelevanceScore, true
	case r.Score != nil:
		return *r.Score, true
	}
	return 0, false
}

// Rerank scores the shortlist with the cross-encoder and returns the best topK.
//
// Unlike the LLM reranker, a failure here is returned rather than swallowed. A
// cross-encoder is opt-in and configured deliberately; silently falling back to
// retrieval order would make a misconfigured endpoint look like a reranker that
// simply never helps, which is the one outcome the measurement cannot
// distinguish from the truth.
func (c *CrossEncoder) Rerank(ctx context.Context, query string, hits []Hit, topK int) ([]Hit, error) {
	if len(hits) == 0 {
		return hits, nil
	}
	docs := make([]string, len(hits))
	for i, h := range hits {
		docs[i] = strings.TrimSpace(h.Chunk.Embedded())
	}
	topN := c.TopN
	if topN <= 0 {
		topN = topK
	}
	if topN <= 0 || topN > len(docs) {
		topN = len(docs)
	}

	body := map[string]any{
		"query":            query,
		"documents":        docs,
		"texts":            docs, // text-embeddings-inference names the field this way
		"top_n":            topN,
		"return_documents": false,
	}
	if c.Model != "" {
		body["model"] = c.Model
	}
	payload, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+"/rerank", bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if c.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.APIKey)
	}
	hc := c.HTTPClient
	if hc == nil {
		hc = &http.Client{Timeout: 60 * time.Second}
	}
	resp, err := hc.Do(req)
	if err != nil {
		return nil, fmt.Errorf("index: rerank: %w", err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return nil, fmt.Errorf("index: rerank: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("index: rerank: %s: %s", resp.Status, truncateStr(string(raw), 200))
	}

	results, err := parseRerankResponse(raw)
	if err != nil {
		return nil, err
	}
	return applyRerank(hits, results, topK)
}

// parseRerankResponse accepts either {"results": [...]} or a bare [...].
func parseRerankResponse(raw []byte) ([]rerankResult, error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) > 0 && trimmed[0] == '[' {
		var bare []rerankResult
		if err := json.Unmarshal(trimmed, &bare); err != nil {
			return nil, fmt.Errorf("index: rerank: parse response: %w", err)
		}
		return bare, nil
	}
	var wrapped struct {
		Results []rerankResult `json:"results"`
		Data    []rerankResult `json:"data"`
	}
	if err := json.Unmarshal(trimmed, &wrapped); err != nil {
		return nil, fmt.Errorf("index: rerank: parse response: %w", err)
	}
	if len(wrapped.Results) > 0 {
		return wrapped.Results, nil
	}
	return wrapped.Data, nil
}

// applyRerank reorders hits by the returned scores.
//
// A server that returns fewer results than it was given is normal — top_n asks
// it to. Those hits keep retrieval order behind everything scored, so a short
// reply degrades to partial reranking rather than dropping candidates. An index
// outside range is a protocol violation and is an error: it means the reply
// does not correspond to the request, and quietly ignoring it would rerank
// against the wrong passages.
func applyRerank(hits []Hit, results []rerankResult, topK int) ([]Hit, error) {
	if len(results) == 0 {
		return nil, errors.New("index: rerank: endpoint returned no scores")
	}
	score := make([]float64, len(hits))
	scored := make([]bool, len(hits))
	for _, r := range results {
		if r.Index < 0 || r.Index >= len(hits) {
			return nil, fmt.Errorf("index: rerank: result index %d outside the %d passages sent", r.Index, len(hits))
		}
		s, ok := r.score()
		if !ok {
			return nil, fmt.Errorf("index: rerank: result %d carries no score field", r.Index)
		}
		if scored[r.Index] {
			return nil, fmt.Errorf("index: rerank: result index %d returned twice", r.Index)
		}
		score[r.Index], scored[r.Index] = s, true
	}

	idx := make([]int, len(hits))
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool {
		ia, ib := idx[a], idx[b]
		if scored[ia] != scored[ib] {
			return scored[ia] // anything scored outranks anything unscored
		}
		if !scored[ia] {
			return false // both unscored: hold retrieval order
		}
		return score[ia] > score[ib]
	})

	out := make([]Hit, 0, len(hits))
	for _, i := range idx {
		h := hits[i]
		if scored[i] {
			h.Score = score[i]
		}
		out = append(out, h)
	}
	return truncate(out, topK), nil
}

func truncateStr(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
