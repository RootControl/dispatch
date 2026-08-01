package index

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/RootControl/dispatch/core"
)

// serveRerank stands up a /rerank endpoint returning body, and records the
// request it was sent.
func serveRerank(t *testing.T, status int, body string) (*CrossEncoder, *map[string]any) {
	t.Helper()
	var got map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/rerank" {
			t.Errorf("posted to %s, want /rerank", r.URL.Path)
		}
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &got)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		io.WriteString(w, body)
	}))
	t.Cleanup(srv.Close)
	return &CrossEncoder{BaseURL: srv.URL, HTTPClient: srv.Client()}, &got
}

// The three response shapes in the wild: Cohere/Jina wrap in "results" with
// "relevance_score", text-embeddings-inference returns a bare array with
// "score". All three must decode, or the reranker works against one vendor.
func TestCrossEncoderResponseShapes(t *testing.T) {
	cases := map[string]string{
		"cohere/jina wrapped": `{"results":[{"index":2,"relevance_score":0.9},{"index":0,"relevance_score":0.5},{"index":1,"relevance_score":0.1}]}`,
		"tei bare array":      `[{"index":2,"score":0.9},{"index":0,"score":0.5},{"index":1,"score":0.1}]`,
		"data-wrapped":        `{"data":[{"index":2,"relevance_score":0.9},{"index":0,"relevance_score":0.5},{"index":1,"relevance_score":0.1}]}`,
	}
	for name, body := range cases {
		t.Run(name, func(t *testing.T) {
			ce, _ := serveRerank(t, 200, body)
			got, err := ce.Rerank(context.Background(), "q", hits("a", "b", "c"), 3)
			if err != nil {
				t.Fatal(err)
			}
			want := []string{"c", "a", "b"}
			if strings.Join(hitIDList(got), ",") != strings.Join(want, ",") {
				t.Fatalf("order = %v, want %v", hitIDList(got), want)
			}
			if got[0].Score != 0.9 {
				t.Errorf("top hit score = %v, want the reranker's 0.9", got[0].Score)
			}
		})
	}
}

func TestCrossEncoderSendsQueryAndPassages(t *testing.T) {
	ce, req := serveRerank(t, 200, `{"results":[{"index":0,"relevance_score":1}]}`)
	ce.Model = "rerank-v3"
	if _, err := ce.Rerank(context.Background(), "what is the budget?", hits("a", "b"), 1); err != nil {
		t.Fatal(err)
	}
	if (*req)["query"] != "what is the budget?" {
		t.Errorf("query = %v", (*req)["query"])
	}
	if (*req)["model"] != "rerank-v3" {
		t.Errorf("model = %v", (*req)["model"])
	}
	// Both field names go out, since servers disagree on which they read.
	for _, field := range []string{"documents", "texts"} {
		docs, ok := (*req)[field].([]any)
		if !ok || len(docs) != 2 {
			t.Errorf("%s = %v, want 2 passages", field, (*req)[field])
		}
	}
}

// truncating to topK is the caller's contract; asking the server for fewer than
// it was given is what top_n is for.
func TestCrossEncoderTruncatesToTopK(t *testing.T) {
	ce, _ := serveRerank(t, 200,
		`{"results":[{"index":3,"relevance_score":0.9},{"index":1,"relevance_score":0.8},{"index":0,"relevance_score":0.7},{"index":2,"relevance_score":0.6}]}`)
	got, err := ce.Rerank(context.Background(), "q", hits("a", "b", "c", "d"), 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 {
		t.Fatalf("got %d hits, want 2", len(got))
	}
	if hitIDList(got)[0] != "d" || hitIDList(got)[1] != "b" {
		t.Fatalf("order = %v, want [d b]", hitIDList(got))
	}
}

// A server honouring top_n scores only some passages. The rest must keep
// retrieval order behind the scored ones rather than disappearing.
func TestCrossEncoderKeepsUnscoredCandidates(t *testing.T) {
	ce, _ := serveRerank(t, 200, `{"results":[{"index":2,"relevance_score":0.9}]}`)
	got, err := ce.Rerank(context.Background(), "q", hits("a", "b", "c"), 3)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"c", "a", "b"} // scored first, then retrieval order
	if strings.Join(hitIDList(got), ",") != strings.Join(want, ",") {
		t.Fatalf("order = %v, want %v", hitIDList(got), want)
	}
	// Unscored hits keep their retrieval score rather than being zeroed.
	if got[1].Score != 3 || got[2].Score != 2 {
		t.Errorf("unscored hits lost their retrieval scores: %v, %v", got[1].Score, got[2].Score)
	}
}

// Errors surface rather than degrading to retrieval order. A misconfigured
// endpoint that silently returned the input would be indistinguishable from a
// reranker that never helps — the one thing the measurement must be able to
// tell apart.
func TestCrossEncoderReportsFailures(t *testing.T) {
	cases := map[string]struct {
		status int
		body   string
		want   string
	}{
		"http error":     {500, `{"message":"boom"}`, "500"},
		"unauthorized":   {401, `{"message":"bad key"}`, "401"},
		"malformed json": {200, `{"results":`, "parse"},
		"no scores":      {200, `{"results":[]}`, "no scores"},
		"index too high": {200, `{"results":[{"index":99,"relevance_score":1}]}`, "outside"},
		"negative index": {200, `{"results":[{"index":-1,"relevance_score":1}]}`, "outside"},
		"duplicate index": {200,
			`{"results":[{"index":0,"relevance_score":1},{"index":0,"relevance_score":2}]}`, "twice"},
		"missing score": {200, `{"results":[{"index":0}]}`, "no score field"},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			ce, _ := serveRerank(t, tc.status, tc.body)
			_, err := ce.Rerank(context.Background(), "q", hits("a", "b"), 2)
			if err == nil {
				t.Fatalf("expected an error for %s", name)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("error %q does not mention %q", err, tc.want)
			}
		})
	}
}

func TestCrossEncoderEmptyShortlist(t *testing.T) {
	ce, _ := serveRerank(t, 500, `should not be called`)
	got, err := ce.Rerank(context.Background(), "q", nil, 5)
	if err != nil || len(got) != 0 {
		t.Fatalf("empty shortlist: got %v, %v", got, err)
	}
}

func TestNewCrossEncoderFromEnv(t *testing.T) {
	t.Setenv("RERANK_BASE_URL", "")
	if _, err := NewCrossEncoderFromEnv(); err == nil {
		t.Fatal("expected an error with no RERANK_BASE_URL")
	}

	t.Setenv("RERANK_BASE_URL", "https://example.test/v2/")
	t.Setenv("RERANK_API_KEY", "sk-test")
	t.Setenv("RERANK_MODEL", "bge-reranker-v2-m3")
	ce, err := NewCrossEncoderFromEnv()
	if err != nil {
		t.Fatal(err)
	}
	if ce.BaseURL != "https://example.test/v2" {
		t.Errorf("BaseURL = %q, want the trailing slash trimmed", ce.BaseURL)
	}
	if ce.APIKey != "sk-test" || ce.Model != "bge-reranker-v2-m3" {
		t.Errorf("key/model = %q/%q", ce.APIKey, ce.Model)
	}
}

// The store wires a reranker in through the same path as the LLM one, and must
// over-fetch for it.
func TestStoreUsesCrossEncoder(t *testing.T) {
	ce, req := serveRerank(t, 200, `{"results":[{"index":1,"relevance_score":0.99}]}`)
	s := filteredStore(t)
	s.SetReranker(ce)

	got, err := s.Search(context.Background(), core.Query{Text: "atlas budget", TopK: 1})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 {
		t.Fatalf("got %d hits, want 1", len(got))
	}
	docs, _ := (*req)["documents"].([]any)
	if len(docs) <= 1 {
		t.Errorf("reranker was sent %d passages for topK=1; Search did not over-fetch", len(docs))
	}
}

// RERANK_TIMEOUT exists because the 60s default is genuinely too tight for
// some real configurations: scoring 20 passages with a 568M cross-encoder on
// CPU ran past it, and the failure reads like a broken endpoint rather than a
// slow one.
func TestCrossEncoderTimeoutFromEnv(t *testing.T) {
	t.Setenv("RERANK_BASE_URL", "http://example.test")

	t.Setenv("RERANK_TIMEOUT", "")
	ce, err := NewCrossEncoderFromEnv()
	if err != nil {
		t.Fatal(err)
	}
	if ce.HTTPClient != nil {
		t.Error("unset RERANK_TIMEOUT should leave the default client in place")
	}

	t.Setenv("RERANK_TIMEOUT", "5m")
	if ce, err = NewCrossEncoderFromEnv(); err != nil {
		t.Fatal(err)
	}
	if ce.HTTPClient == nil || ce.HTTPClient.Timeout != 5*time.Minute {
		t.Errorf("HTTPClient = %+v, want a 5m timeout", ce.HTTPClient)
	}

	for _, bad := range []string{"soon", "5", "-30s", "0"} {
		t.Setenv("RERANK_TIMEOUT", bad)
		if _, err := NewCrossEncoderFromEnv(); err == nil {
			t.Errorf("RERANK_TIMEOUT=%q was accepted", bad)
		}
	}
}
