package llm

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// server spins up a stub endpoint. handler receives the decoded request body.
func server(t *testing.T, handler func(path string, body map[string]any) (int, string)) *Client {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(raw, &body)
		code, reply := handler(r.URL.Path, body)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(code)
		_, _ = io.WriteString(w, reply)
	}))
	t.Cleanup(srv.Close)

	c, err := New(Config{BaseURL: srv.URL, APIKey: "test-key", MaxRetries: 3})
	if err != nil {
		t.Fatal(err)
	}
	// Keep backoff from making tests slow.
	c.http = &http.Client{Timeout: 5 * time.Second}
	return c
}

// chatReply builds a minimal successful chat response.
func chatReply(content string) string {
	b, _ := json.Marshal(content)
	return `{"choices":[{"message":{"role":"assistant","content":` + string(b) + `},"finish_reason":"stop"}]}`
}

func TestChatSendsAuthAndModel(t *testing.T) {
	var gotModel string
	var gotAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(raw, &body)
		gotModel, _ = body["model"].(string)
		_, _ = io.WriteString(w, chatReply("hello"))
	}))
	defer srv.Close()

	c, err := New(Config{BaseURL: srv.URL, APIKey: "sk-abc", ChatModel: "my-model"})
	if err != nil {
		t.Fatal(err)
	}
	got, err := c.Chat(context.Background(), []Message{User("hi")})
	if err != nil {
		t.Fatal(err)
	}
	if got != "hello" {
		t.Errorf("Chat = %q", got)
	}
	if gotAuth != "Bearer sk-abc" {
		t.Errorf("Authorization = %q", gotAuth)
	}
	if gotModel != "my-model" {
		t.Errorf("model = %q", gotModel)
	}
}

// Local servers reject an Authorization header they never asked for, so an
// empty key must send none at all.
func TestChatOmitsAuthWhenKeyIsEmpty(t *testing.T) {
	var hadAuth bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, hadAuth = r.Header["Authorization"]
		_, _ = io.WriteString(w, chatReply("ok"))
	}))
	defer srv.Close()

	c, _ := New(Config{BaseURL: srv.URL})
	if _, err := c.Chat(context.Background(), []Message{User("hi")}); err != nil {
		t.Fatal(err)
	}
	if hadAuth {
		t.Error("Authorization header sent with an empty API key")
	}
}

// The failure found on a real corpus: a thinking model spends its budget
// reasoning and returns no content. That must be an error, not a blank answer.
func TestChatRejectsEmptyCompletion(t *testing.T) {
	for name, reply := range map[string]string{
		"empty content": `{"choices":[{"message":{"role":"assistant","content":""},"finish_reason":"stop"}]}`,
		"only whitespace": `{"choices":[{"message":{"role":"assistant","content":"  \n "},` +
			`"finish_reason":"stop"}]}`,
		"reasoning but no answer": `{"choices":[{"message":{"role":"assistant","content":"",` +
			`"reasoning":"thinking at length about the question"},"finish_reason":"length"}]}`,
	} {
		c := server(t, func(string, map[string]any) (int, string) { return 200, reply })
		_, err := c.Chat(context.Background(), []Message{User("q")})
		if err == nil {
			t.Errorf("%s: expected an error", name)
			continue
		}
		if !errors.Is(err, ErrEmptyCompletion) {
			t.Errorf("%s: error should wrap ErrEmptyCompletion, got %v", name, err)
		}
	}
}

// The reasoning case should say what happened and what to do, since the cause
// is not guessable from a blank answer.
func TestEmptyCompletionErrorIsDiagnostic(t *testing.T) {
	c := server(t, func(string, map[string]any) (int, string) {
		return 200, `{"choices":[{"message":{"role":"assistant","content":"","reasoning":"a long chain of thought"},` +
			`"finish_reason":"length"}]}`
	})
	_, err := c.Chat(context.Background(), []Message{User("q")})
	if err == nil {
		t.Fatal("expected an error")
	}
	for _, want := range []string{"reasoning", "length", "non-thinking model"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error should mention %q: %v", want, err)
		}
	}
}

// An empty completion is worth retrying; a fresh sample may reason less.
func TestChatJSONRetriesEmptyCompletion(t *testing.T) {
	var calls atomic.Int32
	c := server(t, func(string, map[string]any) (int, string) {
		if calls.Add(1) == 1 {
			return 200, `{"choices":[{"message":{"role":"assistant","content":"","reasoning":"x"},"finish_reason":"length"}]}`
		}
		return 200, chatReply(`{"ok":true}`)
	})
	var out struct {
		OK bool `json:"ok"`
	}
	if err := c.ChatJSON(context.Background(), []Message{User("q")}, &out); err != nil {
		t.Fatal(err)
	}
	if !out.OK || calls.Load() != 2 {
		t.Errorf("out=%+v calls=%d, want a retry that then succeeded", out, calls.Load())
	}
}

func TestChatJSONStripsCodeFence(t *testing.T) {
	c := server(t, func(string, map[string]any) (int, string) {
		return 200, chatReply("```json\n{\"tiers\":[\"semantic\"]}\n```")
	})
	var out struct {
		Tiers []string `json:"tiers"`
	}
	if err := c.ChatJSON(context.Background(), []Message{User("q")}, &out); err != nil {
		t.Fatal(err)
	}
	if len(out.Tiers) != 1 || out.Tiers[0] != "semantic" {
		t.Errorf("fenced JSON not decoded: %+v", out)
	}
}

func TestChatJSONSetsJSONMode(t *testing.T) {
	var format string
	c := server(t, func(_ string, body map[string]any) (int, string) {
		if rf, ok := body["response_format"].(map[string]any); ok {
			format, _ = rf["type"].(string)
		}
		return 200, chatReply(`{}`)
	})
	var out map[string]any
	if err := c.ChatJSON(context.Background(), []Message{User("q")}, &out); err != nil {
		t.Fatal(err)
	}
	if format != "json_object" {
		t.Errorf("response_format = %q, want json_object", format)
	}
}

// 429 and 5xx are transient; 4xx is not. Retrying a 400 forever wastes the
// caller's time on a request that will never succeed.
func TestRetryPolicy(t *testing.T) {
	cases := []struct {
		status    int
		wantCalls int32
		wantErr   bool
	}{
		{http.StatusTooManyRequests, 4, true}, // 1 + MaxRetries(3)
		{http.StatusInternalServerError, 4, true},
		{http.StatusBadRequest, 1, true},
		{http.StatusUnauthorized, 1, true},
	}
	for _, tc := range cases {
		var calls atomic.Int32
		c := server(t, func(string, map[string]any) (int, string) {
			calls.Add(1)
			return tc.status, `{"error":{"message":"nope","type":"test"}}`
		})
		_, err := c.Chat(context.Background(), []Message{User("q")})
		if (err != nil) != tc.wantErr {
			t.Errorf("status %d: err = %v", tc.status, err)
		}
		if calls.Load() != tc.wantCalls {
			t.Errorf("status %d: %d attempts, want %d", tc.status, calls.Load(), tc.wantCalls)
		}
	}
}

func TestRetryEventuallySucceeds(t *testing.T) {
	var calls atomic.Int32
	c := server(t, func(string, map[string]any) (int, string) {
		if calls.Add(1) < 3 {
			return http.StatusServiceUnavailable, `{}`
		}
		return 200, chatReply("recovered")
	})
	got, err := c.Chat(context.Background(), []Message{User("q")})
	if err != nil {
		t.Fatal(err)
	}
	if got != "recovered" || calls.Load() != 3 {
		t.Errorf("got %q after %d calls", got, calls.Load())
	}
}

func TestContextCancellationStopsRetrying(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	var calls atomic.Int32
	c := server(t, func(string, map[string]any) (int, string) {
		if calls.Add(1) == 1 {
			cancel()
		}
		return http.StatusServiceUnavailable, `{}`
	})
	if _, err := c.Chat(ctx, []Message{User("q")}); err == nil {
		t.Fatal("expected an error after cancellation")
	}
	if n := calls.Load(); n > 2 {
		t.Errorf("kept retrying after cancellation: %d attempts", n)
	}
}

func TestEmbedNormalizesAndOrders(t *testing.T) {
	// Return vectors out of order to prove Index is honored, not arrival order.
	c := server(t, func(string, map[string]any) (int, string) {
		return 200, `{"data":[{"index":1,"embedding":[0,3,4]},{"index":0,"embedding":[3,0,4]}]}`
	})
	got, err := c.Embed(context.Background(), []string{"first", "second"})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 {
		t.Fatalf("got %d vectors", len(got))
	}
	// [3,0,4] has length 5, so normalized it is [0.6, 0, 0.8].
	if math.Abs(got[0][0]-0.6) > 1e-9 || math.Abs(got[0][2]-0.8) > 1e-9 {
		t.Errorf("vector 0 not unit-normalized: %v", got[0])
	}
	for i, v := range got {
		var sum float64
		for _, x := range v {
			sum += x * x
		}
		if math.Abs(math.Sqrt(sum)-1) > 1e-9 {
			t.Errorf("vector %d has length %v, want 1", i, math.Sqrt(sum))
		}
	}
}

// A count mismatch means vectors would be silently misaligned with their
// chunks, which corrupts every later search.
func TestEmbedRejectsCountMismatch(t *testing.T) {
	c := server(t, func(string, map[string]any) (int, string) {
		return 200, `{"data":[{"index":0,"embedding":[1,0]}]}`
	})
	if _, err := c.Embed(context.Background(), []string{"a", "b"}); err == nil {
		t.Fatal("expected an error when the server returns too few vectors")
	}
}

func TestEmbedEmptyInputMakesNoRequest(t *testing.T) {
	var calls atomic.Int32
	c := server(t, func(string, map[string]any) (int, string) {
		calls.Add(1)
		return 200, `{"data":[]}`
	})
	got, err := c.Embed(context.Background(), nil)
	if err != nil || got != nil {
		t.Errorf("Embed(nil) = %v, %v", got, err)
	}
	if calls.Load() != 0 {
		t.Error("empty input should not hit the network")
	}
}

// Contextual chunking fans out one call per chunk; the semaphore is what keeps
// a large ingest from opening hundreds of connections at once.
func TestConcurrencyLimit(t *testing.T) {
	var mu sync.Mutex
	var inFlight, peak int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		inFlight++
		if inFlight > peak {
			peak = inFlight
		}
		mu.Unlock()
		time.Sleep(20 * time.Millisecond)
		mu.Lock()
		inFlight--
		mu.Unlock()
		_, _ = io.WriteString(w, chatReply("ok"))
	}))
	defer srv.Close()

	c, _ := New(Config{BaseURL: srv.URL, MaxConcurrent: 2})
	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, _ = c.Chat(context.Background(), []Message{User("q")})
		}()
	}
	wg.Wait()

	mu.Lock()
	defer mu.Unlock()
	if peak > 2 {
		t.Errorf("peak concurrency %d exceeded MaxConcurrent 2", peak)
	}
}

func TestNewRequiresBaseURL(t *testing.T) {
	t.Setenv("LLM_BASE_URL", "")
	if _, err := New(Config{}); err == nil {
		t.Fatal("expected an error without a base URL")
	}
}

func TestNewAppliesEnvAndDefaults(t *testing.T) {
	t.Setenv("LLM_BASE_URL", "http://example.test/v1/")
	t.Setenv("LLM_CHAT_MODEL", "")
	c, err := New(Config{})
	if err != nil {
		t.Fatal(err)
	}
	// The trailing slash must be trimmed or every path becomes a double slash.
	if c.cfg.BaseURL != "http://example.test/v1" {
		t.Errorf("BaseURL = %q", c.cfg.BaseURL)
	}
	if c.ChatModel() == "" || c.EmbedModel() == "" {
		t.Errorf("models should default: chat=%q embed=%q", c.ChatModel(), c.EmbedModel())
	}
}

func TestStripFence(t *testing.T) {
	cases := map[string]string{
		"```json\n{\"a\":1}\n```": `{"a":1}`,
		"```\n{\"a\":1}\n```":     `{"a":1}`,
		`{"a":1}`:                 `{"a":1}`,
		"  {\"a\":1}  ":           `{"a":1}`,
	}
	for in, want := range cases {
		if got := stripFence(in); got != want {
			t.Errorf("stripFence(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestBackoffIsBoundedAndIncreasing(t *testing.T) {
	prev := time.Duration(0)
	for attempt := 1; attempt <= 10; attempt++ {
		d := backoff(attempt)
		if d > 30*time.Second {
			t.Errorf("attempt %d: backoff %v exceeds the 30s cap", attempt, d)
		}
		if attempt < 7 && d <= prev {
			t.Errorf("attempt %d: backoff %v did not grow from %v", attempt, d, prev)
		}
		prev = d
	}
}
