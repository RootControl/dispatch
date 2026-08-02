package llm

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// sseServer replays the given SSE lines.
func sseServer(t *testing.T, lines ...string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		raw, _ := io.ReadAll(r.Body)
		_ = decodeJSON(raw, &body)
		if body["stream"] != true {
			t.Errorf("request did not set stream: %v", body)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		for _, l := range lines {
			io.WriteString(w, l+"\n\n")
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
		}
	}))
	t.Cleanup(srv.Close)
	return srv
}

func streamClient(t *testing.T, srv *httptest.Server) *Client {
	t.Helper()
	c, err := New(Config{BaseURL: srv.URL, ChatModel: "m", EmbedModel: "e", HTTPClient: srv.Client()})
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func delta(content string) string {
	return `data: {"choices":[{"delta":{"content":"` + content + `"}}]}`
}

func TestChatStreamAssemblesFragments(t *testing.T) {
	srv := sseServer(t, delta("The "), delta("budget "), delta("is 4M."), "data: [DONE]")
	c := streamClient(t, srv)

	var seen []string
	got, err := c.ChatStream(context.Background(), []Message{User("q")}, func(s string) {
		seen = append(seen, s)
	})
	if err != nil {
		t.Fatal(err)
	}
	if got != "The budget is 4M." {
		t.Errorf("assembled %q", got)
	}
	// The whole point is that fragments arrive separately rather than at the end.
	if len(seen) != 3 {
		t.Errorf("onDelta called %d time(s), want 3: %v", len(seen), seen)
	}
	if strings.Join(seen, "") != got {
		t.Errorf("fragments %q do not reassemble to %q", strings.Join(seen, ""), got)
	}
}

// A nil callback is Chat with extra steps, and must still return the text.
func TestChatStreamWithoutCallback(t *testing.T) {
	srv := sseServer(t, delta("hello"), "data: [DONE]")
	got, err := streamClient(t, srv).ChatStream(context.Background(), []Message{User("q")}, nil)
	if err != nil || got != "hello" {
		t.Fatalf("got %q, %v", got, err)
	}
}

// Keep-alives, comments and vendor extensions ride along on a real stream. One
// malformed event is not worth discarding a good answer over.
func TestChatStreamToleratesNoise(t *testing.T) {
	srv := sseServer(t,
		": keep-alive",
		"",
		delta("a"),
		"data: {not json}",
		`data: {"choices":[]}`,
		delta("b"),
		"data: [DONE]",
	)
	got, err := streamClient(t, srv).ChatStream(context.Background(), []Message{User("q")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got != "ab" {
		t.Errorf("got %q, want the content around the noise", got)
	}
}

// The same empty-completion diagnosis as the non-streaming path. It matters
// more here: the caller watched nothing arrive and deserves the reason.
func TestChatStreamDiagnosesReasoningOnlyOutput(t *testing.T) {
	srv := sseServer(t,
		`data: {"choices":[{"delta":{"reasoning":"thinking at some length about this"}}]}`,
		`data: {"choices":[{"delta":{},"finish_reason":"length"}]}`,
		"data: [DONE]",
	)
	_, err := streamClient(t, srv).ChatStream(context.Background(), []Message{User("q")}, nil)
	if !errors.Is(err, ErrEmptyCompletion) {
		t.Fatalf("err = %v, want ErrEmptyCompletion", err)
	}
	for _, want := range []string{"reasoning", "non-thinking", "length"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not mention %q", err, want)
		}
	}
}

func TestChatStreamReportsServerErrors(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(503)
		io.WriteString(w, `{"error":{"message":"overloaded"}}`)
	}))
	defer srv.Close()

	_, err := streamClient(t, srv).ChatStream(context.Background(), []Message{User("q")}, nil)
	if err == nil || !strings.Contains(err.Error(), "503") {
		t.Fatalf("err = %v, want the status", err)
	}
}

// An in-band error event must surface, not be mistaken for an empty answer.
func TestChatStreamReportsInBandErrors(t *testing.T) {
	srv := sseServer(t, delta("partial"), `data: {"error":{"message":"context length exceeded"}}`)
	_, err := streamClient(t, srv).ChatStream(context.Background(), []Message{User("q")}, nil)
	if err == nil || !strings.Contains(err.Error(), "context length") {
		t.Fatalf("err = %v, want the in-band error", err)
	}
}

// A connection that dies before any content is a truncated stream, not an empty
// answer — different causes, different fixes.
func TestChatStreamDistinguishesATruncatedConnection(t *testing.T) {
	srv := sseServer(t)
	_, err := streamClient(t, srv).ChatStream(context.Background(), []Message{User("q")}, nil)
	if err == nil {
		t.Fatal("a stream with no events returned an answer")
	}
	if !strings.Contains(err.Error(), "before any content") {
		t.Errorf("err = %v, want it to name the truncation", err)
	}
}

func TestChatStreamAccountsUsage(t *testing.T) {
	srv := sseServer(t, delta("hi"),
		`data: {"choices":[],"usage":{"prompt_tokens":80,"completion_tokens":5}}`,
		"data: [DONE]")
	c := streamClient(t, srv)
	if _, err := c.ChatStream(context.Background(), []Message{User("q")}, nil); err != nil {
		t.Fatal(err)
	}
	if u := c.Usage(); u.PromptTokens != 80 || u.CompletionTokens != 5 {
		t.Errorf("usage = %+v", u)
	}
}

func TestChatStreamHonoursCancellation(t *testing.T) {
	srv := sseServer(t, delta("a"), "data: [DONE]")
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := streamClient(t, srv).ChatStream(ctx, []Message{User("q")}, nil); err == nil {
		t.Fatal("a cancelled context streamed an answer")
	}
}

// Client must satisfy Streamer, which is what agent type-asserts for.
func TestClientIsAStreamer(t *testing.T) {
	var _ Streamer = (*Client)(nil)
}

func decodeJSON(b []byte, out any) error { return json.Unmarshal(b, out) }

// Accounting is once per stream, not once per event. Doing it inside the read
// loop reported a single streamed answer as forty calls.
func TestChatStreamCountsOneCallPerStream(t *testing.T) {
	lines := make([]string, 0, 42)
	for range 40 {
		lines = append(lines, delta("x "))
	}
	lines = append(lines,
		`data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":40}}`,
		"data: [DONE]")

	c := streamClient(t, sseServer(t, lines...))
	if _, err := c.ChatStream(context.Background(), []Message{User("q")}, nil); err != nil {
		t.Fatal(err)
	}
	u := c.Usage()
	if u.Calls != 1 {
		t.Errorf("Calls = %d after one stream of 40 events, want 1", u.Calls)
	}
	if u.PromptTokens != 10 || u.CompletionTokens != 40 {
		t.Errorf("usage = %+v, want the final event's totals counted once", u)
	}
}

// A stream that dies halfway still burned what it burned.
func TestChatStreamAccountsFailedStreams(t *testing.T) {
	c := streamClient(t, sseServer(t, delta("partial"),
		`data: {"error":{"message":"context length exceeded"}}`))
	if _, err := c.ChatStream(context.Background(), []Message{User("q")}, nil); err == nil {
		t.Fatal("expected an error")
	}
	if u := c.Usage(); u.Calls != 1 {
		t.Errorf("Calls = %d, want the failed stream counted", u.Calls)
	}
}
