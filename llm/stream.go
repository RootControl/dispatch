package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// Streamer is the optional streaming half of the LLM surface. It is a separate
// interface rather than a method on LLM so the scripted fake, the reranker and
// every other implementation stay as small as they were: a caller type-asserts
// for it and falls back to Chat when it is absent.
type Streamer interface {
	// ChatStream calls onDelta with each content fragment as it arrives and
	// returns the complete text. onDelta may be nil, in which case this is Chat
	// with extra steps.
	//
	// A returned error means the answer is incomplete; whatever was streamed
	// before it should not be treated as an answer.
	ChatStream(ctx context.Context, messages []Message, onDelta func(string)) (string, error)
}

var _ Streamer = (*Client)(nil)

// streamChunk is one server-sent event from an OpenAI-compatible stream.
type streamChunk struct {
	Choices []struct {
		Delta struct {
			Content   string `json:"content"`
			Reasoning string `json:"reasoning"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage *usage    `json:"usage"`
	Error *apiError `json:"error"`
}

// ChatStream implements Streamer.
//
// Streaming is not retried. The retry loop behind Chat exists to paper over a
// transient failure before anything is observable; once fragments have been
// handed to the caller they cannot be unsent, and a silent second attempt would
// emit a second answer over the first. A streaming failure is the caller's to
// retry, at the level where the partial output is visible.
func (c *Client) ChatStream(ctx context.Context, messages []Message, onDelta func(string)) (string, error) {
	payload, err := json.Marshal(streamReq{
		chatReq: chatReq{Model: c.cfg.ChatModel, Messages: messages},
		Stream:  true,
		Options: &streamOptions{IncludeUsage: true},
	})
	if err != nil {
		return "", err
	}

	// Hold a concurrency token for the whole stream: a streamed call occupies
	// the server for its entire duration, so releasing early would let more
	// requests through than MaxConcurrent allows.
	select {
	case c.tokens <- struct{}{}:
		defer func() { <-c.tokens }()
	case <-ctx.Done():
		return "", ctx.Err()
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		c.cfg.BaseURL+"/chat/completions", bytes.NewReader(payload))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	if c.cfg.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.cfg.APIKey)
	}

	resp, err := c.http.Do(req)
	if err != nil {
		return "", fmt.Errorf("llm: stream: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<16))
		return "", fmt.Errorf("llm: stream: %s: %s", resp.Status, truncate(string(body), 200))
	}

	text, st, err := readStream(resp.Body, onDelta)
	// Account once per stream, after it finishes — including on failure, since
	// a stream that died halfway still burned what it burned. Accounting inside
	// the read loop would count one call per server-sent event: a single
	// streamed answer reported as forty calls.
	c.account(st.usage)
	if err != nil {
		return "", err
	}

	// The same empty-completion diagnosis as the non-streaming path. It matters
	// more here, not less: the caller watched nothing arrive for thirty seconds
	// and deserves to be told the output budget went on reasoning rather than
	// being handed a blank answer.
	if strings.TrimSpace(text) == "" {
		if st.reasoning > 0 {
			return "", fmt.Errorf("%w: model streamed %d characters of reasoning but no answer (finish_reason %q): "+
				"the output budget was spent thinking — shorten the prompt, lower the evidence count, or use a non-thinking model",
				ErrEmptyCompletion, st.reasoning, st.finish)
		}
		return "", fmt.Errorf("%w (finish_reason %q)", ErrEmptyCompletion, st.finish)
	}
	return text, nil
}

// streamState is what a stream reports besides its content.
type streamState struct {
	reasoning int    // reasoning characters seen, to diagnose an empty answer
	finish    string // the last finish_reason
	usage     *usage // from the final event, when the server sends one
}

// readStream consumes the SSE body, returning the assembled content and what
// the stream said about itself.
func readStream(body io.Reader, onDelta func(string)) (text string, st streamState, err error) {
	var b strings.Builder
	sc := bufio.NewScanner(body)
	// Server-sent events are line-delimited, but one line carries a whole JSON
	// object — which for a long reasoning delta can exceed the 64 KB default.
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)

	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || !strings.HasPrefix(line, "data:") {
			continue // comments, keep-alives, and the blank line between events
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "[DONE]" {
			break
		}

		var chunk streamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			// One malformed event is not worth discarding a good answer over;
			// the stream carries keep-alives and vendor extensions too.
			continue
		}
		if chunk.Error != nil {
			return "", st, chunk.Error
		}
		if chunk.Usage != nil {
			st.usage = chunk.Usage // present only on the final event, if at all
		}
		if len(chunk.Choices) == 0 {
			continue
		}
		choice := chunk.Choices[0]
		if choice.FinishReason != "" {
			st.finish = choice.FinishReason
		}
		st.reasoning += len(choice.Delta.Reasoning)
		if d := choice.Delta.Content; d != "" {
			b.WriteString(d)
			if onDelta != nil {
				onDelta(d)
			}
		}
	}
	if err := sc.Err(); err != nil {
		return "", st, fmt.Errorf("llm: stream: %w", err)
	}
	// A stream that ends without [DONE] and without content is a truncated
	// connection, not an empty answer — different causes, different fixes.
	if b.Len() == 0 && st.finish == "" && st.reasoning == 0 {
		return "", st, errors.New("llm: stream ended before any content arrived")
	}
	return b.String(), st, nil
}

// streamReq is chatReq plus the streaming fields. Embedded rather than
// duplicated so the two paths cannot drift on model or messages.
type streamReq struct {
	chatReq
	Stream  bool           `json:"stream"`
	Options *streamOptions `json:"stream_options,omitempty"`
}

// streamOptions asks for a final usage event. OpenAI and llama.cpp honour it;
// Ollama ignores it and reports usage on the last chunk anyway, and a server
// that does neither simply leaves the counters at zero.
type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}
