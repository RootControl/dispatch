package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"
)

// Config configures a Client. Zero values fall back to environment variables and
// then to sane defaults, so a caller can pass Config{} and rely on the env.
type Config struct {
	BaseURL    string // LLM_BASE_URL, e.g. https://api.openai.com/v1
	APIKey     string // LLM_API_KEY
	ChatModel  string // LLM_CHAT_MODEL
	EmbedModel string // LLM_EMBED_MODEL
	// MaxConcurrent bounds simultaneous in-flight requests. Contextual chunking
	// fans out one call per chunk, so this is the throttle that keeps a large
	// ingest from hammering the endpoint. Default 4.
	MaxConcurrent int
	// MaxRetries bounds exponential backoff on 429/5xx. Default 4.
	MaxRetries int
	HTTPClient *http.Client
}

// Client is an OpenAI-compatible LLM over net/http. It satisfies LLM.
type Client struct {
	cfg    Config
	http   *http.Client
	tokens chan struct{} // semaphore; cap == MaxConcurrent

	usageMu sync.Mutex
	used    Usage
}

var _ LLM = (*Client)(nil)

// New builds a Client, filling unset Config fields from the environment. It
// returns an error only when no base URL can be determined, since every other
// field has a workable default.
func New(cfg Config) (*Client, error) {
	if cfg.BaseURL == "" {
		cfg.BaseURL = os.Getenv("LLM_BASE_URL")
	}
	if cfg.BaseURL == "" {
		return nil, errors.New("llm: no base URL (set LLM_BASE_URL or Config.BaseURL)")
	}
	cfg.BaseURL = strings.TrimRight(cfg.BaseURL, "/")
	if cfg.APIKey == "" {
		cfg.APIKey = os.Getenv("LLM_API_KEY")
	}
	if cfg.ChatModel == "" {
		cfg.ChatModel = envOr("LLM_CHAT_MODEL", "gpt-4o-mini")
	}
	if cfg.EmbedModel == "" {
		cfg.EmbedModel = envOr("LLM_EMBED_MODEL", "text-embedding-3-small")
	}
	if cfg.MaxConcurrent <= 0 {
		cfg.MaxConcurrent = 4
	}
	if cfg.MaxRetries <= 0 {
		cfg.MaxRetries = 4
	}
	hc := cfg.HTTPClient
	if hc == nil {
		hc = &http.Client{Timeout: 120 * time.Second}
	}
	return &Client{
		cfg:    cfg,
		http:   hc,
		tokens: make(chan struct{}, cfg.MaxConcurrent),
	}, nil
}

// ChatModel and EmbedModel report the resolved model names. Callers use these to
// tag caches and saved indexes so a model swap invalidates derived data rather
// than silently reusing it.
func (c *Client) ChatModel() string  { return c.cfg.ChatModel }
func (c *Client) EmbedModel() string { return c.cfg.EmbedModel }

// Usage is the running token total for a Client.
type Usage struct {
	Calls            int
	PromptTokens     int
	CompletionTokens int
}

// Total returns prompt plus completion tokens.
func (u Usage) Total() int { return u.PromptTokens + u.CompletionTokens }

// Usage reports what this client has spent since it was created.
//
// The counters come from the server's own usage block, not an estimate: a
// server that omits it leaves them at zero, which reads as "not reported"
// rather than as a wrong number. Calls counts every chat request including
// retries, so a model that had to be asked twice shows as two.
func (c *Client) Usage() Usage {
	c.usageMu.Lock()
	defer c.usageMu.Unlock()
	return c.used
}

func (c *Client) account(u *usage) {
	c.usageMu.Lock()
	defer c.usageMu.Unlock()
	c.used.Calls++
	if u != nil {
		c.used.PromptTokens += u.PromptTokens
		c.used.CompletionTokens += u.CompletionTokens
	}
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// ---- OpenAI wire types (only the fields we use) ----

type chatReq struct {
	Model          string          `json:"model"`
	Messages       []Message       `json:"messages"`
	ResponseFormat *responseFormat `json:"response_format,omitempty"`
	Temperature    float64         `json:"temperature"`
}

type responseFormat struct {
	Type string `json:"type"` // "json_object"
}

type chatResp struct {
	Choices []struct {
		Message struct {
			Role    string `json:"role"`
			Content string `json:"content"`
			// Reasoning is non-standard but returned by thinking models via
			// Ollama. It is read only to diagnose an empty Content.
			Reasoning string `json:"reasoning"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage *usage    `json:"usage"`
	Error *apiError `json:"error"`
}

// usage is the token accounting OpenAI-compatible servers return. Ollama and
// llama.cpp both send it; a server that does not simply leaves the counters at
// zero, which Usage reports as such rather than as an estimate.
type usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type embedReq struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

type embedResp struct {
	Data []struct {
		Index     int       `json:"index"`
		Embedding []float64 `json:"embedding"`
	} `json:"data"`
	Error *apiError `json:"error"`
}

type apiError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
}

func (e *apiError) Error() string { return fmt.Sprintf("%s: %s", e.Type, e.Message) }

// ErrEmptyCompletion reports that the model returned no usable content. It is
// worth retrying: thinking models are stochastic, and a second attempt often
// produces a shorter reasoning chain that leaves room for an answer.
var ErrEmptyCompletion = errors.New("llm: empty completion")

// Chat implements LLM.
func (c *Client) Chat(ctx context.Context, messages []Message) (string, error) {
	return c.chat(ctx, messages, false)
}

// ChatJSON implements LLM. It sets JSON response mode and retries on unmarshal
// failure, since even JSON-mode models occasionally emit prose or fenced code.
func (c *Client) ChatJSON(ctx context.Context, messages []Message, out any) error {
	var lastErr error
	for attempt := 0; attempt <= c.cfg.MaxRetries; attempt++ {
		raw, err := c.chat(ctx, messages, true)
		if errors.Is(err, ErrEmptyCompletion) {
			// Retry: a fresh sample may reason less and leave room to answer.
			lastErr = err
			continue
		}
		if err != nil {
			return err
		}
		if err := json.Unmarshal([]byte(stripFence(raw)), out); err != nil {
			lastErr = fmt.Errorf("llm: decode JSON reply: %w (got %q)", err, truncate(raw, 200))
			continue
		}
		return nil
	}
	return lastErr
}

func (c *Client) chat(ctx context.Context, messages []Message, jsonMode bool) (string, error) {
	body := chatReq{Model: c.cfg.ChatModel, Messages: messages}
	if jsonMode {
		body.ResponseFormat = &responseFormat{Type: "json_object"}
	}
	var resp chatResp
	if err := c.do(ctx, "/chat/completions", body, &resp); err != nil {
		return "", err
	}
	// Account before checking for an error: a call that spent tokens and then
	// returned nothing usable is exactly the one worth seeing in the total.
	c.account(resp.Usage)
	if resp.Error != nil {
		return "", resp.Error
	}
	if len(resp.Choices) == 0 {
		return "", errors.New("llm: empty choices in chat response")
	}
	choice := resp.Choices[0]

	// An empty completion is never usable, and returning it silently produces a
	// blank answer that looks like a content problem rather than a model one.
	// Thinking models hit this by spending the whole output budget on reasoning
	// and emitting no content, so say that explicitly when it happens.
	if strings.TrimSpace(choice.Message.Content) == "" {
		if r := strings.TrimSpace(choice.Message.Reasoning); r != "" {
			return "", fmt.Errorf("%w: model returned %d characters of reasoning but no answer (finish_reason %q): "+
				"the output budget was spent thinking — shorten the prompt, lower the evidence count, or use a non-thinking model",
				ErrEmptyCompletion, len(r), choice.FinishReason)
		}
		return "", fmt.Errorf("%w (finish_reason %q)", ErrEmptyCompletion, choice.FinishReason)
	}
	return choice.Message.Content, nil
}

// Embed implements LLM. Vectors are unit-normalized so downstream cosine
// similarity is a plain dot product.
func (c *Client) Embed(ctx context.Context, inputs []string) ([][]float64, error) {
	if len(inputs) == 0 {
		return nil, nil
	}
	var resp embedResp
	if err := c.do(ctx, "/embeddings", embedReq{Model: c.cfg.EmbedModel, Input: inputs}, &resp); err != nil {
		return nil, err
	}
	if resp.Error != nil {
		return nil, resp.Error
	}
	if len(resp.Data) != len(inputs) {
		return nil, fmt.Errorf("llm: embedding count mismatch: got %d for %d inputs", len(resp.Data), len(inputs))
	}
	out := make([][]float64, len(inputs))
	for _, d := range resp.Data {
		if d.Index < 0 || d.Index >= len(out) {
			return nil, fmt.Errorf("llm: embedding index %d out of range", d.Index)
		}
		out[d.Index] = normalize(d.Embedding)
	}
	return out, nil
}

// do performs a JSON POST with retry/backoff, respecting the concurrency
// semaphore and ctx cancellation.
func (c *Client) do(ctx context.Context, path string, reqBody, out any) error {
	payload, err := json.Marshal(reqBody)
	if err != nil {
		return err
	}
	var lastErr error
	for attempt := 0; attempt <= c.cfg.MaxRetries; attempt++ {
		if attempt > 0 {
			if err := sleep(ctx, backoff(attempt)); err != nil {
				return err
			}
		}
		retryable, err := c.attempt(ctx, path, payload, out)
		if err == nil {
			return nil
		}
		lastErr = err
		if !retryable {
			return err
		}
	}
	return fmt.Errorf("llm: exhausted retries: %w", lastErr)
}

// attempt makes one request. The bool reports whether a failure is worth
// retrying (transport error, 429, or 5xx); 4xx other than 429 is terminal.
func (c *Client) attempt(ctx context.Context, path string, payload []byte, out any) (retryable bool, err error) {
	select {
	case c.tokens <- struct{}{}:
		defer func() { <-c.tokens }()
	case <-ctx.Done():
		return false, ctx.Err()
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.cfg.BaseURL+path, bytes.NewReader(payload))
	if err != nil {
		return false, err
	}
	req.Header.Set("Content-Type", "application/json")
	if c.cfg.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.cfg.APIKey)
	}

	resp, err := c.http.Do(req)
	if err != nil {
		return true, err // transport failure — retry
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return true, err
	}
	if resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode >= 500 {
		return true, fmt.Errorf("llm: %s: %s", resp.Status, truncate(string(data), 200))
	}
	if resp.StatusCode >= 400 {
		return false, fmt.Errorf("llm: %s: %s", resp.Status, truncate(string(data), 200))
	}
	if err := json.Unmarshal(data, out); err != nil {
		return false, fmt.Errorf("llm: decode response: %w", err)
	}
	return false, nil
}

// backoff returns an exponential delay: 0.5s, 1s, 2s, 4s ... capped at 30s.
func backoff(attempt int) time.Duration {
	d := time.Duration(math.Pow(2, float64(attempt-1))*500) * time.Millisecond
	return min(d, 30*time.Second)
}

func sleep(ctx context.Context, d time.Duration) error {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func normalize(v []float64) []float64 {
	var sum float64
	for _, x := range v {
		sum += x * x
	}
	if sum == 0 {
		return v
	}
	norm := math.Sqrt(sum)
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = x / norm
	}
	return out
}

// stripFence removes a ```json ... ``` fence if a model wrapped its JSON in one.
func stripFence(s string) string {
	s = strings.TrimSpace(s)
	if !strings.HasPrefix(s, "```") {
		return s
	}
	s = strings.TrimPrefix(s, "```")
	s = strings.TrimPrefix(s, "json")
	if i := strings.LastIndex(s, "```"); i >= 0 {
		s = s[:i]
	}
	return strings.TrimSpace(s)
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
