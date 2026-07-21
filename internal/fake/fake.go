// Package fake provides a scripted LLM and a deterministic embedder so the rest
// of the module can be tested offline, for free, and reproducibly. It satisfies
// llm.LLM. Test-only: nothing in the shipped binary imports it.
package fake

import (
	"context"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"math"
	"strings"
	"sync"
	"unicode"

	"github.com/RootControl/dispatch/llm"
)

// LLM is a scripted implementation of llm.LLM. Chat and ChatJSON replies come
// from ChatFunc; embeddings come from a deterministic hash embedder so identical
// text always yields an identical unit vector.
type LLM struct {
	// ChatFunc computes a reply from the messages. If nil, Chat returns "" and
	// ChatJSON returns an empty JSON object. Set it to script tier/loop behavior.
	ChatFunc func(messages []llm.Message) (string, error)
	// Dims is the embedding width. Default 64.
	Dims int

	mu    sync.Mutex
	calls int // total Chat+ChatJSON invocations, for assertions
}

var _ llm.LLM = (*LLM)(nil)

// Calls reports how many chat calls have been made — handy for asserting that
// the contextual-chunk cache actually prevented repeat LLM work.
func (f *LLM) Calls() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.calls
}

func (f *LLM) Chat(ctx context.Context, messages []llm.Message) (string, error) {
	f.mu.Lock()
	f.calls++
	f.mu.Unlock()
	if f.ChatFunc == nil {
		return "", nil
	}
	return f.ChatFunc(messages)
}

func (f *LLM) ChatJSON(ctx context.Context, messages []llm.Message, out any) error {
	reply, err := f.Chat(ctx, messages)
	if err != nil {
		return err
	}
	if strings.TrimSpace(reply) == "" {
		reply = "{}"
	}
	if err := json.Unmarshal([]byte(reply), out); err != nil {
		return fmt.Errorf("fake: script returned non-JSON %q: %w", reply, err)
	}
	return nil
}

// Embed returns one deterministic unit vector per input. The mapping is a
// bag-of-token-hashes: vectors for texts that share tokens point in similar
// directions, so cosine similarity is meaningful enough to exercise ranking.
func (f *LLM) Embed(ctx context.Context, inputs []string) ([][]float64, error) {
	dims := f.Dims
	if dims <= 0 {
		dims = 64
	}
	out := make([][]float64, len(inputs))
	for i, in := range inputs {
		out[i] = embed(in, dims)
	}
	return out, nil
}

func embed(text string, dims int) []float64 {
	v := make([]float64, dims)
	// Split on non-alphanumeric runes, not whitespace: otherwise "budget." and
	// "budget" would hash to different dimensions and never reinforce each
	// other, an artifact no real tokenizer has.
	toks := strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsNumber(r)
	})
	for _, tok := range toks {
		h := fnv.New32a()
		_, _ = h.Write([]byte(tok))
		v[h.Sum32()%uint32(dims)] += 1
	}
	// Unit-normalize so callers can treat cosine as a dot product, matching the
	// real client's contract.
	var sum float64
	for _, x := range v {
		sum += x * x
	}
	if sum == 0 {
		return v
	}
	norm := math.Sqrt(sum)
	for i := range v {
		v[i] /= norm
	}
	return v
}
