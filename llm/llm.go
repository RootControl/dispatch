// Package llm is an OpenAI-compatible client built on net/http alone. Point
// LLM_BASE_URL at any compatible server (Ollama, llama.cpp, LM Studio,
// mlx-openai-server) or a hosted provider.
//
// The rest of the module depends on the LLM interface, never the concrete
// Client, so tests can substitute a scripted fake and run offline and free.
package llm

import "context"

// Message is one chat message in the OpenAI schema.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// System, User, and Assistant are constructors for the three roles the loop uses.
func System(content string) Message    { return Message{Role: "system", Content: content} }
func User(content string) Message      { return Message{Role: "user", Content: content} }
func Assistant(content string) Message { return Message{Role: "assistant", Content: content} }

// LLM is the capability surface the module needs: chat, schema-constrained chat,
// and embeddings. Everything above this interface is testable without a network.
type LLM interface {
	// Chat returns the assistant's reply to messages.
	Chat(ctx context.Context, messages []Message) (string, error)
	// ChatJSON requests a JSON object reply and unmarshals it into out. The
	// implementation is responsible for retrying malformed output; callers get a
	// populated out or an error, never partial JSON.
	ChatJSON(ctx context.Context, messages []Message, out any) error
	// Embed returns one unit-normalized vector per input, in order.
	Embed(ctx context.Context, inputs []string) ([][]float64, error)
}
