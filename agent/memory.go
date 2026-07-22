package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/RootControl/dispatch/core"
	"github.com/RootControl/dispatch/index"
	"github.com/RootControl/dispatch/llm"
)

// Memory is the tiered write-back store: a bounded "core" block that is always
// in the prompt, plus an unbounded "archival" store that is searchable. When
// core overflows its budget, the oldest entries are evicted into archival rather
// than dropped — they stop being free to recall but remain findable.
//
// This is the MemGPT/Letta pattern, and it is what makes retrieval improve
// across questions and sessions instead of starting cold every time.
type Memory struct {
	llm      llm.LLM
	archival *index.Store
	dir      string
	budget   int // core block size in characters

	mu   sync.Mutex
	core []CoreEntry
	seq  int
}

// CoreEntry is one takeaway held in the always-in-prompt block.
type CoreEntry struct {
	Text  string    `json:"text"`
	Added time.Time `json:"added"`
}

// MemoryConfig configures a Memory.
type MemoryConfig struct {
	LLM        llm.LLM // embeddings for archival search
	Dir        string  // persistence directory; default .dispatch/memory
	CoreBudget int     // core block characters; default 2000
	EmbedTag   string  // embedding-model identity for the archival index
}

// NewMemory builds a Memory. Archival entries are not contextually chunked:
// a takeaway is already written to stand alone, so a situating sentence would
// be a wasted LLM call per memory.
func NewMemory(cfg MemoryConfig) *Memory {
	if cfg.Dir == "" {
		cfg.Dir = filepath.Join(".dispatch", "memory")
	}
	if cfg.CoreBudget <= 0 {
		cfg.CoreBudget = 2000
	}
	return &Memory{
		llm:      cfg.LLM,
		dir:      cfg.Dir,
		budget:   cfg.CoreBudget,
		archival: index.New(index.Config{LLM: cfg.LLM, EmbedTag: cfg.EmbedTag}),
	}
}

// Memory is a Retriever over its archival store, so the agent loop can fan out
// to remembered material exactly like any other tier.
var _ core.Retriever = (*Memory)(nil)

func (m *Memory) Tier() core.Tier { return core.TierMemory }

// Retrieve searches archival memory. The core block is not searched: it is
// already in every prompt, so returning it as evidence would duplicate it.
func (m *Memory) Retrieve(ctx context.Context, q core.Query) ([]core.Result, error) {
	if m.archival.Len() == 0 {
		return nil, nil
	}
	hits, err := m.archival.Search(ctx, q.Text, q.TopK)
	if err != nil {
		return nil, err
	}
	out := make([]core.Result, 0, len(hits))
	for _, h := range hits {
		out = append(out, core.Result{
			Tier:     core.TierMemory,
			SourceID: h.Chunk.ID,
			Text:     h.Chunk.Text,
			Score:    h.Score,
		})
	}
	return out, nil
}

// Remember adds a takeaway to the core block, evicting the oldest entries into
// archival if that pushes core over budget.
func (m *Memory) Remember(ctx context.Context, text string) error {
	text = strings.TrimSpace(text)
	if text == "" {
		return nil
	}

	m.mu.Lock()
	for _, e := range m.core {
		if e.Text == text {
			m.mu.Unlock()
			return nil // already known; nothing to do
		}
	}
	m.core = append(m.core, CoreEntry{Text: text, Added: time.Now()})
	evicted := m.evictLocked()
	m.mu.Unlock()

	// Archiving embeds, so it happens outside the lock.
	for _, e := range evicted {
		if err := m.archive(ctx, e); err != nil {
			return err
		}
	}
	return nil
}

// evictLocked pops the oldest entries until core fits its budget. The newest
// entry is never evicted, so a single oversized takeaway still lands in core
// rather than vanishing straight into archival.
func (m *Memory) evictLocked() []CoreEntry {
	var evicted []CoreEntry
	for len(m.core) > 1 && m.sizeLocked() > m.budget {
		evicted = append(evicted, m.core[0])
		m.core = m.core[1:]
	}
	return evicted
}

func (m *Memory) sizeLocked() int {
	n := 0
	for _, e := range m.core {
		n += len(e.Text) + 3 // "- " prefix and newline
	}
	return n
}

func (m *Memory) archive(ctx context.Context, e CoreEntry) error {
	m.mu.Lock()
	m.seq++
	id := fmt.Sprintf("mem-%d", m.seq)
	m.mu.Unlock()

	_, err := m.archival.Ingest(ctx, []core.Doc{{
		ID:   id,
		Text: e.Text,
		Meta: map[string]string{"added": e.Added.Format(time.RFC3339)},
	}})
	return err
}

// CoreBlock renders the always-in-prompt memory. Empty when nothing is
// remembered, so callers can omit the section entirely.
func (m *Memory) CoreBlock() string {
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(m.core) == 0 {
		return ""
	}
	var b strings.Builder
	for _, e := range m.core {
		fmt.Fprintf(&b, "- %s\n", e.Text)
	}
	return strings.TrimRight(b.String(), "\n")
}

// CoreLen reports how many entries are in the core block.
func (m *Memory) CoreLen() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.core)
}

// ArchivalLen reports how many entries have been evicted into archival.
func (m *Memory) ArchivalLen() int { return m.archival.Len() }

const takeawaySystem = `You decide what is worth remembering from a question-and-answer exchange.

Record a takeaway only if it is a SPECIFIC, durable fact about the subject matter: named
entities, concrete values, dates, relationships. It must be self-contained, because it will
be recalled without the question that produced it.

Never generalize. A fact stripped of its specifics is worthless here.

GOOD: "Project Atlas has an approved budget of four million dollars, including a roughly
twelve percent contingency reserve."
BAD:  "Project budgets are typically allocated across functional areas and include a
contingency reserve." (generic; true of any project, tells us nothing about Atlas)
BAD:  "The budget is four million." (not self-contained; whose budget?)

Also set worth_remembering to false for chit-chat, or when the answer says the evidence was
insufficient.

Reply with a JSON object only:
{"worth_remembering": <bool>, "takeaway": "<one self-contained sentence, or empty>"}`

// WriteBack extracts a durable takeaway from an exchange and remembers it.
// Costs one LLM call, so the loop only calls it when write-back is enabled.
//
// A takeaway must stand alone: it is recalled without the question that produced
// it, so "it is twelve percent" would be useless.
func (m *Memory) WriteBack(ctx context.Context, question, answer string) (string, error) {
	var out struct {
		WorthRemembering bool   `json:"worth_remembering"`
		Takeaway         string `json:"takeaway"`
	}
	msgs := []llm.Message{
		llm.System(takeawaySystem),
		llm.User(fmt.Sprintf("Question: %s\n\nAnswer: %s", question, answer)),
	}
	if err := m.llm.ChatJSON(ctx, msgs, &out); err != nil {
		return "", err
	}
	if !out.WorthRemembering || strings.TrimSpace(out.Takeaway) == "" {
		return "", nil
	}
	return out.Takeaway, m.Remember(ctx, out.Takeaway)
}

// --- persistence ---

// Save writes the core block and archival index under the memory directory.
func (m *Memory) Save() error {
	if err := os.MkdirAll(m.dir, 0o755); err != nil {
		return err
	}
	m.mu.Lock()
	snapshot := struct {
		Core []CoreEntry `json:"core"`
		Seq  int         `json:"seq"`
	}{Core: m.core, Seq: m.seq}
	m.mu.Unlock()

	data, err := json.MarshalIndent(snapshot, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(m.dir, "core.json"), data, 0o644); err != nil {
		return err
	}
	// An empty archival index has nothing worth writing.
	if m.archival.Len() == 0 {
		return nil
	}
	return m.archival.Save(filepath.Join(m.dir, "archival.json"))
}

// Load restores memory from disk. A missing directory is not an error: the first
// run of a fresh agent legitimately has nothing to remember.
func (m *Memory) Load() error {
	data, err := os.ReadFile(filepath.Join(m.dir, "core.json"))
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	var snapshot struct {
		Core []CoreEntry `json:"core"`
		Seq  int         `json:"seq"`
	}
	if err := json.Unmarshal(data, &snapshot); err != nil {
		return fmt.Errorf("agent: parse core memory: %w", err)
	}
	m.mu.Lock()
	m.core, m.seq = snapshot.Core, snapshot.Seq
	m.mu.Unlock()

	archivalPath := filepath.Join(m.dir, "archival.json")
	if _, err := os.Stat(archivalPath); err == nil {
		return m.archival.Load(archivalPath)
	}
	return nil
}
