package main

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/RootControl/dispatch/agent"
	"github.com/RootControl/dispatch/core"
	"github.com/RootControl/dispatch/index"
	"github.com/RootControl/dispatch/llm"
	"github.com/RootControl/dispatch/router"
	"github.com/RootControl/dispatch/tiers"
)

// stackOptions selects which tiers are assembled.
type stackOptions struct {
	IndexPath string
	SQLDir    string // empty disables the structured tier
	MaxHops   int
	Remember  bool // enables the memory tier
	Rerank    bool // rescore the shortlist with a cross-encoder (RERANK_BASE_URL)
	RerankLLM bool // rescore with the chat model; measured worse than not reranking
}

// artifactPath names a derived artifact next to its index, so a second corpus
// under --index does not silently overwrite the first one's tree and graph.
// ".dispatch/index.json" + "hierarchy" -> ".dispatch/index-hierarchy.json".
func artifactPath(indexPath, name string) string {
	dir, base := filepath.Dir(indexPath), filepath.Base(indexPath)
	stem := strings.TrimSuffix(base, filepath.Ext(base))
	return filepath.Join(dir, stem+"-"+name+".json")
}

// stack is the assembled retrieval system. Both `ask` and `eval` build one, so
// an evaluation measures the same wiring the CLI actually answers with — a
// parallel copy would drift and quietly make the numbers meaningless.
type stack struct {
	client    *llm.Client
	store     *index.Store
	registry  map[core.Tier]core.Retriever
	memory    *agent.Memory
	available []core.Tier
}

// buildStack loads the index and registers every tier whose artifact exists.
// A missing hierarchy or graph is normal — those are opt-in at ingest time.
func buildStack(opts stackOptions) (*stack, error) {
	store, client, err := newStore(false, 0, 0) // retrieval doesn't contextualize
	if err != nil {
		return nil, err
	}
	switch {
	case opts.Rerank && opts.RerankLLM:
		return nil, fmt.Errorf("choose one of --rerank and --rerank-llm")
	case opts.Rerank:
		// --rerank means the cross-encoder, because that is the one with a
		// reason to work. A missing endpoint is an error rather than a fallback
		// to the LLM reranker: measurements say that path makes retrieval
		// worse, and silently taking it would attribute the loss to reranking
		// in general.
		ce, err := index.NewCrossEncoderFromEnv()
		if err != nil {
			return nil, fmt.Errorf("%w\n(use --rerank-llm for the chat-model reranker, which measured worse than no reranking)", err)
		}
		store.SetReranker(ce)
	case opts.RerankLLM:
		// Reranking on the utility model, not the answering one. Scoring a
		// shortlist for relevance is a mechanical judgment, and doing it with an
		// 8B thinking model took ~50s per query against ~2s — slow enough that
		// the feature would not be used.
		util, _, _ := utilityLLM(client)
		store.SetReranker(&index.LLMReranker{LLM: util})
	}
	if err := store.Load(opts.IndexPath); err != nil {
		return nil, fmt.Errorf("load index (run `dispatch ingest` first): %w", err)
	}

	st := &stack{
		client: client,
		store:  store,
		registry: map[core.Tier]core.Retriever{
			core.TierSemantic: tiers.NewSemantic(store),
		},
	}

	hierarchyPath := artifactPath(opts.IndexPath, "hierarchy")
	if _, err := os.Stat(hierarchyPath); err == nil {
		h, err := tiers.LoadHierarchy(client, hierarchyPath)
		if err != nil {
			return nil, fmt.Errorf("load hierarchy: %w", err)
		}
		st.registry[core.TierHierarchical] = h
	}

	graphPath := artifactPath(opts.IndexPath, "graph")
	if _, err := os.Stat(graphPath); err == nil {
		rel, err := tiers.LoadGraph(graphPath, opts.MaxHops)
		if err != nil {
			return nil, fmt.Errorf("load graph: %w", err)
		}
		st.registry[core.TierRelational] = rel
	}

	// The structured tier needs a SQLRunner. The CSV runner is a demo backend;
	// production implements SQLRunner over database/sql with a role granted
	// SELECT only.
	if opts.SQLDir != "" {
		runner, err := tiers.LoadCSVDir(opts.SQLDir)
		if err != nil {
			return nil, fmt.Errorf("load sql tables: %w", err)
		}
		st.registry[core.TierStructured] = tiers.NewStructured(client, runner)
	}

	// Memory is opt-in: it costs an extra LLM call per question and writes to
	// disk, neither of which should happen without being asked for.
	if opts.Remember {
		st.memory = agent.NewMemory(agent.MemoryConfig{
			LLM:      client,
			Dir:      defaultMemoryDir,
			EmbedTag: client.EmbedModel(),
		})
		if err := st.memory.Load(); err != nil {
			return nil, fmt.Errorf("load memory: %w", err)
		}
		st.registry[core.TierMemory] = st.memory
	}

	st.available = make([]core.Tier, 0, len(st.registry))
	for t := range st.registry {
		st.available = append(st.available, t)
	}
	slices.Sort(st.available)
	return st, nil
}

// router returns the configured router over the registered tiers.
func (s *stack) router(useLLM bool) router.Router {
	if useLLM {
		return router.LLM{LLM: s.client, Available: s.available}
	}
	return router.Heuristic{Available: s.available}
}

// rerankLabel names the reranker in eval output. The two paths must be
// distinguishable in a results table: they are different enough that reporting
// both as "rerank=true" would make the numbers uninterpretable.
func rerankLabel(cross, viaLLM bool) string {
	switch {
	case cross:
		return "cross-encoder"
	case viaLLM:
		return "llm"
	}
	return "off"
}
