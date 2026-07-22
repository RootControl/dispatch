// Command dispatch ingests a corpus and answers questions over it.
//
//	dispatch ingest --corpus ./testdata/corpus [--dry-run]
//	dispatch ask [--trace] "your question"
//
// Configuration comes from a .env file in the working directory, or from the
// environment directly (shell variables take precedence). See .env.example.
//
//	LLM_BASE_URL    OpenAI-compatible endpoint (required)
//	LLM_API_KEY     bearer token
//	LLM_CHAT_MODEL  chat model (default gpt-4o-mini)
//	LLM_EMBED_MODEL embedding model (default text-embedding-3-small)
package main

import (
	"context"
	"flag"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/RootControl/dispatch/agent"
	"github.com/RootControl/dispatch/core"
	"github.com/RootControl/dispatch/index"
	"github.com/RootControl/dispatch/internal/envfile"
	"github.com/RootControl/dispatch/llm"
	"github.com/RootControl/dispatch/router"
	"github.com/RootControl/dispatch/tiers"
)

const (
	defaultIndexPath = ".dispatch/index.json"
	defaultCacheDir  = ".dispatch/cache"
	defaultMemoryDir = ".dispatch/memory"
	envPath          = ".env"
)

func main() {
	// Load .env before anything reads configuration. Shell variables already set
	// take precedence over the file.
	if err := envfile.Load(envPath); err != nil {
		fmt.Fprintf(os.Stderr, "dispatch: reading %s: %v\n", envPath, err)
		os.Exit(1)
	}
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	var err error
	switch os.Args[1] {
	case "ingest":
		err = runIngest(os.Args[2:])
	case "ask":
		err = runAsk(os.Args[2:])
	default:
		usage()
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "dispatch: %v\n", err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprint(os.Stderr, `dispatch — agentic tiered retrieval

  dispatch ingest --corpus DIR [--dry-run] [--no-context] [--chunk-tokens N]
  dispatch ask [--trace] [--retrieve-only] [-k N] [--max-steps N] [--remember] "question"

Configure first:  cp .env.example .env  and fill in LLM_BASE_URL / LLM_API_KEY.
`)
}

// newStore wires an index.Store to the configured endpoint. Cache and index are
// tagged with the model names so swapping models invalidates derived data.
func newStore(contextualize bool, chunkTokens int) (*index.Store, *llm.Client, error) {
	client, err := llm.New(llm.Config{})
	if err != nil {
		return nil, nil, err
	}
	s := index.New(index.Config{
		LLM:           client,
		Cache:         index.NewCache(defaultCacheDir),
		Contextualize: contextualize,
		CacheTag:      client.ChatModel(),
		EmbedTag:      client.EmbedModel(),
		Chunk:         index.ChunkOptions{TargetTokens: chunkTokens},
	})
	return s, client, nil
}

func runIngest(args []string) error {
	fs := flag.NewFlagSet("ingest", flag.ExitOnError)
	corpus := fs.String("corpus", "./testdata/corpus", "directory of .md/.txt documents")
	indexPath := fs.String("index", defaultIndexPath, "where to write the index")
	dryRun := fs.Bool("dry-run", false, "report chunk count and expected LLM calls, then stop")
	noContext := fs.Bool("no-context", false, "skip contextual chunking (cheaper, worse retrieval)")
	chunkTokens := fs.Int("chunk-tokens", 0, "target chunk size in tokens (0 = default 800)")
	if err := fs.Parse(args); err != nil {
		return err
	}

	docs, err := loadCorpus(*corpus)
	if err != nil {
		return err
	}
	if len(docs) == 0 {
		return fmt.Errorf("no .md or .txt files under %s", *corpus)
	}

	store, client, err := newStore(!*noContext, *chunkTokens)
	if err != nil {
		return err
	}

	if *dryRun {
		plan := store.Plan(docs)
		fmt.Printf("dry run: %d docs -> %d chunks\n", plan.Docs, plan.Chunks)
		fmt.Printf("  context calls to make: %d (%d already cached)\n", plan.LLMCalls, plan.CacheHits)
		fmt.Printf("  embedding calls:       %d batch(es), %d texts\n", plan.Docs, plan.Chunks)
		fmt.Printf("  chat model %s, embed model %s\n", client.ChatModel(), client.EmbedModel())
		return nil
	}

	stats, err := store.Ingest(context.Background(), docs)
	if err != nil {
		return err
	}
	if err := store.Save(*indexPath); err != nil {
		return err
	}
	fmt.Printf("ingested %d docs -> %d chunks (%d context calls, %d cache hits)\n",
		stats.Docs, stats.Chunks, stats.LLMCalls, stats.CacheHits)
	fmt.Printf("index written to %s\n", *indexPath)
	return nil
}

func runAsk(args []string) error {
	fs := flag.NewFlagSet("ask", flag.ExitOnError)
	indexPath := fs.String("index", defaultIndexPath, "index to read")
	topK := fs.Int("k", 5, "results to retrieve per tier")
	trace := fs.Bool("trace", false, "show routing, retrieval, and judge steps")
	retrieveOnly := fs.Bool("retrieve-only", false, "print evidence without running the loop")
	maxSteps := fs.Int("max-steps", 3, "maximum retrieve/judge rounds")
	remember := fs.Bool("remember", false, "search memory, and write back a takeaway after answering")
	if err := fs.Parse(args); err != nil {
		return err
	}
	question := strings.Join(fs.Args(), " ")
	if question == "" {
		return fmt.Errorf("ask what? provide a question")
	}

	store, client, err := newStore(false, 0) // retrieval doesn't contextualize
	if err != nil {
		return err
	}
	if err := store.Load(*indexPath); err != nil {
		return fmt.Errorf("load index (run `dispatch ingest` first): %w", err)
	}

	// Registry of live tiers. The router only routes to what is registered here;
	// the remaining three tiers land in later milestones.
	registry := map[core.Tier]core.Retriever{
		core.TierSemantic: tiers.NewSemantic(store),
	}

	// Memory is opt-in: it costs an extra LLM call per question and writes to
	// disk, neither of which should happen without being asked for.
	var mem *agent.Memory
	if *remember {
		mem = agent.NewMemory(agent.MemoryConfig{
			LLM:      client,
			Dir:      defaultMemoryDir,
			EmbedTag: client.EmbedModel(),
		})
		if err := mem.Load(); err != nil {
			return fmt.Errorf("load memory: %w", err)
		}
		registry[core.TierMemory] = mem
	}

	available := make([]core.Tier, 0, len(registry))
	for t := range registry {
		available = append(available, t)
	}
	slices.Sort(available)

	ctx := context.Background()

	// --retrieve-only skips the loop entirely: route once, retrieve once, print.
	// Useful for inspecting retrieval quality without paying for judge calls.
	if *retrieveOnly {
		decision, err := router.Heuristic{Available: available}.Route(ctx, question)
		if err != nil {
			return err
		}
		if *trace {
			fmt.Printf("route [%s] -> %v (%s)\n\n", decision.Source, decision.Tiers, decision.Reason)
		}
		var evidence []core.Result
		for _, t := range decision.Tiers {
			got, err := registry[t].Retrieve(ctx, core.Query{Text: question, TopK: *topK})
			if err != nil {
				return err
			}
			evidence = append(evidence, got...)
		}
		if len(evidence) == 0 {
			return fmt.Errorf("no evidence retrieved from %d indexed chunks", store.Len())
		}
		fmt.Println(agent.FormatEvidence(evidence))
		return nil
	}

	loop := &agent.Loop{
		LLM:        client,
		Router:     router.Heuristic{Available: available},
		Retrievers: registry,
		MaxSteps:   *maxSteps,
		TopK:       *topK,
		Memory:     mem,
		WriteBack:  *remember,
	}
	answer, err := loop.Run(ctx, question)
	if err != nil {
		if answer.Trace != nil && *trace {
			fmt.Fprintln(os.Stderr, answer.Trace)
		}
		return err
	}
	// Persist whatever was learned, even if nothing new was written back —
	// eviction may still have moved entries into archival.
	if mem != nil {
		if err := mem.Save(); err != nil {
			return fmt.Errorf("save memory: %w", err)
		}
	}
	fmt.Println(answer.Text)
	if *trace {
		fmt.Printf("\n--- trace ---\n%s\n", answer.Trace)
	}
	return nil
}

// loadCorpus reads .md/.txt files under dir into Docs, using the path relative
// to dir as a stable document ID so citations survive a move of the corpus root.
func loadCorpus(dir string) ([]core.Doc, error) {
	var docs []core.Doc
	err := filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		switch strings.ToLower(filepath.Ext(path)) {
		case ".md", ".txt":
		default:
			return nil
		}
		b, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(dir, path)
		if err != nil {
			rel = path
		}
		docs = append(docs, core.Doc{ID: rel, Text: string(b), Meta: map[string]string{"path": path}})
		return nil
	})
	if err != nil {
		return nil, err
	}
	// Stable order so chunk IDs are reproducible across runs.
	slices.SortFunc(docs, func(a, b core.Doc) int { return strings.Compare(a.ID, b.ID) })
	return docs, nil
}
