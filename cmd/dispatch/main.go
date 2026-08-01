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
	case "eval":
		err = runEval(os.Args[2:])
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

  dispatch ingest --corpus DIR [--index PATH] [--dry-run] [--no-context] [--full]
                  [--chunk-tokens N] [--min-words N] [--exclude DIRS] [--hierarchy] [--graph]
                  [--utility-model NAME]

  dispatch ask [--index PATH] [--trace] [-k N] [--max-steps N] [--llm-router]
               [--remember] [--sql-dir DIR] [--max-hops N] [--rerank] [--retrieve-only] "question"

  dispatch eval [--router heuristic|llm|both] [--cases FILE] [-v]   # routing accuracy
  dispatch eval answers [--index PATH] [--cases FILE] [-k N] [-v]   # answer quality
  dispatch eval retrieval [--index PATH] [-k N] [--rerank] [-v]     # recall over every chunk
  dispatch eval scale [--sizes N,N,N] [--queries N] [--dims N]      # latency/memory vs corpus size

Artifacts live beside their index: --index .dispatch/x.json puts the summary tree
at .dispatch/x-hierarchy.json and the entity graph at .dispatch/x-graph.json, so
several corpora can coexist. Both tiers register automatically when present.

Configure first:  cp .env.example .env  and fill in LLM_BASE_URL / LLM_API_KEY.
`)
}

// utilityLLM returns the model for mechanical passes — context sentences,
// entity extraction, summaries, coreference — and reports whether it differs
// from the answering model.
//
// Those passes are the bulk of ingest (~13s and ~42s per chunk on an 8B
// thinking model) and none of them need reasoning: they rewrite or classify
// text. Pointing LLM_UTILITY_MODEL at something small is the largest single
// lever on ingest cost. Unset, it is the chat model and nothing changes.
func utilityLLM(client *llm.Client) (*llm.Client, string, bool) {
	name := utilityOverride
	if name == "" {
		name = os.Getenv("LLM_UTILITY_MODEL")
	}
	if name == "" || name == client.ChatModel() {
		return client, client.ChatModel(), false
	}
	u, err := llm.New(llm.Config{ChatModel: name})
	if err != nil {
		return client, client.ChatModel(), false
	}
	return u, name, true
}

// utilityOverride is `ingest --utility-model`, which takes precedence over
// LLM_UTILITY_MODEL. The env var configures a machine; the flag lets one ingest
// try a cheaper model without editing .env — and since this is the largest
// single lever on ingest cost, trying one should not require a config change.
var utilityOverride string

// newStore wires an index.Store to the configured endpoint. Cache and index are
// tagged with the model names so swapping models invalidates derived data.
func newStore(contextualize bool, chunkTokens, minWords int) (*index.Store, *llm.Client, error) {
	client, err := llm.New(llm.Config{})
	if err != nil {
		return nil, nil, err
	}
	util, utilName, _ := utilityLLM(client)
	s := index.New(index.Config{
		LLM:           client,
		ContextLLM:    util,
		Cache:         index.NewCache(defaultCacheDir),
		Contextualize: contextualize,
		// Cache and index are tagged with the model that PRODUCED them: context
		// sentences come from the utility model, so switching it must invalidate
		// them rather than silently reuse another model's work.
		CacheTag: utilName,
		EmbedTag: client.EmbedModel(),
		Chunk:    index.ChunkOptions{TargetTokens: chunkTokens, MinWords: minWords},
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
	minWords := fs.Int("min-words", 0, "skip chunks with fewer prose words than this (0 = keep all)")
	hierarchy := fs.Bool("hierarchy", false, "also build the RAPTOR summary tree for the hierarchical tier")
	branching := fs.Int("branching", 5, "leaves per cluster when building the hierarchy")
	graph := fs.Bool("graph", false, "also build the entity graph for the relational tier")
	exclude := fs.String("exclude", "", "extra comma-separated directory names to skip (node_modules, .git, dist and friends are always skipped)")
	full := fs.Bool("full", false, "rebuild from scratch instead of updating the existing index")
	utilModel := fs.String("utility-model", "", "cheaper model for context sentences, extraction and summaries (overrides LLM_UTILITY_MODEL)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	utilityOverride = *utilModel

	docs, err := loadCorpus(*corpus, strings.Split(*exclude, ","))
	if err != nil {
		return err
	}
	if len(docs) == 0 {
		return fmt.Errorf("no .md or .txt files under %s", *corpus)
	}

	store, client, err := newStore(!*noContext, *chunkTokens, *minWords)
	if err != nil {
		return err
	}

	// Load whatever is already indexed so this ingest can be incremental. A
	// missing index is the normal first run; a mismatched embedding model is
	// not, and Load says so rather than mixing embedding spaces.
	if !*full {
		switch err := store.Load(*indexPath); {
		case err == nil:
			fmt.Printf("updating %s (%d chunks indexed)\n", *indexPath, store.Len())
		case !os.IsNotExist(err):
			return fmt.Errorf("%w\n(use --full to rebuild from scratch)", err)
		}
	}

	if *minWords > 0 {
		all := index.New(index.Config{LLM: client, Chunk: index.ChunkOptions{TargetTokens: *chunkTokens}})
		if kept, total := store.Plan(docs).Chunks, all.Plan(docs).Chunks; total > kept {
			fmt.Printf("min-words %d: skipping %d of %d chunks with no prose\n", *minWords, total-kept, total)
		}
	}

	if *dryRun {
		plan := store.Plan(docs)
		fmt.Printf("dry run: %d docs -> %d chunks\n", plan.Docs, plan.Chunks)
		if plan.Unchanged > 0 {
			fmt.Printf("  unchanged, skipped:    %d of %d docs\n", plan.Unchanged, plan.Docs)
		}
		if stale := staleDocs(store, docs); len(stale) > 0 {
			fmt.Printf("  gone from the corpus:  %d doc(s), to be pruned\n", len(stale))
		}
		fmt.Printf("  context calls to make: %d (%d already cached)\n", plan.LLMCalls, plan.CacheHits)
		fmt.Printf("  embedding calls:       %d chunk(s) (%d already cached)\n", plan.EmbedCalls, plan.EmbedHits)
		_, utilName, split := utilityLLM(client)
		fmt.Printf("  chat model %s, embed model %s\n", client.ChatModel(), client.EmbedModel())
		if split {
			fmt.Printf("  utility model %s (context sentences, extraction, summaries)\n", utilName)
		}
		return nil
	}

	// Prune before ingesting: a document deleted from the corpus is still in the
	// index, and nothing else will ever tell the index it is gone.
	if pruned := store.Prune(docIDs(docs)); pruned > 0 {
		fmt.Printf("pruned %d chunk(s) from documents no longer in the corpus\n", pruned)
	}

	stats, err := store.Ingest(context.Background(), docs)
	if err != nil {
		return err
	}
	if err := store.Save(*indexPath); err != nil {
		return err
	}
	fmt.Printf("ingested %d docs -> %d chunks (%d context calls, %d cache hits; %d embedded, %d cached)\n",
		stats.Docs, stats.Chunks, stats.LLMCalls, stats.CacheHits, stats.EmbedCalls, stats.EmbedHits)
	if stats.Unchanged > 0 {
		fmt.Printf("  %d of %d docs unchanged and skipped\n", stats.Unchanged, stats.Docs)
	}
	fmt.Printf("index written to %s\n", *indexPath)

	// The tree is opt-in: it costs roughly one LLM call per cluster per level on
	// top of ingestion, and is only useful for corpus-wide questions.
	if *hierarchy {
		util, utilName, _ := utilityLLM(client)
		h, hstats, err := tiers.BuildHierarchy(context.Background(), util, store,
			tiers.HierarchyOptions{
				Branching: *branching,
				Cache:     index.NewCache(defaultCacheDir),
				CacheTag:  utilName,
				EmbedTag:  client.EmbedModel(),
			})
		if err != nil {
			return err
		}
		hierarchyPath := artifactPath(*indexPath, "hierarchy")
		if err := h.Save(hierarchyPath); err != nil {
			return err
		}
		fmt.Printf("hierarchy: %d levels, %d summaries (%d LLM calls, %d cache hits; %d embedded, %d cached) -> %s\n",
			hstats.Levels, hstats.Summaries, hstats.LLMCalls, hstats.CacheHits,
			hstats.EmbedCalls, hstats.EmbedHits, hierarchyPath)
	}

	// Also opt-in: extraction is one LLM call per chunk, cached by content hash
	// so a rebuild over unchanged documents is free.
	if *graph {
		util, utilName, _ := utilityLLM(client)
		rel, gstats, err := tiers.BuildGraph(context.Background(), util, store, tiers.GraphOptions{
			Cache:    index.NewCache(defaultCacheDir),
			CacheTag: utilName,
		})
		if err != nil {
			return err
		}
		graphPath := artifactPath(*indexPath, "graph")
		if err := rel.Save(graphPath); err != nil {
			return err
		}
		fmt.Printf("graph: %d entities, %d relations (%d LLM calls, %d cache hits) -> %s\n",
			gstats.Entities, gstats.Relations, gstats.LLMCalls, gstats.CacheHits, graphPath)
		// Automatic coreference is the riskiest step, so show it rather than
		// letting merged nodes appear as if extraction produced them.
		if n := len(gstats.Merges); n > 0 {
			fmt.Printf("  merged %d coreferent name(s):\n", n)
			for i, m := range gstats.Merges {
				if i == 8 {
					fmt.Printf("    ... and %d more\n", n-8)
					break
				}
				fmt.Printf("    %q -> %q\n", m.From, m.Into)
			}
		}
		// Silent truncation would read as full coverage, so say what was lost.
		// Zero merges from zero failures means the model rejected the
		// candidates; zero merges from many failures means coreference never
		// ran. Those must not look the same.
		if gstats.CorefFailures > 0 {
			fmt.Printf("  WARNING: %d coreference adjudication(s) failed; those names were left unmerged\n",
				gstats.CorefFailures)
			fmt.Printf("  first failure: %v\n", gstats.CorefError)
		}
		if gstats.Skipped > 0 {
			fmt.Printf("  WARNING: %d of %d chunks could not be extracted and contribute no entities\n",
				gstats.Skipped, gstats.Chunks)
			fmt.Printf("  first failure: %v\n", gstats.FirstError)
		}
	}
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
	maxHops := fs.Int("max-hops", 2, "relational graph traversal depth")
	sqlDir := fs.String("sql-dir", "", "directory of CSV tables to enable the structured (text-to-SQL) tier")
	llmRoute := fs.Bool("llm-router", false, "classify with the model instead of keywords (falls back to keywords on failure)")
	rerank := fs.Bool("rerank", false, "rescore the retrieved shortlist with a cross-encoder (needs RERANK_BASE_URL)")
	rerankLLM := fs.Bool("rerank-llm", false, "rescore with the chat model instead — measured worse than no reranking; see the README")
	var filter filterFlag
	fs.Var(&filter, "filter", "restrict retrieval to chunks whose metadata matches, e.g. -filter path=docs/* (repeatable)")
	stream := fs.Bool("stream", false, "print the answer as the model produces it")
	timeout := fs.Duration("timeout", 0, "give up after this long (e.g. 90s, 2m); 0 waits forever")
	maxCalls := fs.Int("max-calls", 0, "cap the LLM calls one question may make (0 = no cap)")
	hyde := fs.Bool("hyde", false, "embed a hypothetical answer alongside the query (helps vague questions, costs one call per round)")
	diverse := fs.Bool("diverse", false, "drop evidence that restates evidence already held")
	perDoc := fs.Int("per-doc", 0, "with --diverse, cap how many evidence items one document may contribute (0 = no cap)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	question := strings.Join(fs.Args(), " ")
	if question == "" {
		return fmt.Errorf("ask what? provide a question")
	}

	st, err := buildStack(stackOptions{
		IndexPath: *indexPath,
		SQLDir:    *sqlDir,
		MaxHops:   *maxHops,
		Remember:  *remember,
		Rerank:    *rerank,
		RerankLLM: *rerankLLM,
	})
	if err != nil {
		return err
	}
	store, client, registry, mem := st.store, st.client, st.registry, st.memory

	// A hung endpoint otherwise hangs the command forever: the HTTP client has a
	// per-request timeout, and one question makes many requests.
	ctx := context.Background()
	if *timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, *timeout)
		defer cancel()
	}
	route := st.router(*llmRoute)

	// --retrieve-only skips the loop entirely: route once, retrieve once, print.
	// Useful for inspecting retrieval quality without paying for judge calls.
	if *retrieveOnly {
		decision, err := route.Route(ctx, question)
		if err != nil {
			return err
		}
		if *trace {
			fmt.Printf("route [%s] -> %v (%s)\n\n", decision.Source, decision.Tiers, decision.Reason)
		}
		// Expansion belongs on this path too: comparing retrieval with and
		// without it is the reason to have it, and that comparison should not
		// require paying for judge and generate calls.
		var expanded string
		if *hyde {
			util, _, _ := utilityLLM(client)
			if expanded, err = (&agent.HyDE{LLM: util}).Expand(ctx, question); err != nil {
				return fmt.Errorf("expand: %w", err)
			}
			if *trace {
				fmt.Printf("hyde -> %s\n\n", expanded)
			}
		}
		var evidence []core.Result
		for _, t := range decision.Tiers {
			// Same rule the loop applies: a tier that cannot honour the filter
			// must not answer around it.
			if _, ok := registry[t].(core.Filterable); !filter.f.Empty() && !ok {
				if *trace {
					fmt.Printf("skipping [%s]: does not apply metadata filters\n", t)
				}
				continue
			}
			got, err := registry[t].Retrieve(ctx, core.Query{
				Text: question, Expanded: expanded, TopK: *topK, Filter: filter.f})
			if err != nil {
				return err
			}
			evidence = append(evidence, got...)
		}
		if len(evidence) == 0 {
			if !filter.f.Empty() {
				return fmt.Errorf("no evidence matched filter %v across %d indexed chunks", filter.f, store.Len())
			}
			return fmt.Errorf("no evidence retrieved from %d indexed chunks", store.Len())
		}
		fmt.Println(agent.FormatEvidence(evidence))
		return nil
	}

	loop := &agent.Loop{
		LLM:        client,
		Router:     route,
		Retrievers: registry,
		MaxSteps:   *maxSteps,
		MaxCalls:   *maxCalls,
		TopK:       *topK,
		Memory:     mem,
		WriteBack:  *remember,
		Filter:     filter.f,
		Diversity:  agent.Diversity{Enabled: *diverse || *perDoc > 0, PerDoc: *perDoc},
	}
	if *hyde {
		// The utility model, like the other mechanical passes: writing a
		// plausible-sounding passage is not reasoning, and this is on the path
		// of every question.
		util, _, _ := utilityLLM(client)
		loop.Expander = &agent.HyDE{LLM: util}
	}
	if *stream {
		loop.OnDelta = func(s string) { fmt.Print(s) }
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
	// When streaming, the answer already went to stdout fragment by fragment.
	// Printing answer.Text as well would double it.
	if *stream {
		fmt.Println()
	} else {
		fmt.Println(answer.Text)
	}
	// A fabricated citation looks exactly like a real one, so it goes to stderr
	// unconditionally rather than hiding behind --trace. Piping the answer
	// somewhere still leaves the warning where a person will see it.
	if !answer.Citations.OK() {
		fmt.Fprintf(os.Stderr, "\nWARNING: %d citation(s) resolve to no retrieved evidence: %s\n",
			len(answer.Citations.Unresolved), strings.Join(answer.Citations.Unresolved, " "))
	}
	if *trace {
		fmt.Printf("\n--- trace ---\n%s\n", answer.Trace)
		// Tokens come from the server's own usage block. A server that does not
		// report them leaves this at zero, so say "not reported" rather than
		// printing a zero that reads as "free".
		if u := client.Usage(); u.Total() > 0 {
			fmt.Printf("tokens: %d prompt + %d completion = %d over %d call(s)\n",
				u.PromptTokens, u.CompletionTokens, u.Total(), u.Calls)
		} else {
			fmt.Printf("tokens: not reported by the endpoint (%d call(s))\n", u.Calls)
		}
	}
	return nil
}

// filterFlag collects repeated -filter key=value pairs into a core.Filter.
type filterFlag struct{ f core.Filter }

func (x *filterFlag) String() string {
	if x == nil || len(x.f) == 0 {
		return ""
	}
	pairs := make([]string, 0, len(x.f))
	for k, v := range x.f {
		pairs = append(pairs, k+"="+v)
	}
	slices.Sort(pairs)
	return strings.Join(pairs, ",")
}

func (x *filterFlag) Set(s string) error {
	k, v, ok := strings.Cut(s, "=")
	if !ok || strings.TrimSpace(k) == "" {
		return fmt.Errorf("want key=value, got %q", s)
	}
	if x.f == nil {
		x.f = core.Filter{}
	}
	// Repeating a key would otherwise silently keep only the last one, and the
	// filter is a safety boundary — a dropped clause widens it.
	if prev, dup := x.f[k]; dup {
		return fmt.Errorf("filter key %q given twice (%q then %q); only one value per key is supported", k, prev, v)
	}
	x.f[strings.TrimSpace(k)] = v
	return nil
}

// docIDs lists the corpus's document IDs, which is what Prune keeps.
func docIDs(docs []core.Doc) []string {
	out := make([]string, len(docs))
	for i, d := range docs {
		out[i] = d.ID
	}
	return out
}

// staleDocs reports indexed documents the corpus no longer contains, so a dry
// run can say what an ingest would prune before it prunes it.
func staleDocs(store *index.Store, docs []core.Doc) []string {
	live := make(map[string]bool, len(docs))
	for _, d := range docs {
		live[d.ID] = true
	}
	var stale []string
	for _, id := range store.DocIDs() {
		if !live[id] {
			stale = append(stale, id)
		}
	}
	return stale
}

// skipDirs are never descended into. Without this, pointing --corpus at any
// real repository ingests its dependencies: a checkout of a Node project here
// held 13 documents worth reading and 4,602 markdown files in total, nearly all
// of them vendored changelogs.
// testdata is deliberately absent: for a docs-in-repo corpus it is often
// exactly what you want indexed.
var skipDirs = map[string]bool{
	"node_modules": true, ".git": true, "vendor": true, "dist": true,
	"build": true, "target": true, ".next": true, ".venv": true,
	"venv": true, "__pycache__": true, ".cache": true, "coverage": true,
	".dispatch": true,
}

// loadCorpus reads .md/.txt files under dir into Docs, using the path relative
// to dir as a stable document ID so citations survive a move of the corpus root.
// Vendored and build directories are skipped; extraSkip adds to that set.
func loadCorpus(dir string, extraSkip []string) ([]core.Doc, error) {
	skip := make(map[string]bool, len(skipDirs)+len(extraSkip))
	for k, v := range skipDirs {
		skip[k] = v
	}
	for _, s := range extraSkip {
		if s = strings.TrimSpace(s); s != "" {
			skip[s] = true
		}
	}

	var docs []core.Doc
	err := filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			// Never skip the root itself, even if it happens to be named "dist".
			if path != dir && skip[d.Name()] {
				return fs.SkipDir
			}
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
		// Metadata is what `ask --filter` matches on, so record the fields a
		// question would actually scope by. "dir" is the corpus-relative
		// directory, which makes `-filter dir=docs*` mean what it looks like;
		// "path" stays absolute for display and would scope by filesystem
		// layout instead.
		dir := filepath.ToSlash(filepath.Dir(rel))
		if dir == "." {
			dir = ""
		}
		docs = append(docs, core.Doc{ID: rel, Text: string(b), Meta: map[string]string{
			"path": path,
			"doc":  filepath.ToSlash(rel),
			"dir":  dir,
			"ext":  strings.ToLower(strings.TrimPrefix(filepath.Ext(path), ".")),
		}})
		return nil
	})
	if err != nil {
		return nil, err
	}
	// Stable order so chunk IDs are reproducible across runs.
	slices.SortFunc(docs, func(a, b core.Doc) int { return strings.Compare(a.ID, b.ID) })
	return docs, nil
}
