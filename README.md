# dispatch

Agentic, tiered retrieval in Go. Retrieval is **a loop the agent drives**, not a
fixed pipeline stage, over a **layered store** where each tier answers a
different *kind* of question. Zero external dependencies — stdlib only.

```
question
   │
   ▼
router ──── classifies ──▶ [structured | semantic | relational | hierarchical | memory]
   │
   ▼
agent loop:  route → fan-out retrieve → judge sufficiency ─┐
   ▲                                                       │ gap?
   └──────────── refine query toward the gap ◀─────────────┘
   │ enough
   ▼
generate (cites [tier:source]) → write-back to memory
```

## Why not just RAG or GraphRAG

Both share a hidden assumption: retrieval runs once, before generation. Different
questions are different retrieval problems.

| Question | Tier | Wrong tool |
|---|---|---|
| "Total on invoice 4471?" | structured (text-to-SQL) | vector top-k |
| "What does the corpus say about X?" | semantic (contextual chunk + hybrid) | — |
| "Who does Bob report to, and what did she sign?" | relational (entity graph, multi-hop) | single chunk |
| "Recurring themes across all docs?" | hierarchical (RAPTOR) | plain RAG (no chunk holds the answer) |

## Quick start

```bash
cp .env.example .env      # then fill in LLM_BASE_URL and LLM_API_KEY
go run ./cmd/dispatch ingest --corpus ./testdata/corpus --dry-run
go run ./cmd/dispatch ingest --corpus ./testdata/corpus
go run ./cmd/dispatch ask --trace "What is the Atlas budget?"
```

Works against any OpenAI-compatible endpoint. For Ollama you need **two** models —
a chat model and a separate embedding model, since chat models cannot embed:

```bash
ollama pull nomic-embed-text
```

Always run `ingest --dry-run` first: it reports chunk count and how many LLM
calls an ingest will actually make, consulting the cache so the estimate reflects
work already done.

## Commands

```
dispatch ingest --corpus DIR [--dry-run] [--no-context] [--chunk-tokens N] [--hierarchy] [--graph]
dispatch ask [--trace] [--retrieve-only] [-k N] [--max-steps N] [--remember] [--sql-dir DIR] [--llm-router] "question"
dispatch eval [--router heuristic|llm|both] [--cases FILE] [-v]
```

The hierarchical and relational tiers are opt-in at ingest time because each
costs LLM calls; `ask` registers a tier only when its artifact exists.

## The two ideas that matter most

**Contextual chunking** (`index/store.go`). Before embedding a chunk, an LLM
writes one sentence situating it in its document; that sentence is prepended,
then embedded and indexed for BM25 too. Anthropic reported this cuts retrieval
failures ~49%. The test in `index/store_test.go` demonstrates the mechanism: a
chunk holding a budget figure but never naming the project moves from rank 1 to
rank 0 once context injects the project name.

**Write-back memory** (`agent/memory.go`). A bounded "core" block always in the
prompt plus an unbounded searchable "archival" store. Core overflow evicts the
oldest entries into archival rather than dropping them. Core memory enters the
prompt marked explicitly **not citable** — it is the agent's own prior
conclusion, not a retrieved source, and citing it would manufacture citations
that resolve to nothing.

## Layout

- **core** — domain types + the single `Retriever` interface every tier satisfies
- **llm** — OpenAI-compatible client over `net/http`, behind an interface so
  everything above it tests offline
- **index** — contextual chunking + hybrid store (cosine + BM25 fused by RRF)
- **tiers/** — the four retrievers
- **router** — LLM classification with a keyword-heuristic fallback
- **agent** — the loop (`loop.go`), write-back memory (`memory.go`), trace
- **internal/fake** — scripted LLM + deterministic embedder, so `go test ./...`
  runs offline and free

## Verifying

```bash
go vet ./... && go test ./...
```

Hermetic: no network, no API key, no cost.

Routing accuracy is measured separately, because a misrouted question retrieves
plausible evidence from the wrong tier and the answer still looks fine:

```bash
go run ./cmd/dispatch eval --router both -v
```

On the 13 bundled cases with `gemma4:e4b`: heuristic 85% top-1, LLM router 92%
top-1 and 100% top-2. Top-2 matters because the loop fans out — a correct tier
ranked second is still searched in the same round.

## Known limitations

Measured, not guessed:

- **Entity resolution is normalization only** — case, punctuation, a leading
  "the". The live graph over the sample corpus holds `atlas`, `project atlas`,
  and `project` as three separate nodes. Multi-hop traversal can miss paths a
  human would consider connected. Fixing it needs embedding-based coreference or
  an alias table.
- **Ingest is slow on a local thinking model.** Contextual chunking ~13s/chunk
  and graph extraction ~42s/chunk on an 8B model. Both are cached by content
  hash, so re-ingest is free, but a first run over a large corpus is long.
  `index.Config.ContextLLM` accepts a separate cheaper model; the CLI does not
  expose it yet.
- **RAPTOR summaries are not cached.** Unlike chunking and extraction, a rebuild
  with `--hierarchy` pays again.
- **`isReadOnly` is defense in depth, not the primary guard.** Grant the
  executing database role SELECT only. A generated-SQL allowlist is string
  matching against an adversary who controls the model's input.
- **`TableRunner` is not a SQL engine.** It is an in-memory CSV runner covering a
  documented `SELECT` subset so the structured tier is demonstrable without a
  driver. Anything outside the subset is a clear error, never a wrong answer.
  Production implements `SQLRunner` over `database/sql`.
- **The bundled corpus is tiny** (2 docs, 6 chunks). Enough to exercise the
  machinery, not enough to judge retrieval quality. Loop refinement only triggers
  at small `-k` because one round otherwise retrieves half the corpus.

## Swapping in production backends

The reference `index.Store` is in-memory. Keep the `Ingest` contextual-chunking
logic and replace the store with **pgvector** or **DuckDB**. Keep the `Retriever`
interface and everything upstream is unchanged.

`tiers.SQLRunner` is an interface — back it with `database/sql`.

## Lineage

Anthropic Contextual Retrieval, RAPTOR (hierarchical summary tree), LightRAG /
HippoRAG (graph-guided multi-hop), MemGPT/Letta (tiered write-back memory), and
the agentic-retrieval line where a router picks retrieve/reflect/answer with an
evidence-gap tracker — which is what `agent/loop.go` implements.

No single architecture wins across all query types, which is the whole reason
this routes instead of picking one.
