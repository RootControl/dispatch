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

Beyond that there are two evaluations against the live endpoint, kept separate
because routing and answering fail for different reasons and a combined score
would hide which half broke.

**Routing** — a misrouted question retrieves plausible evidence from the wrong
tier and the answer still looks fine, so this is scored on its own:

```bash
go run ./cmd/dispatch eval --router both -v
```

On the 13 bundled cases with `gemma4:e4b`: heuristic 85% top-1, LLM router 92%
top-1 and 100% top-2. Top-2 matters because the loop fans out — a correct tier
ranked second is still searched in the same round.

**Answer quality** — end to end, reporting three numbers that need different
fixes:

```bash
go run ./cmd/dispatch eval answers -v
```

- **retrieval** — did the expected document reach the evidence at all?
- **facts** — given that evidence, did the answer state the required facts?
- **citations** — did every `[tier:source]` marker resolve to evidence that was
  actually retrieved? A marker that doesn't is a fabricated citation, the exact
  failure a grounded system exists to prevent and one that is invisible without
  this check.

Splitting retrieval from facts is the point: a wrong answer with `retrieval ok`
is a generation problem, and the same answer with `retrieval miss` is a chunking
or embedding problem. Facts accept alternative surface forms (`12%`, `twelve
percent`) so the score measures correctness rather than phrasing.

On the 8 bundled cases over 18 chunks with `gemma4:e4b`, all three metrics come
out 8/8. **Read that with suspicion rather than satisfaction** — see below.

Run against a real corpus (15 documents from a LibreChat checkout, 70 chunks,
`testdata/eval/nexus-answers.json`) the numbers are more informative:

```
retrieval:  6/6    facts: 7/8    citations: 6/6    end-to-end: 5/6
```

The single failure is the *easiest* question — "what is this project and what is
it for?" — and it is worth more than the five passes. See **Thinking models**
below.

## Known limitations

Measured, not guessed:

- **Entity resolution is normalization only** — case, punctuation, a leading
  "the". The live graph over the sample corpus holds `atlas`, `project atlas`,
  and `project` as three separate nodes. Multi-hop traversal can miss paths a
  human would consider connected. Fixing it needs embedding-based coreference or
  an alias table.
- **Thinking models can spend their whole output budget reasoning and return no
  answer.** Found on the real corpus: `gemma4:e4b` given four evidence chunks
  produced 1,114 characters of reasoning, hit `finish_reason: "length"`, and
  emitted empty content. dispatch used to pass that through as a blank answer,
  which reads like "the corpus doesn't say" when the real cause is the model.
  It is now a hard error naming the cause and the fix, and the fix is verified:
  the same question answers correctly at `-k 2`. If you see it, lower `-k`,
  shorten chunks, or use a non-thinking model.
- **Vague queries retrieve noise.** "What is this project and what is it for?"
  has no distinctive content words, so both BM25 and the embedding latch onto
  incidental matches — the top hit was a translations how-to that merely says
  "in the project". Contextual chunking helps but does not rescue a query with
  nothing to match on.
- **Ingest is slow on a local thinking model.** Contextual chunking ~13s/chunk
  and graph extraction ~42s/chunk on an 8B model. Both are cached by content
  hash, so re-ingest is free, but a first run over a large corpus is long
  (70 chunks took ~15 minutes). `index.Config.ContextLLM` accepts a separate
  cheaper model; the CLI does not expose it yet.
- **RAPTOR summaries are not cached.** Unlike chunking and extraction, a rebuild
  with `--hierarchy` pays again.
- **`isReadOnly` is defense in depth, not the primary guard.** Grant the
  executing database role SELECT only. A generated-SQL allowlist is string
  matching against an adversary who controls the model's input.
- **`TableRunner` is not a SQL engine.** It is an in-memory CSV runner covering a
  documented `SELECT` subset so the structured tier is demonstrable without a
  driver. Anything outside the subset is a clear error, never a wrong answer.
  Production implements `SQLRunner` over `database/sql`.
- **The bundled corpus is small** (6 docs, 18 chunks) and the answer eval scores
  8/8 on it. That number is weaker evidence than it looks, for three reasons
  worth stating plainly:
  1. The cases were written against a corpus written for them. Self-consistent
     by construction; it shows the machinery works, not that retrieval is good.
  2. Every case answers in `rounds 1`. At `-k 4` across three tiers, up to 12 of
     18 chunks reach the evidence — so retrieval barely has to *rank*, and the
     loop never has to refine. Both are the corpus being small, not the system
     being strong.
  3. A perfect score is a reason to distrust the eval first. The harness was
     verified against negative controls: a case with a deliberately wrong
     `expect_sources` reports `RETRIEVAL` while still scoring `facts 1/1`
     (proving the two are genuinely independent), and a case demanding a fact
     absent from the corpus reports `FACTS` with the miss named.

  The real test is your own documents. Retrieval quality claims cannot be
  settled on 18 chunks.

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
