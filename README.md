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

Works against any OpenAI-compatible endpoint. For a local Ollama setup, the
measured recommendation is three models — see [Models](#models) for why:

```bash
ollama pull qwen2.5:7b        # answering: non-thinking, not a reasoning model
ollama pull llama3.2:3b       # mechanical passes: ~11x faster, better index
ollama pull nomic-embed-text  # embeddings: a chat model cannot do this
```

Always run `ingest --dry-run` first: it reports chunk count and how many LLM
calls an ingest will actually make, consulting the cache so the estimate reflects
work already done. On a local 8B model that estimate is the difference between a
15-minute ingest and an overnight one.

Against your own documents, with all four tiers:

```bash
go run ./cmd/dispatch ingest --corpus ~/some/repo --index .dispatch/mine.json \
    --chunk-tokens 300 --dry-run          # check the cost first
go run ./cmd/dispatch ingest --corpus ~/some/repo --index .dispatch/mine.json \
    --chunk-tokens 300 --hierarchy --graph
go run ./cmd/dispatch ask --index .dispatch/mine.json --trace -k 4 "your question"
```

`--graph` is the expensive one (see the timing table below) and only pays off for
questions about how things relate. Skip it if you only need content lookup.

## Commands

```
dispatch ingest --corpus DIR [--index PATH] [--dry-run] [--no-context]
                [--chunk-tokens N] [--exclude DIRS] [--hierarchy] [--graph]

dispatch ask [--index PATH] [--trace] [-k N] [--max-steps N] [--llm-router]
             [--remember] [--sql-dir DIR] [--max-hops N] [--retrieve-only] "question"

dispatch eval [--router heuristic|llm|both] [--cases FILE] [-v]   # routing accuracy
dispatch eval answers [--index PATH] [--cases FILE] [-k N] [-v]   # answer quality
dispatch eval retrieval [--index PATH] [-k N] [--rerank] [-v]     # recall over every chunk
```

The hierarchical and relational tiers are opt-in at ingest time because each
costs LLM calls; `ask` registers a tier only when its artifact exists.

Artifacts live beside their index — `--index .dispatch/x.json` writes
`.dispatch/x-hierarchy.json` and `.dispatch/x-graph.json` — so several corpora
can coexist without overwriting each other.

Pointing `--corpus` at a repository skips `node_modules`, `.git`, `dist`,
`vendor`, `build` and similar by default; `--exclude` adds more. Without this a
Node checkout offers 4,602 markdown files where 13 are worth reading.

`--min-words N` skips chunks carrying less prose than that — markup and
attribute tokens are not counted, so a block of badges is short by this measure
however long it looks. On the real corpus `--min-words 8` skipped 2 of 70: a
file whose whole content was `CLAUDE.md`, and a header of image badges. Both
cost an LLM call, sat in the index, and could answer nothing.

It is **off by default and reports what it drops**, because silently discarding
a user's content is worse than indexing some noise — a corpus of short entries,
a glossary or one-line FAQ answers, would lose them with no error and no way to
notice.

## Models

Three roles, three different requirements. Everything below was measured on this
project's own evals against the same corpus and embeddings.

| Role | Env var | Recommended | Why |
|---|---|---|---|
| Answering | `LLM_CHAT_MODEL` | `qwen2.5:7b` | Non-thinking. Matches a 9.6 GB reasoning model on routing and answers at half the time and memory. |
| Mechanical | `LLM_UTILITY_MODEL` | `llama3.2:3b` | Context sentences, extraction, summaries, coreference. ~11x faster and produced a *better* index. |
| Embedding | `LLM_EMBED_MODEL` | `nomic-embed-text` | Required separately — a chat model cannot embed. |
| Reranking | `--rerank` | **leave off** | Measured worse than no reranking with every chat model tried. |

```bash
ollama pull qwen2.5:7b && ollama pull llama3.2:3b && ollama pull nomic-embed-text
```

### Prefer a non-thinking answering model

| Answering model | Answer eval | Routing top-1 | Judge converges | Size |
|---|---|---|---|---|
| `gemma4:e4b` (thinking) | 4m58s | 12/13 | 8/8 in round 1 | 9.6 GB |
| **`qwen2.5:7b`** | **2m44s** | **12/13** | 5/8 in round 1 | **4.7 GB** |
| `llama3.2:3b` | 1m32s | 11/13 | 0/8 in round 1 | 2.0 GB |

All three scored 8/8 on answer quality, so that column cannot choose between
them — it is saturated. Routing accuracy and judge behaviour can.

A reasoning model spends its output budget thinking before answering: measured,
~880 characters of reasoning for a one-sentence reply. That is slow, and on long
prompts it is the cause of empty completions — the model runs out of budget
mid-thought and returns nothing. Its one real advantage here is a judge that
converges in a single round rather than three, which costs retrieval rounds but
changed no answer.

Size matters more than the table suggests. At 4.7 GB the answering, utility and
embedding models are all resident together on a 17 GB machine; at 9.6 GB they
evict one another and reload — invisible in a per-call benchmark, expensive in a
real run.

`llama3.2:3b` is the fastest and the wrong choice for this role: its judge never
returns "sufficient", so every question burns the full step budget, and routing
drops to 85%.

### Use a smaller model for the mechanical passes

`LLM_UTILITY_MODEL` routes context sentences, entity extraction, summaries and
coreference to a separate model. These are the bulk of ingest and none of them
reason — they rewrite or classify text.

On 18 chunks: **20 seconds with `llama3.2:3b` against ~234 seconds with
`gemma4:e4b`, about 11x** — and the cheaper model produced the *better* index
(recall@1 61% vs 50%). Cache and index tags follow the model that produced each
artifact, so switching invalidates rather than silently reusing another model's
work.

Set only `LLM_CHAT_MODEL` and nothing changes: the utility model defaults to it.

### Do not rerank with a chat model

Off by default, and it should stay off — see the limitations section for the
measurements. Briefly: RRF's own top-5 is already 94% correct, and a chat model
asked to reorder 20 candidates evicts right answers rather than promoting them.
A purpose-built cross-encoder is a different thing and `index.Reranker` exists so
one can be dropped in.

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
- **tiers/** — the four retrievers, plus the entity graph the relational tier
  traverses (`graph.go` builds and canonicalizes it, `coref.go` adjudicates
  which names denote one thing)
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

On the 13 bundled cases, by router and answering model:

| Router | Model | top-1 | top-2 |
|---|---|---|---|
| heuristic | — (keywords) | 12/13 (92%) | 12/13 (92%) |
| llm | `gemma4:e4b` | 12/13 (92%) | 13/13 (100%) |
| llm | `qwen2.5:7b` | 12/13 (92%) | 12/13 (92%) |
| llm | `llama3.2:3b` | 11/13 (85%) | 13/13 (100%) |

Top-2 matters because the loop fans out — a correct tier ranked second is still
searched in the same round, so a top-2 hit is a near-miss rather than a miss.

The keyword router matches every model on top-1 while costing nothing and taking
no time, which is why it remains the default. `--llm-router` is worth it only
when questions use vocabulary the keywords do not cover.

The heuristic was at 85% until a real corpus showed its relational vocabulary
was entirely org-chart shaped (`reports to`, `manager`, `signed`) with nothing
for how technical documents state relationships. Adding `depends on`, `requires`,
`part of` and friends moved it to parity on top-1. Its one remaining miss —
"what is the approved budget figure?" — carries no arithmetic keyword, which is
the irreducible limit of keyword routing.

**Retrieval recall** — the broadest signal, and the one hand-written cases
cannot give:

```bash
go run ./cmd/dispatch eval retrieval -k 5
```

It generates one question per chunk *from that chunk alone* and asks whether the
chunk comes back. The generator sees no other chunk, no index and no tier, so it
cannot flatter the system the way an author who has read the whole corpus can,
and every chunk gets probed rather than the eight someone chose. Questions are
cached, so the benchmark does not quietly rewrite itself between runs.

| Corpus | Questions by | Coverage | recall@1 | recall@5 |
|---|---|---|---|---|
| bundled, 18 chunks | `llama3.2:3b` | 18/18 | 50% | 94% |
| bundled, 18 chunks | `qwen2.5:7b` | 17/18 | 53% | 94% |
| real, 70 chunks | `llama3.2:3b` | 65/70 | 54% | 100% |
| real, 70 chunks | `qwen2.5:7b` | 62/70 | 55% | 95% |

recall@1 is the stable number — 50-55% regardless of who writes the questions,
which is the useful signal: **the right chunk is the top hit about half the
time, and in the top five almost always.** Run with `-k 4` or `-k 5`; `-k 1`
would be wrong as often as right.

recall@5 moves between 94% and 100% with the question set, so read it as "about
95%" rather than any single figure. Inspecting the real corpus's three misses
under `qwen2.5:7b` found that two were not retrieval failures at all: one chunk
was a file whose entire content was the string `CLAUDE.md`, the other a block of
badge markup. Nothing could retrieve them because they answer nothing — see
`--min-words` under Commands.

Coverage is reported because it can be gamed. A question that refers to its
source rather than naming its subject — "what configurations are mentioned in
the passage?" — fits every chunk about configuration, so no retriever can pick
the right one and scoring against it measures the generator. Those are skipped,
which means 5 of the real corpus's 70 chunks go unmeasured. Printing 100% while
quietly dropping the hard cases is how a benchmark flatters itself, so the
denominator is always shown.

On the bundled corpus the context sentences themselves were compared:
`gemma4:e4b` gave recall@1 50%, `llama3.2:3b` gave 61% — the cheaper model wrote
the better index.

**This is the number the answer eval was hiding.** Both corpora score 8/8 there,
but the top-ranked chunk is wrong roughly four times in ten — `-k 4` simply
supplies enough evidence to answer anyway. Recall@5 at 94% is what makes the
answer scores possible; recall@1 is what they conceal.

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

Two corpora are scored, and the second is the one that means anything:

| Corpus | Chunks | retrieval | facts | citations | end-to-end | Time |
|---|---|---|---|---|---|---|
| bundled (`testdata/corpus`) | 18 | 8/8 | 14/14 | 8/8 | 8/8 | 2m44s |
| real (15-doc repo checkout) | 70 | 8/8 | 11/11 | 8/8 | 8/8 | 4m56s |

Both with the recommended pairing — `qwen2.5:7b` answering, `llama3.2:3b` for
the mechanical passes.

Both at `-k 4` over semantic + hierarchical + relational. The structured tier is
absent from these runs because it needs `--sql-dir`; it is exercised separately
by the design doc's own example, "the total on invoice 4471", which routes
`[structured semantic]` and answers 162500 from `[structured:line_items]`.

The artifacts behind those rows:

| Corpus | Docs | Chunks | Summary tree | Entities | Relations |
|---|---|---|---|---|---|
| bundled | 6 | 18 | 5 summaries, 2 levels | 54 | 69 |
| real | 15 | 70 | 18 summaries, 3 levels | 468 | 193 |

```bash
# the real one, against an index built from your own documents
go run ./cmd/dispatch eval answers --index .dispatch/mine.json --cases mine.json -k 4
```

Both rows were re-measured after coreference merging landed, and neither moved.
That is the result worth having from a change like that: entity merging is a
destructive rewrite of the graph, so the useful evidence is that it changed
nothing downstream. A question naming a merged-away variant still resolves —
asked about the "statement mailer", which is now an alias of "customer statement
mailer", the relational tier returns that node's edges.

The bundled score is close to meaningless on its own: those cases were written
against a corpus written for them, and at `-k 4` across three tiers up to 12 of
18 chunks reach the evidence, so retrieval barely has to *rank*. The real corpus
at 70 chunks shows about a tenth of the corpus per query, which is a genuine
ranking test.

**Neither score should be read as "it works".** The reason is not the one it
used to be. An earlier run failed on "what is this project and what is it for?"
with an empty answer, and passed on retry with nothing changed — an intermittent
failure, which is worse than a consistent one. That is now understood and fixed
rather than merely absent: it was a reasoning model exhausting its output budget
before answering, and the recommended answering model emits no reasoning tokens
at all, so the mechanism is gone rather than unobserved. The case has passed four
consecutive times on the real corpus under the new configuration.

What remains true:

1. **A perfect score is a reason to distrust the eval first.** The harness was
   checked against negative controls before its numbers were believed: a case
   with a deliberately wrong `expect_sources` reports `RETRIEVAL` while still
   scoring `facts 1/1` — proving the two metrics are genuinely independent — and
   a case demanding a fact absent from the corpus reports `FACTS` with the miss
   named. A check that cannot fail looks exactly like a check that passes.

## Known limitations

Measured, not guessed:

- **Coreference trades recall for precision, deliberately.** Variants like
  `Atlas` / `Project Atlas` are merged so both reach one node. Candidates are
  lexical — a unique token-suffix of exactly one other entity, within two tokens
  of growth — and the merge decision goes to the model, three votes, majority
  required.

  Measured across both corpora: **31 candidates → 11 merges, all 11 correct**,
  and every dangerous pair correctly refused (`openai`/`azure openai`,
  `model`/`user model`, `api`/`packages api`). The cost is recall: 20 candidates
  were refused, some of which were probably valid — `external vendor` /
  `single external vendor` merged on one corpus and was refused on a direct
  probe. That is the intended direction, though: a refused merge loses a
  connection, a wrong one invents relationships nothing downstream can detect.

  A pure-lexical version was tried first and rejected: it produced 23 merges on
  the real corpus and got roughly a third wrong, including `openai` →
  `azure openai`. That pair is the proof no string rule suffices — identical in
  shape to `atlas` → `project atlas`, opposite in meaning.

  Merged entities stay reachable under their old names: the variant becomes an
  alias, because folding `atlas` into `project atlas` would otherwise delete the
  key a question saying only "Atlas" matches, making retrieval *worse*. Every
  merge is printed at ingest for audit, and `GraphOptions.NoCoreference` turns
  the whole pass off.

  Partial personal names are handled separately, in seeding rather than
  merging. "Who does Priya report to?" reaches `Priya Raman` even though no
  `priya` node exists — the corpus only ever writes the full name, so there was
  nothing to merge and the real gap was that the question matched no key.
  Seeding is also the safer place for it: a wrong seed adds evidence that
  hop-ranking pushes down, where a wrong merge would be permanent. It is
  restricted to `person`-typed entities and requires the first name to be
  unambiguous, so `packages` never seeds `packages/api` and two people sharing a
  first name seed neither. It therefore depends on the extractor typing people
  correctly; anyone typed otherwise keeps full-name-only matching.

  Two things make this fragile on a small model, both found by measurement:
  a single sample is a coin flip (gemma4 answered both ways on the same pair,
  which is why it votes), and a one-sided prompt collapses it to always-no
  (four "false" examples and no "true" ones produced zero merges from nine
  candidates — indistinguishable from a broken feature).
- **Thinking models can spend their whole output budget reasoning and return no
  answer.** Found on the real corpus: `gemma4:e4b` given four evidence chunks
  produced 1,114 characters of reasoning, hit `finish_reason: "length"`, and
  emitted empty content. dispatch used to pass that through as a blank answer,
  which reads like "the corpus doesn't say" when the real cause is the model.
  It is now a hard error naming the cause and the fix, and the fix is verified:
  the same question answers correctly at `-k 2`.

  It was **intermittent**, which is worse than consistent: sampling decided
  whether the model reasoned past the limit, and anything enlarging the prompt
  raised the odds — a larger `-k`, larger chunks, more registered tiers, or the
  loop refining and carrying a second round of evidence into generation. The
  agent loop could push a model over its budget by working correctly.

  **This is why the recommended answering model is non-thinking.** Given the same
  prompt, `gemma4:e4b` emitted 1,242 characters of reasoning and `qwen2.5:7b`
  emitted none. With no reasoning tokens the failure mode does not exist, rather
  than being rare — which is the difference between a fix and a lucky run. The
  error and its diagnostic message stay, because a hosted reasoning model can
  still hit it.
- **Vague queries retrieve noise.** "What is this project and what is it for?"
  has no distinctive content words, so both BM25 and the embedding latch onto
  incidental matches — the top hit was a translations how-to that merely says
  "in the project". Contextual chunking helps but does not rescue a query with
  nothing to match on.
- **Ingest is slow on a local thinking model.** Measured on `gemma4:e4b` over
  70 chunks of ~300 tokens:

  | Stage | Rate | 70 chunks | Cached? |
  |---|---|---|---|
  | contextual chunking | ~13s/chunk | ~15 min | yes |
  | graph extraction | ~42s/chunk (p90 65s) | ~40 min | yes |
  | coreference adjudication | ~46s/candidate (3 votes) | ~18 min | yes |
  | RAPTOR tree | ~30s/summary | ~10 min | **no** |

  Coreference scales with candidate count, not chunk count — 22 candidates for
  70 chunks — so it grows far more slowly than the rest.

  Chunking and extraction are cached by content hash, so re-ingest is free — a
  second run over unchanged documents reported 70 cache hits and 0 calls. The
  tree is not cached and pays again on every `--hierarchy`.
  `index.Config.ContextLLM` accepts a separate cheaper model for the chunking
  pass, which is the biggest single lever; the CLI does not expose it yet.
- **`isReadOnly` is defense in depth, not the primary guard.** Grant the
  executing database role SELECT only. A generated-SQL allowlist is string
  matching against an adversary who controls the model's input.
- **`TableRunner` is not a SQL engine.** It is an in-memory CSV runner covering a
  documented `SELECT` subset so the structured tier is demonstrable without a
  driver. Anything outside the subset is a clear error, never a wrong answer.
  Production implements `SQLRunner` over `database/sql`.
- **Reranking with a general chat model makes retrieval worse. Tested twice.**
  `index.Reranker` exists, `--rerank` enables it, the tests pass — and it is off
  by default because every measurement says it should be:

  | Reranker | recall@1 | recall@5 | Time |
  |---|---|---|---|
  | none | 53% | 94% | 31s |
  | `llama3.2:3b` | 33% | 39% | 3m39s |
  | `qwen2.5:7b` | 35% | 88% | 6m47s |

  The first result could be dismissed as a 3B model being too weak. The second
  cannot: a 2.3x larger model that matches a 9.6 GB reasoning model on routing
  and answer quality is *still* worse than no reranking, at 13x the wall-clock.
  Size was not the problem.

  The mechanism is straightforward once stated. RRF's own top-5 is already 94%
  correct, and Search over-fetches 20 candidates for the reranker to choose 5
  from. Reranking can only help if the reranker orders better than RRF already
  does; otherwise the extra 15 candidates are 15 chances to evict a right answer.
  A chat model asked to score passages 0-10 does not order better than RRF.

  The implementation is not at fault: it parses scores, ignores bogus ids, keeps
  unscored candidates and falls back to retrieval order on failure. The
  literature this came from (Anthropic's 49% → 67%) assumes a purpose-built
  cross-encoder — Cohere Rerank, `bge-reranker`, Voyage — trained for exactly
  this ordering task. `index.Reranker` is a one-method interface so one can be
  dropped in. Until then, off.
- **Concurrency is bounded by the server, and then by memory.** The eval loops
  run `--jobs` items at once (default 4), but Ollama defaults to
  `OLLAMA_NUM_PARALLEL=1` and serves one request at a time, so client
  concurrency alone bought 9%:

  | Configuration | Answer eval | vs serial |
  |---|---|---|
  | `--jobs 1`, `NUM_PARALLEL=1` | 5m27s | — |
  | `--jobs 4`, `NUM_PARALLEL=1` | 4m58s | 9% |
  | `--jobs 4`, `NUM_PARALLEL=2` | 4m17s | 21% |

  Raising it helps, but far less than it should. Two concurrent `gemma4:e4b`
  calls in isolation ran 2.15x faster than serial; inside the eval that became
  21%. The gap is memory — a 9.6 GB model, a second context slot and the
  embedding model on a 17 GB machine leaves nothing spare, and each case is
  internally sequential anyway (judge must finish before generate).

  `OLLAMA_NUM_PARALLEL` is read by the server at startup, so it needs the Ollama
  app restarted rather than an exported shell variable — and on macOS the app
  spawns its server with a curated environment that ignores `launchctl setenv`.
  Note also that a separately-installed `ollama` CLI may be older than the one
  the app bundles: mine could not load `gemma4` at all
  (`unknown model architecture`). Use
  `/Applications/Ollama.app/Contents/Resources/ollama` if running the server by
  hand.

  The larger levers remain a smaller answering model or a hosted endpoint.
- **Both eval corpora are still small** — 18 and 70 chunks. 70 is enough to make
  ranking matter; it is not enough to say anything about behaviour at 10,000,
  where a flat cosine scan and an in-memory graph both stop being reasonable.
- **The loop rarely refines on these corpora.** Almost every case answers in
  `rounds 1`, because `-k` across several tiers already surfaces enough. That is
  the corpora being small rather than the judge being lenient — refinement is
  demonstrable at `-k 1` — but it does mean the refine path has far less real
  mileage than the retrieve path.

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
