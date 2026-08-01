# dispatch

Tiered retrieval in Go, **with the evals to tell you whether it works**.
Zero dependencies — stdlib only. Runs against any OpenAI-compatible endpoint,
including a local Ollama.

Most RAG systems cannot answer a basic question about themselves: *how often is
the top result the right one?* dispatch measures it, with no labelled data —
it writes a question for each chunk and checks whether that chunk comes back.

On its own corpora that number is **54%** at rank 1 and **~95%** in the top five
— rising to **74%** at rank 1 with a cross-encoder reranker in front of it.
Which is the point: retrieve at `-k 4` or `-k 5`, never `-k 1`, and now you know
why rather than guessing — and you know what moves it. Run it against your own
documents and you will get your own number in a couple of minutes.

Four findings from doing that here, all reproducible from this repo:

- **Reranking depends on the model class, and then not on its size.** A 7B
  *chat* model took recall@1 to 35%. A 278M *cross-encoder* took it from 54% to
  **74%** — the largest single improvement measured here. Doubling that
  cross-encoder to 568M then made it *worse* (69%). `--rerank` means the
  cross-encoder; `--rerank-llm` is the path that fails.
  ([numbers](#known-limitations))
- **HyDE query expansion did nothing, at 21x the cost.** The standard fix for
  vague queries left recall@1 at 54% on the real corpus, cost a point of
  recall@5, and turned 7.8s into 2m44s. Also off by default.
  ([why](#known-limitations))
- **A 3B model built a better index than an 8B reasoning model**, 11x faster,
  and a non-thinking answering model matched a reasoning one on every quality
  measure at half the memory. ([models](#models))
- **The flat in-memory index is linear in both latency and memory** — 10ms and
  87 MB at 10,000 chunks, 54ms and 431 MB at 50,000. That is the number behind
  "swap it for pgvector past a few thousand", and
  [`index/pgvector`](index/pgvector) is the swap. ([measured](#known-limitations))

```bash
go get github.com/RootControl/dispatch
```

```go
client, _ := llm.New(llm.Config{}) // LLM_BASE_URL, LLM_CHAT_MODEL, LLM_EMBED_MODEL

store := index.New(index.Config{LLM: client, Contextualize: true})
store.Ingest(ctx, []core.Doc{
    {ID: "handbook", Text: "Expenses over $500 need director approval."},
})

loop := &agent.Loop{
    LLM:        client,
    Router:     router.Heuristic{Available: []core.Tier{core.TierSemantic}},
    Retrievers: map[core.Tier]core.Retriever{core.TierSemantic: tiers.NewSemantic(store)},
}
answer, _ := loop.Run(ctx, "What approval do large expenses need?")
fmt.Println(answer.Text)  // cites [semantic:handbook#0]
fmt.Println(answer.Trace) // shows every tier searched and every judge verdict
```

Runnable version: [`examples/minimal`](examples/minimal/main.go). There is a CLI
too — see [Commands](#commands) — but the library is the product.

**`Ingest` takes `[]core.Doc`, so bring your own parser.** dispatch ships no
format handling on purpose: point your PDF, docx, Confluence or database
extractor at it and hand over `{ID, Text}`. The CLI reads `.md` and `.txt`
because that is all a CLI needs to demonstrate; the library has no such limit.

Everything is behind a one-method interface — `core.Retriever`, `index.Reranker`,
`index.Searcher`, `router.Router`, `tiers.SQLRunner` — so a tier, a reranker or
the whole store can be replaced without touching anything upstream. The bundled
`index.Store` is a flat in-memory index, comfortable to a few thousand chunks
([measured](#known-limitations)); [`index/pgvector`](index/pgvector) is the
drop-in for past that, over `database/sql` with no dependency added.

## Why not just RAG or GraphRAG

Both share a hidden assumption: retrieval runs once, before generation. Different
questions are different retrieval problems.

| Question | Tier | Wrong tool |
|---|---|---|
| "Total on invoice 4471?" | structured (text-to-SQL) | vector top-k |
| "What does the corpus say about X?" | semantic (contextual chunk + hybrid) | — |
| "Who does Bob report to, and what did she sign?" | relational (entity graph, multi-hop) | single chunk |
| "Recurring themes across all docs?" | hierarchical (RAPTOR) | plain RAG (no chunk holds the answer) |

## Trying it from the CLI

The CLI exists to demonstrate and evaluate the library — it is how you get a
number for your own corpus without writing any code.

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
dispatch ingest --corpus DIR [--index PATH] [--dry-run] [--no-context] [--full]
                [--chunk-tokens N] [--min-words N] [--exclude DIRS]
                [--hierarchy] [--graph] [--utility-model NAME]

dispatch ask [--index PATH] [--trace] [-k N] [--max-steps N] [--llm-router]
             [--remember] [--sql-dir DIR] [--max-hops N] [--filter K=V]
             [--rerank] [--retrieve-only] [--stream] [--diverse] [--per-doc N]
             [--hyde] [--timeout D] [--max-calls N] "question"

dispatch eval [--router heuristic|llm|both] [--cases FILE] [-v]   # routing accuracy
dispatch eval answers [--index PATH] [--cases FILE] [-k N] [-v]   # answer quality
dispatch eval retrieval [--index PATH] [-k N] [--rerank] [--hyde] [-v]  # recall over every chunk
dispatch eval scale [--sizes N,N,N] [--queries N]                 # latency/memory vs corpus size
```

The hierarchical and relational tiers are opt-in at ingest time because each
costs LLM calls; `ask` registers a tier only when its artifact exists.

Artifacts live beside their index — `--index .dispatch/x.json` writes
`.dispatch/x-hierarchy.json` and `.dispatch/x-graph.json` — so several corpora
can coexist without overwriting each other.

### Ingest is incremental

`ingest` loads the existing index, re-derives only the documents that changed,
and prunes the ones the corpus has lost. A document is unchanged when a
fingerprint over its text, its metadata, the splitter settings and both model
identities matches what it was last ingested under — so editing a chunk size or
switching the utility model re-derives everything, as it must, while editing one
file costs one file.

```
$ dispatch ingest --corpus ~/docs --index .dispatch/mine.json
updating .dispatch/mine.json (3 chunks indexed)
ingested 3 docs -> 3 chunks (0 context calls, 0 cache hits; 0 embedded, 0 cached)
  3 of 3 docs unchanged and skipped

$ # edit one file, add one, delete one
$ dispatch ingest --corpus ~/docs --index .dispatch/mine.json
pruned 1 chunk(s) from documents no longer in the corpus
ingested 3 docs -> 3 chunks (2 context calls, 0 cache hits; 2 embedded, 0 cached)
  1 of 3 docs unchanged and skipped
```

`--full` rebuilds from scratch. Embeddings are content-addressed in the same
cache as context sentences, so even a full rebuild re-embeds only chunks whose
text actually changed — budget roughly 15 KB of cache per chunk at 768
dimensions.

### Scoping a query with `--filter`

`--filter key=value` restricts retrieval to chunks whose metadata matches; a
trailing `*` matches by prefix. It is repeatable and every clause must match.
The CLI records `path`, `doc`, `dir` and `ext` per document.

```bash
dispatch ask --filter dir=notes "who is staffed on this?"
dispatch ask --filter 'dir=docs*' --filter ext=md "what are the risks?"
```

**A filter is enforced, not merely applied.** Only tiers backed by an
`index.Store` — semantic and archival memory — can match on chunk metadata; the
hierarchical, relational and structured tiers retrieve over derived artifacts
that carry none. Rather than let those answer from outside the scope, the loop
drops any retriever that does not implement `core.Filterable` and says so in the
trace. A filter that leaves no usable tier is an error, not an empty answer.

That is the deliberate cost: filtering a question the hierarchical tier would
have answered gives you a narrower search, not a wider one that quietly ignores
you. Archival memory is filtered on the same rule, which means a filter naming a
document field excludes every takeaway — a query scoped to part of the corpus
should not be answered from a conclusion drawn somewhere else.

### Stopping one document from eating the evidence budget

Evidence was deduplicated by citation identity, which catches nothing but
literally the same chunk. Measured on the real 70-chunk index, "who is staffed
on the project?" returned this:

```
[semantic:config/translations/README.md#0]
[semantic:CLAUDE.md#0]
[semantic:CLAUDE.md#1]
[semantic:CLAUDE.md#6]
```

Three of four slots to one document, and `#0`/`#1` are adjacent — so they share
the default 100-token chunk overlap verbatim. The budget bought one file said
three ways. It compounds across tiers: a passage can arrive as a semantic chunk
and again inside a hierarchical summary built from it, under two citations that
look like two sources and are not.

`--diverse` drops evidence that restates evidence already held (Jaccard over
content tokens, stopwords excluded, 0.6 by default). `--per-doc N` caps how many
items one document may contribute — the blunter instrument, and the effective
one on the case above:

```
$ dispatch ask --per-doc 2 --trace "who is staffed on the project?"
 4 diverse  3 evidence item(s) kept — dropped 1: [semantic:CLAUDE.md#6] (doc cap)
```

Suppression runs over the whole set each round, not just new arrivals, and runs
before the budget is applied — so a dropped duplicate frees its slot for
something distinct rather than having already spent it. **Off by default**: it
discards retrieved evidence, and doing that silently to someone who did not ask
is the wrong default.

### Citations are checked before you see the answer

Every `[tier:source]` marker in a generated answer is resolved against the
evidence that was actually retrieved. A marker that resolves to nothing is a
fabricated source, and it looks exactly like a real one — which is why this runs
on every answer rather than only in the eval, and why the warning goes to stderr
whether or not you asked for `--trace`:

```
$ dispatch ask "what is the budget?"
The budget is 4M [semantic:atlas-charter.md#1] with a reserve [semantic:invented.md#9].

WARNING: 1 citation(s) resolve to no retrieved evidence: [semantic:invented.md#9]
```

`Answer.Citations` carries the same report to a library caller — `Resolved`,
`Unresolved`, and `Uncited` (evidence the answer never used, which is normal,
since evidence is over-fetched on purpose). `Loop.CitationPolicy` chooses what
happens: `CiteReport` (default, tell the caller and change nothing),
`CiteStrip` (remove the bad markers, still reporting them), or `CiteError`
(fail the answer). The default does not rewrite the model's text, because
silently editing an answer is its own kind of dishonesty.

The check costs no LLM call — the answer and the evidence are both already in
hand — and `agent.ExtractCitations` is the single implementation the eval uses
too. Two copies would drift, and the copy that drifts is the one that stops
catching fabrications.

### Streaming, budgets and cost

```bash
dispatch ask --stream --timeout 90s --max-calls 4 --trace "what is the budget?"
```

`--stream` prints the answer as the model produces it. `llm.Streamer` is a
separate optional interface, so a model that cannot stream falls back to a
single call and every other implementation stays as small as it was. Only
generation streams — the judge emits JSON nobody wants to watch assemble.

`--timeout` bounds the whole question. Nothing did before: the HTTP client has a
per-request timeout and one question makes many requests, so a hung endpoint
hung the command forever. A deadline that expires mid-retrieval is reported as a
deadline — `fanOut` records a tier's failure and carries on, which is right for
one tier and wrong for a dead endpoint, where it would have surfaced as "no
evidence retrieved" and blamed the corpus.

`--max-calls` caps LLM calls per question, which is the unit that is billed;
`--max-steps` bounds rounds, which is not the same thing once an expander adds a
call per round and write-back adds one at the end. Reaching the cap is not an
error — the loop answers from what it has — and generation is always reserved,
so the budget can never run out at the one moment where everything has been paid
for and nothing produced.

`--trace` now reports tokens, read from the server's own `usage` block rather
than estimated. A server that omits it says so instead of printing a zero that
reads as free:

```
rounds: 1, llm calls: 2
tokens: 1509 prompt + 64 completion = 1573 over 2 call(s)
```

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
  everything above it tests offline; optional SSE streaming (`stream.go`) and
  token accounting from the server's own `usage` block
- **index** — contextual chunking + hybrid store (cosine + BM25 fused by RRF),
  the content-addressed cache, and `index.Searcher`, the seam a backend swaps at
- **index/pgvector** — the same tiers over Postgres + pgvector, via
  `database/sql` and no dependency
- **tiers/** — the four retrievers, plus the entity graph the relational tier
  traverses (`graph.go` builds and canonicalizes it, `coref.go` adjudicates
  which names denote one thing)
- **router** — LLM classification with a keyword-heuristic fallback
- **agent** — the loop (`loop.go`), write-back memory (`memory.go`), trace,
  citation verification (`cite.go`), evidence diversity (`diversity.go`),
  query expansion (`expand.go`)
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

That is the number *before reranking*, and it is the one worth quoting, because
it describes what the retrieval stack does on its own. A cross-encoder in front
of it takes the real corpus to 74% — see [reranking](#known-limitations), where
the same table also shows two chat models taking it to 33% and 35%. Everything
below in this section is measured without a reranker.

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
- **Vague queries retrieve noise, and HyDE did not fix it. Measured.**
  "What is this project and what is it for?" has no distinctive content words,
  so both BM25 and the embedding latch onto incidental matches — the top hit was
  a translations how-to that merely says "in the project". Contextual chunking
  helps but does not rescue a query with nothing to match on.

  Query expansion is the standard answer: write a hypothetical answer passage
  and embed *that*, on the theory that a real answer resembles a fake answer far
  more than it resembles a question. `agent.HyDE` implements it and `--hyde`
  turns it on. The measurement:

  | Corpus | Expansion | recall@1 | recall@5 | Time |
  |---|---|---|---|---|
  | bundled, 18 chunks | none | 50% | 94% | — |
  | bundled, 18 chunks | HyDE | 56% | 94% | — |
  | real, 70 chunks | none | **54%** | **100%** | **7.8s** |
  | real, 70 chunks | HyDE | 54% | 98% | 2m44s |

  On the corpus that matters it changed recall@1 not at all, cost a point of
  recall@5, and took **21x the wall-clock**. The bundled corpus moved by one
  chunk out of eighteen, which is noise and should not be read as a gain.

  And on the query that motivated it, the failure is unchanged: the translations
  how-to is still the top hit. The hypothetical explains why — asked what this
  project is, `llama3.2:3b` confidently invented an unrelated "EcoCycle"
  initiative. A vague query gives the expander nothing to ground on either, so
  it invents a subject and retrieves confidently about the wrong one.

  So: **off by default**, like the LLM reranker, for the same reason. It is kept
  because the implementation is sound and a stronger expander model or a corpus
  with more distinctive vocabulary might change the result — measure it before
  believing it. Two mitigations are built in: the question stays in the embedded
  text so a bad hypothetical degrades the query rather than replacing it, and
  the expansion reaches only the dense half, since BM25 would treat every
  invented noun as a high-idf term (`core.Query.Embedding` explains that).
- **Ingest is slow on a local thinking model.** Measured on `gemma4:e4b` over
  70 chunks of ~300 tokens:

  | Stage | Rate | 70 chunks | Cached? |
  |---|---|---|---|
  | contextual chunking | ~13s/chunk | ~15 min | yes |
  | graph extraction | ~42s/chunk (p90 65s) | ~40 min | yes |
  | coreference adjudication | ~46s/candidate (3 votes) | ~18 min | yes |
  | RAPTOR tree | ~30s/summary | ~10 min | yes |

  Coreference scales with candidate count, not chunk count — 22 candidates for
  70 chunks — so it grows far more slowly than the rest.

  Every stage is now cached by content hash, including the two that were not:
  chunk embeddings and the RAPTOR summaries' embeddings. A re-ingest over
  unchanged documents is skipped outright at the document level and makes no
  calls of any kind. The lever that remains is the utility model —
  `--utility-model`, or `LLM_UTILITY_MODEL`, or `index.Config.ContextLLM` from
  the library — which is worth ~11x on the passes that dominate a first ingest.
- **`isReadOnly` is defense in depth, not the primary guard.** Grant the
  executing database role SELECT only. A generated-SQL allowlist is string
  matching against an adversary who controls the model's input.
- **`TableRunner` is not a SQL engine.** It is an in-memory CSV runner covering a
  documented `SELECT` subset so the structured tier is demonstrable without a
  driver. Anything outside the subset is a clear error, never a wrong answer.
  Production implements `SQLRunner` over `database/sql`.
- **Reranking with a general chat model makes retrieval worse. Tested twice.**
  `index.Reranker` exists, `--rerank-llm` enables it, the tests pass — and it is
  off by default because every measurement says it should be:

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
  this ordering task.

  **`index.CrossEncoder` is now that path**, and `--rerank` means it:

  ```bash
  # A local cross-encoder. Any /rerank endpoint works — Cohere, Jina, Voyage,
  # or text-embeddings-inference; this one is what the table below was measured
  # against, and it runs on a laptop.
  docker run -d --name rerank -p 7997:7997 michaelf34/infinity:latest \
      v2 --model-id BAAI/bge-reranker-base --port 7997 --engine torch

  export RERANK_BASE_URL=http://localhost:7997   # or https://api.cohere.com/v2
  export RERANK_API_KEY=...                      # omit for a local server
  export RERANK_MODEL=BAAI/bge-reranker-base
  export RERANK_TIMEOUT=5m                       # optional; default 60s

  dispatch eval retrieval -k 5              # baseline
  dispatch eval retrieval -k 5 --rerank     # with the cross-encoder
  ```

  `RERANK_TIMEOUT` exists because the 60s default is genuinely too tight for
  some real configurations: `bge-reranker-v2-m3` scoring 20 passages on an
  emulated CPU ran past it, and the failure surfaces as a bare "context deadline
  exceeded" that reads like a broken endpoint rather than a slow one.

  It speaks the Cohere/Jina `/rerank` shape and the bare-array shape
  text-embeddings-inference returns, over `net/http` — no new dependency. Unlike
  the LLM reranker it returns errors rather than degrading to retrieval order,
  because a misconfigured endpoint that silently passed the input through would
  be indistinguishable from a reranker that simply never helps, which is the one
  thing the measurement has to be able to tell apart.

  **And it works.** Two cross-encoders, measured against both corpora:

  | Corpus | Reranker | Params | recall@1 | recall@5 |
  |---|---|---|---|---|
  | bundled, 18 chunks | none | — | 50% | 94% |
  | bundled, 18 chunks | `bge-reranker-base` | 278M | 56% | **100%** |
  | bundled, 18 chunks | `bge-reranker-v2-m3` | 568M | 56% | **100%** |
  | real, 70 chunks | none | — | 54% | **100%** |
  | real, 70 chunks | `bge-reranker-base` | 278M | **74%** | 97% |
  | real, 70 chunks | `bge-reranker-v2-m3` | 568M | 69% | 98% |
  | real, 70 chunks | `qwen2.5:7b` (chat) | 7B | 35% | 88% |
  | real, 70 chunks | `llama3.2:3b` (chat) | 3B | 33% | 39% |

  **+20 points of recall@1 on the real corpus** — 35/65 top hits to 48/65. That
  is the largest single improvement measured anywhere in this project, and it
  lands on exactly the number the rest of the README calls its weakest.

  So the earlier finding was too broad. It is not that reranking does not work;
  it is that **reranking with a chat model does not work**, and the distinction
  is the model class rather than the model size. A 7B chat model took recall@1
  to 35%. A 278M cross-encoder took it to 74%. The literature's assumption was
  load-bearing all along.

  Within the right model class, though, size is not the lever either — and the
  two rerankers make that concrete. `bge-reranker-base` lost two chunks from the
  top five. Both were explainable, and one carried a prediction:
  `README.zh.md#0` is a Chinese document against an English-trained reranker, so
  the multilingual `bge-reranker-v2-m3` should recover it.

  **It does — and costs more than it recovers.** v2-m3 has twice the parameters,
  fixes exactly the miss predicted, takes recall@5 to 98%, and gives up three
  first-place hits to do it: 74% → 69% at rank 1. The corpus is overwhelmingly
  English, so the multilingual model spends capacity on a problem it barely has
  and is worse at the one it does. Use v2-m3 when you actually have
  mixed-language documents, not because it is the bigger model.

  The other miss survives both: `AGENTS.md#0`, probed with "Who is Claude?" — a
  query with nothing distinctive to match, which is the same class of failure
  HyDE could not fix either. No reranker rescues a query that under-specifies
  what it wants.

  A reranker reorders a 20-candidate pool down to 5, so it can evict as well as
  promote. At `-k 5` `bge-reranker-base` traded two top-five hits for thirteen
  first-place ones, which is the trade worth making when the answer reads
  `-k 4`-worth of evidence anyway.

  **Still off by default**, because it needs an endpoint that is not part of
  this repo — not because the evidence is against it. If you have somewhere to
  run a cross-encoder, the measurement above says turn it on, and the two
  commands above say so for your corpus rather than this one.

  *No latency figures here, deliberately.* Both runs were under x86 emulation on
  an arm64 host, which is not a number anyone should plan with — and v2-m3
  additionally needed `--batch-size 2` to stop the server being OOM-killed, and
  `RERANK_TIMEOUT` above the 60s default. Reranking 20 passages per query is
  real work; measure it natively before believing any speed claim in either
  direction.
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
  ranking matter; it is not enough to say anything about *retrieval quality* at
  10,000. What the flat index costs at that size is now measured rather than
  asserted (`dispatch eval scale`, 768 dimensions, this machine):

  | chunks | ingest | search p50 | search p90 | resident | per chunk |
  |---|---|---|---|---|---|
  | 100 | 2ms | 233µs | 244µs | 931 KB | 9.3 KB |
  | 1,000 | 23ms | 1.06ms | 1.21ms | 8.8 MB | 9.0 KB |
  | 10,000 | 677ms | 9.78ms | 10.4ms | 86.6 MB | 8.9 KB |
  | 50,000 | 15.6s | 54.3ms | 55.0ms | 431 MB | 8.8 KB |

  Cleanly linear in both, which is what a full scan of every vector plus every
  BM25 posting predicts. Read it as: a few thousand chunks is comfortable,
  10,000 costs 10ms and 87 MB per process, and 50,000 is where a 54ms floor per
  query and 431 MB resident stop being reasonable.

  **That eval measures the index structure and nothing else.** It uses synthetic
  documents and a deterministic local embedder, so it needs no API key and costs
  nothing — and reports no recall number, because recall over synthetic text
  measures the generator. Quality at scale still needs a real corpus.
  The in-memory entity graph has the same problem and no equivalent measurement.
- **The loop rarely refines on these corpora.** Almost every case answers in
  `rounds 1`, because `-k` across several tiers already surfaces enough. That is
  the corpora being small rather than the judge being lenient — refinement is
  demonstrable at `-k 1` — but it does mean the refine path has far less real
  mileage than the retrieve path.

## Swapping in production backends

The reference `index.Store` is in-memory, and the table above says roughly where
that stops working. `index/pgvector` is the replacement, over `database/sql`:

```go
import _ "github.com/jackc/pgx/v5/stdlib"   // you bring the driver

db, _ := sql.Open("pgx", os.Getenv("DATABASE_URL"))
store, _ := pgvector.New(pgvector.Config{DB: db, LLM: client, Dims: 768})
store.Migrate(ctx)                          // table + HNSW + GIN indexes

loop.Retrievers[core.TierSemantic] = tiers.NewSemantic(store)
```

**dispatch stays dependency-free because the driver is yours.** The package
imports only `database/sql`; nothing enters `go.mod`.

The split is deliberate. `index.Store` keeps chunking, contextual chunking, the
caches and the ingest planner; `pgvector.Store` owns storage and search. That is
why its `Upsert` takes chunks and vectors rather than documents — one
contextual-chunking pass feeds either backend. It runs the same hybrid (cosine
pool + Postgres full-text pool) and fuses with the **same exported
`index.FuseRankings`** as the in-memory path, because ranking is the one place
where two backends drifting apart would be invisible in the results.

The seam is `index.Searcher` (`Search`, `Len`), which `tiers.NewSemantic` takes;
everything above it is unchanged. Metadata filters translate to `meta ->> $n`
over a JSONB column with values bound, never interpolated, and LIKE
metacharacters escaped so a filter value cannot widen its own prefix match.

### Verifying it against a real Postgres

Two test suites, because they check different things. The unit tests are
hermetic — a stdlib `database/sql` fake driver covers query construction,
parameter binding, scan order and error paths — but cannot check that a real
server accepts any of it. [`index/pgvector/integration`](index/pgvector/integration)
does, and is **a separate module so the pgx driver never enters dispatch's
`go.mod`**; the parent's `./...` skips any directory holding its own `go.mod`,
so `go test ./...` at the repo root stays hermetic and dependency-free.

```bash
docker run -d --name pgtest -e POSTGRES_PASSWORD=dispatch \
    -e POSTGRES_DB=dispatch -p 55432:5432 pgvector/pgvector:pg17

cd index/pgvector/integration
DISPATCH_PG_DSN='postgres://postgres:dispatch@localhost:55432/dispatch?sslmode=disable' \
    go test -v ./...
```

12 tests, run against PostgreSQL 17.10 + pgvector: migration accepted and
repeatable, upsert/search round-tripping every column, both halves of the
hybrid, filters (including LIKE-metacharacter escaping), delete, and the
semantic tier over Postgres with and without a filter. Without `DISPATCH_PG_DSN`
they skip.

The DDL builds what it claims, and the planner uses it:

```
"inspect_me_embedding_idx" hnsw (embedding vector_cosine_ops)
"inspect_me_fts_idx"       gin (to_tsvector('english'::regconfig, embedded))
"inspect_me_meta_idx"      gin (meta)

->  Index Scan using inspect_me_embedding_idx   -- ORDER BY embedding <=> $1
->  Bitmap Index Scan on inspect_me_fts_idx     -- to_tsvector @@ plainto_tsquery
```

**The suite was checked against negative controls before its passes were
believed**, the same standard the answer eval is held to. Six deliberate breaks
— unescaped LIKE wildcards, rows returned in scan order instead of fused order,
full-text indexing the body without its context sentence, the lexical half
returning nothing, filters never reaching the SQL, delete as a no-op — each
fail the test that covers them.

That mattered: **the two lexical-half tests originally passed while the
full-text column was deliberately broken**, and took three attempts to isolate.
Searching for a term in the text proved nothing because the vector contained
that term too; giving the target a body-only vector proved nothing because it
was the only row in the table; giving every row one identical vector proved
nothing because Postgres returns tied rows in reverse insertion order, handing
the last-inserted target the top slot for free. The version that works points
the dense half *the wrong way* — fillers at cosine distance 0 from the query,
the target orthogonal — so a target that still ranks first can only have been
lifted lexically. A check that cannot fail looks exactly like a check that
passes.

`tiers.SQLRunner` is an interface — back it with `database/sql` the same way.

## API changes

Pre-1.0, and two signatures moved to carry filters and to let a backend swap in:

| Before | Now |
|---|---|
| `store.Search(ctx, "text", k)` | `store.Search(ctx, core.Query{Text: "text", TopK: k})` |
| `tiers.NewSemantic(*index.Store)` | `tiers.NewSemantic(index.Searcher)` — `*index.Store` still satisfies it |

`core.Query` gained `Filter` and `Expanded`, `core.Retriever` gained an optional
companion interface `core.Filterable`, `llm.LLM` gained an optional companion
interface `llm.Streamer`, and `--rerank` now means the cross-encoder; the
chat-model reranker moved to `--rerank-llm`. Saved indexes load unchanged: one
written before this carries no document fingerprints, so the next ingest
re-derives each document once and is incremental from then on.

`agent.Answer` gained `Citations`, and `Loop` gained `CitationPolicy`,
`Diversity`, `Expander`, `MaxCalls` and `OnDelta` — all zero-valued to the
previous behaviour, except citation verification, which now always runs and
populates `Answer.Citations` (it changes no text under the default policy).
`extractCitations` moved from the eval to `agent.ExtractCitations`.

## Lineage

Anthropic Contextual Retrieval, RAPTOR (hierarchical summary tree), LightRAG /
HippoRAG (graph-guided multi-hop), MemGPT/Letta (tiered write-back memory), and
the agentic-retrieval line where a router picks retrieve/reflect/answer with an
evidence-gap tracker — which is what `agent/loop.go` implements.

No single architecture wins across all query types, which is the whole reason
this routes instead of picking one.
