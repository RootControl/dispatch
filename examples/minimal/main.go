// Command minimal is the smallest useful dispatch program: index some
// documents, then answer a question with citations.
//
// It is also the answer to "how do I ingest PDFs?" — Ingest takes
// []core.Doc, so any parser that produces text works. dispatch deliberately
// ships no format handling of its own.
//
//	LLM_BASE_URL=http://localhost:11434/v1 \
//	LLM_CHAT_MODEL=qwen2.5:7b \
//	LLM_EMBED_MODEL=nomic-embed-text \
//	go run ./examples/minimal
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/RootControl/dispatch/agent"
	"github.com/RootControl/dispatch/core"
	"github.com/RootControl/dispatch/index"
	"github.com/RootControl/dispatch/llm"
	"github.com/RootControl/dispatch/router"
	"github.com/RootControl/dispatch/tiers"
)

func main() {
	ctx := context.Background()

	// Reads LLM_BASE_URL, LLM_API_KEY, LLM_CHAT_MODEL, LLM_EMBED_MODEL.
	client, err := llm.New(llm.Config{})
	if err != nil {
		log.Fatal(err)
	}

	// Contextualize costs one LLM call per chunk and is the single biggest
	// retrieval win. Set a Cache to make re-ingesting free.
	store := index.New(index.Config{
		LLM:           client,
		Contextualize: true,
		Cache:         index.NewCache(".dispatch/cache"),
		CacheTag:      client.ChatModel(),
		EmbedTag:      client.EmbedModel(),
	})

	// Documents come from anywhere — your PDF extractor, a database row, an
	// HTTP response. dispatch only wants an ID and text.
	docs := []core.Doc{
		{ID: "handbook", Text: "Expenses over $500 need director approval. " +
			"Travel is booked through Navan. Receipts are due within 30 days."},
		{ID: "security", Text: "Production access requires hardware MFA. " +
			"Access reviews happen quarterly and are owned by the platform team."},
	}
	if _, err := store.Ingest(ctx, docs); err != nil {
		log.Fatal(err)
	}

	// One tier here; register more and the loop fans out across them.
	semantic := tiers.NewSemantic(store)
	loop := &agent.Loop{
		LLM:        client,
		Router:     router.Heuristic{Available: []core.Tier{core.TierSemantic}},
		Retrievers: map[core.Tier]core.Retriever{core.TierSemantic: semantic},
	}

	answer, err := loop.Run(ctx, "What approval do large expenses need?")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(answer.Text)

	// Every claim cites [tier:source]; Evidence resolves those back to chunks.
	for _, e := range answer.Evidence {
		fmt.Printf("  %s\n", e.Cite())
	}

	// The trace shows what the agent actually did: which tiers it searched,
	// whether the judge asked for more, and how the query was refined.
	fmt.Println(answer.Trace)
}
