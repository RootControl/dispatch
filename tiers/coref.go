package tiers

import (
	"context"
	"fmt"
	"sync"

	"github.com/RootControl/dispatch/index"
	"github.com/RootControl/dispatch/internal/par"
	"github.com/RootControl/dispatch/llm"
)

// corefVotes is how many independent samples adjudicate each candidate pair.
// Odd, so a majority always exists.
const corefVotes = 3

const corefSystem = `You decide whether two names refer to the SAME thing.

B is always A plus extra words. The question is what those extra words do.

SAME (answer true) — the extra words name the SAME thing more fully: a category word, an
owner, or a descriptive adjective.
- "Atlas" / "Project Atlas" -> true          (extra word is the category)
- "mailer" / "customer statement mailer" -> true   (extra words describe the same mailer)
- "RAG API" / "LibreChat's RAG API" -> true  (extra word is the owner)
- "steering group" / "Project Atlas Steering Group" -> true

DIFFERENT (answer false) — the extra words pick out a DIFFERENT member of a family, so
both can exist at once and mean different things.
- "OpenAI" / "Azure OpenAI" -> false         (two providers; both exist)
- "model" / "User model" -> false            (many models; User is one)
- "api" / "packages/api" -> false            (two modules; both exist)
- "cert.pem" / "certs/server-cert.pem" -> false

The test: could a document mention BOTH, meaning different things? Then false. Would a
document use them interchangeably for one thing? Then true.

If both readings are equally plausible, answer false.

Reply with a JSON object only: {"same": <bool>}`

// confirmCoreference returns a predicate that asks the model whether two names
// denote one entity, caching each verdict so a rebuild is free.
//
// Candidates come from a lexical pass, which is high-recall and cannot tell
// "Atlas"/"Project Atlas" (one thing) from "OpenAI"/"Azure OpenAI" (two). The
// volume is small — tens of pairs for a 70-chunk corpus — so one call each is
// affordable in a way that per-chunk work is not.
// The returned counters record adjudication failures. Without them a systematic
// failure — a wrong endpoint, a model that cannot answer — is indistinguishable
// from a model that simply rejected every pair, since both produce zero merges.
func confirmCoreference(ctx context.Context, l llm.LLM, cache *index.Cache, tag string) (
	confirm func(short, long string) bool, failures *int, firstErr *error,
) {
	var n int
	var first error
	return func(short, long string) bool {
		key := index.Key("coref|"+tag, short, long)
		if raw, ok := cache.Get(key); ok {
			return raw == "true"
		}
		msgs := []llm.Message{
			llm.System(corefSystem),
			llm.User(fmt.Sprintf("A: %q\nB: %q\n\nDo A and B name the same entity?", short, long)),
		}

		// Vote rather than sample once. A small local model is not consistent on
		// this judgment: asked whether "Atlas" and "Project Atlas" are the same
		// thing, gemma4 answered yes on one sample and no on another. A single
		// sample is a coin flip, and caching it makes one unlucky flip permanent.
		//
		// A tie counts as no. Combined with treating errors as no, the whole
		// mechanism biases toward leaving entities split — which loses some
		// connections, where the opposite error invents them.
		//
		// The votes run concurrently. They are independent samples of the same
		// prompt and only their tally is used, so nothing about the verdict
		// depends on the order they return in — which makes this the one place
		// in the build where concurrency cannot change the result at all.
		var mu sync.Mutex
		yes, asked := 0, 0
		_ = par.ForEach(ctx, corefVotes, corefVotes, func(ctx context.Context, _ int) error {
			var out struct {
				Same bool `json:"same"`
			}
			err := l.ChatJSON(ctx, msgs, &out)
			mu.Lock()
			defer mu.Unlock()
			if err != nil {
				n++
				if first == nil {
					first = fmt.Errorf("adjudicate %q vs %q: %w", short, long, err)
				}
				return nil
			}
			asked++
			if out.Same {
				yes++
			}
			return nil
		})
		if asked == 0 {
			return false // never cache a verdict nobody gave
		}
		same := yes*2 > corefVotes
		_ = cache.Put(key, fmt.Sprint(same))
		return same
	}, &n, &first
}
