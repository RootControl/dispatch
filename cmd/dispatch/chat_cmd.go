package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/RootControl/dispatch/agent"
)

// runChat answers questions in a conversation, resolving follow-ups against
// what was already asked.
//
// `ask` is single-shot, and every retrieval path in this project matches a
// question against the corpus. That is exactly wrong for the second question in
// a conversation: "what about its budget?" has one content word and the subject
// lives in the previous turn. The result is not a worse answer — it is evidence
// about the wrong thing, or, on this corpus, no evidence at all.
//
// The fix is in the query rather than in the prompt. Stuffing the history into
// the generation prompt would leave retrieval still fetching the wrong chunks
// and merely give the model a better chance of noticing.
func runChat(args []string) error {
	fs := flag.NewFlagSet("chat", flag.ExitOnError)
	indexPath := fs.String("index", defaultIndexPath, "index to read")
	topK := fs.Int("k", 5, "results to retrieve per tier")
	maxSteps := fs.Int("max-steps", 3, "maximum retrieve/judge rounds")
	maxHops := fs.Int("max-hops", 2, "relational graph traversal depth")
	sqlDir := fs.String("sql-dir", "", "directory of CSV tables to enable the structured (text-to-SQL) tier")
	llmRoute := fs.Bool("llm-router", false, "classify with the model instead of keywords")
	rerank := fs.Bool("rerank", false, "rescore the retrieved shortlist with a cross-encoder (needs RERANK_BASE_URL)")
	remember := fs.Bool("remember", false, "search memory, and write back a takeaway after answering")
	trace := fs.Bool("trace", false, "show routing, retrieval and judge steps for each answer")
	stream := fs.Bool("stream", false, "print each answer as the model produces it")
	turns := fs.Int("turns", 4, "conversation turns the rewriter may consult")
	always := fs.Bool("always-rewrite", false, "rewrite every question, not only those that look like follow-ups")
	noRewrite := fs.Bool("no-rewrite", false, "keep history but never rewrite — for seeing what the rewriting buys")
	allowStale := fs.Bool("allow-stale", false, "answer from a hierarchy or graph built against a different index")
	if err := fs.Parse(args); err != nil {
		return err
	}

	st, err := buildStack(stackOptions{
		IndexPath:  *indexPath,
		SQLDir:     *sqlDir,
		MaxHops:    *maxHops,
		Remember:   *remember,
		Rerank:     *rerank,
		AllowStale: *allowStale,
	})
	if err != nil {
		return err
	}

	loop := &agent.Loop{
		LLM:        st.client,
		Router:     st.router(*llmRoute),
		Retrievers: st.registry,
		MaxSteps:   *maxSteps,
		TopK:       *topK,
		Memory:     st.memory,
		WriteBack:  *remember,
	}
	if *stream {
		loop.OnDelta = func(s string) { fmt.Print(s) }
	}

	session := &agent.Session{Loop: loop, MaxTurns: *turns}
	if !*noRewrite {
		// The utility model, like the other mechanical passes: restating a
		// question is not reasoning, and this sits on the path of every turn.
		util, _, _ := utilityLLM(st.client)
		session.Rewriter = &agent.Condenser{LLM: util, Always: *always}
	}

	fmt.Printf("dispatch chat over %s (%d chunks). Ctrl-D to leave, /reset to forget, /history to review.\n",
		*indexPath, st.store.Len())
	if *noRewrite {
		fmt.Println("--no-rewrite: follow-ups are searched exactly as typed.")
	}
	fmt.Println()

	in := bufio.NewScanner(os.Stdin)
	in.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for {
		fmt.Print("> ")
		if !in.Scan() {
			fmt.Println()
			return in.Err()
		}
		question := strings.TrimSpace(in.Text())
		switch {
		case question == "":
			continue
		case question == "/reset":
			session.Reset()
			fmt.Println("(conversation forgotten)")
			continue
		case question == "/history":
			printHistory(session)
			continue
		case question == "/quit" || question == "/exit":
			return nil
		}

		answer, err := session.Ask(context.Background(), question)
		if err != nil {
			// One bad question must not end the conversation: the commonest
			// cause is a follow-up that found nothing, and the natural next
			// move is to rephrase it.
			fmt.Fprintf(os.Stderr, "error: %v\n\n", err)
			continue
		}
		if *stream {
			fmt.Println()
		} else {
			fmt.Println(answer.Text)
		}
		if !answer.Citations.OK() {
			fmt.Fprintf(os.Stderr, "\nWARNING: %d citation(s) resolve to no retrieved evidence: %s\n",
				len(answer.Citations.Unresolved), strings.Join(answer.Citations.Unresolved, " "))
		}
		// Show the rewrite even without --trace. A question silently searched as
		// something else is the one thing about this mode a user cannot infer
		// from the output, and it is also where it goes wrong.
		if !*trace {
			for _, s := range answer.Trace.Steps {
				if s.Kind == agent.StepRewrite {
					fmt.Printf("\n(searched as: %s)\n", s.Query)
				}
			}
		}
		if *trace {
			fmt.Printf("\n--- trace ---\n%s\n", answer.Trace)
		}
		fmt.Println()
	}
}

func printHistory(s *agent.Session) {
	h := s.History()
	if len(h) == 0 {
		fmt.Println("(nothing asked yet)")
		return
	}
	for i, t := range h {
		fmt.Printf("%d. Q: %s\n   A: %s\n", i+1, t.Question, truncateLine(strings.ReplaceAll(t.Answer, "\n", " "), 120))
	}
}
