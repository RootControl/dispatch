package main

import (
	"flag"
	"fmt"
	"os"
	"sort"
	"strings"
	"text/tabwriter"

	"github.com/RootControl/dispatch/tiers"
)

// runGraph inspects the entity graph.
//
// The graph is the least legible thing this project builds. Its entities come
// from an LLM reading one chunk at a time, its edges are whatever that model
// chose to state, and its aliases come from an automatic coreference pass that
// the README itself calls the riskiest decision the tier makes — a wrong merge
// invents relationships nothing downstream can detect. Every merge was printed
// at ingest and then gone with the scrollback.
//
// It also answers the question the relational tier otherwise cannot: when a
// question returns nothing, was it that the graph holds nothing relevant, or
// that the question named no entity the graph knows? Those have opposite fixes
// — extract more, or phrase differently — and until `graph seeds` they looked
// identical from the outside.
func runGraph(args []string) error {
	if len(args) > 0 {
		switch args[0] {
		case "entities":
			return graphEntities(args[1:])
		case "aliases":
			return graphAliases(args[1:])
		case "seeds":
			return graphSeeds(args[1:])
		case "show":
			return graphShow(args[1:])
		}
	}
	return graphStats(args)
}

// openGraph loads the graph beside an index. No LLM is needed to read one, and
// requiring an endpoint to inspect a file would make this useless in exactly
// the situation it is for: something is wrong and you are trying to find out
// what.
func openGraph(indexPath string, maxHops int) (*tiers.Relational, string, error) {
	path := artifactPath(indexPath, "graph")
	if !fileExists(path) {
		return nil, path, fmt.Errorf("no graph at %s\n(build one with `dispatch ingest --graph`)", path)
	}
	rel, err := tiers.LoadGraph(path, maxHops)
	if err != nil {
		return nil, path, err
	}
	return rel, path, nil
}

func graphStats(args []string) error {
	fs := flag.NewFlagSet("graph", flag.ExitOnError)
	indexPath := fs.String("index", defaultIndexPath, "index whose graph to inspect")
	if err := fs.Parse(args); err != nil {
		return err
	}
	rel, path, err := openGraph(*indexPath, 2)
	if err != nil {
		return err
	}

	entities, relations := rel.Stats()
	byType := map[string]int{}
	for _, e := range rel.EntityList() {
		t := e.Type
		if t == "" {
			t = "(untyped)"
		}
		byType[t]++
	}
	// An entity reached only as a relation endpoint carries no type, which is
	// worth surfacing: partial-name seeding is restricted to people, so an
	// untyped person is a person the seeder cannot reach by first name.
	types := make([]string, 0, len(byType))
	for t := range byType {
		types = append(types, t)
	}
	sort.Slice(types, func(i, j int) bool {
		if byType[types[i]] != byType[types[j]] {
			return byType[types[i]] > byType[types[j]]
		}
		return types[i] < types[j]
	})

	fmt.Printf("graph:      %s\n", path)
	fmt.Printf("entities:   %d\n", entities)
	for _, t := range types {
		fmt.Printf("  %-12s %d\n", t, byType[t])
	}
	fmt.Printf("relations:  %d\n", relations)
	fmt.Printf("aliases:    %d merged variant(s)\n", len(rel.Aliases()))

	// Isolated entities are the graph's dead weight: extracted, never connected,
	// so they can seed a traversal that then goes nowhere.
	var isolated int
	for _, e := range rel.EntityList() {
		if rel.Degree(e.Key) == 0 {
			isolated++
		}
	}
	if isolated > 0 {
		fmt.Printf("  %d entity(ies) have no edges: they can seed a traversal that reaches nothing\n", isolated)
	}

	gen := rel.SourceGeneration()
	switch {
	case gen == "":
		fmt.Println("generation: (none recorded — built before artifacts were stamped)")
	default:
		fmt.Printf("generation: %s\n", gen)
		if store, err := openIndex(*indexPath); err == nil {
			if cur := store.Generation(); cur != gen {
				fmt.Printf("            STALE — the index is now %s; re-run `ingest --graph`\n", cur)
			} else {
				fmt.Println("            current")
			}
		}
	}
	return nil
}

func graphEntities(args []string) error {
	fs := flag.NewFlagSet("graph entities", flag.ExitOnError)
	indexPath := fs.String("index", defaultIndexPath, "index whose graph to inspect")
	typ := fs.String("type", "", "only entities of this type (person, org, system, document, other)")
	prefix := fs.String("prefix", "", "only entities whose key starts with this")
	isolated := fs.Bool("isolated", false, "only entities with no edges")
	if err := fs.Parse(args); err != nil {
		return err
	}
	rel, _, err := openGraph(*indexPath, 2)
	if err != nil {
		return err
	}

	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "edges\ttype\tkey\tdisplay name")
	fmt.Fprintln(w, "-----\t----\t---\t------------")
	var shown int
	for _, e := range rel.EntityList() {
		deg := rel.Degree(e.Key)
		switch {
		case *typ != "" && !strings.EqualFold(e.Type, *typ):
			continue
		case *prefix != "" && !strings.HasPrefix(e.Key, *prefix):
			continue
		case *isolated && deg != 0:
			continue
		}
		t := e.Type
		if t == "" {
			t = "-"
		}
		fmt.Fprintf(w, "%d\t%s\t%s\t%s\n", deg, t, e.Key, e.Name)
		shown++
	}
	w.Flush()
	fmt.Printf("\n%d entity(ies)\n", shown)
	return nil
}

func graphAliases(args []string) error {
	fs := flag.NewFlagSet("graph aliases", flag.ExitOnError)
	indexPath := fs.String("index", defaultIndexPath, "index whose graph to inspect")
	if err := fs.Parse(args); err != nil {
		return err
	}
	rel, _, err := openGraph(*indexPath, 2)
	if err != nil {
		return err
	}

	aliases := rel.Aliases()
	if len(aliases) == 0 {
		fmt.Println("no merges: every entity kept its own name")
		fmt.Println("(zero merges from a model that rejected every candidate looks exactly like")
		fmt.Println(" zero merges from coreference never running — `ingest --graph` prints which)")
		return nil
	}

	variants := make([]string, 0, len(aliases))
	for v := range aliases {
		variants = append(variants, v)
	}
	sort.Strings(variants)

	fmt.Printf("%d merged variant(s). Each was judged to name the same thing as its canonical\n", len(aliases))
	fmt.Println("entity by a majority of three model votes. A wrong merge invents relationships,")
	fmt.Println("so this is the list worth reading if an answer connects two things it should not.")
	fmt.Println()
	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "variant\tmerged into\tedges on the canonical entity")
	fmt.Fprintln(w, "-------\t-----------\t----------------------------")
	for _, v := range variants {
		fmt.Fprintf(w, "%s\t%s\t%d\n", v, aliases[v], rel.Degree(aliases[v]))
	}
	w.Flush()
	fmt.Println("\nBoth names still reach the entity: a variant stays a valid way to name it,")
	fmt.Println("because dropping the commonest phrasing would make retrieval worse, not better.")
	return nil
}

// graphSeeds answers "why did the relational tier return nothing for this?"
func graphSeeds(args []string) error {
	fs := flag.NewFlagSet("graph seeds", flag.ExitOnError)
	indexPath := fs.String("index", defaultIndexPath, "index whose graph to inspect")
	maxHops := fs.Int("max-hops", 2, "how far to traverse from each seed")
	if err := fs.Parse(args); err != nil {
		return err
	}
	question := strings.Join(fs.Args(), " ")
	if question == "" {
		return fmt.Errorf("seeds what? give a question")
	}
	rel, _, err := openGraph(*indexPath, *maxHops)
	if err != nil {
		return err
	}

	seeds := rel.Seeds(question)
	fmt.Printf("question: %s\n\n", question)
	if len(seeds) == 0 {
		fmt.Println("no seeds: the question names no entity this graph knows.")
		fmt.Println("The tier returns nothing here, which is honest — it has no starting point,")
		fmt.Println("and traversing from a guess would invent connections.")
		fmt.Println("\nThat is a question-side failure, not a retrieval one. Either the entity was")
		fmt.Println("never extracted (`dispatch graph entities --prefix ...` to check), or it is")
		fmt.Println("written differently in the corpus than in the question.")
		return nil
	}

	fmt.Printf("seeds (%d):\n", len(seeds))
	for _, s := range seeds {
		fmt.Printf("  %-40s %d edge(s)\n", s, rel.Degree(s))
	}

	hits := rel.Traverse(seeds, *maxHops)
	if len(hits) == 0 {
		fmt.Println("\nthe seeds are isolated: no edges to traverse, so the tier still returns nothing")
		return nil
	}
	fmt.Printf("\nreached %d edge(s) within %d hop(s):\n", len(hits), *maxHops)
	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "hop\trelationship\tasserted by")
	fmt.Fprintln(w, "---\t------------\t-----------")
	for _, h := range hits {
		fmt.Fprintf(w, "%d\t%s\t%s\n", h.Hop, rel.Render(h.Edge), h.Edge.ChunkID)
	}
	w.Flush()
	return nil
}

func graphShow(args []string) error {
	fs := flag.NewFlagSet("graph show", flag.ExitOnError)
	indexPath := fs.String("index", defaultIndexPath, "index whose graph to inspect")
	excerpts := fs.Bool("excerpts", false, "print the chunk text that asserted each edge")
	if err := fs.Parse(args); err != nil {
		return err
	}
	want := strings.ToLower(strings.Join(fs.Args(), " "))
	if want == "" {
		return fmt.Errorf("show what? give an entity key (see `dispatch graph entities`)")
	}
	rel, _, err := openGraph(*indexPath, 2)
	if err != nil {
		return err
	}

	var found bool
	for _, e := range rel.EntityList() {
		if e.Key == want {
			found = true
			fmt.Printf("%s (%s), key %q\n\n", e.Name, orDash(e.Type), e.Key)
			break
		}
	}
	if !found {
		// An alias is a legitimate way to name an entity, so resolve rather than
		// reporting a miss.
		if canonical, ok := rel.Aliases()[want]; ok {
			fmt.Printf("%q was merged into %q; showing that entity.\n\n", want, canonical)
			want = canonical
			found = true
		}
	}
	if !found {
		var near []string
		for _, e := range rel.EntityList() {
			if strings.Contains(e.Key, want) {
				near = append(near, e.Key)
			}
		}
		if len(near) > 0 {
			return fmt.Errorf("no entity %q; did you mean one of %s", want, joinCapped(near, 5))
		}
		return fmt.Errorf("no entity %q in the graph", want)
	}

	var n int
	for _, e := range rel.Edges() {
		if e.From != want && e.To != want {
			continue
		}
		n++
		fmt.Printf("  %s   [%s]\n", rel.Render(e), e.ChunkID)
		if *excerpts {
			if ex := rel.Excerpt(e.ChunkID); ex != "" {
				fmt.Printf("      %s\n", truncateLine(strings.ReplaceAll(ex, "\n", " "), 160))
			}
		}
	}
	if n == 0 {
		fmt.Println("  (no edges: this entity was extracted but never connected to anything)")
	}
	return nil
}

func orDash(s string) string {
	if s == "" {
		return "untyped"
	}
	return s
}
