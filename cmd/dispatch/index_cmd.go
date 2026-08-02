package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"slices"
	"sort"
	"strings"
	"text/tabwriter"

	"github.com/RootControl/dispatch/core"
	"github.com/RootControl/dispatch/index"
)

// runIndex inspects an index without answering anything.
//
// Until this existed, "did my document get ingested, and into how many chunks"
// had no answer short of asking a question and reading the citations — which
// conflates a chunking problem with a retrieval one. It is also what makes the
// stale-artifact and cache-growth problems diagnosable rather than theoretical.
func runIndex(args []string) error {
	if len(args) > 0 && args[0] == "docs" {
		return indexDocs(args[1:])
	}
	if len(args) > 0 && args[0] == "show" {
		return indexShow(args[1:])
	}
	return indexStats(args)
}

// openIndex loads an index for inspection. Contextualisation and chunk options
// are irrelevant to reading one, so they are left at zero.
func openIndex(path string) (*index.Store, error) {
	store, _, err := newStore(false, 0, 0, false)
	if err != nil {
		return nil, err
	}
	if err := store.Load(path); err != nil {
		return nil, fmt.Errorf("load index: %w", err)
	}
	return store, nil
}

func indexStats(args []string) error {
	fs := flag.NewFlagSet("index", flag.ExitOnError)
	indexPath := fs.String("index", defaultIndexPath, "index to inspect")
	if err := fs.Parse(args); err != nil {
		return err
	}
	store, err := openIndex(*indexPath)
	if err != nil {
		return err
	}

	chunks, vectors := store.Entries()
	docs := store.DocIDs()

	var totalChars, withContext int
	dims := 0
	for i, c := range chunks {
		totalChars += len(c.Text)
		if c.Context != "" {
			withContext++
		}
		if i == 0 && len(vectors) > 0 {
			dims = len(vectors[0])
		}
	}

	fmt.Printf("index:      %s\n", *indexPath)
	fmt.Printf("generation: %s\n", store.Generation())
	fmt.Printf("documents:  %d\n", len(docs))
	fmt.Printf("chunks:     %d", len(chunks))
	if len(docs) > 0 {
		fmt.Printf("  (%.1f per document)", float64(len(chunks))/float64(len(docs)))
	}
	fmt.Println()
	if len(chunks) > 0 {
		fmt.Printf("chunk size: %d chars mean\n", totalChars/len(chunks))
		// Contextual chunking is the single largest quality lever and the
		// easiest to have silently skipped with --no-context, so say plainly
		// whether it actually ran.
		fmt.Printf("contextual: %d of %d chunks carry a situating sentence\n", withContext, len(chunks))
	}
	fmt.Printf("dimensions: %d\n", dims)

	// Derived artifacts, and whether they still describe this index — the check
	// that turns a silent staleness bug into a visible one.
	gen := store.Generation()
	for _, a := range []struct{ kind, path string }{
		{"hierarchy", artifactPath(*indexPath, "hierarchy")},
		{"graph", artifactPath(*indexPath, "graph")},
	} {
		if !fileExists(a.path) {
			fmt.Printf("%-11s (none)\n", a.kind+":")
			continue
		}
		state := artifactState(a.kind, a.path, gen)
		fmt.Printf("%-11s %s — %s\n", a.kind+":", a.path, state)
	}
	return nil
}

// artifactState reports whether an artifact still matches the index.
func artifactState(kind, path, indexGen string) string {
	var artifactGen string
	switch kind {
	case "hierarchy":
		h, err := loadHierarchyQuiet(path)
		if err != nil {
			return "UNREADABLE: " + err.Error()
		}
		artifactGen = h
	case "graph":
		g, err := loadGraphGeneration(path)
		if err != nil {
			return "UNREADABLE: " + err.Error()
		}
		artifactGen = g
	}
	switch {
	case artifactGen == "":
		return "no generation stamp (built before they were recorded)"
	case artifactGen == indexGen:
		return "current"
	}
	return fmt.Sprintf("STALE — built from %s, index is %s; re-run `ingest --%s`", artifactGen, indexGen, kind)
}

func indexDocs(args []string) error {
	fs := flag.NewFlagSet("index docs", flag.ExitOnError)
	indexPath := fs.String("index", defaultIndexPath, "index to inspect")
	filterPrefix := fs.String("prefix", "", "only documents whose ID starts with this")
	if err := fs.Parse(args); err != nil {
		return err
	}
	store, err := openIndex(*indexPath)
	if err != nil {
		return err
	}

	chunks, _ := store.Entries()
	perDoc := map[string]int{}
	chars := map[string]int{}
	for _, c := range chunks {
		perDoc[c.DocID]++
		chars[c.DocID] += len(c.Text)
	}

	ids := make([]string, 0, len(perDoc))
	for id := range perDoc {
		if strings.HasPrefix(id, *filterPrefix) {
			ids = append(ids, id)
		}
	}
	sort.Strings(ids)
	if len(ids) == 0 {
		return fmt.Errorf("no documents in %s matching %q", *indexPath, *filterPrefix)
	}

	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "chunks\tchars\tdocument")
	fmt.Fprintln(w, "------\t-----\t--------")
	for _, id := range ids {
		fmt.Fprintf(w, "%d\t%d\t%s\n", perDoc[id], chars[id], id)
	}
	w.Flush()
	fmt.Printf("\n%d document(s), %d chunk(s)\n", len(ids), len(chunks))
	return nil
}

func indexShow(args []string) error {
	fs := flag.NewFlagSet("index show", flag.ExitOnError)
	indexPath := fs.String("index", defaultIndexPath, "index to inspect")
	full := fs.Bool("full", false, "print whole chunks rather than the first line")
	if err := fs.Parse(args); err != nil {
		return err
	}
	want := strings.Join(fs.Args(), " ")
	if want == "" {
		return fmt.Errorf("show what? give a chunk ID or a document ID")
	}
	store, err := openIndex(*indexPath)
	if err != nil {
		return err
	}

	chunks, _ := store.Entries()
	var matched []core.Chunk
	for _, c := range chunks {
		if c.ID == want || c.DocID == want {
			matched = append(matched, c)
		}
	}
	if len(matched) == 0 {
		// A near-miss is far more useful than "not found": the usual cause is a
		// path prefix that does not match how the corpus was loaded.
		var near []string
		for _, c := range chunks {
			if strings.Contains(c.DocID, want) {
				near = append(near, c.DocID)
			}
		}
		slices.Sort(near)
		near = slices.Compact(near)
		if len(near) > 0 {
			return fmt.Errorf("no chunk or document %q; did you mean one of %s", want, joinCapped(near, 5))
		}
		return fmt.Errorf("no chunk or document %q in %s", want, *indexPath)
	}

	slices.SortFunc(matched, func(a, b core.Chunk) int { return a.Position - b.Position })
	for _, c := range matched {
		fmt.Printf("--- %s (doc %s, position %d)\n", c.ID, c.DocID, c.Position)
		if len(c.Meta) > 0 {
			keys := make([]string, 0, len(c.Meta))
			for k := range c.Meta {
				keys = append(keys, k)
			}
			sort.Strings(keys)
			var pairs []string
			for _, k := range keys {
				pairs = append(pairs, k+"="+c.Meta[k])
			}
			fmt.Printf("    meta: %s\n", strings.Join(pairs, " "))
		}
		if c.Context != "" {
			fmt.Printf("    context: %s\n", c.Context)
		}
		body := c.Text
		if !*full {
			if i := strings.IndexByte(body, '\n'); i > 0 {
				body = body[:i] + " ..."
			}
			body = truncateLine(body, 200)
		}
		fmt.Printf("%s\n\n", body)
	}
	return nil
}

func truncateLine(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

// loadHierarchyQuiet returns a tree's source generation without needing an
// endpoint: inspection must work with no LLM configured.
func loadHierarchyQuiet(path string) (string, error) {
	var snap struct {
		SourceGeneration string `json:"source_generation"`
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	if err := json.Unmarshal(data, &snap); err != nil {
		return "", err
	}
	return snap.SourceGeneration, nil
}

// loadGraphGeneration reads only the stamp, for the same reason.
func loadGraphGeneration(path string) (string, error) {
	var g struct {
		SourceGeneration string `json:"source_generation"`
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	if err := json.Unmarshal(data, &g); err != nil {
		return "", err
	}
	return g.SourceGeneration, nil
}
