package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/RootControl/dispatch/index"
)

// runCache reports on and prunes the content-addressed cache.
//
// The cache is keyed by content hash and nothing ever removed an entry, so
// every re-chunking, model swap and edited document left its entries behind
// permanently. The embedding cache makes that expensive rather than untidy: a
// 768-dimension vector is roughly 15 KB of JSON, so a corpus re-chunked twice
// carries three full sets of vectors and uses none of two of them.
func runCache(args []string) error {
	if len(args) > 0 && args[0] == "gc" {
		return cacheGC(args[1:])
	}
	return cacheStats(args)
}

func cacheStats(args []string) error {
	fs := flag.NewFlagSet("cache", flag.ExitOnError)
	dir := fs.String("dir", defaultCacheDir, "cache directory")
	if err := fs.Parse(args); err != nil {
		return err
	}
	byKind, err := scanCache(*dir)
	if err != nil {
		return err
	}
	fmt.Printf("cache: %s\n", *dir)
	var total int64
	var files int
	for _, kind := range []string{".txt", ".vec"} {
		e := byKind[kind]
		total += e.bytes
		files += e.count
		fmt.Printf("  %-5s %5d entries  %10s   %s\n", kind, e.count, humanBytes(e.bytes), kindLabel(kind))
	}
	fmt.Printf("  total %5d entries  %10s\n", files, humanBytes(total))
	fmt.Println("\nRun `dispatch cache gc --index PATH` to drop what no index references.")
	return nil
}

func kindLabel(ext string) string {
	if ext == ".vec" {
		return "chunk and summary embeddings"
	}
	return "context sentences, extractions, summaries"
}

type cacheEntry struct {
	count int
	bytes int64
}

func scanCache(dir string) (map[string]cacheEntry, error) {
	out := map[string]cacheEntry{".txt": {}, ".vec": {}}
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return out, nil
		}
		return nil, err
	}
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		ext := filepath.Ext(e.Name())
		info, err := e.Info()
		if err != nil {
			continue
		}
		c := out[ext]
		c.count++
		c.bytes += info.Size()
		out[ext] = c
	}
	return out, nil
}

// cacheGC removes cache entries that no named index can use.
//
// Only vector entries are collected. Their keys are derivable — a vector is
// keyed by the embedding-model tag and the exact text embedded, both of which a
// loaded index holds — so "is this entry reachable" has an exact answer. The
// text entries are keyed by a model tag the index does not record, so deciding
// they are dead would be a guess, and guessing wrong means paying an LLM to
// regenerate something that was already correct. Sweeping the expensive half
// exactly is worth more than sweeping both approximately.
func cacheGC(args []string) error {
	fs := flag.NewFlagSet("cache gc", flag.ExitOnError)
	dir := fs.String("dir", defaultCacheDir, "cache directory")
	var indexes multiFlag
	fs.Var(&indexes, "index", "an index whose entries must be kept (repeatable)")
	dryRun := fs.Bool("dry-run", false, "report what would be removed, then stop")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if len(indexes) == 0 {
		indexes = multiFlag{defaultIndexPath}
	}

	// Everything reachable from every named index. Missing an index here would
	// delete live entries, so an unreadable one is an error rather than a skip.
	live := map[string]bool{}
	for _, path := range indexes {
		store, err := openIndex(path)
		if err != nil {
			return fmt.Errorf("%w\n(every index that shares this cache must be listed, or gc would drop entries it still needs)", err)
		}
		embedTag := store.EmbedTag()
		chunks, _ := store.Entries()
		for _, c := range chunks {
			live[index.VecKey(embedTag, c.Embedded())] = true
		}
		// The hierarchy embeds its summaries into the same cache.
		hp := artifactPath(path, "hierarchy")
		if fileExists(hp) {
			h, _, err := loadHierarchyChunks(hp)
			if err != nil {
				return fmt.Errorf("read %s: %w", hp, err)
			}
			for _, text := range h {
				live[index.VecKey(embedTag, text)] = true
			}
		}
		fmt.Printf("keeping entries for %s (%d chunks)\n", path, len(chunks))
	}

	entries, err := os.ReadDir(*dir)
	if err != nil {
		if os.IsNotExist(err) {
			fmt.Printf("cache %s does not exist; nothing to do\n", *dir)
			return nil
		}
		return err
	}

	var removed int
	var freed int64
	for _, e := range entries {
		if e.IsDir() || filepath.Ext(e.Name()) != ".vec" {
			continue
		}
		key := strings.TrimSuffix(e.Name(), ".vec")
		if live[key] {
			continue
		}
		info, err := e.Info()
		if err != nil {
			continue
		}
		removed++
		freed += info.Size()
		if *dryRun {
			continue
		}
		if err := os.Remove(filepath.Join(*dir, e.Name())); err != nil {
			return err
		}
	}

	verb := "removed"
	if *dryRun {
		verb = "would remove"
	}
	fmt.Printf("%s %d unreferenced vector entries, freeing %s\n", verb, removed, humanBytes(freed))
	if removed > 0 {
		// Naming too few indexes deletes live entries, and nothing here can
		// detect that. It is survivable — a dropped vector costs one embedding
		// call to rebuild, not an answer — but worth saying rather than leaving
		// someone to discover it as an unexplained bill.
		fmt.Println("(if an index that shares this cache was not listed, its entries went too;\n" +
			"that costs one embedding call each to rebuild on the next ingest, nothing more)")
	}
	if removed == 0 {
		fmt.Println("(text entries are never collected: their keys fold in a model tag the index does not record,\nso deciding they are dead would be a guess, and a wrong guess costs an LLM call to undo)")
	}
	return nil
}

// multiFlag collects a repeatable string flag.
type multiFlag []string

func (m *multiFlag) String() string     { return strings.Join(*m, ",") }
func (m *multiFlag) Set(s string) error { *m = append(*m, s); return nil }

// loadHierarchyChunks reads a saved tree's summary texts without needing an
// endpoint, so gc works with no LLM configured.
func loadHierarchyChunks(path string) ([]string, string, error) {
	store := index.New(index.Config{})
	if err := store.Load(path); err != nil {
		return nil, "", err
	}
	chunks, _ := store.Entries()
	out := make([]string, 0, len(chunks))
	for _, c := range chunks {
		out = append(out, c.Embedded())
	}
	return out, store.SourceGeneration(), nil
}
