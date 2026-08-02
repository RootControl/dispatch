package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"slices"
	"sort"

	"github.com/RootControl/dispatch/internal/atomicfile"
)

// retrievalRun is one `eval retrieval` run, saved so a later run can be
// compared against it.
//
// Every reranker and expansion comparison in this project's history was done by
// reading two recall figures and diffing two miss lists by eye. That hides the
// thing that actually matters: an aggregate can hold still while half the
// corpus moves underneath it, and a change that gains thirteen chunks and loses
// two looks identical to one that gains one and loses nothing if you only read
// recall@1. Per-chunk ranks make the trade visible.
type retrievalRun struct {
	Label string `json:"label"`
	TopK  int    `json:"k"`
	// Ranks maps chunk ID to the 0-based rank its own question retrieved it at,
	// or -1 for "not in the top k". Chunks that could not be probed are absent
	// rather than recorded as misses.
	Ranks map[string]int `json:"ranks"`
	// Texts maps chunk ID to a hash of its content.
	//
	// Without this the comparison is unsound across a re-chunking, and silently
	// so. Chunk IDs are "doc#position", so changing the splitter keeps every ID
	// and changes what is behind it: atlas-charter.md#0 exists in both runs and
	// is different text in each. Comparing their ranks produces confident
	// per-chunk deltas about two things that were never the same chunk — which
	// is exactly what this tool did on its first real use.
	Texts map[string]string `json:"texts,omitempty"`
}

func (r retrievalRun) recallAt(n int) (hits, total int) {
	for _, rank := range r.Ranks {
		total++
		if rank >= 0 && rank < n {
			hits++
		}
	}
	return hits, total
}

func saveRun(path string, run retrievalRun) error {
	data, err := json.MarshalIndent(run, "", "  ")
	if err != nil {
		return err
	}
	return atomicfile.Write(path, append(data, '\n'), 0o644)
}

func loadRun(path string) (retrievalRun, error) {
	var run retrievalRun
	data, err := os.ReadFile(path)
	if err != nil {
		return run, err
	}
	if err := json.Unmarshal(data, &run); err != nil {
		return run, fmt.Errorf("parse %s: %w", path, err)
	}
	if len(run.Ranks) == 0 {
		return run, fmt.Errorf("%s records no per-chunk ranks", path)
	}
	return run, nil
}

// change is one chunk whose rank moved between two runs.
type change struct {
	chunk    string
	from, to int
}

// compareRuns prints a per-chunk diff of two runs.
//
// Chunks present in only one run are reported separately rather than counted as
// changes: they usually mean the corpus or the question cache moved between the
// runs, which makes the aggregate comparison unsound, and saying so is more
// useful than folding them into a delta.
func compareRuns(base, now retrievalRun) {
	var improved, regressed []change
	var onlyBase, onlyNow, rechunked []string

	for id, to := range now.Ranks {
		from, ok := base.Ranks[id]
		if !ok {
			onlyNow = append(onlyNow, id)
			continue
		}
		// Same ID, different text: the splitter changed under it, so these are
		// two different chunks wearing one name and their ranks are not
		// comparable. Counting them as improved or regressed would be a
		// confident statement about nothing.
		if bh, nh := base.Texts[id], now.Texts[id]; bh != "" && nh != "" && bh != nh {
			rechunked = append(rechunked, id)
			continue
		}
		if from == to {
			continue
		}
		// -1 means "not retrieved", which sorts worst rather than best.
		if rankBetter(to, from) {
			improved = append(improved, change{id, from, to})
		} else {
			regressed = append(regressed, change{id, from, to})
		}
	}
	for id := range base.Ranks {
		if _, ok := now.Ranks[id]; !ok {
			onlyBase = append(onlyBase, id)
		}
	}

	baseLabel := base.Label
	if baseLabel == "" {
		baseLabel = "baseline"
	}
	fmt.Printf("\n=== vs %s ===\n", baseLabel)
	if base.TopK != now.TopK {
		fmt.Printf("WARNING: baseline used k=%d, this run k=%d; the recall figures are not comparable\n",
			base.TopK, now.TopK)
	}

	for _, at := range []int{1, now.TopK} {
		bh, bt := base.recallAt(at)
		nh, nt := now.recallAt(at)
		fmt.Printf("recall@%-2d %d/%d (%s) -> %d/%d (%s)   %s\n", at,
			bh, bt, pct(bh, bt), nh, nt, pct(nh, nt), delta(bh, nh))
	}

	fmt.Printf("\nchunks improved: %d, regressed: %d, unchanged: %d\n",
		len(improved), len(regressed),
		len(now.Ranks)-len(improved)-len(regressed)-len(onlyNow)-len(rechunked))

	printChanges("improved", improved)
	printChanges("regressed", regressed)

	// This is the loudest warning here because it is the one that looks like a
	// result. The others produce visibly odd output; this one produces a clean
	// table of deltas that mean nothing.
	if len(rechunked) > 0 {
		sort.Strings(rechunked)
		fmt.Printf("\nWARNING: %d chunk ID(s) hold different text in the two runs, so their\n", len(rechunked))
		fmt.Println("ranks are not comparable and they are excluded from the counts above.")
		fmt.Println("This is what a change to chunking looks like: the IDs are positional, so")
		fmt.Println("the splitter moved and every per-chunk delta across it would be spurious.")
		fmt.Println("Compare chunking changes on the aggregate only, and read even that with")
		fmt.Println("care — more, smaller chunks are easier to retrieve for their own question.")
		fmt.Printf("  %s\n", joinCapped(rechunked, 5))
	}

	// The aggregate is only meaningful over the same chunks.
	if len(onlyBase) > 0 || len(onlyNow) > 0 {
		fmt.Printf("\nWARNING: the two runs do not cover the same chunks — %d only in the baseline, %d only here.\n",
			len(onlyBase), len(onlyNow))
		fmt.Println("The totals above are over different denominators; re-run the baseline to compare cleanly.")
		for _, s := range [][2]any{{"only in baseline", onlyBase}, {"only in this run", onlyNow}} {
			ids := s[1].([]string)
			if len(ids) == 0 {
				continue
			}
			sort.Strings(ids)
			fmt.Printf("  %s: %s\n", s[0], joinCapped(ids, 5))
		}
	}
}

// rankBetter reports whether rank a is better than b, treating -1 (not
// retrieved) as worse than any real position.
func rankBetter(a, b int) bool {
	switch {
	case a < 0:
		return false
	case b < 0:
		return true
	}
	return a < b
}

func printChanges(label string, cs []change) {
	if len(cs) == 0 {
		return
	}
	slices.SortFunc(cs, func(x, y change) int {
		if x.chunk != y.chunk {
			if x.chunk < y.chunk {
				return -1
			}
			return 1
		}
		return 0
	})
	fmt.Printf("\n%s:\n", label)
	for i, c := range cs {
		if i == 12 {
			fmt.Printf("  ... and %d more\n", len(cs)-12)
			break
		}
		fmt.Printf("  %-52s %s -> %s\n", c.chunk, rankLabel(c.from), rankLabel(c.to))
	}
}

func rankLabel(r int) string {
	if r < 0 {
		return "miss"
	}
	return fmt.Sprintf("#%d", r+1)
}

func pct(n, total int) string {
	if total == 0 {
		return "n/a"
	}
	return fmt.Sprintf("%d%%", n*100/total)
}

func delta(from, to int) string {
	switch {
	case to > from:
		return fmt.Sprintf("+%d", to-from)
	case to < from:
		return fmt.Sprintf("%d", to-from)
	}
	return "="
}

func joinCapped(ids []string, n int) string {
	if len(ids) <= n {
		return fmt.Sprint(ids)
	}
	return fmt.Sprintf("%v ... and %d more", ids[:n], len(ids)-n)
}

// chunkDigest fingerprints a chunk's content, so a comparison can tell a chunk
// that moved from a chunk that was re-split under the same positional ID.
func chunkDigest(text string) string {
	sum := sha256.Sum256([]byte(text))
	return hex.EncodeToString(sum[:8])
}
