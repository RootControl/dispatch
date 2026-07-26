package main

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/RootControl/dispatch/core"
)

// write creates path with content, making parents as needed.
func write(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func docIDs(docs []core.Doc) []string {
	out := make([]string, len(docs))
	for i, d := range docs {
		out[i] = d.ID
	}
	return out
}

// Pointing --corpus at a real repository must not ingest its dependencies. A
// checkout here held 13 documents worth reading and 4,602 markdown files total.
func TestLoadCorpusSkipsVendoredDirectories(t *testing.T) {
	root := t.TempDir()
	write(t, filepath.Join(root, "README.md"), "real doc")
	write(t, filepath.Join(root, "docs", "guide.md"), "real doc")
	write(t, filepath.Join(root, "notes.txt"), "real doc")
	write(t, filepath.Join(root, "node_modules", "left-pad", "README.md"), "vendored")
	write(t, filepath.Join(root, ".git", "COMMIT_EDITMSG.md"), "git internals")
	write(t, filepath.Join(root, "dist", "bundle.md"), "build output")
	write(t, filepath.Join(root, "web", "vendor", "lib.md"), "vendored, nested")

	docs, err := loadCorpus(root, nil)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"README.md", filepath.Join("docs", "guide.md"), "notes.txt"}
	gotIDs := docIDs(docs)
	slices.Sort(gotIDs)
	slices.Sort(want)
	if !slices.Equal(gotIDs, want) {
		t.Errorf("loaded %v, want %v", gotIDs, want)
	}
}

func TestLoadCorpusExtraExclusions(t *testing.T) {
	root := t.TempDir()
	write(t, filepath.Join(root, "keep.md"), "x")
	write(t, filepath.Join(root, "drafts", "wip.md"), "x")

	docs, err := loadCorpus(root, []string{"drafts"})
	if err != nil {
		t.Fatal(err)
	}
	if len(docs) != 1 || docs[0].ID != "keep.md" {
		t.Errorf("extra exclusion ignored: %v", docs)
	}

	// Blank entries from strings.Split on an empty flag must not exclude
	// everything or match a directory named "".
	all, err := loadCorpus(root, []string{"", "  "})
	if err != nil {
		t.Fatal(err)
	}
	if len(all) != 2 {
		t.Errorf("blank exclusions changed the result: %v", all)
	}
}

// A corpus root that happens to be named like a skipped directory must still
// load — the skip applies to descendants, not the root itself.
func TestLoadCorpusDoesNotSkipItsOwnRoot(t *testing.T) {
	root := filepath.Join(t.TempDir(), "dist")
	write(t, filepath.Join(root, "doc.md"), "x")

	docs, err := loadCorpus(root, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(docs) != 1 {
		t.Fatalf("root named 'dist' was skipped: %v", docs)
	}
}

func TestLoadCorpusIgnoresUnsupportedExtensions(t *testing.T) {
	root := t.TempDir()
	write(t, filepath.Join(root, "doc.md"), "x")
	write(t, filepath.Join(root, "image.png"), "x")
	write(t, filepath.Join(root, "paper.pdf"), "x")

	docs, err := loadCorpus(root, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(docs) != 1 || docs[0].ID != "doc.md" {
		t.Errorf("unsupported extensions were not filtered: %v", docIDs(docs))
	}
}
