package atomicfile

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestWriteCreatesFileAndDirs(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nested", "deeper", "index.json")
	if err := Write(path, []byte(`{"ok":true}`), 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != `{"ok":true}` {
		t.Errorf("content = %q", got)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode().Perm() != 0o644 {
		t.Errorf("mode = %v, want 0644", info.Mode().Perm())
	}
}

func TestWriteReplacesExistingContent(t *testing.T) {
	path := filepath.Join(t.TempDir(), "index.json")
	if err := Write(path, []byte("old and rather longer"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := Write(path, []byte("new"), 0o644); err != nil {
		t.Fatal(err)
	}
	got, _ := os.ReadFile(path)
	// Not "newand rather longer": a rename replaces, it does not overwrite in place.
	if string(got) != "new" {
		t.Errorf("content = %q, want the file fully replaced", got)
	}
}

// The whole point. A write that fails part-way must leave the previous file
// exactly as it was — os.Create truncates first, so the old index would already
// be gone, and since ingest became incremental that index is the only record of
// which documents have been embedded.
func TestFailedWriteLeavesTheOldFileIntact(t *testing.T) {
	path := filepath.Join(t.TempDir(), "index.json")
	const good = `{"chunks":["a","b","c"],"embed_model":"nomic"}`
	if err := Write(path, []byte(good), 0o644); err != nil {
		t.Fatal(err)
	}

	boom := errors.New("disk full")
	err := WriteFunc(path, 0o644, func(f *os.File) error {
		// Write a partial document, then fail — exactly the shape of a crash
		// or a full disk mid-encode.
		if _, werr := f.Write([]byte(`{"chunks":["a`)); werr != nil {
			return werr
		}
		return boom
	})
	if !errors.Is(err, boom) {
		t.Fatalf("err = %v, want the write error surfaced", err)
	}

	got, readErr := os.ReadFile(path)
	if readErr != nil {
		t.Fatalf("the original file is gone after a failed write: %v", readErr)
	}
	if string(got) != good {
		t.Errorf("content = %q, want the previous good copy untouched", got)
	}
}

// A failure must not leave temporary files behind, or a directory accumulates
// one per crash.
func TestFailedWriteCleansUpItsTempFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "index.json")

	err := WriteFunc(path, 0o644, func(*os.File) error { return errors.New("nope") })
	if err == nil {
		t.Fatal("expected an error")
	}
	entries, _ := os.ReadDir(dir)
	for _, e := range entries {
		if strings.Contains(e.Name(), "tmp-") {
			t.Errorf("left a temporary file behind: %s", e.Name())
		}
	}
}

func TestSuccessfulWriteLeavesNoTempFile(t *testing.T) {
	dir := t.TempDir()
	if err := Write(filepath.Join(dir, "index.json"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	entries, _ := os.ReadDir(dir)
	if len(entries) != 1 || entries[0].Name() != "index.json" {
		var names []string
		for _, e := range entries {
			names = append(names, e.Name())
		}
		t.Errorf("directory holds %v, want just the target", names)
	}
}

// The temporary file must live in the target's directory. Rename is only atomic
// within a filesystem, and TMPDIR is very often a different one — where this
// would silently degrade to a copy that can tear.
func TestTempFileIsCreatedBesideTheTarget(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "index.json")

	var tmpDir string
	err := WriteFunc(path, 0o644, func(f *os.File) error {
		tmpDir = filepath.Dir(f.Name())
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if tmpDir != dir {
		t.Errorf("temp file was created in %s, want %s (the target's own directory)", tmpDir, dir)
	}
}
