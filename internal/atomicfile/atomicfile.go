// Package atomicfile writes a file so that a reader never sees a partial one.
//
// The failure it prevents is specific and expensive. Writing an index with
// os.Create truncates the existing file before the first byte of the new one is
// written, so a crash, a kill, or a full disk part-way through leaves truncated
// JSON *and* the previous good copy already gone. Since ingest became
// incremental, that index is the only record of which documents have been
// embedded — losing it means paying to embed the whole corpus again.
//
// Write instead goes to a temporary file in the same directory, is flushed to
// disk, and is renamed over the target. Rename within a directory is atomic on
// POSIX and on Windows via MoveFileEx, so a reader sees either the old file or
// the new one, never a half-written one.
package atomicfile

import (
	"fmt"
	"os"
	"path/filepath"
)

// Write saves data to path atomically, creating parent directories as needed.
func Write(path string, data []byte, perm os.FileMode) error {
	return WriteFunc(path, perm, func(f *os.File) error {
		_, err := f.Write(data)
		return err
	})
}

// WriteFunc is Write for callers that stream their output — a json.Encoder,
// say — rather than holding it all in memory first.
//
// The temporary file is created in the target's own directory, not TMPDIR:
// rename is only atomic within a filesystem, and /tmp is very often a different
// one, where this would silently degrade to a copy that can tear.
func WriteFunc(path string, perm os.FileMode, write func(*os.File) error) error {
	dir := filepath.Dir(path)
	if dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}

	tmp, err := os.CreateTemp(dir, "."+filepath.Base(path)+".tmp-*")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	// Remove the temporary file on any failure. After a successful rename it no
	// longer exists, so the failed Remove there is expected and ignored.
	defer func() {
		tmp.Close()
		os.Remove(tmpName)
	}()

	if err := write(tmp); err != nil {
		return err
	}
	// Flush the contents before the rename. Without this the rename can reach
	// disk first and a crash leaves an entry pointing at an empty file — the
	// same data loss, arrived at the long way round.
	if err := tmp.Sync(); err != nil {
		return fmt.Errorf("atomicfile: sync %s: %w", tmpName, err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("atomicfile: close %s: %w", tmpName, err)
	}
	if err := os.Chmod(tmpName, perm); err != nil {
		return err
	}
	if err := os.Rename(tmpName, path); err != nil {
		return fmt.Errorf("atomicfile: rename %s -> %s: %w", tmpName, path, err)
	}
	return nil
}
