package index

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
)

// Cache is a content-addressed disk cache for contextual-chunk sentences. Each
// context sentence costs one LLM call to produce, so caching by content hash
// makes a re-ingest of unchanged documents free — which is the whole reason a
// hosted API is affordable here. Keys fold in a tag identifying the model, so
// swapping the context model invalidates stale entries rather than reusing them.
type Cache struct {
	dir string
}

// NewCache roots a cache at dir (e.g. .dispatch/cache). The directory is created
// lazily on the first Put, so constructing a cache is cheap and side-effect free.
func NewCache(dir string) *Cache { return &Cache{dir: dir} }

// Key derives the on-disk key for a chunk's context sentence. tag is the model
// identity; docID and chunkText together identify the chunk content.
func Key(tag, docID, chunkText string) string {
	h := sha256.New()
	h.Write([]byte(tag))
	h.Write([]byte{0})
	h.Write([]byte(docID))
	h.Write([]byte{0})
	h.Write([]byte(chunkText))
	return hex.EncodeToString(h.Sum(nil))
}

func (c *Cache) path(key string) string { return filepath.Join(c.dir, key+".txt") }

// Get returns the cached value and whether it was present. A cached empty string
// (a chunk the model declined to contextualize) is a valid hit.
func (c *Cache) Get(key string) (string, bool) {
	if c == nil {
		return "", false
	}
	data, err := os.ReadFile(c.path(key))
	if err != nil {
		return "", false
	}
	return string(data), true
}

// Has reports whether a key is cached without reading its value — used by the
// dry-run planner to estimate how many LLM calls an ingest will actually make.
func (c *Cache) Has(key string) bool {
	if c == nil {
		return false
	}
	_, err := os.Stat(c.path(key))
	return err == nil
}

// Put writes a value, creating the cache directory on first use.
func (c *Cache) Put(key, val string) error {
	if c == nil {
		return nil
	}
	if err := os.MkdirAll(c.dir, 0o755); err != nil {
		return err
	}
	return os.WriteFile(c.path(key), []byte(val), 0o644)
}
