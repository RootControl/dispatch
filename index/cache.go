package index

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"

	"github.com/RootControl/dispatch/internal/atomicfile"
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

// VecKey derives the on-disk key for a chunk's embedding. tag is the embedding
// model identity and text is the text actually embedded — the situating context
// sentence included, since that is what the vector is of. No document ID: an
// identical passage under two document IDs embeds identically, so keying on
// content alone shares the entry rather than paying twice.
func VecKey(tag, text string) string {
	h := sha256.New()
	h.Write([]byte(tag))
	h.Write([]byte{0})
	h.Write([]byte(text))
	return hex.EncodeToString(h.Sum(nil))
}

func (c *Cache) path(key string) string { return filepath.Join(c.dir, key+".txt") }

func (c *Cache) vecPath(key string) string { return filepath.Join(c.dir, key+".vec") }

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
//
// The write is atomic because a torn text entry cannot be detected downstream.
// os.WriteFile opens with O_TRUNC and then writes, so a crash or a concurrent
// writer leaves a prefix — and half a context sentence is a perfectly valid
// string, so it is returned as a hit, embedded into the chunk, and indexed. An
// empty read is worse still: Get documents the empty string as a legitimate hit
// (a chunk the model declined to contextualize), so the truncation window and a
// real decision are indistinguishable. Neither is recoverable by a later
// ingest, since the cache is precisely what a re-ingest trusts in place of the
// model.
func (c *Cache) Put(key, val string) error {
	if c == nil {
		return nil
	}
	return atomicfile.Write(c.path(key), []byte(val), 0o644)
}

// GetVector returns a cached embedding. A malformed or empty entry is reported
// as a miss rather than an error: the cost of a miss is one embedding call,
// while returning a truncated vector would silently corrupt every ranking it
// takes part in.
func (c *Cache) GetVector(key string) ([]float64, bool) {
	if c == nil {
		return nil, false
	}
	data, err := os.ReadFile(c.vecPath(key))
	if err != nil {
		return nil, false
	}
	var v []float64
	if err := json.Unmarshal(data, &v); err != nil || len(v) == 0 {
		return nil, false
	}
	return v, true
}

// HasVector reports whether an embedding is cached without decoding it, for the
// dry-run planner.
func (c *Cache) HasVector(key string) bool {
	if c == nil {
		return false
	}
	_, err := os.Stat(c.vecPath(key))
	return err == nil
}

// PutVector caches an embedding.
//
// Vectors are stored as JSON float64 rather than anything more compact on
// purpose: a narrower encoding would make a cached run rank fractionally
// differently from an uncached one, and reproducible eval numbers are worth
// more here than disk. Budget roughly 15 KB per chunk at 768 dimensions.
// A truncated vector is already caught by GetVector's decode, so this is the
// cheaper half of the problem — but it is written atomically too, because
// "reported as a miss" still costs an embedding call for every entry a crash
// happened to be holding open, and the fix is the same one line.
func (c *Cache) PutVector(key string, v []float64) error {
	if c == nil {
		return nil
	}
	data, err := json.Marshal(v)
	if err != nil {
		return err
	}
	return atomicfile.Write(c.vecPath(key), data, 0o644)
}
