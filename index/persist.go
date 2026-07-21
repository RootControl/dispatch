package index

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/RootControl/dispatch/core"
)

// snapshot is the on-disk index format. Only chunks and vectors are stored; the
// BM25 index is rebuilt on load, since it is derived from the chunk text and
// storing it would just be a second source of truth to keep in sync.
type snapshot struct {
	EmbedModel string       `json:"embed_model"`
	Chunks     []core.Chunk `json:"chunks"`
	Vectors    [][]float64  `json:"vectors"`
}

// Save writes the index to path, creating parent directories as needed. Chunks
// and vectors are written in vector-index order so the two stay aligned.
func (s *Store) Save(path string) error {
	s.mu.RLock()
	defer s.mu.RUnlock()

	snap := snapshot{
		EmbedModel: s.opts.EmbedTag,
		Chunks:     make([]core.Chunk, 0, len(s.vec.ids)),
		Vectors:    make([][]float64, 0, len(s.vec.ids)),
	}
	for i, id := range s.vec.ids {
		snap.Chunks = append(snap.Chunks, s.chunks[id])
		snap.Vectors = append(snap.Vectors, s.vec.vecs[i])
	}

	if dir := filepath.Dir(path); dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	return enc.Encode(snap)
}

// Load replaces the store's contents with the index at path and rebuilds the
// BM25 side.
//
// It refuses to load an index built with a different embedding model: mixing
// embedding spaces silently returns plausible-looking nonsense rather than
// failing, so this is one of the few cases worth a hard error over a warning.
func (s *Store) Load(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var snap snapshot
	if err := json.Unmarshal(data, &snap); err != nil {
		return fmt.Errorf("index: parse %s: %w", path, err)
	}
	if len(snap.Chunks) != len(snap.Vectors) {
		return fmt.Errorf("index: corrupt snapshot: %d chunks, %d vectors", len(snap.Chunks), len(snap.Vectors))
	}
	if s.opts.EmbedTag != "" && snap.EmbedModel != "" && snap.EmbedModel != s.opts.EmbedTag {
		return fmt.Errorf("index: %s was built with embedding model %q but the current model is %q; re-run ingest",
			path, snap.EmbedModel, s.opts.EmbedTag)
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	s.chunks = make(map[string]core.Chunk, len(snap.Chunks))
	s.vec = &vectorIndex{}
	s.bm = newBM25()
	for i, c := range snap.Chunks {
		s.chunks[c.ID] = c
		s.vec.add(c.ID, snap.Vectors[i])
		s.bm.add(c.ID, c.Embedded())
	}
	return nil
}
