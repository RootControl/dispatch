package index

import (
	"encoding/json"
	"fmt"
	"maps"
	"os"

	"github.com/RootControl/dispatch/core"
	"github.com/RootControl/dispatch/internal/atomicfile"
)

// snapshot is the on-disk index format. Only chunks and vectors are stored; the
// BM25 index is rebuilt on load, since it is derived from the chunk text and
// storing it would just be a second source of truth to keep in sync.
type snapshot struct {
	EmbedModel string       `json:"embed_model"`
	Chunks     []core.Chunk `json:"chunks"`
	Vectors    [][]float64  `json:"vectors"`
	// DocHashes carries the per-document fingerprints so an incremental ingest
	// after a restart can tell unchanged documents from new ones. An older
	// snapshot without it simply loads empty, and the next ingest re-derives
	// every document once — correct, just not free.
	DocHashes map[string]string `json:"doc_hashes,omitempty"`
	// SourceGeneration is set on derived artifacts (the RAPTOR tree) and names
	// the index generation they were built from, so a stale one can be refused
	// rather than quietly answering from documents the corpus has dropped.
	SourceGeneration string `json:"source_generation,omitempty"`
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
		DocHashes:  maps.Clone(s.docHash),

		SourceGeneration: s.sourceGen,
	}
	for i, id := range s.vec.ids {
		snap.Chunks = append(snap.Chunks, s.chunks[id])
		snap.Vectors = append(snap.Vectors, s.vec.vecs[i])
	}

	// Atomic: a crash part-way through an os.Create would leave truncated JSON
	// with the previous good index already gone, and since ingest became
	// incremental that index is the only record of what has been embedded.
	return atomicfile.WriteFunc(path, 0o644, func(f *os.File) error {
		enc := json.NewEncoder(f)
		enc.SetIndent("", "  ")
		return enc.Encode(snap)
	})
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
	s.docHash = map[string]string{}
	maps.Copy(s.docHash, snap.DocHashes)
	s.sourceGen = snap.SourceGeneration
	for i, c := range snap.Chunks {
		// A snapshot written before Ingest became an upsert can hold the same
		// chunk twice. Take the first and drop the rest rather than rebuilding
		// the duplication in memory — loading a corrupt index is the one moment
		// where it can be repaired for free.
		if _, dup := s.chunks[c.ID]; dup {
			continue
		}
		s.chunks[c.ID] = c
		s.vec.add(c.ID, snap.Vectors[i])
		s.bm.add(c.ID, c.Embedded())
	}
	return nil
}
