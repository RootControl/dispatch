// Package pgvector backs the semantic tier with Postgres + pgvector instead of
// the reference in-memory store.
//
// It imports only database/sql from the standard library — the caller supplies
// the driver, so dispatch itself stays dependency-free:
//
//	import _ "github.com/jackc/pgx/v5/stdlib"
//
//	db, _ := sql.Open("pgx", os.Getenv("DATABASE_URL"))
//	store, _ := pgvector.New(ctx, pgvector.Config{DB: db, LLM: client, Dims: 768})
//	loop.Retrievers[core.TierSemantic] = tiers.NewSemantic(store)
//
// The split of responsibilities is the point. index.Store owns chunking,
// contextual chunking, the caches and the ingest planner; this owns storage and
// search. Keep the first, replace the second — which is why Ingest here takes
// chunks and vectors rather than documents, and why the same
// contextual-chunking pass feeds both backends.
//
// Requires the pgvector extension and a full-text configuration. Call Migrate
// once to create the table and its indexes.
package pgvector

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"strconv"
	"strings"

	"github.com/RootControl/dispatch/core"
	"github.com/RootControl/dispatch/index"
	"github.com/RootControl/dispatch/llm"
)

// Config configures a Store.
type Config struct {
	DB  *sql.DB // required
	LLM llm.LLM // required: embeds the query at search time
	// Dims is the embedding width, needed to declare the vector column. It must
	// match the embedding model; pgvector fixes it at table creation.
	Dims int
	// Table defaults to "dispatch_chunks".
	Table string
	// TextSearchConfig is the Postgres full-text configuration for the lexical
	// half of the hybrid. Defaults to "english".
	TextSearchConfig string
	// Pool is how many candidates each half of the hybrid contributes before
	// fusion. Defaults to max(5*topK, 20), matching the reference store.
	Pool int
	// Rerank, when set, reorders the fused shortlist. Search over-fetches for it.
	Rerank index.Reranker
}

// Store is a pgvector-backed index.Searcher.
type Store struct {
	cfg   Config
	table string
	tsCfg string
}

var _ index.Searcher = (*Store)(nil)

// New validates cfg and returns a Store. It does not touch the database; call
// Migrate to create the schema, or point it at a table you manage yourself.
func New(cfg Config) (*Store, error) {
	if cfg.DB == nil {
		return nil, errors.New("pgvector: Config.DB is required")
	}
	if cfg.LLM == nil {
		return nil, errors.New("pgvector: Config.LLM is required to embed queries")
	}
	if cfg.Dims <= 0 {
		return nil, errors.New("pgvector: Config.Dims must match the embedding model's width")
	}
	table := cfg.Table
	if table == "" {
		table = "dispatch_chunks"
	}
	if err := validIdentifier(table); err != nil {
		return nil, err
	}
	tsCfg := cfg.TextSearchConfig
	if tsCfg == "" {
		tsCfg = "english"
	}
	if err := validIdentifier(tsCfg); err != nil {
		return nil, err
	}
	return &Store{cfg: cfg, table: table, tsCfg: tsCfg}, nil
}

// validIdentifier guards the two settings that must be interpolated into SQL
// rather than bound as parameters — a table name and a text-search
// configuration cannot be placeholders. Everything else in this package is
// parameterised, so this is the only place an injection could originate.
func validIdentifier(s string) error {
	if s == "" || len(s) > 63 {
		return fmt.Errorf("pgvector: %q is not a usable identifier", s)
	}
	for i, r := range s {
		alpha := r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r == '_'
		digit := r >= '0' && r <= '9'
		if !alpha && !(digit && i > 0) {
			return fmt.Errorf("pgvector: %q is not a usable identifier: only letters, digits and underscore, not starting with a digit", s)
		}
	}
	return nil
}

// Migrate creates the table and its indexes if they do not exist.
//
// The vector index is HNSW over cosine distance. It is created after the table
// so an existing deployment can skip it; on a large corpus, build it after the
// bulk load rather than before, since maintaining it during ingest is far
// slower than one build at the end.
func (s *Store) Migrate(ctx context.Context) error {
	stmts := []string{
		`CREATE EXTENSION IF NOT EXISTS vector`,
		fmt.Sprintf(`CREATE TABLE IF NOT EXISTS %s (
			id        TEXT PRIMARY KEY,
			doc_id    TEXT NOT NULL,
			text      TEXT NOT NULL,
			context   TEXT NOT NULL DEFAULT '',
			position  INTEGER NOT NULL DEFAULT 0,
			meta      JSONB NOT NULL DEFAULT '{}'::jsonb,
			embedding vector(%d) NOT NULL,
			embedded  TEXT NOT NULL
		)`, s.table, s.cfg.Dims),
		fmt.Sprintf(`CREATE INDEX IF NOT EXISTS %s_doc_id_idx ON %s (doc_id)`, s.table, s.table),
		fmt.Sprintf(`CREATE INDEX IF NOT EXISTS %s_meta_idx ON %s USING gin (meta)`, s.table, s.table),
		fmt.Sprintf(`CREATE INDEX IF NOT EXISTS %s_fts_idx ON %s USING gin (to_tsvector('%s', embedded))`,
			s.table, s.table, s.tsCfg),
		fmt.Sprintf(`CREATE INDEX IF NOT EXISTS %s_embedding_idx ON %s USING hnsw (embedding vector_cosine_ops)`,
			s.table, s.table),
	}
	for _, q := range stmts {
		if _, err := s.cfg.DB.ExecContext(ctx, q); err != nil {
			return fmt.Errorf("pgvector: migrate: %w", err)
		}
	}
	return nil
}

// Upsert stores chunks and their vectors, replacing any chunk with the same ID.
// It mirrors index.Store.Add, in bulk, so the shared contextual-chunking pass
// can write to either backend.
func (s *Store) Upsert(ctx context.Context, chunks []core.Chunk, vectors [][]float64) error {
	if len(chunks) != len(vectors) {
		return fmt.Errorf("pgvector: %d chunks but %d vectors", len(chunks), len(vectors))
	}
	if len(chunks) == 0 {
		return nil
	}
	tx, err := s.cfg.DB.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()

	q := fmt.Sprintf(`INSERT INTO %s (id, doc_id, text, context, position, meta, embedding, embedded)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
		ON CONFLICT (id) DO UPDATE SET
			doc_id = EXCLUDED.doc_id, text = EXCLUDED.text, context = EXCLUDED.context,
			position = EXCLUDED.position, meta = EXCLUDED.meta,
			embedding = EXCLUDED.embedding, embedded = EXCLUDED.embedded`, s.table)
	stmt, err := tx.PrepareContext(ctx, q)
	if err != nil {
		return fmt.Errorf("pgvector: upsert: %w", err)
	}
	defer stmt.Close()

	for i, c := range chunks {
		if len(vectors[i]) != s.cfg.Dims {
			return fmt.Errorf("pgvector: chunk %q has a %d-dimension vector, table expects %d",
				c.ID, len(vectors[i]), s.cfg.Dims)
		}
		meta, err := encodeMeta(c.Meta)
		if err != nil {
			return err
		}
		if _, err := stmt.ExecContext(ctx, c.ID, c.DocID, c.Text, c.Context, c.Position,
			meta, encodeVector(vectors[i]), c.Embedded()); err != nil {
			return fmt.Errorf("pgvector: upsert %q: %w", c.ID, err)
		}
	}
	return tx.Commit()
}

// Delete removes every chunk belonging to the named documents.
func (s *Store) Delete(ctx context.Context, docIDs ...string) (int, error) {
	if len(docIDs) == 0 {
		return 0, nil
	}
	// pq array literals vary by driver, so bind each ID rather than an array.
	holders := make([]string, len(docIDs))
	args := make([]any, len(docIDs))
	for i, id := range docIDs {
		holders[i] = "$" + strconv.Itoa(i+1)
		args[i] = id
	}
	res, err := s.cfg.DB.ExecContext(ctx,
		fmt.Sprintf(`DELETE FROM %s WHERE doc_id IN (%s)`, s.table, strings.Join(holders, ",")), args...)
	if err != nil {
		return 0, fmt.Errorf("pgvector: delete: %w", err)
	}
	n, err := res.RowsAffected()
	return int(n), err
}

// Len reports the number of stored chunks.
func (s *Store) Len() int {
	var n int
	if err := s.cfg.DB.QueryRow(fmt.Sprintf(`SELECT count(*) FROM %s`, s.table)).Scan(&n); err != nil {
		return 0
	}
	return n
}

// Search runs the same hybrid as the reference store — a cosine pool and a
// full-text pool, fused by RRF — so switching backends changes where the data
// lives and not how ranking behaves.
//
// Fusion happens in Go rather than SQL deliberately: it reuses the same tested
// RRF as the in-memory path, so the two backends cannot drift apart in the one
// place a difference would be invisible in the results.
func (s *Store) Search(ctx context.Context, q core.Query) ([]index.Hit, error) {
	topK := q.TopK
	if topK <= 0 {
		topK = 10
	}
	pool := s.cfg.Pool
	if pool <= 0 {
		pool = max(topK*5, 20)
	}

	// q.Embedding() is the HyDE expansion when one was written; the full-text
	// half below stays on the literal query. See core.Query.Embedding.
	vecs, err := s.cfg.LLM.Embed(ctx, []string{q.Embedding()})
	if err != nil {
		return nil, fmt.Errorf("pgvector: embed query: %w", err)
	}
	if len(vecs) != 1 || len(vecs[0]) != s.cfg.Dims {
		return nil, fmt.Errorf("pgvector: query embedding is %d-dimensional, table expects %d",
			len(vecs[0]), s.cfg.Dims)
	}

	where, args := s.filterClause(q.Filter)

	// Ranked id lists from each half. Only ids and order matter — RRF ignores
	// the scores — so neither query needs to return the row.
	vecArgs := append([]any{encodeVector(vecs[0])}, args...)
	vecSQL := fmt.Sprintf(`SELECT id FROM %s %s ORDER BY embedding <=> $1 LIMIT %d`,
		s.table, where, pool)

	ftsArgs := append([]any{q.Text}, args...)
	ftsWhere := where
	match := fmt.Sprintf(`to_tsvector('%s', embedded) @@ plainto_tsquery('%s', $1)`, s.tsCfg, s.tsCfg)
	if ftsWhere == "" {
		ftsWhere = "WHERE " + match
	} else {
		ftsWhere += " AND " + match
	}
	ftsSQL := fmt.Sprintf(`SELECT id FROM %s %s ORDER BY ts_rank_cd(to_tsvector('%s', embedded), plainto_tsquery('%s', $1)) DESC LIMIT %d`,
		s.table, ftsWhere, s.tsCfg, s.tsCfg, pool)

	vecIDs, err := s.queryIDs(ctx, vecSQL, vecArgs)
	if err != nil {
		return nil, fmt.Errorf("pgvector: vector search: %w", err)
	}
	ftsIDs, err := s.queryIDs(ctx, ftsSQL, ftsArgs)
	if err != nil {
		return nil, fmt.Errorf("pgvector: text search: %w", err)
	}

	fuseTo := topK
	if s.cfg.Rerank != nil {
		fuseTo = max(topK*4, 20)
	}
	fused := index.FuseRankings([][]string{vecIDs, ftsIDs}, fuseTo)
	if len(fused) == 0 {
		return nil, nil
	}

	hits, err := s.load(ctx, fused)
	if err != nil {
		return nil, err
	}
	if s.cfg.Rerank != nil {
		return s.cfg.Rerank.Rerank(ctx, q.Text, hits, topK)
	}
	return hits, nil
}

// filterClause turns a core.Filter into a WHERE over the meta JSONB column.
// Values are bound, never interpolated; a trailing "*" becomes a LIKE prefix
// with the literal's own wildcards escaped so a value containing % or _ cannot
// widen the filter.
func (s *Store) filterClause(f core.Filter) (string, []any) {
	if f.Empty() {
		return "", nil
	}
	// Sorted keys so the generated SQL is stable and reviewable.
	keys := make([]string, 0, len(f))
	for k := range f {
		keys = append(keys, k)
	}
	sortStrings(keys)

	var conds []string
	var args []any
	n := 1 // $1 is always the query (vector or text)
	for _, k := range keys {
		v := f[k]
		n++
		keyArg := n
		args = append(args, k)
		n++
		valArg := n
		if pre, isPrefix := strings.CutSuffix(v, "*"); isPrefix {
			args = append(args, escapeLike(pre)+"%")
			conds = append(conds, fmt.Sprintf(`meta ->> $%d LIKE $%d ESCAPE '\'`, keyArg, valArg))
		} else {
			args = append(args, v)
			conds = append(conds, fmt.Sprintf(`meta ->> $%d = $%d`, keyArg, valArg))
		}
	}
	return "WHERE " + strings.Join(conds, " AND "), args
}

func escapeLike(s string) string {
	r := strings.NewReplacer(`\`, `\\`, `%`, `\%`, `_`, `\_`)
	return r.Replace(s)
}

func (s *Store) queryIDs(ctx context.Context, query string, args []any) ([]string, error) {
	rows, err := s.cfg.DB.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		out = append(out, id)
	}
	return out, rows.Err()
}

// load fetches the fused ids and returns them in fused order. Postgres does not
// preserve the IN-list order, so the rows are reordered here rather than
// trusting the scan order — a mistake that would silently discard the ranking.
func (s *Store) load(ctx context.Context, fused []index.Ranked) ([]index.Hit, error) {
	holders := make([]string, len(fused))
	args := make([]any, len(fused))
	for i, f := range fused {
		holders[i] = "$" + strconv.Itoa(i+1)
		args[i] = f.ID
	}
	rows, err := s.cfg.DB.QueryContext(ctx, fmt.Sprintf(
		`SELECT id, doc_id, text, context, position, meta FROM %s WHERE id IN (%s)`,
		s.table, strings.Join(holders, ",")), args...)
	if err != nil {
		return nil, fmt.Errorf("pgvector: load chunks: %w", err)
	}
	defer rows.Close()

	byID := make(map[string]core.Chunk, len(fused))
	for rows.Next() {
		var c core.Chunk
		var meta []byte
		if err := rows.Scan(&c.ID, &c.DocID, &c.Text, &c.Context, &c.Position, &meta); err != nil {
			return nil, err
		}
		if c.Meta, err = decodeMeta(meta); err != nil {
			return nil, err
		}
		byID[c.ID] = c
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}

	out := make([]index.Hit, 0, len(fused))
	for _, f := range fused {
		c, ok := byID[f.ID]
		if !ok {
			continue // deleted between the ranking query and this one
		}
		out = append(out, index.Hit{Chunk: c, Score: f.Score})
	}
	return out, nil
}
