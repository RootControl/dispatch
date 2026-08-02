// Package sqldb backs the structured tier with a real database over
// database/sql, instead of the in-memory CSV runner.
//
// It imports only database/sql from the standard library — the caller supplies
// the driver, so dispatch itself stays dependency-free:
//
//	import _ "github.com/jackc/pgx/v5/stdlib"
//
//	db, _ := sql.Open("pgx", os.Getenv("DATABASE_URL"))
//	runner, _ := sqldb.New(sqldb.Config{DB: db, Dialect: sqldb.Postgres})
//	loop.Retrievers[core.TierStructured] = tiers.NewStructured(client, runner)
//
// # The read-only guarantee
//
// tiers.isReadOnly scans the generated SQL and rejects anything that is not a
// single read. Its own documentation calls that defense in depth rather than
// the primary guard, because it is string matching against an adversary who
// controls the model's input, and string matching loses that game eventually.
//
// This package supplies the guard that does not: every query runs inside a
// transaction opened with sql.TxOptions{ReadOnly: true}. On PostgreSQL that
// issues SET TRANSACTION READ ONLY, so a write is refused by the server, in the
// engine, after any string-level trick has already succeeded. The refusal is
// verified against a real Postgres in ./integration, including through a
// statement the scanner is deliberately made to pass.
//
// It is still not a substitute for GRANT SELECT. A read-only transaction stops
// writes; only privileges stop reads of tables this question had no business
// touching. Use both — that is what defense in depth means.
package sqldb

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/RootControl/dispatch/tiers"
)

// Dialect selects how the schema is introspected. Everything else — the
// read-only transaction, the row cap, the timeout — is dialect-independent.
type Dialect string

const (
	// Postgres reads information_schema and pg_class.
	Postgres Dialect = "postgres"
	// Generic reads information_schema only, which SQL Server, MySQL and
	// several others also expose. No row estimates: there is no portable way to
	// get them, and inventing a number a model will reason about is worse than
	// having none.
	Generic Dialect = "generic"
)

// Config configures a Runner.
type Config struct {
	DB      *sql.DB // required
	Dialect Dialect // default Postgres
	// Schema is the namespace to introspect. Default "public" for Postgres, and
	// for Generic the empty string, meaning every schema the role can see.
	Schema string
	// Tables, when set, restricts BOTH introspection and what the model is told
	// exists. It is not a security boundary — the transaction and the role's
	// privileges are — but a schema listing only the tables a question could
	// legitimately need is a smaller target and a better prompt.
	Tables []string
	// MaxRows caps rows read from a result set; default 1000. The tier renders
	// far fewer than this into evidence, but the runner is what stands between
	// a generated cross join and the process's memory.
	MaxRows int
	// Timeout bounds one query; default 30s. A model that writes an accidental
	// cartesian product should cost a timeout, not a hung agent.
	Timeout time.Duration
	// RowCounts includes approximate row counts in the schema description
	// (Postgres only). They help a model choose between similar tables, and
	// they are estimates from the planner's statistics rather than counts, so
	// they cost nothing on a large table.
	RowCounts bool
}

// Runner executes read-only queries for the structured tier.
type Runner struct {
	cfg Config
}

var _ tiers.SQLRunner = (*Runner)(nil)

// New validates cfg and returns a Runner. It does not touch the database.
func New(cfg Config) (*Runner, error) {
	if cfg.DB == nil {
		return nil, errors.New("sqldb: Config.DB is required")
	}
	switch cfg.Dialect {
	case "":
		cfg.Dialect = Postgres
	case Postgres, Generic:
	default:
		return nil, fmt.Errorf("sqldb: unknown dialect %q (want %q or %q)", cfg.Dialect, Postgres, Generic)
	}
	if cfg.Schema == "" && cfg.Dialect == Postgres {
		cfg.Schema = "public"
	}
	if cfg.MaxRows <= 0 {
		cfg.MaxRows = 1000
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = 30 * time.Second
	}
	return &Runner{cfg: cfg}, nil
}

// Query runs a read-only statement inside a read-only transaction.
//
// The transaction is the point. Everything else here — the timeout, the row
// cap — bounds cost; this bounds damage, and it does so in the engine rather
// than in a regexp.
func (r *Runner) Query(ctx context.Context, query string) (tiers.Rows, error) {
	ctx, cancel := context.WithTimeout(ctx, r.cfg.Timeout)
	defer cancel()

	tx, err := r.cfg.DB.BeginTx(ctx, &sql.TxOptions{ReadOnly: true, Isolation: sql.LevelDefault})
	if err != nil {
		return tiers.Rows{}, fmt.Errorf("sqldb: begin read-only transaction: %w", err)
	}
	// Always roll back. Nothing here should have anything to commit, and if a
	// driver somehow let a write through, this is the last thing that undoes it.
	defer tx.Rollback()

	rows, err := tx.QueryContext(ctx, query)
	if err != nil {
		return tiers.Rows{}, err
	}
	defer rows.Close()
	return scan(rows, r.cfg.MaxRows)
}

// scan reads a result set into strings, which is what the tier feeds the model.
func scan(rows *sql.Rows, maxRows int) (tiers.Rows, error) {
	cols, err := rows.Columns()
	if err != nil {
		return tiers.Rows{}, err
	}
	out := tiers.Rows{Columns: cols}

	// []byte rather than string, so a driver returning raw bytes (pgx does, for
	// several types) is scanned rather than rejected. A NULL stays nil and
	// becomes "NULL" below — distinct from the empty string, which a model
	// asked about a missing value should not confuse with one.
	cells := make([]any, len(cols))
	ptrs := make([]any, len(cols))
	for i := range cells {
		ptrs[i] = &cells[i]
	}

	for rows.Next() {
		if len(out.Rows) >= maxRows {
			// Stop reading rather than truncate after the fact: the cap exists
			// to bound memory, and a cap applied after materializing everything
			// would not.
			break
		}
		if err := rows.Scan(ptrs...); err != nil {
			return tiers.Rows{}, err
		}
		row := make([]string, len(cols))
		for i, c := range cells {
			row[i] = render(c)
		}
		out.Rows = append(out.Rows, row)
	}
	return out, rows.Err()
}

func render(v any) string {
	switch t := v.(type) {
	case nil:
		return "NULL"
	case []byte:
		return string(t)
	case time.Time:
		// RFC3339 rather than the Go default: a model reading a date is far
		// likelier to have seen this shape.
		return t.Format(time.RFC3339)
	case bool:
		if t {
			return "true"
		}
		return "false"
	default:
		return fmt.Sprint(t)
	}
}

// Schema describes the available tables for the text-to-SQL prompt.
//
// It is read from the database rather than configured, because a hand-written
// schema description drifts: the model keeps writing queries against a column
// that was renamed a year ago, and the failure looks like a bad model.
func (r *Runner) Schema(ctx context.Context) (string, error) {
	ctx, cancel := context.WithTimeout(ctx, r.cfg.Timeout)
	defer cancel()

	cols, err := r.columns(ctx)
	if err != nil {
		return "", err
	}
	if len(cols) == 0 {
		where := "the database"
		if r.cfg.Schema != "" {
			where = "schema " + r.cfg.Schema
		}
		return "", fmt.Errorf("sqldb: no tables visible in %s\n"+
			"(the role may lack privileges, or Config.Tables may name tables that do not exist)", where)
	}

	counts := map[string]int64{}
	if r.cfg.RowCounts && r.cfg.Dialect == Postgres {
		// Best-effort: an unreadable pg_class is not a reason to fail a query
		// that only wanted column names.
		counts, _ = r.rowEstimates(ctx)
	}

	var b strings.Builder
	names := make([]string, 0, len(cols))
	for t := range cols {
		names = append(names, t)
	}
	sort.Strings(names)
	for _, t := range names {
		fmt.Fprintf(&b, "TABLE %s (\n", t)
		for _, c := range cols[t] {
			fmt.Fprintf(&b, "  %s %s", c.name, c.typ)
			if !c.nullable {
				b.WriteString(" NOT NULL")
			}
			b.WriteString("\n")
		}
		b.WriteString(")")
		if n, ok := counts[t]; ok && n >= 0 {
			fmt.Fprintf(&b, "  -- roughly %d row(s)", n)
		}
		b.WriteString("\n\n")
	}
	return strings.TrimRight(b.String(), "\n"), nil
}

type column struct {
	name     string
	typ      string
	nullable bool
}

func (r *Runner) columns(ctx context.Context) (map[string][]column, error) {
	// Bound parameters throughout: this is introspection driven by
	// configuration rather than by a model, but a table name arriving from a
	// config file is still not something to concatenate into SQL.
	q := `SELECT table_name, column_name, data_type, is_nullable
	      FROM information_schema.columns
	      WHERE ($1 = '' OR table_schema = $1)
	      ORDER BY table_name, ordinal_position`
	args := []any{r.cfg.Schema}
	if r.cfg.Dialect == Generic {
		// Placeholder syntax is not portable; ? is the commoner form outside
		// Postgres. Nothing else in this query is dialect-specific.
		q = strings.ReplaceAll(q, "$1", "?")
		args = []any{r.cfg.Schema, r.cfg.Schema}
	}

	tx, err := r.cfg.DB.BeginTx(ctx, &sql.TxOptions{ReadOnly: true})
	if err != nil {
		return nil, fmt.Errorf("sqldb: begin read-only transaction: %w", err)
	}
	defer tx.Rollback()

	rows, err := tx.QueryContext(ctx, q, args...)
	if err != nil {
		return nil, fmt.Errorf("sqldb: read information_schema: %w", err)
	}
	defer rows.Close()

	allow := map[string]bool{}
	for _, t := range r.cfg.Tables {
		allow[strings.ToLower(t)] = true
	}

	out := map[string][]column{}
	for rows.Next() {
		var table, name, typ, nullable string
		if err := rows.Scan(&table, &name, &typ, &nullable); err != nil {
			return nil, err
		}
		if len(allow) > 0 && !allow[strings.ToLower(table)] {
			continue
		}
		out[table] = append(out[table], column{name: name, typ: typ, nullable: nullable == "YES"})
	}
	return out, rows.Err()
}

// rowEstimates reads the planner's row estimates. They are estimates, not
// counts: SELECT count(*) on every table would make describing a schema cost
// more than answering the question.
func (r *Runner) rowEstimates(ctx context.Context) (map[string]int64, error) {
	q := `SELECT c.relname, c.reltuples::bigint
	      FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
	      WHERE c.relkind IN ('r','p','m','v') AND ($1 = '' OR n.nspname = $1)`
	rows, err := r.cfg.DB.QueryContext(ctx, q, r.cfg.Schema)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := map[string]int64{}
	for rows.Next() {
		var name string
		var n int64
		if err := rows.Scan(&name, &n); err != nil {
			return nil, err
		}
		// -1 means "never analyzed", which is not the same as empty. Omitting
		// it is better than telling a model a table has minus one row.
		if n >= 0 {
			out[name] = n
		}
	}
	return out, rows.Err()
}
