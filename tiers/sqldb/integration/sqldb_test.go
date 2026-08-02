// Package integration runs tiers/sqldb against a real PostgreSQL server.
//
// It is a separate module so the pgx driver never enters dispatch's go.mod. The
// unit tests use a stdlib fake driver and can verify that a read-only
// transaction was REQUESTED; only a real server can show that the request is
// honoured, which is the entire claim this package makes.
//
//	docker run -d --name sqldbtest -e POSTGRES_PASSWORD=dispatch \
//	    -e POSTGRES_DB=dispatch -p 55433:5432 pgvector/pgvector:pg17
//
//	cd tiers/sqldb/integration
//	DISPATCH_PG_DSN='postgres://postgres:dispatch@localhost:55433/dispatch?sslmode=disable' \
//	    go test -v ./...
package integration

import (
	"context"
	"database/sql"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/RootControl/dispatch/tiers"
	"github.com/RootControl/dispatch/tiers/sqldb"

	_ "github.com/jackc/pgx/v5/stdlib"
)

func open(t *testing.T) *sql.DB {
	t.Helper()
	dsn := os.Getenv("DISPATCH_PG_DSN")
	if dsn == "" {
		t.Skip("set DISPATCH_PG_DSN to run the sqldb integration tests")
	}
	db, err := sql.Open("pgx", dsn)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := db.PingContext(ctx); err != nil {
		t.Fatalf("cannot reach Postgres at DISPATCH_PG_DSN: %v", err)
	}
	t.Cleanup(func() { db.Close() })
	return db
}

// seed builds a small billing schema in its own namespace, so several tests can
// share one server without colliding.
func seed(t *testing.T, db *sql.DB, schema string) {
	t.Helper()
	ctx := context.Background()
	stmts := []string{
		`DROP SCHEMA IF EXISTS ` + schema + ` CASCADE`,
		`CREATE SCHEMA ` + schema,
		`CREATE TABLE ` + schema + `.vendors (
			id integer PRIMARY KEY,
			name text NOT NULL,
			country text)`,
		`CREATE TABLE ` + schema + `.invoices (
			id integer PRIMARY KEY,
			vendor_id integer NOT NULL REFERENCES ` + schema + `.vendors(id),
			total numeric,
			issued_at timestamptz NOT NULL,
			paid boolean NOT NULL)`,
		`INSERT INTO ` + schema + `.vendors VALUES
			(1, 'Northwind Systems', 'GB'),
			(2, 'Contoso Ltd', NULL)`,
		`INSERT INTO ` + schema + `.invoices VALUES
			(4471, 1, 12500.00, '2026-03-14T09:30:00Z', true),
			(4472, 1, 3200.50,  '2026-04-01T10:00:00Z', false),
			(4473, 2, NULL,     '2026-04-02T11:15:00Z', false)`,
		`ANALYZE ` + schema + `.invoices`,
	}
	for _, s := range stmts {
		if _, err := db.ExecContext(ctx, s); err != nil {
			t.Fatalf("seed %q: %v", s, err)
		}
	}
	t.Cleanup(func() {
		db.ExecContext(context.Background(), `DROP SCHEMA IF EXISTS `+schema+` CASCADE`)
	})
}

func runner(t *testing.T, db *sql.DB, cfg sqldb.Config) *sqldb.Runner {
	t.Helper()
	cfg.DB = db
	r, err := sqldb.New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return r
}

// THE CLAIM. tiers.isReadOnly is string matching and its own documentation says
// so; this shows the server refusing a write that has already got past every
// string-level check, because it is the transaction that stops it and not the
// scanner.
//
// Each statement below is passed straight to Query — the tier's scanner is
// bypassed deliberately. That is the threat model: assume the scanner lost.
func TestReadOnlyTransactionRefusesWrites(t *testing.T) {
	db := open(t)
	seed(t, db, "sqldb_ro")
	r := runner(t, db, sqldb.Config{Schema: "sqldb_ro"})

	writes := []struct {
		name string
		sql  string
	}{
		{"insert", `INSERT INTO sqldb_ro.vendors VALUES (99, 'Injected', 'XX')`},
		{"update", `UPDATE sqldb_ro.invoices SET total = 0`},
		{"delete", `DELETE FROM sqldb_ro.invoices`},
		{"drop", `DROP TABLE sqldb_ro.invoices`},
		{"create", `CREATE TABLE sqldb_ro.smuggled (x integer)`},
		// A write smuggled inside a SELECT via a data-modifying CTE. This is
		// the shape a string scanner is worst at: it starts with WITH, and the
		// dangerous verb is nested.
		{"data-modifying CTE", `WITH gone AS (DELETE FROM sqldb_ro.invoices RETURNING id) SELECT count(*) FROM gone`},
	}
	for _, w := range writes {
		t.Run(w.name, func(t *testing.T) {
			if _, err := r.Query(context.Background(), w.sql); err == nil {
				t.Fatalf("the server accepted a write in a read-only transaction: %s", w.sql)
			} else if !strings.Contains(strings.ToLower(err.Error()), "read-only") &&
				!strings.Contains(strings.ToLower(err.Error()), "read only") {
				// Any error stops the write, but the message matters: this is
				// what tells an operator the guard fired rather than the query
				// simply being wrong.
				t.Logf("refused, though not by the read-only guard: %v", err)
			}
		})
	}

	// And nothing got through: the data is untouched.
	var vendors, invoices int
	if err := db.QueryRow(`SELECT
		(SELECT count(*) FROM sqldb_ro.vendors),
		(SELECT count(*) FROM sqldb_ro.invoices)`).Scan(&vendors, &invoices); err != nil {
		t.Fatal(err)
	}
	if vendors != 2 || invoices != 3 {
		t.Fatalf("data changed: %d vendors, %d invoices, want 2 and 3", vendors, invoices)
	}
}

// The negative control for the test above. If a plain transaction also refused
// these writes — because the role lacked privileges, say — the read-only flag
// would be proving nothing, and the test would pass for the wrong reason.
func TestTheSameWritesSucceedWithoutTheReadOnlyFlag(t *testing.T) {
	db := open(t)
	seed(t, db, "sqldb_control")

	ctx := context.Background()
	tx, err := db.BeginTx(ctx, &sql.TxOptions{}) // the only difference
	if err != nil {
		t.Fatal(err)
	}
	defer tx.Rollback()
	if _, err := tx.ExecContext(ctx, `INSERT INTO sqldb_control.vendors VALUES (99, 'Control', 'XX')`); err != nil {
		t.Fatalf("the control write failed, so the read-only test proves nothing: %v", err)
	}
}

func TestSchemaIntrospectsRealTables(t *testing.T) {
	db := open(t)
	seed(t, db, "sqldb_schema")
	r := runner(t, db, sqldb.Config{Schema: "sqldb_schema", RowCounts: true})

	got, err := r.Schema(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"TABLE invoices", "TABLE vendors",
		"vendor_id integer NOT NULL",
		"total numeric",       // nullable, so no NOT NULL
		"issued_at timestamp", // Postgres reports "timestamp with time zone"
		"country text",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("schema missing %q:\n%s", want, got)
		}
	}
	if strings.Contains(got, "total numeric NOT NULL") {
		t.Errorf("nullable column described as NOT NULL:\n%s", got)
	}
	// Row estimates come from the planner, and the seed ANALYZEs, so this one
	// should be exact. It is the number that helps a model pick a table.
	if !strings.Contains(got, "roughly 3 row(s)") {
		t.Errorf("expected a row estimate for invoices:\n%s", got)
	}
}

// A different schema must not leak in. Two tenants on one server is the normal
// case, and a schema listing both is both a bad prompt and a bad boundary.
func TestSchemaIsScopedToItsNamespace(t *testing.T) {
	db := open(t)
	seed(t, db, "sqldb_a")
	seed(t, db, "sqldb_b")
	r := runner(t, db, sqldb.Config{Schema: "sqldb_a"})

	got, err := r.Schema(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	// Both schemas hold identically-named tables, so the check is that the
	// listing is not doubled.
	if n := strings.Count(got, "TABLE invoices"); n != 1 {
		t.Errorf("invoices listed %d times: another schema leaked in\n%s", n, got)
	}
}

func TestQueryReturnsRowsTheTierCanRender(t *testing.T) {
	db := open(t)
	seed(t, db, "sqldb_q")
	r := runner(t, db, sqldb.Config{Schema: "sqldb_q"})

	rows, err := r.Query(context.Background(),
		`SELECT v.name, sum(i.total) AS total
		 FROM sqldb_q.invoices i JOIN sqldb_q.vendors v ON v.id = i.vendor_id
		 GROUP BY v.name ORDER BY v.name`)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows.Columns) != 2 || rows.Columns[0] != "name" || rows.Columns[1] != "total" {
		t.Fatalf("columns = %v", rows.Columns)
	}
	if len(rows.Rows) != 2 {
		t.Fatalf("rows = %v, want 2", rows.Rows)
	}
	// Contoso's only invoice has a NULL total, so SUM is NULL — and the render
	// has to say NULL rather than "0" or "". A model told "0" would report that
	// Contoso billed nothing, which is a different claim from "unknown".
	if rows.Rows[0][0] != "Contoso Ltd" || rows.Rows[0][1] != "NULL" {
		t.Errorf("row 0 = %v, want [Contoso Ltd NULL]", rows.Rows[0])
	}
	if rows.Rows[1][1] != "15700.50" {
		t.Errorf("row 1 total = %q, want 15700.50", rows.Rows[1][1])
	}
}

func TestNullAndTimestampRendering(t *testing.T) {
	db := open(t)
	seed(t, db, "sqldb_render")
	r := runner(t, db, sqldb.Config{Schema: "sqldb_render"})

	rows, err := r.Query(context.Background(),
		`SELECT country, issued_at, paid FROM sqldb_render.vendors v
		 JOIN sqldb_render.invoices i ON i.vendor_id = v.id
		 WHERE i.id = 4473`)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows.Rows) != 1 {
		t.Fatalf("rows = %v", rows.Rows)
	}
	got := rows.Rows[0]
	if got[0] != "NULL" {
		t.Errorf("NULL country rendered as %q", got[0])
	}
	if !strings.HasPrefix(got[1], "2026-04-02T") {
		t.Errorf("timestamp rendered as %q, want RFC3339", got[1])
	}
	if got[2] != "false" {
		t.Errorf("boolean rendered as %q", got[2])
	}
}

func TestMaxRowsCapsARealResultSet(t *testing.T) {
	db := open(t)
	r := runner(t, db, sqldb.Config{Schema: "public", MaxRows: 5})
	rows, err := r.Query(context.Background(), `SELECT g FROM generate_series(1, 10000) g`)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows.Rows) != 5 {
		t.Fatalf("read %d rows under a cap of 5", len(rows.Rows))
	}
}

// A generated cartesian product must cost a timeout, not a hung agent.
func TestTimeoutCancelsARunawayQuery(t *testing.T) {
	db := open(t)
	r := runner(t, db, sqldb.Config{Schema: "public", Timeout: 500 * time.Millisecond})
	start := time.Now()
	_, err := r.Query(context.Background(), `SELECT pg_sleep(30)`)
	if err == nil {
		t.Fatal("a 30-second sleep completed under a 500ms timeout")
	}
	if el := time.Since(start); el > 10*time.Second {
		t.Errorf("timeout took %s to fire", el)
	}
}

// The runner is what the structured tier actually takes.
func TestSatisfiesTheTierInterface(t *testing.T) {
	db := open(t)
	seed(t, db, "sqldb_iface")
	var _ tiers.SQLRunner = runner(t, db, sqldb.Config{Schema: "sqldb_iface"})
}
