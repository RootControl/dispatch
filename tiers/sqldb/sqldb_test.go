package sqldb

import (
	"context"
	"database/sql/driver"
	"errors"
	"strings"
	"testing"
	"time"
)

func newRunner(t *testing.T, db *fakeDB, cfg Config) *Runner {
	t.Helper()
	cfg.DB = openFake(db)
	r, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return r
}

func TestNewRejectsMissingDB(t *testing.T) {
	if _, err := New(Config{}); err == nil {
		t.Fatal("a runner with no database should not construct")
	}
}

func TestNewRejectsUnknownDialect(t *testing.T) {
	if _, err := New(Config{DB: openFake(&fakeDB{}), Dialect: "oracle"}); err == nil {
		t.Fatal("an unknown dialect should be an error, not a silent default")
	}
}

// The guarantee this package exists to add: every query runs in a transaction
// the driver was asked to make read-only. tiers.isReadOnly is string matching
// and says so about itself; this is the engine refusing.
func TestQueryRunsInAReadOnlyTransaction(t *testing.T) {
	db := &fakeDB{
		cols: map[string][]string{"line_items": {"total"}},
		rows: map[string][][]driver.Value{"line_items": {{int64(42)}}},
	}
	r := newRunner(t, db, Config{})

	if _, err := r.Query(context.Background(), "SELECT sum(amount) AS total FROM line_items"); err != nil {
		t.Fatal(err)
	}
	txs := db.transactions()
	if len(txs) == 0 {
		t.Fatal("no transaction was opened: the query ran outside one")
	}
	for i, tx := range txs {
		if !tx.ReadOnly {
			t.Errorf("transaction %d was not requested read-only", i)
		}
	}
}

// Introspection reads too, so it takes the same transaction. A read-only path
// with one un-flagged branch is not a read-only path.
func TestSchemaAlsoUsesAReadOnlyTransaction(t *testing.T) {
	db := &fakeDB{
		cols: map[string][]string{"information_schema.columns": {"table_name", "column_name", "data_type", "is_nullable"}},
		rows: map[string][][]driver.Value{"information_schema.columns": {
			{[]byte("invoices"), []byte("id"), []byte("integer"), []byte("NO")},
		}},
	}
	r := newRunner(t, db, Config{})
	if _, err := r.Schema(context.Background()); err != nil {
		t.Fatal(err)
	}
	for i, tx := range db.transactions() {
		if !tx.ReadOnly {
			t.Errorf("introspection transaction %d was not requested read-only", i)
		}
	}
}

func TestSchemaRendersTablesAndColumns(t *testing.T) {
	db := &fakeDB{
		cols: map[string][]string{"information_schema.columns": {"table_name", "column_name", "data_type", "is_nullable"}},
		rows: map[string][][]driver.Value{"information_schema.columns": {
			{[]byte("invoices"), []byte("id"), []byte("integer"), []byte("NO")},
			{[]byte("invoices"), []byte("total"), []byte("numeric"), []byte("YES")},
			{[]byte("vendors"), []byte("name"), []byte("text"), []byte("NO")},
		}},
	}
	r := newRunner(t, db, Config{})
	got, err := r.Schema(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"TABLE invoices", "id integer NOT NULL", "total numeric", "TABLE vendors"} {
		if !strings.Contains(got, want) {
			t.Errorf("schema missing %q:\n%s", want, got)
		}
	}
	// Nullability has to be visible: a model that does not know a column can be
	// NULL writes aggregates that silently skip rows.
	if strings.Contains(got, "total numeric NOT NULL") {
		t.Errorf("a nullable column was described as NOT NULL:\n%s", got)
	}
	// Sorted, so the prompt is stable and the model's output is reproducible.
	if strings.Index(got, "TABLE invoices") > strings.Index(got, "TABLE vendors") {
		t.Errorf("tables should be sorted:\n%s", got)
	}
}

// An empty schema must fail loudly. A model handed no tables writes SQL against
// invented ones, and the resulting error names a missing table rather than the
// real problem, which is privileges.
func TestSchemaFailsWhenNoTablesAreVisible(t *testing.T) {
	r := newRunner(t, &fakeDB{
		cols: map[string][]string{"information_schema.columns": {"table_name", "column_name", "data_type", "is_nullable"}},
	}, Config{})
	_, err := r.Schema(context.Background())
	if err == nil {
		t.Fatal("no visible tables should be an error")
	}
	if !strings.Contains(err.Error(), "privileges") {
		t.Errorf("the error should point at the likely cause, got: %v", err)
	}
}

func TestTablesRestrictsWhatTheModelIsTold(t *testing.T) {
	db := &fakeDB{
		cols: map[string][]string{"information_schema.columns": {"table_name", "column_name", "data_type", "is_nullable"}},
		rows: map[string][][]driver.Value{"information_schema.columns": {
			{[]byte("invoices"), []byte("id"), []byte("integer"), []byte("NO")},
			{[]byte("secrets"), []byte("token"), []byte("text"), []byte("NO")},
		}},
	}
	r := newRunner(t, db, Config{Tables: []string{"invoices"}})
	got, err := r.Schema(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(got, "secrets") {
		t.Errorf("an excluded table reached the prompt:\n%s", got)
	}
	if !strings.Contains(got, "invoices") {
		t.Errorf("the allowed table is missing:\n%s", got)
	}
}

// The schema query binds its parameter rather than concatenating it. A schema
// name usually comes from a config file rather than a model, but "usually" is
// not a reason to build the injectable version.
func TestSchemaNameIsBound(t *testing.T) {
	db := &fakeDB{
		cols: map[string][]string{"information_schema.columns": {"table_name", "column_name", "data_type", "is_nullable"}},
		rows: map[string][][]driver.Value{"information_schema.columns": {
			{[]byte("t"), []byte("c"), []byte("text"), []byte("YES")},
		}},
	}
	r := newRunner(t, db, Config{Schema: "reporting'; DROP TABLE x --"})
	if _, err := r.Schema(context.Background()); err != nil {
		t.Fatal(err)
	}
	q, ok := db.find("information_schema.columns")
	if !ok {
		t.Fatal("no introspection query was issued")
	}
	if strings.Contains(q.SQL, "DROP TABLE") {
		t.Errorf("the schema name was interpolated into SQL:\n%s", q.SQL)
	}
	if len(q.Args) == 0 {
		t.Fatal("the schema name was not passed as a parameter")
	}
}

func TestMaxRowsStopsReading(t *testing.T) {
	rows := make([][]driver.Value, 500)
	for i := range rows {
		rows[i] = []driver.Value{int64(i)}
	}
	db := &fakeDB{
		cols: map[string][]string{"big": {"n"}},
		rows: map[string][][]driver.Value{"big": rows},
	}
	r := newRunner(t, db, Config{MaxRows: 10})
	got, err := r.Query(context.Background(), "SELECT n FROM big")
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Rows) != 10 {
		t.Errorf("read %d rows under a cap of 10", len(got.Rows))
	}
}

// NULL and the empty string are different answers to "what is this value", and
// a model asked about a missing total must be able to tell them apart.
func TestNullRendersDistinctlyFromEmpty(t *testing.T) {
	db := &fakeDB{
		cols: map[string][]string{"t": {"a", "b"}},
		rows: map[string][][]driver.Value{"t": {{nil, []byte("")}}},
	}
	r := newRunner(t, db, Config{})
	got, err := r.Query(context.Background(), "SELECT a, b FROM t")
	if err != nil {
		t.Fatal(err)
	}
	if got.Rows[0][0] != "NULL" {
		t.Errorf("NULL rendered as %q", got.Rows[0][0])
	}
	if got.Rows[0][1] != "" {
		t.Errorf("empty string rendered as %q", got.Rows[0][1])
	}
}

func TestValuesRenderReadably(t *testing.T) {
	when := time.Date(2026, 3, 14, 9, 30, 0, 0, time.UTC)
	db := &fakeDB{
		cols: map[string][]string{"t": {"when", "ok", "n"}},
		rows: map[string][][]driver.Value{"t": {{when, true, int64(7)}}},
	}
	r := newRunner(t, db, Config{})
	got, err := r.Query(context.Background(), "SELECT * FROM t")
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"2026-03-14T09:30:00Z", "true", "7"}
	for i, w := range want {
		if got.Rows[0][i] != w {
			t.Errorf("column %d = %q, want %q", i, got.Rows[0][i], w)
		}
	}
}

func TestQueryPropagatesDriverErrors(t *testing.T) {
	boom := errors.New("relation \"nope\" does not exist")
	r := newRunner(t, &fakeDB{err: boom}, Config{})
	if _, err := r.Query(context.Background(), "SELECT * FROM nope"); err == nil {
		t.Fatal("a failing query should return its error")
	}
}

// A driver that cannot give a read-only transaction must fail the query rather
// than fall back to a writable one.
func TestQueryFailsWhenAReadOnlyTransactionCannotBeOpened(t *testing.T) {
	r := newRunner(t, &fakeDB{beginErr: errors.New("read-only transactions unsupported")}, Config{})
	_, err := r.Query(context.Background(), "SELECT 1")
	if err == nil {
		t.Fatal("the query ran despite the read-only transaction failing")
	}
	if !strings.Contains(err.Error(), "read-only") {
		t.Errorf("the error should name what failed, got: %v", err)
	}
}

func TestTimeoutBoundsAQuery(t *testing.T) {
	r := newRunner(t, &fakeDB{}, Config{Timeout: time.Nanosecond})
	// Deadline exceeded before or during the call; either way the caller must
	// get an error rather than wait.
	if _, err := r.Query(context.Background(), "SELECT pg_sleep(60)"); err == nil {
		t.Fatal("a query past its timeout should error")
	}
}
