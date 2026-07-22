package tiers

import (
	"context"
	"strings"
	"testing"

	"github.com/RootControl/dispatch/core"
	"github.com/RootControl/dispatch/internal/fake"
	"github.com/RootControl/dispatch/llm"
)

func testRunner() *TableRunner {
	return NewTableRunner(
		&Table{Name: "invoices", Columns: []string{"id", "customer", "status"}, Rows: [][]string{
			{"4471", "Acme Corp", "paid"},
			{"4472", "Globex", "open"},
			{"4473", "Initech", "paid"},
		}},
		&Table{Name: "line_items", Columns: []string{"invoice_id", "description", "amount"}, Rows: [][]string{
			{"4471", "Migration services", "120000"},
			{"4471", "Support retainer", "30000"},
			{"4471", "Training workshop", "12500"},
			{"4472", "Migration services", "80000"},
		}},
	)
}

func query(t *testing.T, sql string) Rows {
	t.Helper()
	rows, err := testRunner().Query(context.Background(), sql)
	if err != nil {
		t.Fatalf("Query(%q): %v", sql, err)
	}
	return rows
}

// The design doc's own example question.
func TestRunnerTotalOnInvoice(t *testing.T) {
	rows := query(t, "SELECT SUM(amount) FROM line_items WHERE invoice_id = 4471")
	if len(rows.Rows) != 1 || rows.Rows[0][0] != "162500" {
		t.Fatalf("expected 162500, got %v", rows.Rows)
	}
}

func TestRunnerAggregates(t *testing.T) {
	cases := []struct{ sql, want string }{
		{"SELECT COUNT(*) FROM invoices", "3"},
		{"SELECT COUNT(*) FROM invoices WHERE status = 'paid'", "2"},
		{"SELECT SUM(amount) FROM line_items", "242500"},
		{"SELECT MAX(amount) FROM line_items", "120000"},
		{"SELECT MIN(amount) FROM line_items", "12500"},
		{"SELECT AVG(amount) FROM line_items WHERE invoice_id = 4471", "54166.666666666664"},
		// String comparison is case-insensitive for equality.
		{"SELECT COUNT(*) FROM invoices WHERE customer = 'acme corp'", "1"},
		// Numeric comparison, not lexicographic: "9" must not exceed "12500".
		{"SELECT COUNT(*) FROM line_items WHERE amount > 100000", "1"},
	}
	for _, c := range cases {
		rows := query(t, c.sql)
		if len(rows.Rows) != 1 || rows.Rows[0][0] != c.want {
			t.Errorf("%s = %v, want %s", c.sql, rows.Rows, c.want)
		}
	}
}

func TestRunnerGroupByOrderLimit(t *testing.T) {
	rows := query(t, "SELECT invoice_id, SUM(amount) AS total FROM line_items GROUP BY invoice_id ORDER BY total DESC")
	if len(rows.Rows) != 2 {
		t.Fatalf("expected 2 groups, got %v", rows.Rows)
	}
	if rows.Rows[0][0] != "4471" || rows.Rows[0][1] != "162500" {
		t.Errorf("first group = %v, want [4471 162500]", rows.Rows[0])
	}
	if rows.Columns[1] != "total" {
		t.Errorf("alias lost: columns = %v", rows.Columns)
	}

	limited := query(t, "SELECT invoice_id, SUM(amount) AS total FROM line_items GROUP BY invoice_id ORDER BY total DESC LIMIT 1")
	if len(limited.Rows) != 1 {
		t.Errorf("LIMIT ignored: %v", limited.Rows)
	}
}

func TestRunnerProjectionAndStar(t *testing.T) {
	rows := query(t, "SELECT customer, status FROM invoices WHERE id = 4471")
	if len(rows.Rows) != 1 || rows.Rows[0][0] != "Acme Corp" || rows.Rows[0][1] != "paid" {
		t.Fatalf("projection wrong: %v", rows.Rows)
	}
	star := query(t, "SELECT * FROM invoices WHERE id = 4472")
	if len(star.Rows) != 1 || len(star.Rows[0]) != 3 {
		t.Fatalf("SELECT * wrong: %v", star.Rows)
	}
}

// Unsupported input must fail loudly, never silently return a wrong answer.
func TestRunnerRejectsUnsupported(t *testing.T) {
	bad := []string{
		"SELECT a FROM t1 JOIN t2 ON t1.id = t2.id",
		"SELECT * FROM (SELECT * FROM invoices)",
		"SELECT amount * 2 FROM line_items",
		"SELECT * FROM nonexistent_table",
		"SELECT no_such_column FROM invoices",
		"DELETE FROM invoices",
	}
	for _, sql := range bad {
		if _, err := testRunner().Query(context.Background(), sql); err == nil {
			t.Errorf("Query(%q) succeeded; expected a clear error", sql)
		}
	}
}

func TestRunnerSchema(t *testing.T) {
	got, err := testRunner().Schema(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"invoices", "line_items", "amount", "3 rows"} {
		if !strings.Contains(got, want) {
			t.Errorf("schema missing %q:\n%s", want, got)
		}
	}
}

// --- the tier ---

func sqlFake(sql string) *fake.LLM {
	return &fake.LLM{ChatFunc: func([]llm.Message) (string, error) {
		return `{"sql": ` + jsonString(sql) + `}`, nil
	}}
}

func jsonString(s string) string {
	return `"` + strings.ReplaceAll(strings.ReplaceAll(s, `\`, `\\`), `"`, `\"`) + `"`
}

func TestStructuredRetrieve(t *testing.T) {
	f := sqlFake("SELECT SUM(amount) FROM line_items WHERE invoice_id = 4471")
	var r core.Retriever = NewStructured(f, testRunner())
	if r.Tier() != core.TierStructured {
		t.Fatalf("Tier() = %s", r.Tier())
	}

	got, err := r.Retrieve(context.Background(), core.Query{Text: "Total on invoice 4471?", TopK: 5})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 {
		t.Fatalf("expected 1 result, got %d", len(got))
	}
	if !strings.Contains(got[0].Text, "162500") {
		t.Errorf("result missing the total:\n%s", got[0].Text)
	}
	// The SQL travels with the answer so a number can be audited.
	if !strings.Contains(got[0].Text, "SELECT SUM(amount)") {
		t.Errorf("result should include the query that produced it:\n%s", got[0].Text)
	}
	if got[0].Cite() != "[structured:line_items]" {
		t.Errorf("Cite() = %q, want [structured:line_items]", got[0].Cite())
	}
	if got[0].Meta["sql"] == "" {
		t.Error("Meta should carry the SQL")
	}
}

// A model that emits a write must be stopped by the tier, even though the
// runner would also refuse it.
func TestStructuredRefusesGeneratedWrite(t *testing.T) {
	f := sqlFake("DELETE FROM invoices WHERE id = 4471")
	_, err := NewStructured(f, testRunner()).Retrieve(context.Background(), core.Query{Text: "remove invoice 4471"})
	if err == nil {
		t.Fatal("expected the tier to refuse a generated write")
	}
	if !strings.Contains(err.Error(), "refused") {
		t.Errorf("error should say the SQL was refused, got %v", err)
	}
}

// An unanswerable question is a gap for the loop, not an error.
func TestStructuredEmptySQLIsAGap(t *testing.T) {
	f := sqlFake("")
	got, err := NewStructured(f, testRunner()).Retrieve(context.Background(), core.Query{Text: "what is love?"})
	if err != nil {
		t.Fatalf("unanswerable question should not error: %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("expected no results, got %d", len(got))
	}
}

func TestStructuredEmptyResultIsAGap(t *testing.T) {
	f := sqlFake("SELECT * FROM invoices WHERE id = 999999")
	got, err := NewStructured(f, testRunner()).Retrieve(context.Background(), core.Query{Text: "invoice 999999?"})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 0 {
		t.Fatalf("expected no results for an empty table scan, got %d", len(got))
	}
}

// The schema is fetched once, not per question.
func TestStructuredCachesSchema(t *testing.T) {
	runner := &countingRunner{TableRunner: testRunner()}
	s := NewStructured(sqlFake("SELECT COUNT(*) FROM invoices"), runner)
	for range 3 {
		if _, err := s.Retrieve(context.Background(), core.Query{Text: "how many?"}); err != nil {
			t.Fatal(err)
		}
	}
	if runner.schemaCalls != 1 {
		t.Errorf("Schema called %d times, want 1", runner.schemaCalls)
	}
}

type countingRunner struct {
	*TableRunner
	schemaCalls int
}

func (c *countingRunner) Schema(ctx context.Context) (string, error) {
	c.schemaCalls++
	return c.TableRunner.Schema(ctx)
}

func TestLoadCSVDir(t *testing.T) {
	runner, err := LoadCSVDir("../testdata/sql")
	if err != nil {
		t.Fatal(err)
	}
	rows, err := runner.Query(context.Background(), "SELECT SUM(amount) FROM line_items WHERE invoice_id = 4471")
	if err != nil {
		t.Fatal(err)
	}
	if rows.Rows[0][0] != "162500" {
		t.Errorf("CSV total = %v, want 162500", rows.Rows)
	}
	if _, err := LoadCSVDir(t.TempDir()); err == nil {
		t.Error("expected an error for a directory with no CSVs")
	}
}
