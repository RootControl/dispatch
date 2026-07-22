package tiers

import (
	"strings"
	"testing"
)

// The guard is the security-relevant part of this tier, so it is tested against
// bypasses rather than only against happy paths.
func TestIsReadOnlyAcceptsReads(t *testing.T) {
	ok := []string{
		`SELECT * FROM invoices`,
		`select sum(amount) from line_items where invoice_id = 4471`,
		`SELECT COUNT(*) FROM invoices;`,
		`WITH recent AS (SELECT * FROM invoices) SELECT * FROM recent`,
		// A write verb inside a string literal is data, not a statement.
		`SELECT * FROM invoices WHERE customer = 'Drop Table Industries'`,
		`SELECT * FROM t WHERE note = 'it''s fine'`,
	}
	for _, q := range ok {
		if err := isReadOnly(q); err != nil {
			t.Errorf("isReadOnly(%q) = %v, want nil", q, err)
		}
	}
}

func TestIsReadOnlyRejectsWrites(t *testing.T) {
	bad := []struct {
		query, because string
	}{
		{`DELETE FROM invoices`, "delete"},
		{`INSERT INTO invoices VALUES (1)`, "insert"},
		{`UPDATE invoices SET status = 'paid'`, "update"},
		{`DROP TABLE invoices`, "drop"},
		{`TRUNCATE invoices`, "truncate"},
		{`GRANT ALL ON invoices TO public`, "grant"},

		// Stacked statements: the read is a decoy for what follows.
		{`SELECT 1; DROP TABLE invoices`, "stacked statements"},
		{`SELECT 1;DELETE FROM invoices;`, "stacked statements"},

		// Comments can hide a payload from a naive scan.
		{`SELECT * FROM t -- ; DROP TABLE x`, "line comment"},
		{`SELECT /* sneaky */ * FROM t`, "block comment"},
		{`SELECT * FROM t # mysql comment`, "hash comment"},

		// Writes that wear a SELECT's clothes.
		{`SELECT * INTO other FROM invoices`, "SELECT ... INTO writes"},
		{`WITH x AS (INSERT INTO t VALUES (1) RETURNING *) SELECT * FROM x`, "writing CTE"},

		// Reaching outside the query.
		{`ATTACH DATABASE '/tmp/x.db' AS x`, "attach"},
		{`SELECT * FROM t INTO OUTFILE '/tmp/x'`, "outfile"},
		{`PRAGMA table_info(t)`, "pragma"},
		{`EXEC sp_who`, "exec"},

		// Not a read at all.
		{`SHOW TABLES`, "not a SELECT"},
		{``, "empty"},
		{`   `, "blank"},

		// An unparseable quote state must not be guessed at.
		{`SELECT * FROM t WHERE a = 'unbalanced`, "unbalanced quote"},
	}
	for _, c := range bad {
		if err := isReadOnly(c.query); err == nil {
			t.Errorf("isReadOnly(%q) = nil, want rejection (%s)", c.query, c.because)
		}
	}
}

// Case and whitespace must not be a bypass.
func TestIsReadOnlyIgnoresCaseAndSpacing(t *testing.T) {
	for _, q := range []string{
		"  \n\t SeLeCt 1 FROM t  ",
		"dElEtE   FROM t",
		"SELECT 1 ;  drop table t",
	} {
		err := isReadOnly(q)
		wantErr := !strings.HasPrefix(strings.ToUpper(strings.TrimSpace(q)), "SELECT 1 FROM")
		if (err != nil) != wantErr {
			t.Errorf("isReadOnly(%q) = %v", q, err)
		}
	}
}

func TestStripLiterals(t *testing.T) {
	got, err := stripLiterals(`SELECT * FROM t WHERE a = 'DROP TABLE x' AND b = 2`)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(strings.ToUpper(got), "DROP") {
		t.Errorf("literal contents leaked into the scanned text: %q", got)
	}
	if !strings.Contains(got, "SELECT") || !strings.Contains(got, "AND b = 2") {
		t.Errorf("non-literal text was lost: %q", got)
	}
	if _, err := stripLiterals(`SELECT 'oops`); err == nil {
		t.Error("unbalanced quote should error")
	}
}
