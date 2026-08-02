package sqldb

import (
	"context"
	"database/sql"
	"database/sql/driver"
	"fmt"
	"io"
	"strings"
	"sync"
)

// A minimal database/sql driver, stdlib-only, so query construction, parameter
// binding, scanning and — the point of this package — the transaction options
// are exercised without a server. Importing a real driver would put a
// dependency in a module that has none; see ./integration for the real thing.
//
// It implements driver.ConnBeginTx specifically so the test can observe that a
// READ-ONLY transaction was requested. Whether the server then honours it is
// exactly what a fake cannot tell you, which is why the enforcement test runs
// against Postgres.

type recordedQuery struct {
	SQL  string
	Args []driver.NamedValue
}

type fakeDB struct {
	mu       sync.Mutex
	queries  []recordedQuery
	txOpts   []driver.TxOptions
	rows     map[string][][]driver.Value
	cols     map[string][]string
	err      error
	beginErr error
}

func (f *fakeDB) record(q string, args []driver.NamedValue) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.queries = append(f.queries, recordedQuery{SQL: q, Args: args})
}

func (f *fakeDB) seen() []recordedQuery {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]recordedQuery(nil), f.queries...)
}

func (f *fakeDB) transactions() []driver.TxOptions {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]driver.TxOptions(nil), f.txOpts...)
}

func (f *fakeDB) find(substr string) (recordedQuery, bool) {
	for _, q := range f.seen() {
		if strings.Contains(q.SQL, substr) {
			return q, true
		}
	}
	return recordedQuery{}, false
}

func (f *fakeDB) resultFor(q string) ([]string, [][]driver.Value) {
	for key, rows := range f.rows {
		if strings.Contains(q, key) {
			return f.cols[key], rows
		}
	}
	return []string{"id"}, nil
}

// --- driver plumbing ---

type fakeDriver struct{ db *fakeDB }

func (d fakeDriver) Open(string) (driver.Conn, error) { return &fakeConn{db: d.db}, nil }

type fakeConn struct{ db *fakeDB }

func (c *fakeConn) Prepare(q string) (driver.Stmt, error) { return &fakeStmt{db: c.db, q: q}, nil }
func (c *fakeConn) Close() error                          { return nil }
func (c *fakeConn) Begin() (driver.Tx, error)             { return fakeTx{}, nil }

// BeginTx records what was asked for. A driver that ignored ReadOnly would look
// identical from the outside, which is the limit of a fake and the reason the
// integration test exists.
func (c *fakeConn) BeginTx(_ context.Context, opts driver.TxOptions) (driver.Tx, error) {
	c.db.mu.Lock()
	c.db.txOpts = append(c.db.txOpts, opts)
	err := c.db.beginErr
	c.db.mu.Unlock()
	if err != nil {
		return nil, err
	}
	return fakeTx{}, nil
}

func (c *fakeConn) QueryContext(_ context.Context, q string, args []driver.NamedValue) (driver.Rows, error) {
	c.db.record(q, args)
	if c.db.err != nil {
		return nil, c.db.err
	}
	cols, rows := c.db.resultFor(q)
	return &fakeRows{cols: cols, rows: rows}, nil
}

type fakeTx struct{}

func (fakeTx) Commit() error   { return nil }
func (fakeTx) Rollback() error { return nil }

type fakeStmt struct {
	db *fakeDB
	q  string
}

func (s *fakeStmt) Close() error  { return nil }
func (s *fakeStmt) NumInput() int { return -1 }

func (s *fakeStmt) Exec([]driver.Value) (driver.Result, error) { return nil, io.EOF }

func (s *fakeStmt) Query(args []driver.Value) (driver.Rows, error) {
	named := make([]driver.NamedValue, len(args))
	for i, a := range args {
		named[i] = driver.NamedValue{Ordinal: i + 1, Value: a}
	}
	s.db.record(s.q, named)
	if s.db.err != nil {
		return nil, s.db.err
	}
	cols, rows := s.db.resultFor(s.q)
	return &fakeRows{cols: cols, rows: rows}, nil
}

type fakeRows struct {
	cols []string
	rows [][]driver.Value
	i    int
}

func (r *fakeRows) Columns() []string { return r.cols }
func (r *fakeRows) Close() error      { return nil }
func (r *fakeRows) Next(dest []driver.Value) error {
	if r.i >= len(r.rows) {
		return io.EOF
	}
	copy(dest, r.rows[r.i])
	r.i++
	return nil
}

var (
	registerCount int
	registerMu    sync.Mutex
)

// openFake registers a uniquely-named driver backed by db and opens it.
// database/sql panics on a duplicate driver name, so each call gets its own.
func openFake(db *fakeDB) *sql.DB {
	registerMu.Lock()
	registerCount++
	name := fmt.Sprintf("dispatch-sqldb-fake-%d", registerCount)
	registerMu.Unlock()
	sql.Register(name, fakeDriver{db: db})
	conn, err := sql.Open(name, "")
	if err != nil {
		panic(err)
	}
	return conn
}
