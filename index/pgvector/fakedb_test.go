package pgvector

import (
	"database/sql"
	"database/sql/driver"
	"fmt"
	"io"
	"strings"
	"sync"
)

// A minimal database/sql driver, so the SQL this package generates and the rows
// it scans are exercised without a Postgres server. It is stdlib-only, which is
// the whole reason it is here: an integration test needs a real driver, and
// importing one would put a dependency in a module that has none.
//
// It verifies query construction, parameter binding and scanning. It cannot
// verify that Postgres accepts the SQL — see TestAgainstLivePostgres for that.

type recordedQuery struct {
	SQL  string
	Args []driver.Value
}

type fakeDB struct {
	mu      sync.Mutex
	queries []recordedQuery
	// rows keyed by a substring of the query; the first match wins.
	rows map[string][][]driver.Value
	cols map[string][]string
	err  error
}

func (f *fakeDB) record(q string, args []driver.Value) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.queries = append(f.queries, recordedQuery{SQL: q, Args: args})
}

func (f *fakeDB) seen() []recordedQuery {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]recordedQuery(nil), f.queries...)
}

// find returns the recorded query containing substr.
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

type fakeTx struct{}

func (fakeTx) Commit() error   { return nil }
func (fakeTx) Rollback() error { return nil }

type fakeStmt struct {
	db *fakeDB
	q  string
}

func (s *fakeStmt) Close() error  { return nil }
func (s *fakeStmt) NumInput() int { return -1 } // let database/sql pass anything

func (s *fakeStmt) Exec(args []driver.Value) (driver.Result, error) {
	s.db.record(s.q, args)
	if s.db.err != nil {
		return nil, s.db.err
	}
	return fakeResult{}, nil
}

func (s *fakeStmt) Query(args []driver.Value) (driver.Rows, error) {
	s.db.record(s.q, args)
	if s.db.err != nil {
		return nil, s.db.err
	}
	cols, rows := s.db.resultFor(s.q)
	return &fakeRows{cols: cols, rows: rows}, nil
}

type fakeResult struct{}

func (fakeResult) LastInsertId() (int64, error) { return 0, nil }
func (fakeResult) RowsAffected() (int64, error) { return 1, nil }

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

var registerOnce sync.Once
var registerCount int
var registerMu sync.Mutex

// openFake registers a uniquely-named driver backed by db and opens it.
// database/sql panics on a duplicate driver name, so each call gets its own.
func openFake(db *fakeDB) *sql.DB {
	registerOnce.Do(func() {})
	registerMu.Lock()
	registerCount++
	name := fmt.Sprintf("dispatch-fake-%d", registerCount)
	registerMu.Unlock()
	sql.Register(name, fakeDriver{db: db})
	conn, err := sql.Open(name, "")
	if err != nil {
		panic(err)
	}
	return conn
}
