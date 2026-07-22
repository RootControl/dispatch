package tiers

import (
	"context"
	"fmt"
	"regexp"
	"strings"
	"sync"

	"github.com/RootControl/dispatch/core"
	"github.com/RootControl/dispatch/llm"
)

// Rows is a query result. Values are strings: this tier feeds text to an LLM,
// so preserving the driver's Go types would buy nothing and force every
// implementation to agree on conversions.
type Rows struct {
	Columns []string
	Rows    [][]string
}

// SQLRunner is the extension point for the structured tier. Back it with
// database/sql against Postgres, SQL Server, or anything else.
//
// GRANT THE EXECUTING ROLE SELECT ONLY. isReadOnly in this package is a second
// line of defense, not the first.
type SQLRunner interface {
	// Schema describes the tables available, in whatever form helps a model
	// write correct SQL — table names, columns, and ideally row counts.
	Schema(ctx context.Context) (string, error)
	// Query runs a read-only statement.
	Query(ctx context.Context, query string) (Rows, error)
}

// Structured answers questions with exact answers that live in a relational
// store — totals, counts, lookups by key — the questions vector search is worst
// at, because "the total on invoice 4471" is arithmetic, not similarity.
type Structured struct {
	llm     llm.LLM
	runner  SQLRunner
	MaxRows int // rows rendered into evidence; default 50

	once   sync.Once
	schema string
	schErr error
}

var _ core.Retriever = (*Structured)(nil)

// NewStructured builds the tier over a runner.
func NewStructured(l llm.LLM, r SQLRunner) *Structured {
	return &Structured{llm: l, runner: r}
}

func (s *Structured) Tier() core.Tier { return core.TierStructured }

const sqlSystem = `You translate a question into ONE read-only SQL SELECT query.

Rules:
- Use only the tables and columns in the schema. Never invent names.
- Emit a single SELECT (or WITH ... SELECT). Never INSERT, UPDATE, DELETE, or DDL.
- No semicolons beyond the end, and no SQL comments.
- Prefer explicit aggregates (SUM, COUNT, AVG) when the question asks for a total,
  a count, or an average.
- If the question cannot be answered from this schema, set "sql" to an empty string.

Reply with a JSON object only:
{"sql": "<the query, or empty>"}`

// Retrieve writes SQL for the question, checks it, runs it, and returns the
// result table as evidence alongside the query that produced it — so an answer
// built on these numbers can be audited back to the arithmetic that made them.
func (s *Structured) Retrieve(ctx context.Context, q core.Query) ([]core.Result, error) {
	schema, err := s.cachedSchema(ctx)
	if err != nil {
		return nil, fmt.Errorf("tiers: read schema: %w", err)
	}

	var out struct {
		SQL string `json:"sql"`
	}
	msgs := []llm.Message{
		llm.System(sqlSystem),
		llm.User(fmt.Sprintf("<schema>\n%s\n</schema>\n\nQuestion: %s", schema, q.Text)),
	}
	if err := s.llm.ChatJSON(ctx, msgs, &out); err != nil {
		return nil, fmt.Errorf("tiers: generate SQL: %w", err)
	}
	query := strings.TrimSpace(out.SQL)
	if query == "" {
		// The model judged the question unanswerable from this schema. That is
		// a gap for the loop to route around, not a failure.
		return nil, nil
	}
	if err := isReadOnly(query); err != nil {
		return nil, fmt.Errorf("tiers: refused generated SQL (%w): %s", err, query)
	}

	rows, err := s.runner.Query(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("tiers: run %q: %w", query, err)
	}
	if len(rows.Rows) == 0 {
		return nil, nil
	}

	maxRows := s.MaxRows
	if maxRows <= 0 {
		maxRows = 50
	}
	return []core.Result{{
		Tier:     core.TierStructured,
		SourceID: sourceName(query),
		Text:     renderRows(query, rows, maxRows),
		Score:    1,
		Meta:     map[string]string{"sql": query, "rows": fmt.Sprint(len(rows.Rows))},
	}}, nil
}

func (s *Structured) cachedSchema(ctx context.Context) (string, error) {
	s.once.Do(func() { s.schema, s.schErr = s.runner.Schema(ctx) })
	return s.schema, s.schErr
}

var fromTable = regexp.MustCompile(`(?i)\bFROM\s+([A-Za-z_][A-Za-z0-9_.]*)`)

// sourceName gives the citation something readable: the table queried, so a
// citation reads [structured:line_items] rather than an opaque identifier.
func sourceName(query string) string {
	if m := fromTable.FindStringSubmatch(query); len(m) == 2 {
		return m[1]
	}
	return "query"
}

// renderRows formats a result table, including the SQL so the number in an
// answer can be traced to the query that produced it.
func renderRows(query string, rows Rows, maxRows int) string {
	var b strings.Builder
	fmt.Fprintf(&b, "Query:\n%s\n\nResult:\n", query)
	fmt.Fprintf(&b, "%s\n", strings.Join(rows.Columns, " | "))
	for i, r := range rows.Rows {
		if i >= maxRows {
			fmt.Fprintf(&b, "... %d more row(s) not shown\n", len(rows.Rows)-maxRows)
			break
		}
		fmt.Fprintf(&b, "%s\n", strings.Join(r, " | "))
	}
	return strings.TrimRight(b.String(), "\n")
}
