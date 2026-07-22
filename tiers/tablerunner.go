package tiers

import (
	"context"
	"encoding/csv"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"sort"
	"strconv"
	"strings"
)

// TableRunner is an in-memory SQLRunner over CSV tables. It exists so the
// structured tier can be demonstrated and tested without a database driver,
// which the zero-dependency rule forbids.
//
// It is NOT a SQL engine. It understands one deliberately small shape:
//
//	SELECT <cols | * | COUNT(*) | SUM(c) | AVG(c) | MIN(c) | MAX(c)>
//	FROM <table>
//	[WHERE <col> <op> <literal> [AND ...]]
//	[GROUP BY <col>] [ORDER BY <col> [ASC|DESC]] [LIMIT <n>]
//
// No joins, no subqueries, no expressions. Anything else is a clear error
// rather than a wrong answer. For real work, implement SQLRunner over
// database/sql and grant the role SELECT only.
type TableRunner struct {
	tables map[string]*Table
}

// Table is a CSV-backed table. Values stay strings; comparisons try numeric
// first and fall back to lexicographic.
type Table struct {
	Name    string
	Columns []string
	Rows    [][]string
}

// NewTableRunner builds a runner over the given tables.
func NewTableRunner(tables ...*Table) *TableRunner {
	m := make(map[string]*Table, len(tables))
	for _, t := range tables {
		m[strings.ToLower(t.Name)] = t
	}
	return &TableRunner{tables: m}
}

// LoadCSVDir reads every .csv file under dir as a table named after the file.
// The first record is the header row.
func LoadCSVDir(dir string) (*TableRunner, error) {
	var tables []*Table
	err := filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() || !strings.EqualFold(filepath.Ext(path), ".csv") {
			return err
		}
		f, err := os.Open(path)
		if err != nil {
			return err
		}
		defer f.Close()
		records, err := csv.NewReader(f).ReadAll()
		if err != nil {
			return fmt.Errorf("tiers: read %s: %w", path, err)
		}
		if len(records) == 0 {
			return nil
		}
		name := strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
		tables = append(tables, &Table{Name: name, Columns: records[0], Rows: records[1:]})
		return nil
	})
	if err != nil {
		return nil, err
	}
	if len(tables) == 0 {
		return nil, fmt.Errorf("tiers: no .csv files under %s", dir)
	}
	return NewTableRunner(tables...), nil
}

// Schema describes the tables for the text-to-SQL prompt. Row counts are
// included because they help a model choose between counting and aggregating.
func (r *TableRunner) Schema(ctx context.Context) (string, error) {
	names := make([]string, 0, len(r.tables))
	for n := range r.tables {
		names = append(names, n)
	}
	sort.Strings(names)

	var b strings.Builder
	for _, n := range names {
		t := r.tables[n]
		fmt.Fprintf(&b, "TABLE %s (%s)  -- %d rows\n", t.Name, strings.Join(t.Columns, ", "), len(t.Rows))
	}
	return strings.TrimRight(b.String(), "\n"), nil
}

// Query runs the supported subset. It re-checks isReadOnly: a runner must not
// assume its caller validated anything.
func (r *TableRunner) Query(ctx context.Context, query string) (Rows, error) {
	if err := isReadOnly(query); err != nil {
		return Rows{}, err
	}
	q, err := parseSelect(query)
	if err != nil {
		return Rows{}, err
	}
	t, ok := r.tables[strings.ToLower(q.table)]
	if !ok {
		return Rows{}, fmt.Errorf("unknown table %q", q.table)
	}
	return q.run(t)
}

// --- parsing ---

type selItem struct {
	fn    string // "", COUNT, SUM, AVG, MIN, MAX
	col   string // "*" for COUNT(*)
	label string
}

type cond struct {
	col, op, val string
}

type selectQuery struct {
	items     []selItem
	table     string
	where     []cond
	groupBy   string
	orderBy   string
	orderDesc bool
	limit     int
}

var clauseRE = regexp.MustCompile(`(?is)^\s*SELECT\s+(.*?)\s+FROM\s+([A-Za-z_][A-Za-z0-9_]*)` +
	`(?:\s+WHERE\s+(.*?))?` +
	`(?:\s+GROUP\s+BY\s+([A-Za-z_][A-Za-z0-9_]*))?` +
	`(?:\s+ORDER\s+BY\s+([A-Za-z_][A-Za-z0-9_]*)(?:\s+(ASC|DESC))?)?` +
	`(?:\s+LIMIT\s+(\d+))?\s*;?\s*$`)

var aggRE = regexp.MustCompile(`(?i)^(COUNT|SUM|AVG|MIN|MAX)\s*\(\s*(\*|[A-Za-z_][A-Za-z0-9_]*)\s*\)$`)

func parseSelect(query string) (*selectQuery, error) {
	m := clauseRE.FindStringSubmatch(strings.TrimSpace(query))
	if m == nil {
		return nil, fmt.Errorf("unsupported query shape (this runner handles a small SELECT subset only)")
	}
	q := &selectQuery{table: m[2], groupBy: m[4], orderBy: m[5], limit: -1}
	q.orderDesc = strings.EqualFold(m[6], "DESC")
	if m[7] != "" {
		n, err := strconv.Atoi(m[7])
		if err != nil {
			return nil, fmt.Errorf("bad LIMIT: %w", err)
		}
		q.limit = n
	}

	for _, raw := range splitTopLevel(m[1]) {
		item := strings.TrimSpace(raw)
		if item == "" {
			continue
		}
		// Strip an alias: "SUM(amount) AS total".
		label := item
		if i := regexp.MustCompile(`(?i)\s+AS\s+`).FindStringIndex(item); i != nil {
			label = strings.TrimSpace(item[i[1]:])
			item = strings.TrimSpace(item[:i[0]])
		}
		if am := aggRE.FindStringSubmatch(item); am != nil {
			q.items = append(q.items, selItem{fn: strings.ToUpper(am[1]), col: am[2], label: label})
			continue
		}
		if !regexp.MustCompile(`^(\*|[A-Za-z_][A-Za-z0-9_]*)$`).MatchString(item) {
			return nil, fmt.Errorf("unsupported select expression %q", item)
		}
		q.items = append(q.items, selItem{col: item, label: label})
	}
	if len(q.items) == 0 {
		return nil, fmt.Errorf("no columns selected")
	}

	if where := strings.TrimSpace(m[3]); where != "" {
		for _, clause := range regexp.MustCompile(`(?i)\s+AND\s+`).Split(where, -1) {
			c, err := parseCond(clause)
			if err != nil {
				return nil, err
			}
			q.where = append(q.where, c)
		}
	}
	return q, nil
}

var condRE = regexp.MustCompile(`(?is)^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(>=|<=|!=|<>|=|>|<)\s*(.+?)\s*$`)

func parseCond(clause string) (cond, error) {
	m := condRE.FindStringSubmatch(clause)
	if m == nil {
		return cond{}, fmt.Errorf("unsupported WHERE clause %q", strings.TrimSpace(clause))
	}
	val := strings.TrimSpace(m[3])
	if len(val) >= 2 && val[0] == '\'' && val[len(val)-1] == '\'' {
		val = strings.ReplaceAll(val[1:len(val)-1], "''", "'")
	}
	return cond{col: m[1], op: m[2], val: val}, nil
}

// splitTopLevel splits a select list on commas outside parentheses, so
// "SUM(a), COUNT(*)" yields two items rather than four.
func splitTopLevel(s string) []string {
	var out []string
	depth, start := 0, 0
	for i, c := range s {
		switch c {
		case '(':
			depth++
		case ')':
			depth--
		case ',':
			if depth == 0 {
				out = append(out, s[start:i])
				start = i + 1
			}
		}
	}
	return append(out, s[start:])
}

// --- evaluation ---

func (q *selectQuery) run(t *Table) (Rows, error) {
	idx := map[string]int{}
	for i, c := range t.Columns {
		idx[strings.ToLower(c)] = i
	}
	colIndex := func(name string) (int, error) {
		i, ok := idx[strings.ToLower(name)]
		if !ok {
			return 0, fmt.Errorf("unknown column %q in table %s", name, t.Name)
		}
		return i, nil
	}

	// WHERE
	filtered := make([][]string, 0, len(t.Rows))
	for _, row := range t.Rows {
		keep := true
		for _, c := range q.where {
			i, err := colIndex(c.col)
			if err != nil {
				return Rows{}, err
			}
			if !compare(row[i], c.op, c.val) {
				keep = false
				break
			}
		}
		if keep {
			filtered = append(filtered, row)
		}
	}

	hasAgg := slices.ContainsFunc(q.items, func(s selItem) bool { return s.fn != "" })

	var out Rows
	for _, it := range q.items {
		out.Columns = append(out.Columns, it.label)
	}

	switch {
	case q.groupBy != "":
		gi, err := colIndex(q.groupBy)
		if err != nil {
			return Rows{}, err
		}
		groups := map[string][][]string{}
		var order []string
		for _, row := range filtered {
			k := row[gi]
			if _, seen := groups[k]; !seen {
				order = append(order, k)
			}
			groups[k] = append(groups[k], row)
		}
		sort.Strings(order) // deterministic without an ORDER BY
		for _, k := range order {
			row, err := q.project(groups[k], colIndex, gi)
			if err != nil {
				return Rows{}, err
			}
			out.Rows = append(out.Rows, row)
		}

	case hasAgg:
		row, err := q.project(filtered, colIndex, -1)
		if err != nil {
			return Rows{}, err
		}
		out.Rows = append(out.Rows, row)

	default:
		for _, r := range filtered {
			row, err := q.projectPlain(r, colIndex)
			if err != nil {
				return Rows{}, err
			}
			out.Rows = append(out.Rows, row)
		}
	}

	if q.orderBy != "" {
		if err := q.sort(&out); err != nil {
			return Rows{}, err
		}
	}
	if q.limit >= 0 && len(out.Rows) > q.limit {
		out.Rows = out.Rows[:q.limit]
	}
	return out, nil
}

// project builds one output row for an aggregated group.
func (q *selectQuery) project(rows [][]string, colIndex func(string) (int, error), groupIdx int) ([]string, error) {
	out := make([]string, 0, len(q.items))
	for _, it := range q.items {
		if it.fn == "" {
			// A plain column alongside aggregates only makes sense as the
			// GROUP BY key; anything else has no single value.
			if groupIdx < 0 || len(rows) == 0 {
				return nil, fmt.Errorf("column %q must appear in GROUP BY", it.col)
			}
			i, err := colIndex(it.col)
			if err != nil {
				return nil, err
			}
			out = append(out, rows[0][i])
			continue
		}
		if it.fn == "COUNT" && it.col == "*" {
			out = append(out, strconv.Itoa(len(rows)))
			continue
		}
		i, err := colIndex(it.col)
		if err != nil {
			return nil, err
		}
		out = append(out, aggregate(it.fn, rows, i))
	}
	return out, nil
}

func (q *selectQuery) projectPlain(row []string, colIndex func(string) (int, error)) ([]string, error) {
	var out []string
	for _, it := range q.items {
		if it.col == "*" {
			out = append(out, row...)
			continue
		}
		i, err := colIndex(it.col)
		if err != nil {
			return nil, err
		}
		out = append(out, row[i])
	}
	return out, nil
}

func aggregate(fn string, rows [][]string, col int) string {
	var sum float64
	var count int
	var minV, maxV float64
	var minS, maxS string
	numeric := true
	for _, r := range rows {
		v := r[col]
		f, err := strconv.ParseFloat(strings.TrimSpace(v), 64)
		if err != nil {
			numeric = false
		} else {
			sum += f
			if count == 0 || f < minV {
				minV = f
			}
			if count == 0 || f > maxV {
				maxV = f
			}
		}
		if count == 0 || v < minS {
			minS = v
		}
		if count == 0 || v > maxS {
			maxS = v
		}
		count++
	}
	switch fn {
	case "COUNT":
		return strconv.Itoa(count)
	case "SUM":
		return trimFloat(sum)
	case "AVG":
		if count == 0 {
			return ""
		}
		return trimFloat(sum / float64(count))
	case "MIN":
		if numeric && count > 0 {
			return trimFloat(minV)
		}
		return minS
	case "MAX":
		if numeric && count > 0 {
			return trimFloat(maxV)
		}
		return maxS
	}
	return ""
}

func (q *selectQuery) sort(out *Rows) error {
	pos := slices.IndexFunc(out.Columns, func(c string) bool { return strings.EqualFold(c, q.orderBy) })
	if pos < 0 {
		return fmt.Errorf("ORDER BY %q is not in the select list", q.orderBy)
	}
	sort.SliceStable(out.Rows, func(i, j int) bool {
		a, b := out.Rows[i][pos], out.Rows[j][pos]
		af, aerr := strconv.ParseFloat(a, 64)
		bf, berr := strconv.ParseFloat(b, 64)
		if aerr == nil && berr == nil {
			if q.orderDesc {
				return af > bf
			}
			return af < bf
		}
		if q.orderDesc {
			return a > b
		}
		return a < b
	})
	return nil
}

// compare evaluates one condition, numerically when both sides parse as
// numbers and lexicographically otherwise.
func compare(got, op, want string) bool {
	gf, gerr := strconv.ParseFloat(strings.TrimSpace(got), 64)
	wf, werr := strconv.ParseFloat(strings.TrimSpace(want), 64)
	if gerr == nil && werr == nil {
		switch op {
		case "=":
			return gf == wf
		case "!=", "<>":
			return gf != wf
		case "<":
			return gf < wf
		case "<=":
			return gf <= wf
		case ">":
			return gf > wf
		case ">=":
			return gf >= wf
		}
		return false
	}
	switch op {
	case "=":
		return strings.EqualFold(got, want)
	case "!=", "<>":
		return !strings.EqualFold(got, want)
	case "<":
		return got < want
	case "<=":
		return got <= want
	case ">":
		return got > want
	case ">=":
		return got >= want
	}
	return false
}

func trimFloat(f float64) string {
	return strconv.FormatFloat(f, 'f', -1, 64)
}
