package tiers

import (
	"fmt"
	"regexp"
	"strings"
)

// isReadOnly rejects any statement that is not a single read.
//
// THIS IS DEFENSE IN DEPTH, NOT THE PRIMARY GUARD. The executing database role
// must be granted SELECT only. A generated-SQL allowlist is a string-matching
// exercise against an adversary who controls the model's input, and string
// matching loses that game eventually. This function exists to catch mistakes
// and to make a compromise need two failures instead of one.
//
// It deliberately over-rejects: a legitimate query containing the word "update"
// inside a string literal is refused rather than parsed precisely. A refused
// read is an inconvenience; an executed write is a breach.
func isReadOnly(query string) error {
	q := strings.TrimSpace(query)
	if q == "" {
		return fmt.Errorf("empty query")
	}

	// Comments can hide a payload from a naive scan and have no place in
	// generated SQL.
	if strings.Contains(q, "--") || strings.Contains(q, "/*") || strings.Contains(q, "#") {
		return fmt.Errorf("comments are not permitted in generated SQL")
	}

	// One optional trailing semicolon is allowed; anything else means stacked
	// statements, where the second statement is the dangerous one.
	q = strings.TrimSuffix(q, ";")
	stripped, err := stripLiterals(q)
	if err != nil {
		return err
	}
	if strings.Contains(stripped, ";") {
		return fmt.Errorf("multiple statements are not permitted")
	}

	upper := strings.ToUpper(strings.TrimSpace(stripped))
	if !readStart.MatchString(upper) {
		return fmt.Errorf("only SELECT and WITH queries are permitted")
	}
	if m := forbidden.FindString(upper); m != "" {
		return fmt.Errorf("forbidden keyword %q", strings.ToLower(m))
	}
	return nil
}

var (
	readStart = regexp.MustCompile(`^(SELECT|WITH)\b`)

	// Write and privilege verbs, plus statement forms that write through a
	// SELECT (SELECT ... INTO) or reach outside the query (ATTACH, COPY, LOAD).
	forbidden = regexp.MustCompile(`\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|REPLACE|MERGE|UPSERT|GRANT|REVOKE|EXEC|EXECUTE|CALL|ATTACH|DETACH|PRAGMA|VACUUM|COPY|LOAD|OUTFILE|DUMPFILE|INTO|SET|COMMIT|ROLLBACK|BEGIN|SAVEPOINT|SHUTDOWN)\b`)
)

// stripLiterals removes single-quoted string literals so keywords inside data
// do not trigger the scan. Unbalanced quotes are an error: a query we cannot
// parse confidently is a query we must not run.
func stripLiterals(q string) (string, error) {
	var b strings.Builder
	inLiteral := false
	for i := 0; i < len(q); i++ {
		c := q[i]
		if c != '\'' {
			if !inLiteral {
				b.WriteByte(c)
			}
			continue
		}
		// '' inside a literal is an escaped quote, not a terminator.
		if inLiteral && i+1 < len(q) && q[i+1] == '\'' {
			i++
			continue
		}
		inLiteral = !inLiteral
	}
	if inLiteral {
		return "", fmt.Errorf("unbalanced quote in query")
	}
	return b.String(), nil
}
