package pgvector

import (
	"encoding/json"
	"sort"
	"strconv"
	"strings"
)

// encodeVector renders a vector in pgvector's text input format, "[1,2,3]".
// Every driver can send that as a string parameter, which keeps this package
// free of a driver-specific type and therefore free of a dependency.
//
// 'g' emits the shortest representation that round-trips to the same float64,
// so a stored vector reads back bit-identical and a pgvector-backed ranking
// matches an in-memory one exactly.
func encodeVector(v []float64) string {
	var b strings.Builder
	b.Grow(len(v)*12 + 2)
	b.WriteByte('[')
	for i, f := range v {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString(strconv.FormatFloat(f, 'g', -1, 64))
	}
	b.WriteByte(']')
	return b.String()
}

// encodeMeta renders chunk metadata as JSON for the jsonb column. Nil becomes
// an empty object rather than SQL NULL, so `meta ->> key` is a clean miss
// instead of NULL-propagating through the filter comparison.
func encodeMeta(m map[string]string) (string, error) {
	if len(m) == 0 {
		return "{}", nil
	}
	b, err := json.Marshal(m)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

func decodeMeta(b []byte) (map[string]string, error) {
	if len(b) == 0 {
		return nil, nil
	}
	var m map[string]string
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, err
	}
	if len(m) == 0 {
		return nil, nil
	}
	return m, nil
}

func sortStrings(s []string) { sort.Strings(s) }
