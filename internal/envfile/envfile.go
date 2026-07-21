// Package envfile loads KEY=VALUE pairs from a .env file into the process
// environment. It exists so the project can keep its zero-dependency promise
// rather than pulling in a dotenv library for forty lines of parsing.
package envfile

import (
	"bufio"
	"os"
	"strings"
)

// Load reads path and sets any variables not already present in the
// environment. Variables already set win, so an explicit shell value or a
// per-command override always beats the file.
//
// A missing file is not an error: .env is optional, and configuring entirely
// through the shell is a legitimate setup.
func Load(path string) error {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
	for sc.Scan() {
		key, val, ok := parseLine(sc.Text())
		if !ok {
			continue
		}
		if _, exists := os.LookupEnv(key); exists {
			continue
		}
		if err := os.Setenv(key, val); err != nil {
			return err
		}
	}
	return sc.Err()
}

// parseLine extracts a KEY=VALUE pair, tolerating blank lines, # comments, a
// leading `export `, and single- or double-quoted values.
func parseLine(line string) (key, val string, ok bool) {
	line = strings.TrimSpace(line)
	if line == "" || strings.HasPrefix(line, "#") {
		return "", "", false
	}
	line = strings.TrimPrefix(line, "export ")

	key, val, ok = strings.Cut(line, "=")
	if !ok {
		return "", "", false
	}
	key = strings.TrimSpace(key)
	val = strings.TrimSpace(val)
	if key == "" {
		return "", "", false
	}

	// Strip matching surrounding quotes. Unquoted values keep any trailing
	// inline comment, since stripping it would break values containing '#'.
	if len(val) >= 2 {
		if (val[0] == '"' && val[len(val)-1] == '"') || (val[0] == '\'' && val[len(val)-1] == '\'') {
			val = val[1 : len(val)-1]
		}
	}
	return key, val, true
}
