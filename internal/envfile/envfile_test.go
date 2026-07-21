package envfile

import (
	"os"
	"path/filepath"
	"testing"
)

func TestParseLine(t *testing.T) {
	cases := []struct {
		line          string
		wantKey, want string
		wantOK        bool
	}{
		{`LLM_API_KEY=sk-abc123`, "LLM_API_KEY", "sk-abc123", true},
		{`  LLM_BASE_URL = https://x/v1  `, "LLM_BASE_URL", "https://x/v1", true},
		{`export LLM_CHAT_MODEL=gpt-4o`, "LLM_CHAT_MODEL", "gpt-4o", true},
		{`QUOTED="value here"`, "QUOTED", "value here", true},
		{`SINGLE='value'`, "SINGLE", "value", true},
		{`EMPTY=`, "EMPTY", "", true},
		{`# a comment`, "", "", false},
		{``, "", "", false},
		{`   `, "", "", false},
		{`no-equals-sign`, "", "", false},
		{`=novalue`, "", "", false},
	}
	for _, tc := range cases {
		k, v, ok := parseLine(tc.line)
		if ok != tc.wantOK || k != tc.wantKey || v != tc.want {
			t.Errorf("parseLine(%q) = (%q, %q, %v), want (%q, %q, %v)",
				tc.line, k, v, ok, tc.wantKey, tc.want, tc.wantOK)
		}
	}
}

func TestLoadSetsUnsetVariables(t *testing.T) {
	path := filepath.Join(t.TempDir(), ".env")
	content := "# config\nLLM_BASE_URL=https://example.test/v1\nLLM_API_KEY=sk-from-file\n"
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("LLM_BASE_URL", "")
	os.Unsetenv("LLM_BASE_URL")
	os.Unsetenv("LLM_API_KEY")

	if err := Load(path); err != nil {
		t.Fatal(err)
	}
	if got := os.Getenv("LLM_BASE_URL"); got != "https://example.test/v1" {
		t.Errorf("LLM_BASE_URL = %q", got)
	}
	if got := os.Getenv("LLM_API_KEY"); got != "sk-from-file" {
		t.Errorf("LLM_API_KEY = %q", got)
	}
}

// A value already set in the shell must win, so a per-command override works.
func TestLoadDoesNotOverrideExisting(t *testing.T) {
	path := filepath.Join(t.TempDir(), ".env")
	if err := os.WriteFile(path, []byte("LLM_CHAT_MODEL=from-file\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("LLM_CHAT_MODEL", "from-shell")

	if err := Load(path); err != nil {
		t.Fatal(err)
	}
	if got := os.Getenv("LLM_CHAT_MODEL"); got != "from-shell" {
		t.Errorf("shell value should win, got %q", got)
	}
}

// Configuring purely through the shell is legitimate, so a missing file is fine.
func TestLoadMissingFileIsNotAnError(t *testing.T) {
	if err := Load(filepath.Join(t.TempDir(), "nope.env")); err != nil {
		t.Fatalf("missing .env should not error: %v", err)
	}
}
